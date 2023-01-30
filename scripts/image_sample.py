"""
Train a diffusion model on images.
"""

import argparse
import torch
from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop
import torch.multiprocessing as mp
import torch.distributed as dist

import datetime
import wandb
import numpy as np
import os 

import matplotlib.pyplot as plt

from tqdm import tqdm
from  time import time

from pytorch_fid.fid_score import calculate_fid_given_paths
import torchvision

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    print("logger dir: ",logger.get_dir())
    
    args.distributed = True

    ngpus_per_node = int(os.environ["WORLD_SIZE"])
    args.gpu = int(os.environ["RANK"])
    gpu = args.gpu

    print("My rank: ", gpu, ", world size: ", ngpus_per_node)
    logger.log("Looking for existing log file...")
    folder_name = f"{args.save_dir}/{args.name}"
    resume_checkpoint = ""
    if os.path.exists(args.save_dir):
        if os.path.exists(folder_name):
            most = -1
            for chk_f in os.listdir(folder_name):
                if chk_f.endswith(".pt"):
                    if "ema" in chk_f:
                        split1 = chk_f.split("_")[-1].split(".")[0]
                        try:
                            if int(split1) > most:
                                most = int(split1)
                                resume_checkpoint = folder_name  + "/" + chk_f
                        except:
                            pass
        else:
            raise Exception(f"folder {folder_name} does not exist")

    if resume_checkpoint == "":
        raise Exception(f"Checkpoint not found in {folder_name}")


    dist.barrier()


    args.dropout_args = {"conv_op_dropout": args.conv_op_dropout,
                         "conv_op_dropout_max": args.conv_op_dropout_max,
                         "conv_op_dropout_type": args.conv_op_dropout_type}


    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    print("resume checkpoint: ", resume_checkpoint)

    model.load_state_dict(
        dist_util.load_state_dict(resume_checkpoint, map_location="cpu")
    )

    model.to(dist_util.dev())
    

    if args.inference_drop:
        sampling_name = f"samples_{args.conv_op_dropout_type}_{args.conv_op_dropout_max}_{args.conv_op_dropout}"
        model.train()
    else:
        sampling_name = f"samples_deterministic"
        model.eval()

    start_sample = time()
    sample(model, diffusion, args, sampling_name, gpu = gpu, ngpus_per_node = ngpus_per_node)
    sample_time = time() - start_sample
    if gpu==0:
        logger.log(f"Sample Time: {sample_time}")



    if gpu == 0:
        ##clear GPU memory
        torch.cuda.empty_cache()

        imgs_dir = f"{args.save_dir}/{args.name}/{sampling_name}"

        ##calculate FID
        fid = calculate_fid_given_paths([imgs_dir, args.data_dir], 16, torch.cuda.current_device(), dims = 2048)
        print("FID: ", fid)

    dist.barrier()


# def rename_samples(imgs_dir, num_samples, num_gpus):

#     for f in os.listdir(imgs_dir):
        
#         S = f.split(".")[0].split("_")

#         gpu = int(S[-1])
#         idx = int(S[-2])

#         if idx >= num_sample:
#             new_idx = idx % num_sample
#             new_gpu = gpu * (idx // num_sample)
#             new_name = "sample_" + f"_{new_idx}_{new_gpu}.jpg"
#             if not os.path.exists(imgs_dir + "/" + new_name):
#                 os.rename(imgs_dir + "/" + f, imgs_dir + "/" + new_name)
#         elif gpu >= num_gpus:
#             new_idx = idx +  (gpu * num_sample)
#             new_gpu = gpu % num_gpus
#             new_name = "sample_" + f"_{new_idx}_{new_gpu}.jpg"
#             if not os.path.exists(imgs_dir + "/" + new_name):
#                 os.rename(imgs_dir + "/" + f, imgs_dir + "/" + new_name)


def check_images(num_images, figure_path, gpu = -1, start = 0, cutoff = -1):

    for i in range(num_images):
        if not os.path.exists(figure_path + f"_{i+start}_{gpu}.jpg"):
            return False
        if cutoff > 0:
            if os.path.exists(figure_path + f"_{i+start+ cutoff*gpu}_{0}.jpg"):
                return False

    return True

def save_images(images, figure_path, gpu = -1, start = 0):


    imgs = []
    for i in range(images.shape[0]):
        torchvision.utils.save_image(images[i], figure_path + f"_{i+start}_{gpu}.jpg")



    if start == 0 and gpu == 0 and  wandb.run is not None:
        images = ((images + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        images = images.permute(0, 2, 3, 1)
        images = images.contiguous().cpu().numpy()

        for i in range(images.shape[0]):
            imgs.append(wandb.Image(images[i], caption=f"image_{i}"))

    if gpu == 0 and  wandb.run is not None and len(imgs) > 0:
        wandb.log({"Samples": imgs}, commit=False)

    # print(f"saved image samples at {figure_path}")

def sample(model,diffusion,args, sampling_name, gpu, ngpus_per_node = 1):

    ##imgs_dir = f"{args.save_dir}/samples"
    imgs_dir = f"{args.save_dir}/{args.name}/{sampling_name}"


    if gpu == 0:
        if not os.path.exists(imgs_dir):
            os.makedirs(imgs_dir)

    dist.barrier()


    if model.num_classes and args.guidance_scale != 1.0:
        model_fns = [diffusion.make_classifier_free_fn(model, args.guidance_scale)]

        def denoised_fn(x0):
            s = torch.quantile(torch.abs(x0).reshape([x0.shape[0], -1]), 0.995, dim=-1, interpolation='nearest')
            s = torch.maximum(s, torch.ones_like(s))
            s = s[:, None, None, None]
            x0 = x0.clamp(-s, s) / s
            return x0    
    else:
        model_fns = model
        denoised_fn = None

    logger.log("sampling...")
    all_images = []
    all_labels = []

    out_path = os.path.join(imgs_dir, f"sample_")

    count = 0

    to_sample = (args.num_samples // ngpus_per_node) + (1 if args.num_samples % ngpus_per_node > gpu else 0)

    cutoff = (args.num_samples // ngpus_per_node) + (1 if args.num_samples % ngpus_per_node > 0 else 0)

    num_interations = (to_sample+args.batch_size -1 ) // args.batch_size

    bar = tqdm(range(num_interations), total=num_interations, desc = "Sampling") if gpu == 0 else range(num_interations)

    count = 0
    for j in bar:
        model_kwargs = {}
        if args.class_cond:
            classes = torch.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes

        if check_images(args.batch_size, out_path, gpu, start = count, cutoff = cutoff):
            count = count + args.batch_size
            continue

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model_fns,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            denoised_fn=denoised_fn,
            device=dist_util.dev()
        )

        if count + sample.size(0) > to_sample:
            sample = sample[: to_sample - count]

        ##print(f"sample size {sample.size()}")
        save_images(sample.cpu(), out_path, gpu, start =  count)
        count  = count + sample.size(0)

    dist.barrier()
    logger.log("sampling complete")



def create_argparser():
    defaults = dict(
        name="MISC",
        data_dir="",

        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        weight_schedule="sqrt_snr",
        
        
        dist_url='tcp://224.66.41.62:23456',
        world_size = 1,
        misc = False,
        rank = 0,
        num_samples = 4,
        num_eval = 1000,
        clip_denoised=True,
        use_ddim=False,
        guidance_scale=1.0,
        save_dir="./checkpoints",
        ##figdims="4,4",
        ##figscale="5",
        generate_every = 5000,
        eval_every = 100000,

        # loss args
        predict_xstart=False,

        # model args
        conv_op_dropout=0.0,
        conv_op_dropout_max=1.0,
        conv_op_dropout_type=0,

        inference_drop = True,

    )

    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
