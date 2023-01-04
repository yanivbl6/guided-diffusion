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
                    if "model" in chk_f:
                        split1 = chk_f.split("model")[-1].split(".")[0]
                        try:
                            if int(split1) > most:
                                most = int(split1)
                                resume_checkpoint = folder_name  + "/" + chk_f
                        except:
                            pass
                        
                        

    wandb_id = None
    if gpu == 0 and resume_checkpoint != "":
        logger.log(f"Found existing log file: {resume_checkpoint}")

        try:
            with open(f"{folder_name}/wandb_id.txt", "rb") as f:
                wandb_id = torch.load(f)
        except:
            pass


    if not args.misc and args.gpu == 0:

        if wandb_id is None:
            wandb.init(project="dropoutDiffusion", entity="dl-projects", config=args)
            wandb.run.name = args.name
            wandb.run.save()
            with open(f"{folder_name}/wandb_id.txt", "wb") as f:
                torch.save(wandb.run.id ,f)

        else:
            wandb.init(project="dropoutDiffusion", entity="dl-projects", config=args, id=wandb_id, resume="must")


    if not args.misc:
        wandb.Table.MAX_ROWS = args.num_samples *  ngpus_per_node


    args.dropout_args = {"conv_op_dropout": args.conv_op_dropout,
                         "conv_op_dropout_max": args.conv_op_dropout_max,
                         "conv_op_dropout_type": args.conv_op_dropout_type}





    logger.log("creating model and diffusion...")



    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )


    print("resume checkpoint: ", resume_checkpoint)


    model.to(dist_util.dev())


    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
    schedule_sampler2 = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")




    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        start = args.num_eval,
    )

    data_eval = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        end = args.num_eval,
    )

    logger.log("training...")
    trainer = TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        weight_schedule=args.weight_schedule,
        log = (gpu == -1),
        folder_name = folder_name

    )

    evaluator = TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data_eval,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint="",
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler2,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        weight_schedule=args.weight_schedule,
        log = False,
        folder_name = "",
    )

    
    
    eval_every = args.eval_every
    generate_every = args.generate_every
    assert(generate_every % eval_every == 0)

    evals_to_generate = generate_every // eval_every

    cont = True
    Ts = [50,100,200,400,600,800]

    steps = 0

    while cont:

        model.train()
        cont = trainer.run_loop_n(eval_every)
        steps += eval_every

        model.eval()
    
        if steps % generate_every == 0:
            start_sample = time()
            sample(model, diffusion, args, step = steps, gpu = gpu)
            sample_time = time() - start_sample
            if gpu==0:
                logger.log(f"Sample Time: {sample_time}")


        results = evaluator.evaluate(Ts, int(args.num_eval / ngpus_per_node), ngpus_per_node)

        if steps % generate_every == 0:

            if gpu == 0:
                ##clear GPU memory


                ##calculate FID
                torch.cuda.empty_cache()
                
                imgs_dir = f"{args.save_dir}/samples"

                fid = calculate_fid_given_paths([imgs_dir, args.data_dir], 16, torch.cuda.current_device(), dims = 2048)
                print("FID: ", fid)
                results["FID"] = fid


        if wandb.run is not None and gpu == 0:
            wandb.log(results)

        dist.barrier()



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

def sample(model,diffusion,args, step, gpu):

    imgs_dir = f"{args.save_dir}/samples"

    if gpu == 0:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

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

    ngpus_per_node = torch.cuda.device_count()
    to_sample = (args.num_samples // ngpus_per_node) + (1 if args.num_samples % ngpus_per_node > gpu else 0)

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
        
    )

    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
