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

    args.distributed = True
    ngpus_per_node = torch.cuda.device_count()

    if not os.path.exists("./checkpoints/"):
        os.makedirs("./checkpoints/")


    print("Number of GPUs: ", ngpus_per_node)
    if args.distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))
    else:
        main_worker(0, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):

    ##dist_util.setup_dist()
    ##logger.configure()

    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        
        args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend="nccl", init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank,
                                timeout=datetime.timedelta(minutes=1))

        if not args.misc and args.rank == 0:
            wandb.init(project="dropoutDiffusion", entity="dl-projects", config=args)
    else:
        wandb.init(project="dropoutDiffusion", entity="dl-projects", config=args)

    if not args.misc:
        wandb.Table.MAX_ROWS = args.num_samples *  ngpus_per_node


    args.dropout_args = {"conv_op_dropout": args.conv_op_dropout,
                         "conv_op_dropout_max": args.conv_op_dropout_max,
                         "conv_op_dropout_type": args.conv_op_dropout_type}

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    if gpu==0:
        print(model)



    if args.distributed:

        ##classifier_free = model.classifier_free
        num_classes = model.num_classes


        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            # args.workers = int(
            #     (args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu])
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)


        ##model.classifier_free = classifier_free
        model.num_classes = num_classes

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()


    ##model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    


    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        distributed = args.distributed,
        start = args.num_eval,
    )

    data_eval = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        distributed = args.distributed,
        end = args.num_eval,
    )

    # for batch, cond in data_eval:
    #     batch = batch.cuda()
    #     print(batch.cpu().sum(1).sum(1).sum(1))
    #     dist.all_reduce(batch, op=dist.ReduceOp.SUM)
    #     print(batch.cpu().sum(1).sum(1).sum(1))

    #     print("batch of eval data: ", batch.shape)

    # for batch, cond in tqdm(data, desc = f"Iterating over training data"):
    #     pass

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
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        weight_schedule=args.weight_schedule,
        log = (gpu == -1)

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
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        weight_schedule=args.weight_schedule,
        log = False
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

        # if gpu == 0:
        #     print(results)






        if steps % generate_every == 0:

            if gpu == 0:
                ##clear GPU memory

                ##model = model.cpu()


                ## save the model to CPU 
                if wandb.run is not None:
                    filename = f"./checkpoints/{wandb.run.name}.pt"
                else:
                    filename = f"./checkpoints/checkpoint.pt"

                torch.save({
                            'step': steps,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': trainer.opt.state_dict(),
                            }, filename)

                ##calculate FID

                torch.cuda.empty_cache()

                fid = calculate_fid_given_paths(["samples_P4_misc", args.data_dir], 16, torch.cuda.current_device(), dims = 2048)
                print("FID: ", fid)
                results["FID"] = fid

                ##torch.cuda.empty_cache()

                ##load the model back to GPU
                ##model = model.cuda()


                


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

    args.save_dir = f"samples_P4_misc"


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

    out_path = os.path.join(args.save_dir, f"sample_")

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
        # sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        # sample = sample.permute(0, 2, 3, 1)
        # sample = sample.contiguous()

        ##all_images.extend(sample.cpu().numpy())

        # gathered_samples = [torch.zeros_like(sample) for _ in range(dist.get_world_size())]
        # dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        # all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        # if args.class_cond:
        #     gathered_labels = [
        #         torch.zeros_like(classes) for _ in range(dist.get_world_size())
        #     ]
        #     dist.all_gather(gathered_labels, classes)
        #     all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        # logger.log(f"created {len(all_images) * args.batch_size} samples")

        if count + sample.size(0) > to_sample:
            sample = sample[: to_sample - count]

        ##print(f"sample size {sample.size()}")
        save_images(sample.cpu(), out_path, gpu, start =  count)
        count  = count + sample.size(0)

    dist.barrier()
    logger.log("sampling complete")



def create_argparser():
    defaults = dict(
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
        save_dir="",
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
