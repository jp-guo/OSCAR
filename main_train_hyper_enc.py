import os
import os.path
import argparse
import torch
from torch.utils.data import DataLoader
import transformers
from accelerate import Accelerator
from torchvision import transforms
from tqdm.auto import tqdm
from PIL import Image
import diffusers
from diffusers.optimization import get_scheduler

from accelerate.utils import set_seed
from accelerate import DistributedDataParallelKwargs

from data.dataset import Dataset
from utils import utils_image as util
from utils import utils_option as option
from diffusion.encoder import Encoder

import warnings

warnings.filterwarnings("ignore")

import wandb
from datetime import datetime


def parse_args(input_args=None):
    """
    Parses command-line arguments used for configuring an paired session (pix2pix-Turbo).
    This function sets up an argument parser to handle various training options.

    Returns:
    argparse.Namespace: The parsed command-line arguments.
   """
    parser = argparse.ArgumentParser()

    # training details
    parser.add_argument("--seed", type=int, default=123, help="A seed for reproducible training.")
    parser.add_argument("--num_training_epochs", type=int, default=10000)
    parser.add_argument("--max_train_steps", type=int, default=100000, ) # 100000
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of updates steps to accumulate before performing a backward/update pass.", )
    parser.add_argument("--checkpointing_steps", type=int, default=5000, )
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_scheduler", type=str, default="constant",
                        help=(
                            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
                            ' "constant", "constant_with_warmup"]'
                        ),
                        )
    parser.add_argument("--lr_warmup_steps", type=int, default=500,
                        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1,
                        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
                        )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")

    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--allow_tf32", action="store_true",
                        help=(
                            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
                            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
                        ),
                        )
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"], )
    parser.add_argument("--set_grads_to_none", action="store_true", )

    parser.add_argument("--tracker_project_name", type=str, default="train_oscar",
                        help="The name of the wandb project to log to.")
    parser.add_argument("--pretrained_model_name_or_path", default=None, type=str)

    parser.add_argument('--test_epoch', type=int, default=1)
    # dataset setting
    parser.add_argument("--datasets", default='options/train_hyper_enc.json')

    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--bpp', type=float, required=True)
    parser.add_argument('--hyper_dim', type=int, default=320, help='output dim of hyper encoder')

    parser.add_argument('--val_path', type=str, default=None)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def main(args):
    args.tracker_project_name = os.path.join("encoder_training_results", args.tracker_project_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        kwargs_handlers=[ddp_kwargs],
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.tracker_project_name, "checkpoints"), exist_ok=True)
        if not args.debug:
            wandb.init(project="diff-compress", name=args.tracker_project_name)

    model_gen = Encoder(args)
    model_gen.set_train()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # make the optimizer
    layers_to_opt = model_gen.get_trainable_params()

    optimizer = torch.optim.AdamW(layers_to_opt, lr=args.learning_rate,
                                  betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                                  eps=args.adam_epsilon, )
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
                                 num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                                 num_training_steps=args.max_train_steps,
                                 num_cycles=args.lr_num_cycles, power=args.lr_power, )

    # dataset_type = args.dataset_type
    dataset_opt = option.parse_dataset(args.datasets)['datasets']
    train_set = Dataset(dataset_opt)
    dl_train = DataLoader(train_set,
                          batch_size=dataset_opt['dataloader_batch_size'],
                          shuffle=dataset_opt['dataloader_shuffle'],
                          num_workers=dataset_opt['dataloader_num_workers'],
                          drop_last=True)

    # Prepare everything with our `accelerator`.
    model_gen, optimizer, dl_train, lr_scheduler= accelerator.prepare(
        model_gen, optimizer, dl_train, lr_scheduler
    )

    if accelerator.is_main_process:
        del args.datasets
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    progress_bar = tqdm(range(0, args.max_train_steps), initial=0, desc="Steps",
                        disable=not accelerator.is_local_main_process, total=args.max_train_steps)

    # start the training loop
    global_step = 0
    for epoch in range(0, args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            m_acc = [model_gen]
            with accelerator.accumulate(*m_acc):
                x_tgt = batch["H"].to("cuda")

                z_repa, z = model_gen(x_tgt)

                z = torch.nn.functional.normalize(z, dim=-1)
                z_repa = torch.nn.functional.normalize(z_repa, dim=-1)
                loss = -(z * z_repa).sum(-1).mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    logs = {}
                    # log all the losses
                    logs["loss"] = loss.detach().item()
                    progress_bar.set_postfix(**logs)
                    if not args.debug:
                        wandb.log({'epoch': epoch, 'repa_loss': logs["loss"]})
                    # checkpoint the model
                    save_dir = os.path.join(args.tracker_project_name, "checkpoints")

                    accelerator.log(logs, step=global_step)

                    if global_step == args.max_train_steps:
                        accelerator.unwrap_model(model_gen).save_model(os.path.join(save_dir, "encoder.pkl"))
                        exit()

                    if global_step % args.checkpointing_steps == 0:
                        outf = os.path.join(save_dir, f"model_{global_step}.pkl")
                        accelerator.unwrap_model(model_gen).save_model(outf)

                        if torch.cuda.device_count() > 1:
                            model_gen.module.set_eval()
                        else:
                            model_gen.set_eval()

                        H_paths = util.get_image_paths(args.val_path)
                        avg_loss = 0
                        for idx, img in tqdm(enumerate(H_paths)):
                            img_H = Image.open(img).convert('RGB')
                            # num_pixels = img_H.width * img_H.height
                            tensor_transforms = transforms.Compose([
                                transforms.ToTensor(),
                            ])
                            img = img_H.copy()
                            img = tensor_transforms(img).unsqueeze(0).to("cuda")
                            with torch.no_grad():
                                img = img * 2 - 1
                                if torch.cuda.device_count() > 1:
                                    z_repa, z = model_gen.module(img)
                                else:
                                    z_repa, z = model_gen(img)
                                z = torch.nn.functional.normalize(z, dim=-1)
                                z_repa = torch.nn.functional.normalize(z_repa, dim=-1)
                                loss = -(z * z_repa).sum(-1).mean()
                            avg_loss += loss.detach().item() / len(H_paths)
                        if not args.debug:
                            wandb.log({'val_loss': avg_loss})
                        if torch.cuda.device_count() > 1:
                            model_gen.module.set_train()
                        else:
                            model_gen.set_train()

if __name__ == "__main__":
    args = parse_args()
    main(args)
