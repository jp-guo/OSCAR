import os
import os.path
import random
import argparse
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.distributed as dist
from torch.utils.data import DataLoader
import transformers
from accelerate import Accelerator
from torchvision import transforms
from tqdm.auto import tqdm
import numpy as np
from PIL import Image
from collections import OrderedDict
import pyiqa
import diffusers
from diffusers.utils.import_utils import is_xformers_available
from transformers import get_constant_schedule_with_warmup
from diffusion.models.discriminator import Discriminator
from accelerate.utils import set_seed
from accelerate import DistributedDataParallelKwargs
import wandb
from datetime import datetime

from data.dataset import Dataset
from utils import utils_image as util
from utils import utils_option as option
from diffusion.oscar import OSCAR

import warnings

warnings.filterwarnings("ignore")


def is_dist_on():
    return dist.is_available() and dist.is_initialized()

def cleanup_ddp():
    if is_dist_on():
        torch.cuda.synchronize()
        dist.barrier()
        dist.destroy_process_group()


class EdgeDetectionModel(nn.Module):
    def __init__(self):
        super(EdgeDetectionModel, self).__init__()
        # Sobel filters for edge detection
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)

        sobel_x_kernel = torch.tensor([[-1., 0., 1.],
                                       [-2., 0., 2.],
                                       [-1., 0., 1.]])
        sobel_y_kernel = torch.tensor([[-1., -2., -1.],
                                       [0., 0., 0.],
                                       [1., 2., 1.]])

        self.sobel_x.weight = nn.Parameter(sobel_x_kernel.view(1, 1, 3, 3))
        self.sobel_y.weight = nn.Parameter(sobel_y_kernel.view(1, 1, 3, 3))
        self.sobel_x.weight.requires_grad = False
        self.sobel_y.weight.requires_grad = False

    def forward(self, x):
        # Convert to grayscale if needed
        if x.shape[1] == 3:
            x = transforms.Grayscale()(x)

        # Apply Sobel filters
        edge_x = self.sobel_x(x)
        edge_y = self.sobel_y(x)

        # Calculate gradient magnitude (edge detection result)
        edges = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)

        return edges


def parse_float_list(arg):
    try:
        return [float(x) for x in arg.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError("List elements should be floats")


def parse_int_list(arg):
    try:
        return [int(x) for x in arg.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError("List elements should be integers")


def parse_str_list(arg):
    return arg.split(',')


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
    parser.add_argument("--max_train_steps", type=int, default=100000, ) # 100000
    parser.add_argument("--checkpointing_steps", type=int, default=5000, )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of updates steps to accumulate before performing a backward/update pass.", )
    parser.add_argument("--gradient_checkpointing", action="store_true", )
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
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true",
                        help="Whether or not to use xformers.")
    parser.add_argument("--set_grads_to_none", action="store_true", )

    parser.add_argument("--tracker_project_name", type=str, default="train_oscar",
                        help="The name of the wandb project to log to.")
    parser.add_argument("--pretrained_model_name_or_path", default=None, type=str)

    parser.add_argument('--test_epoch', type=int, default=1)
    # lora setting
    parser.add_argument("--lora_rank", default=16, type=int)
    # dataset setting
    parser.add_argument("--datasets", default='options/train_diff.json')

    parser.add_argument('--val_path', required=True)
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default='adain')

    parser.add_argument('--train_decoder', action='store_true')

    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--aux_loss', default='dists')

    parser.add_argument('--edge_loss', action='store_true')
    parser.add_argument('--gan_loss', action='store_true')

    parser.add_argument('--gan_dis_weight', default=1e-2, type=float)
    parser.add_argument('--gan_gen_weight', default=5e-3, type=float)

    parser.add_argument('--hyper_dim', type=int, default=320, help='output dim of hyper encoder')
    parser.add_argument('--latents_loss_weight', type=float, default=2)

    parser.add_argument('--enc_paths', nargs="+")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def main(args):
    args.tracker_project_name = os.path.join("training_results", args.tracker_project_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        kwargs_handlers=[ddp_kwargs]
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

    model_gen = OSCAR(args)
    tot_bpp = len(model_gen.timesteps)
    if args.enc_paths is not None:
        model_gen.load_hyper_encoders(args.enc_paths)
    model_gen.set_train()

    if args.gan_loss:
        model_reg = Discriminator(args=args, accelerator=accelerator)
        model_reg.set_train()

    loss_fn = pyiqa.create_metric(args.aux_loss, device=accelerator.device, as_loss=True)
    edge_detection_model = EdgeDetectionModel()
    edge_detection_model.requires_grad_(False)
    edge_detection_model.to("cuda")

    model_gen.unet.set_adapter(['default_encoder', 'default_decoder', 'default_others'])

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            model_gen.unet.enable_xformers_memory_efficient_attention()
            if args.gan_loss:
                model_reg.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if args.gradient_checkpointing:
        model_gen.unet.enable_gradient_checkpointing()
        model_reg.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # make the optimizer
    layers_to_opt = model_gen.get_trainable_params()

    optimizer = torch.optim.AdamW(layers_to_opt, lr=args.learning_rate,
                                  betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                                  eps=args.adam_epsilon, )
    lr_scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=10 * accelerator.num_processes
    )

    if args.gan_loss:
        layers_to_opt_reg = model_reg.get_trainable_params()
        optimizer_reg = torch.optim.AdamW(layers_to_opt_reg, lr=args.learning_rate,
                                          betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                                          eps=args.adam_epsilon, )
        lr_scheduler_reg = get_constant_schedule_with_warmup(
            optimizer_reg,
            num_warmup_steps=10 * accelerator.num_processes
        )

    dataset_opt = option.parse_dataset(args.datasets)['datasets']
    train_set = Dataset(dataset_opt)
    dl_train = DataLoader(train_set,
                          batch_size=dataset_opt['dataloader_batch_size'],
                          shuffle=dataset_opt['dataloader_shuffle'],
                          num_workers=dataset_opt['dataloader_num_workers'],
                          drop_last=True)

    if args.gan_loss:
        model_gen, model_reg, optimizer, optimizer_reg, dl_train, lr_scheduler, lr_scheduler_reg = accelerator.prepare(
            model_gen, model_reg, optimizer, optimizer_reg, dl_train, lr_scheduler, lr_scheduler_reg
        )
    else:
        model_gen, optimizer, dl_train, lr_scheduler= accelerator.prepare(
            model_gen, optimizer, dl_train, lr_scheduler
        )

    if accelerator.is_main_process:
        del args.datasets
        del args.enc_paths
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    progress_bar = tqdm(range(0, args.max_train_steps), initial=0, desc="Steps",
                        disable=not accelerator.is_local_main_process, total=args.max_train_steps)

    # start the training loop
    global_step = 0
    accelerator.wait_for_everyone()
    for r in range(accelerator.num_processes):
        if accelerator.process_index == r:
            print(f"rank={accelerator.process_index}, local_rank={accelerator.local_process_index}")
        accelerator.wait_for_everyone()

    while True:
        for step, batch in enumerate(dl_train):
            global_step += 1
            if global_step > args.max_train_steps:
                if int(os.getenv("RANK", "0")) == 0:
                    wandb.finish()
                cleanup_ddp()
                exit()
            if args.gan_loss:
                m_acc = [model_gen, model_reg]
            else:
                m_acc = [model_gen]
            with accelerator.accumulate(*m_acc):
                x_tgt = batch["H"].to("cuda")

                idx = random.randint(0, tot_bpp - 1)
                x_pred, latents_pred, latents_loss = model_gen(x_tgt, idx)
                x_pred = x_pred * 0.5 + 0.5
                x_tgt = x_tgt * 0.5 + 0.5
                aux_loss = loss_fn(x_pred.float(), x_tgt.float())
                edge_loss = loss_fn(
                    edge_detection_model(x_pred),
                    edge_detection_model(x_tgt.float())
                )
                lp_loss = F.l1_loss(x_pred.float(), x_tgt.float(), reduction="mean")

                if args.gan_loss:
                    if torch.cuda.device_count() > 1:
                        generator_loss = model_reg.module.compute_generator_loss(latents_pred)
                    else:
                        generator_loss = model_reg.compute_generator_loss(latents_pred)
                    loss = lp_loss + aux_loss + generator_loss * args.gan_gen_weight
                else:
                    loss = lp_loss + aux_loss

                if args.edge_loss:
                    loss = loss + edge_loss

                loss = loss + args.latents_loss_weight * latents_loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)
                """
                diff loss: let lora model closed to generator
                """
                if args.gan_loss:
                    x_tgt = 2 * x_tgt - 1
                    if torch.cuda.device_count() > 1:
                        gt_latents = model_reg.module.compute_gt_latents(x_tgt)
                        loss_d = model_reg.module.compute_discriminator_loss(gt_latents, latents_pred) * args.gan_dis_weight
                    else:
                        gt_latents = model_reg.compute_gt_latents(x_tgt)
                        loss_d = model_reg.compute_discriminator_loss(gt_latents, latents_pred) * args.gan_dis_weight
                    accelerator.backward(loss_d)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model_reg.parameters(), args.max_grad_norm)
                    optimizer_reg.step()
                    lr_scheduler_reg.step()
                    optimizer_reg.zero_grad(set_to_none=args.set_grads_to_none)

            if accelerator.is_main_process:
                progress_bar.update(1)
                logs = {}
                # log all the losses
                logs[f"lp_loss/{idx}"] = lp_loss.detach().item()
                logs[f"{args.aux_loss}_loss/{idx}"] = aux_loss.detach().item()
                logs[f"edge_loss/{idx}"] = edge_loss.detach().item()
                logs[f"latents_loss/{idx}"] = latents_loss.detach().item()
                if args.gan_loss:
                    logs[f"loss_d/{idx}"] = loss_d.detach().item()
                    logs[f"loss_g/{idx}"] = generator_loss.detach().item()
                progress_bar.set_postfix(**logs)
                if not args.debug:
                    if args.gan_loss:
                        wandb.log({f'lp_loss/{idx}': logs[f"lp_loss/{idx}"],
                                   f"{args.aux_loss}_loss/{idx}": logs[f"{args.aux_loss}_loss/{idx}"],
                                   f'loss_d/{idx}': logs[f'loss_d/{idx}'], f'loss_g/{idx}': logs[f'loss_g/{idx}'],
                                   f"latents_loss/{idx}": logs[f"latents_loss/{idx}"], "lr": optimizer.param_groups[0]["lr"]})
                    else:
                        wandb.log({f'lp_loss/{idx}': logs[f"lp_loss/{idx}"],
                                   f"{args.aux_loss}_loss/{idx}": logs[f"{args.aux_loss}_loss/{idx}"],
                                   f'edge_loss/{idx}': logs[f"edge_loss/{idx}"],
                                   f"latents_loss/{idx}": logs[f"latents_loss/{idx}"], "lr": optimizer.param_groups[0]["lr"]})
                # checkpoint the model
                save_dir = os.path.join(args.tracker_project_name, "checkpoints")

                if global_step % args.checkpointing_steps == 0:
                    outf = os.path.join(save_dir, f"model_{global_step}.pkl")
                    accelerator.unwrap_model(model_gen).save_model(outf)

                    psnr_metric = pyiqa.create_metric('psnr', device="cuda")
                    msssim_metric = pyiqa.create_metric('ms_ssim', device="cuda")
                    lpips_metric = pyiqa.create_metric('lpips-vgg', device="cuda")
                    dists_metric = pyiqa.create_metric('dists', device="cuda")
                    musiq_metric = pyiqa.create_metric('musiq', device="cuda")
                    clipiqa_metric = pyiqa.create_metric('clipiqa+', device="cuda")

                    if torch.cuda.device_count() > 1:
                        model_gen.module.set_eval()
                    else:
                        model_gen.set_eval()

                    H_paths = util.get_image_paths(args.val_path)

                    for idx in range(0, tot_bpp):
                        test_results = OrderedDict()
                        test_results['psnr'] = []
                        test_results['msssim'] = []
                        test_results['lpips'] = []
                        test_results['dists'] = []
                        test_results['musiq'] = []
                        test_results['clipiqa'] = []

                        for _, img in tqdm(enumerate(H_paths)):
                            img_name, ext = os.path.splitext(os.path.basename(img))

                            img_H = Image.open(img).convert('RGB')

                            # vae can only process images with height and width multiples of 8
                            new_width = img_H.width - img_H.width % 8
                            new_height = img_H.height - img_H.height % 8
                            img_H = img_H.resize((new_width, new_height), Image.LANCZOS)

                            tensor_transforms = transforms.Compose([
                                transforms.ToTensor(),
                            ])
                            img = img_H.copy()
                            img = tensor_transforms(img).unsqueeze(0).to("cuda")
                            with torch.no_grad():
                                img = img * 2 - 1
                                if torch.cuda.device_count() > 1:
                                    img_E, _, _, = model_gen.module(img, idx)
                                else:
                                    img_E, _, _ = model_gen(img, idx)
                                img_E = transforms.ToPILImage()(img_E[0].cpu() * 0.5 + 0.5)
                            img_H = np.array(img_H)
                            img_E = np.array(img_E)

                            util.imsave(img_E, os.path.join(args.tracker_project_name, img_name + '.png'))

                            img_E, img_H = img_E / 255., img_H / 255.
                            img_E, img_H = torch.tensor(img_E, device="cuda").permute(2, 0, 1).unsqueeze(
                                0), torch.tensor(img_H,
                                                 device="cuda").permute(
                                2, 0, 1).unsqueeze(0)
                            img_E, img_H = img_E.type(torch.float32), img_H.type(torch.float32)

                            psnr = psnr_metric(img_E, img_H)
                            msssim = msssim_metric(img_E, img_H)
                            lpips_score = lpips_metric(img_E, img_H)
                            dists = dists_metric(img_E, img_H)
                            musiq = musiq_metric(img_E, img_H)
                            clipiqa = clipiqa_metric(img_E, img_H)

                            test_results['psnr'].append(psnr)
                            test_results['msssim'].append(msssim)
                            test_results['lpips'].append(lpips_score.item())
                            test_results['dists'].append(dists.item())
                            test_results['musiq'].append(musiq.item())
                            test_results['clipiqa'].append(clipiqa.item())

                        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
                        ave_msssim = sum(test_results['msssim']) / len(test_results['msssim'])
                        avg_lpips = sum(test_results['lpips']) / len(test_results['lpips'])
                        avg_dists = sum(test_results['dists']) / len(test_results['dists'])
                        avg_musiq = sum(test_results['musiq']) / len(test_results['musiq'])
                        avg_clipiqa = sum(test_results['clipiqa']) / len(test_results['clipiqa'])

                        if not args.debug:
                            wandb.log(
                                {f'PSNR/{idx}': ave_psnr, f'MS-SSIM/{idx}': ave_msssim, f'LPIPS/{idx}': avg_lpips,
                                 f'DISTS/{idx}': avg_dists, f'MUSIQ/{idx}': avg_musiq,
                                 f'CLIPIQA/{idx}': avg_clipiqa})

                    if torch.cuda.device_count() > 1:
                        model_gen.module.set_train()
                    else:
                        model_gen.set_train()

            accelerator.wait_for_everyone()

if __name__ == "__main__":
    args = parse_args()
    main(args)
