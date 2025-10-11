import os
import sys

# sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'diffusion'))
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler
from models.autoencoder_kl import AutoencoderKL
from models.unet_2d_condition import UNet2DConditionModel
from peft import LoraConfig


def initialize_unet(args):
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    unet.requires_grad_(False)
    unet.train()
    lora_target_modules = [
        "to_q", "to_k", "to_v", "to_out.0",
        "proj_in", "proj_out",
        "ff.net.0.proj", "ff.net.2",
        "conv1", "conv2", "conv_shortcut",
        "downsamplers.0.conv", "upsamplers.0.conv",
        "time_emb_proj",
    ]
    lora_config = LoraConfig(
        r=args.lora_rank,
        target_modules=lora_target_modules,
    )
    unet.add_adapter(lora_config)
    return unet, lora_target_modules


class Discriminator(torch.nn.Module):
    def __init__(self, args, accelerator):
        super().__init__()
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self.args = args

        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        self.weight_dtype = weight_dtype

        self.unet, self.lora_target_modules = initialize_unet(args)

        self.vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
        self.vae.requires_grad_(False)

        # self.text_encoder.to(accelerator.device, dtype=weight_dtype)
        self.unet.to(accelerator.device)
        self.vae.to(accelerator.device)

        del self.unet.up_blocks
        del self.unet.conv_out
        del self.vae.decoder

        self.cls_pred_branch = nn.Sequential(
            nn.Conv2d(kernel_size=4, in_channels=1280, out_channels=2560, stride=1, padding=1),  # 8x8 -> 8x8
            nn.GroupNorm(num_groups=32, num_channels=2560),
            nn.SiLU(),
            nn.Conv2d(kernel_size=4, in_channels=2560, out_channels=1, stride=1, padding=1),  # 8x8 -> 8x8
        )
        self.cls_pred_branch.requires_grad_(True)

        self.embeddings = nn.Parameter(torch.zeros(77, 1024, device="cuda"), requires_grad=True)
        self.embeddings.data.normal_(mean=0.0, std=0.02)

    def set_train(self):
        self.unet.train()
        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.cls_pred_branch.requires_grad_(True)
        # self.embeddings.requires_grad_(True)

    def set_eval(self):
        self.unet.eval()
        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = False
        self.cls_pred_branch.requires_grad_(False)
        # self.embeddings.requires_grad_(False)

    def get_trainable_params(self):
        layers_to_opt = []
        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                layers_to_opt.append(_p)
        for _p in self.cls_pred_branch.parameters():
            layers_to_opt.append(_p)
        layers_to_opt += list(self.unet.conv_in.parameters())
        layers_to_opt.append(self.embeddings)

        return layers_to_opt

    def compute_cls_logits(self, latents):
        bsz = latents.shape[0]
        timesteps = torch.randint(20, 980, (bsz,), device=latents.device).long()
        noise = torch.randn_like(latents)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        bz = len(noisy_latents)
        # with torch.no_grad():
        #     rep = self.unet(
        #         noisy_latents,
        #         timestep=timesteps,
        #         encoder_hidden_states=prompt_embeds.float(),
        #         as_discriminator=True
        #     )
        with torch.no_grad():
            rep = self.unet(
                noisy_latents.to(torch.float32),
                timestep=timesteps,
                encoder_hidden_states=self.embeddings.unsqueeze(0).repeat(bz, 1, 1).to(torch.float32),
                as_discriminator=True
            )
        logits = self.cls_pred_branch(rep)

        return logits

    def compute_generator_loss(self, latents):
        self.set_eval()
        logits = self.compute_cls_logits(latents)

        return F.softplus(-logits).mean()

    def compute_discriminator_loss(self, gt_latents, latents):
        self.set_train()
        pred_realism_on_real = self.compute_cls_logits(gt_latents.detach())
        pred_realism_on_fake = self.compute_cls_logits(latents.detach())

        criterion = torch.nn.BCEWithLogitsLoss()
        real_loss = criterion(pred_realism_on_real, torch.ones_like(pred_realism_on_real))
        fake_loss = criterion(pred_realism_on_fake, torch.zeros_like(pred_realism_on_fake))
        loss = (real_loss + fake_loss) / 2

        return loss

    def compute_gt_latents(self, lq):
        with torch.no_grad():
            return self.vae.encode(lq).latent_dist.sample() * self.vae.config.scaling_factor
