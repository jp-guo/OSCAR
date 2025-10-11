import os
import sys

sys.path.append(os.path.join(os.getcwd(), 'diffusion'))
import torch
import torch.nn as nn
from diffusers import DDPMScheduler
from models.autoencoder_kl import AutoencoderKL
from models.unet_2d_condition import UNet2DConditionModel
from models.hyper_encoder import HyperEncoder
from peft import LoraConfig
from utils.rate_config import rate_cfg

from compressai.models.utils import conv, deconv


def initialize_vae(args):
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    vae.requires_grad_(False)

    return vae


def initialize_unet(args):
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    unet.requires_grad_(False)
    unet.train()

    l_target_modules_encoder, l_target_modules_decoder, l_modules_others = [], [], []
    l_grep = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_in", "conv_shortcut", "conv_out",
              "proj_out", "proj_in", "ff.net.2", "ff.net.0.proj"]
    for n, p in unet.named_parameters():
        if "bias" in n or "norm" in n:
            continue
        for pattern in l_grep:
            if pattern in n and ("down_blocks" in n or "conv_in" in n):
                l_target_modules_encoder.append(n.replace(".weight", ""))
                break
            elif pattern in n and ("up_blocks" in n or "conv_out" in n):
                l_target_modules_decoder.append(n.replace(".weight", ""))
                break
            elif pattern in n:
                l_modules_others.append(n.replace(".weight", ""))
                break

    lora_conf_encoder = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian",
                                   target_modules=l_target_modules_encoder)
    lora_conf_decoder = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian",
                                   target_modules=l_target_modules_decoder)
    lora_conf_others = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian", target_modules=l_modules_others)
    unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
    unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
    unet.add_adapter(lora_conf_others, adapter_name="default_others")

    return unet, l_target_modules_encoder, l_target_modules_decoder, l_modules_others


def initialize_hyper_encoder(hyper_dim, bpp):
    hyper_enc = HyperEncoder(M=hyper_dim, cfg_ss=rate_cfg[bpp][0], cfg_cs=rate_cfg[bpp][1])
    hyper_enc.train()

    return hyper_enc


class OSCAR(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self.noise_scheduler.set_timesteps(1, device="cuda")
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.cuda()
        self.args = args

        self.vae = initialize_vae(args)
        self.unet, self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others = initialize_unet(
            self.args)
        self.lora_rank_unet = self.args.lora_rank

        self.bpps = [0.0019, 0.0098, 0.0313, 0.0430, 0.0625, 0.0781, 0.0937, 0.1250]
        self.hyper_encoders = nn.ModuleList([
            initialize_hyper_encoder(args.hyper_dim, bpp) for bpp in
            self.bpps
        ])

        self.unet.to("cuda")
        self.vae.to("cuda")
        self.hyper_encoders.to("cuda")
        self.embeddings = nn.Parameter(torch.zeros(77, 1024, device="cuda"), requires_grad=True)
        self.embeddings.data.normal_(mean=0.0, std=0.02)
        self.timesteps = torch.tensor([810, 780, 310, 280, 210, 190, 170, 150], device="cuda").long()

    def set_train(self):
        self.unet.train()
        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.hyper_encoders.train()
        self.embeddings.requires_grad = True

    def set_eval(self):
        self.unet.eval()
        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = False
        self.hyper_encoders.eval()
        self.embeddings.requires_grad = False

    def load_hyper_encoders(self, paths):
        for i in range(len(paths)):
            print(f'Loading hyper-encoder {i} from {paths[i]}')
            print(self.hyper_encoders[i].load_state_dict(torch.load(paths[i])['hyper_enc']))

    def vae_forward(self, img):
        latents = self.vae.encode(img).latent_dist.sample() * self.vae.config.scaling_factor
        output_image = (self.vae.decode(latents / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        return output_image, None, None

    # ours
    def forward(self, img, idx):
        latents = self.vae.encode(img).latent_dist.sample() * self.vae.config.scaling_factor  # b, 4, 16, 16
        latents = latents.detach()
        bz = latents.shape[0]

        z_hat = self.hyper_encoders[idx](latents)
        b, c, h, w = z_hat.shape
        z = torch.nn.functional.normalize(latents.flatten(2), dim=-1)
        z_repa = torch.nn.functional.normalize(z_hat.flatten(2), dim=-1)
        cos_loss = -(z * z_repa).sum(-1).mean()

        A_z = torch.norm(latents.flatten(2), p=2, dim=-1, keepdim=True)
        A_z_hat = torch.norm(z_hat.flatten(2), p=2, dim=-1, keepdim=True)
        new_z_hat = z_hat.flatten(2) * A_z / A_z_hat
        new_z_hat = new_z_hat.reshape(b, c, h, w)
        model_pred = self.unet(new_z_hat, self.timesteps[idx],
                               encoder_hidden_states=self.embeddings.unsqueeze(0).repeat(bz, 1, 1).to(
                                   torch.float32)).sample
        x_denoised = self.noise_scheduler.step(model_pred, self.timesteps[idx], new_z_hat, return_dict=True).pred_original_sample
        output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        return output_image, x_denoised, cos_loss

    def save_model(self, outf):
        sd = {}
        sd["hyper_encoders"] = self.hyper_encoders.state_dict()
        sd["embeddings"] = self.embeddings
        sd["unet_lora_encoder_modules"], sd["unet_lora_decoder_modules"], sd["unet_lora_others_modules"] = \
            self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others
        sd["rank_unet"] = self.lora_rank_unet
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        torch.save(sd, outf)

    def get_trainable_params(self):
        layers_to_opt = []
        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                layers_to_opt.append(_p)
        layers_to_opt += list(self.unet.conv_in.parameters())
        layers_to_opt += list(self.hyper_encoders.parameters())
        layers_to_opt.append(self.embeddings)

        return layers_to_opt

    def load_ckpt(self, model):
        for n, p in self.unet.named_parameters():
            if "lora" in n or "conv_in" in n:
                p.data.copy_(model["state_dict_unet"][n])
        self.unet.set_adapter(["default_encoder", "default_decoder", "default_others"])
        self.hyper_encoders.load_state_dict(model["hyper_encoders"])
        self.embeddings.data.copy_(model["embeddings"])
