import os
import sys

sys.path.append(os.path.join(os.getcwd(), 'diffusion'))
import torch
from models.autoencoder_kl import AutoencoderKL
from models.hyper_encoder import HyperEncoder
from utils.rate_config import rate_cfg


def initialize_vae(args):
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    vae.requires_grad_(False)

    return vae


def initialize_hyper_encoder(args):
    hyper_enc = HyperEncoder(M=args.hyper_dim, cfg_ss=rate_cfg[args.bpp][0], cfg_cs=rate_cfg[args.bpp][1])
    hyper_enc.train()

    return hyper_enc


class Encoder(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.vae = initialize_vae(args)
        self.hyper_enc = initialize_hyper_encoder(args)

        self.vae.to("cuda")
        self.hyper_enc.to("cuda")

    def set_train(self):
        self.hyper_enc.train()

    def set_eval(self):
        self.hyper_enc.eval()

    def forward(self, img):
        latents = self.vae.encode(img).latent_dist.sample() * self.vae.config.scaling_factor

        # (B,320,H,W)
        z_repa = self.hyper_enc(latents)
        # b, 32*32, 320 for 256*256 patch
        z_repa = z_repa.flatten(2).permute(0, 2, 1)

        latents = latents.flatten(2).permute(0, 2, 1)
        return z_repa, latents

    def save_model(self, outf):
        sd = {}
        sd["hyper_enc"] = self.hyper_enc.state_dict()
        torch.save(sd, outf)

    def get_trainable_params(self):
        layers_to_opt = []
        layers_to_opt += list(self.hyper_enc.parameters())

        return layers_to_opt

    def load_ckpt(self, sd):
        self.hyper_enc.load_state_dict(sd["hyper_enc"])
