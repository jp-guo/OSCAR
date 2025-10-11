import os
import sys

import utils.utils_image as utils

sys.path.append(os.getcwd())
import argparse
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from collections import OrderedDict

from diffusion.oscar import OSCAR

import pyiqa
from tqdm import tqdm


tensor_transforms = transforms.Compose([
    transforms.ToTensor(),
])

def center_crop(image, target_width, target_height):
    # 获取原始图像的尺寸
    width, height = image.size

    # 计算裁剪区域的左上角坐标
    left = (width - target_width) / 2
    top = (height - target_height) / 2
    right = (width + target_width) / 2
    bottom = (height + target_height) / 2

    # 裁剪并返回图像
    return image.crop((left, top, right, bottom))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', '-i', type=str, help='path to the input image')
    parser.add_argument('--output_dir', '-o', type=str, help='the directory to save the output')
    parser.add_argument('--pretrained_model_name_or_path', type=str, default=None, help='sd model path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed to be used')
    parser.add_argument("--oscar_path", type=str)
    parser.add_argument("--lora_rank", type=int, default=16)
    # precision setting
    parser.add_argument("--mixed_precision", type=str, choices=['fp16', 'fp32'], default="fp32")
    # merge lora
    parser.add_argument("--merge_and_unload_lora", default=False)  # merge lora weights before inference
    # tile setting
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224)
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024)
    parser.add_argument("--latent_tiled_size", type=int, default=96)
    parser.add_argument("--latent_tiled_overlap", type=int, default=32)

    parser.add_argument('--hyper_dim', type=int, default=320, help='output dim of hyper encoder')

    args = parser.parse_args()

    # initialize the model
    model = OSCAR(args).to("cuda")
    model.set_eval()
    sd = torch.load(args.oscar_path)
    model.load_ckpt(sd)

    # weight type
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16

    # make the output dir
    os.makedirs(args.output_dir, exist_ok=True)

    H_paths = utils.get_image_paths(args.input_image)
    print(f'There are {len(H_paths)} images.')

    device = 'cuda'
    psnr_metric = pyiqa.create_metric('psnr', device="cuda")
    lpips_metric = pyiqa.create_metric('lpips-vgg', device="cuda")
    dists_metric = pyiqa.create_metric('dists', device="cuda")
    musiq_metric = pyiqa.create_metric('musiq', device="cuda")
    clipiqa_metric = pyiqa.create_metric('clipiqa+', device="cuda")
    msssim_metric = pyiqa.create_metric('ms_ssim', device="cuda")
    fid_metric = pyiqa.create_metric('fid', device="cuda")

    os.makedirs(os.path.join(args.output_dir), exist_ok=True)

    f = open(os.path.join(args.output_dir, 'results.csv'), 'w')
    print('bpp,PSNR,MS-SSIM,DISTS,LPIPS,MUSIQ,CLIPIQA,FID', file=f)
    for level in range(len(model.timesteps)):
        bpp = model.bpps[level]
        os.makedirs(os.path.join(args.output_dir, str(bpp)), exist_ok=True)
        test_results = OrderedDict()
        test_results['psnr'] = []
        test_results['dists'] = []
        test_results['lpips'] = []
        test_results['niqe'] = []
        test_results['musiq'] = []
        test_results['clipiqa'] = []
        test_results['msssim'] = []
        for img in tqdm(H_paths):
            img_name, ext = os.path.splitext(os.path.basename(img))

            img_H = Image.open(img).convert('RGB')
            new_width = img_H.width - img_H.width % 16
            new_height = img_H.height - img_H.height % 16
            img_H = img_H.resize((new_width, new_height), Image.LANCZOS)

            # get caption
            lq = tensor_transforms(img_H.copy()).unsqueeze(0).to(device)
            lq = lq * 2 - 1
            with torch.no_grad():
                img_E, _, _ = model(lq, level)

            img_H = np.array(img_H)
            img_E = transforms.ToPILImage()(img_E[0].cpu() * 0.5 + 0.5)
            img_E = np.array(img_E)
            utils.imsave(img_E, os.path.join(args.output_dir, str(bpp), img_name + '.png'))

            img_E, img_H = img_E / 255., img_H / 255.
            img_E = torch.tensor(img_E, device="cuda").permute(2, 0, 1).unsqueeze(0)
            img_H = torch.tensor(img_H, device="cuda").permute(2, 0, 1).unsqueeze(0)
            img_E, img_H = img_E.type(torch.float32), img_H.type(torch.float32)

            psnr = psnr_metric(img_E, img_H)
            lpips = lpips_metric(img_E, img_H)
            dists = dists_metric(img_E, img_H)
            musiq = musiq_metric(img_E, img_H)
            clipiqa = clipiqa_metric(img_E, img_H)
            msssim = msssim_metric(img_E, img_H)

            test_results['psnr'].append(psnr.item())
            test_results['lpips'].append(lpips.item())
            test_results['dists'].append(dists.item())
            test_results['musiq'].append(musiq.item())
            test_results['clipiqa'].append(clipiqa.item())
            test_results['msssim'].append(msssim.item())

        avg_fid = fid_metric(os.path.join(args.output_dir, str(bpp)), args.input_image)
        avg_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        avg_dists = sum(test_results['dists']) / len(test_results['dists'])
        avg_lpips = sum(test_results['lpips']) / len(test_results['lpips'])
        avg_musiq = sum(test_results['musiq']) / len(test_results['musiq'])
        avg_clipiqa = sum(test_results['clipiqa']) / len(test_results['clipiqa'])
        avg_mssim = sum(test_results['msssim']) / len(test_results['msssim'])
        print(bpp, 'PSNR:', avg_psnr, 'MS-SSIM:', avg_mssim, 'DISTS:', avg_dists, 'LPIPS:', avg_lpips, 'MUSIQ:', avg_musiq,
              'CLIP-IQA:', avg_clipiqa, "FID:", avg_fid)

        print(bpp, avg_psnr, avg_mssim, avg_dists, avg_lpips, avg_musiq, avg_clipiqa, avg_fid, sep=',', end='\n', file=f)
    f.close()