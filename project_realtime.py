# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Project given image to the latent space of pretrained network pickle."""

import copy
import os
from time import perf_counter

import click
import imageio
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

import dnnlib
import legacy

from typing import List, Optional
import time
import cv2
from Library.Spout import Spout

to_tensor = transforms.Compose([
    transforms.ToTensor()
])


def project(
    G,
    *,
    num_steps                  = 1000,
    w_avg_samples              = 10000,
    initial_learning_rate      = 0.02,
    initial_noise_factor       = 0.05,
    regularize_noise_weight    = 1e5,
    verbose                    = False,
    device: torch.device,
    is_spout: bool
):
    # assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore

    # Compute w stats.
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, 1, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    # Setup noise inputs.
    noise_bufs = { name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name }
    w_noise_scale = w_std * initial_noise_factor # noise scale is constant

    # Load VGG16 feature detector.
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True) # pylint: disable=not-callable
    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)

    lr = initial_learning_rate   # learning rate is constant
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    # Setup camera
    cap = cv2.VideoCapture(0)

    if is_spout:
        spout_src = Spout(silent=True, width=G.img_resolution, height=G.img_resolution)
        spout_src.createSender('GAN-src')

    while True:
        start_time = time.time()

        ret, frame = cap.read()
        cv2.imshow('camera', frame)

        target_pil = PIL.Image.fromarray(frame[:, :, ::-1])
        target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.Resampling.LANCZOS)
        target_uint8 = np.array(target_pil, dtype=np.uint8)
        target = torch.tensor(target_uint8.transpose([2, 0, 1]), device=device)
        # target = to_tensor(target_pil)

        # Features for target image.
        target_images = target.unsqueeze(0).to(device).to(torch.float32)
        if target_images.shape[2] > 256:
            target_images = F.interpolate(target_images, size=(256, 256), mode='area')
        target_features = vgg16(target_images, resize_images=False, return_lpips=True)


        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])
        synth_images = G.synthesis(ws, noise_mode='const')


        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images = (synth_images + 1) * (255/2)
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')


        # Features for synth images.
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        dist = (target_features - synth_features).square().sum()

        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None,None,:,:] # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        loss = dist + reg_loss * regularize_noise_weight

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

        # Show image
        img = G.synthesis(w_opt.detach()[0].repeat([1, G.mapping.num_ws, 1]), noise_mode='const')
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img = img[0].cpu().numpy()

        if is_spout:
            spout_src.check()
            spout_src.send(img)

        img = img[:, :, ::-1]
        cv2.imshow('result', img)

        end_time = time.time()
        print(f'fps: {1.0 / (end_time - start_time)}')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

#----------------------------------------------------------------------------
def size_range(s: str) -> List[int]:
    '''Accept a range 'a-c' and return as a list of 2 ints.'''
    return [int(v) for v in s.split('-')][::-1]



@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--num-steps',              help='Number of optimization steps', type=int, default=1000, show_default=True)
@click.option('--seed',                   help='Random seed', type=int, default=303, show_default=True)
@click.option('--scale-type',
                type=click.Choice(['pad', 'padside', 'symm','symmside']),
                default='pad', help='scaling method for --size', required=False)
@click.option('--size', type=size_range, help='size of output (in format x-y)')
@click.option('--is_spout', type=bool, default=False, help='Spout option', required=False)

def run_projection(
    network_pkl: str,
    seed: int,
    num_steps: int,
    scale_type: Optional[str],
    size: Optional[List[int]],
    is_spout: bool
):
    """Project given image to the latent space of pretrained network pickle.

    Examples:

    \b
    python projector.py --outdir=out --target=~/mytargetimg.png \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    if(size): 
        print('render custom size: ',size)
        print('padding method:', scale_type )
        custom = True
    else:
        custom = False

    G_kwargs = dnnlib.EasyDict()
    G_kwargs.size = size 
    G_kwargs.scale_type = scale_type

    # Load networks.
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as fp:
        # G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: ignore
        G = legacy.load_network_pkl(fp, custom=custom, **G_kwargs)['G_ema'].requires_grad_(False).to(device) # type: ignore

    # Optimize projection.
    project(
        G,
        num_steps=num_steps,
        device=device,
        verbose=True,
        is_spout=is_spout
    )

#----------------------------------------------------------------------------

if __name__ == "__main__":
    run_projection() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
