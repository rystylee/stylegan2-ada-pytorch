# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import subprocess
import re
from typing import List, Optional

import click
import dnnlib
import numpy as np
from numpy import linalg
import PIL.Image
import torch

import legacy

from opensimplex import OpenSimplex

import cv2
import torchvision.utils as tvutils
import subprocess
from tqdm import tqdm


#----------------------------------------------------------------------------
def size_range(s: str) -> List[int]:
    '''Accept a range 'a-c' and return as a list of 2 ints.'''
    return [int(v) for v in s.split('-')][::-1]


#----------------------------------------------------------------------------
def interpolate(src, dst, coef):
    return src + (dst - src) * coef


#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--scale-type',
                type=click.Choice(['pad', 'padside', 'symm','symmside']),
                default='pad', help='scaling method for --size', required=False)
@click.option('--size', type=size_range, help='size of output (in format x-y)')
@click.option('--space', type=click.Choice(['z', 'w']), default='z', help='latent space', required=True)

@click.option('--batch_size', type=int, default=1, help='batch size', required=False)
@click.option('--fps', type=int, default=30, help='fps of the video', required=False)
@click.option('--num_frames', type=int, default=900, help='the number of frames to be generated', required=False)
@click.option('--num_interpolation', type=int, default=6, help='the number of interpolation between the latent vector', required=False)
@click.option('--video_name', type=str, default='out.mp4', help='the name of output video', required=True)


def generate_images(
    ctx: click.Context,
    network_pkl: str,
    scale_type: Optional[str],
    size: Optional[List[int]],
    space: str,
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    batch_size: int,
    fps: int,
    num_frames: int,
    num_interpolation: int,
    video_name: str,
):
    """Generate images using pretrained network pickle.
    """
    
    # custom size code from https://github.com/eps696/stylegan2ada/blob/master/src/_genSGAN2.py
    if(size): 
        print('render custom size: ',size)
        print('padding method:', scale_type )
        custom = True
    else:
        custom = False

    G_kwargs = dnnlib.EasyDict()
    G_kwargs.size = size 
    G_kwargs.scale_type = scale_type

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        # G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
        G = legacy.load_network_pkl(f, custom=custom, **G_kwargs)['G_ema'].to(device) # type: ignore

    os.makedirs(outdir, exist_ok=True)

    # Labels.
    # label = torch.zeros([1, G.c_dim], device=device)
    label = None

    import random
    import time

    src_z = torch.from_numpy(np.random.RandomState(random.randint(0, 500)).randn(batch_size, G.z_dim)).to(device)
    dst_z_list = []
    for i in range(num_interpolation):
        if i == num_interpolation - 1:
            dst_z_list.append(src_z)
        else:
            dst_z_list.append(torch.from_numpy(np.random.RandomState(random.randint(0, 500)).randn(batch_size, G.z_dim)).to(device))

    interpolate_frame_counter = 0
    num_interpolate_frame = num_frames / num_interpolation

    for i in tqdm(range(num_frames)):
        z = interpolate(src_z, dst_z_list[int(i // num_interpolate_frame)], float(interpolate_frame_counter) / float(num_interpolate_frame))
        img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        if batch_size == 2:
            img = tvutils.make_grid(img, nrow=2, padding=0, normalize=False).unsqueeze(0)
            # img = tvutils.make_grid(img, nrow=1, padding=1, normalize=False).unsqueeze(0)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img = img[0].cpu().numpy()
        img = img[:, :, ::-1]

        cv2.imshow('result', img)
        cv2.imwrite(f'{outdir}/{i:04d}.png', img)

        interpolate_frame_counter += 1

        if interpolate_frame_counter >= num_interpolate_frame:
            interpolate_frame_counter = 0
            src_z = dst_z_list[int(i // num_interpolate_frame) - 1]
            src_z = dst_z_list[int(i // num_interpolate_frame)]

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    subprocess.run(f'ffmpeg -framerate {fps} -i {outdir}/%04d.png -vcodec libx264 -pix_fmt yuv420p -r {fps} {outdir}/{video_name}')



#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
