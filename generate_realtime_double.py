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
from Library.Spout import Spout


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
@click.option('--network1', 'network1_pkl', help='Network pickle filename', required=True)
@click.option('--network2', 'network2_pkl', help='Network pickle filename', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--scale-type',
                type=click.Choice(['pad', 'padside', 'symm','symmside']),
                default='pad', help='scaling method for --size', required=False)
@click.option('--size', type=size_range, help='size of output (in format x-y)')
@click.option('--space', type=click.Choice(['z', 'w']), default='z', help='latent space', required=True)
@click.option('--batch_size', type=int, default=1, help='batch size', required=False)
@click.option('--is_spout', type=bool, default=False, help='Spout option', required=False)


def generate_images(
    ctx: click.Context,
    network1_pkl: str,
    network2_pkl: str,
    scale_type: Optional[str],
    size: Optional[List[int]],
    space: str,
    truncation_psi: float,
    noise_mode: str,
    batch_size: int,
    is_spout: bool
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

    G_1_kwargs = dnnlib.EasyDict()
    G_1_kwargs.size = size
    G_1_kwargs.scale_type = scale_type

    G_2_kwargs = dnnlib.EasyDict()
    # G_2_kwargs.size = [256, 256]
    G_1_kwargs.size = size
    G_2_kwargs.scale_type = scale_type

    print('Loading networks 1 from "%s"...' % network1_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network1_pkl) as f:
        G_1 = legacy.load_network_pkl(f, custom=custom, **G_1_kwargs)['G_ema'].to(device) # type: ignore
    with dnnlib.util.open_url(network2_pkl) as f:
        G_2 = legacy.load_network_pkl(f, custom=False, **G_2_kwargs)['G_ema'].to(device) # type: ignore

    # Labels.
    label = torch.zeros([batch_size, G_1.c_dim], device=device)

    if is_spout:
        if size:
            spout_src = Spout(silent=True, width=size[0], height=size[1])
            spout_effect = Spout(silent=True, width=256, height=256)
        else:
            spout_src = Spout(silent=True, width=G_1.img_resolution, height=G_1.img_resolution)
            spout_effect = Spout(silent=True, width=256, height=256)
        spout_src.createSender('GAN-src')
        spout_effect.createSender('GAN-effect')

    import random
    import time

    src_z = torch.from_numpy(np.random.RandomState(random.randint(0, 500)).randn(batch_size, G_1.z_dim)).to(device)
    dst_z = torch.from_numpy(np.random.RandomState(random.randint(0, 500)).randn(batch_size, G_1.z_dim)).to(device)

    counter = 0.0
    while True:
        start_time = time.time()

        z = interpolate(src_z, dst_z, counter)
        with torch.no_grad():
            img = G_1(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img = img[0].cpu().numpy()

        if is_spout:
            spout_src.check()
            spout_src.send(img)

        img = img[:, :, ::-1]
        cv2.imshow('src', img)


        with torch.no_grad():
            img = G_2(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img = img[0].cpu().numpy()

        if is_spout:
            spout_effect.check()
            spout_effect.send(img)

        img = img[:, :, ::-1]
        cv2.imshow('effect', img)

        end_time = time.time()
        print(f'fps: {1.0 / (end_time - start_time)}')

        counter += 0.005
        if counter >= 1.0:
            counter = 0.0
            src_z = dst_z
            dst_z = torch.from_numpy(np.random.RandomState(random.randint(0, 500)).randn(batch_size, G_1.z_dim)).to(device)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
