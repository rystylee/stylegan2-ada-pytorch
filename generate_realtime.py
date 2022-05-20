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
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
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
    network_pkl: str,
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

    G_kwargs = dnnlib.EasyDict()
    G_kwargs.size = size 
    G_kwargs.scale_type = scale_type

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        # G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
        G = legacy.load_network_pkl(f, custom=custom, **G_kwargs)['G_ema'].to(device) # type: ignore

    # Labels.
    label = torch.zeros([batch_size, G.c_dim], device=device)

    if is_spout:
        spout = Spout(silent=True, width=1024, height=1024)
        spout.createSender('StyleGAN2-ada')

    import random
    import time

    src_z = torch.from_numpy(np.random.RandomState(random.randint(0, 500)).randn(batch_size, G.z_dim)).to(device)
    dst_z = torch.from_numpy(np.random.RandomState(random.randint(0, 500)).randn(batch_size, G.z_dim)).to(device)

    counter = 0.0
    while True:
        start_time = time.time()

        z = interpolate(src_z, dst_z, counter)
        with torch.no_grad():
            img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
            if batch_size == 2:
                img = tvutils.make_grid(img, nrow=2, padding=0, normalize=False).unsqueeze(0)
                # img = tvutils.make_grid(img, nrow=1, padding=1, normalize=False).unsqueeze(0)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img = img[0].cpu().numpy()

        if is_spout:
            spout.check()
            spout.send(img)

        img = img[:, :, ::-1]

        end_time = time.time()
        print(f'fps: {1.0 / (end_time - start_time)}')

        cv2.imshow('result', img)

        counter += 0.005
        if counter >= 1.0:
            counter = 0.0
            src_z = dst_z
            dst_z = torch.from_numpy(np.random.RandomState(random.randint(0, 500)).randn(batch_size, G.z_dim)).to(device)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
