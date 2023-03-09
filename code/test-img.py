#! /usr/bin/env python

from aist import image

image \
    .stable_diffusion_inpaint(
        'pearl_earring_orig.png',
        'pearl_earring_mask.png',
        'A stately portrait of a cat',
        dims=(512,605),
        strength=0.95,
        guidance_scale=7,
        rounds=50
    ).save('pearl_output.png')
