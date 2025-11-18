import os

from data import common

import numpy as np
import imageio

import torch
import torch.utils.data as data

class Demo(data.Dataset):
    def __init__(self, args, name='Demo', train=False, benchmark=False):
        self.args = args
        self.name = name
        self.scale = args.scale
        self.idx_scale = 0
        self.train = False
        self.benchmark = benchmark

        self.filelist = []
        for f in os.listdir(args.dir_demo):
            if f.find('.png') >= 0 or f.find('.jp') >= 0 or f.find('.raw') >= 0:
                self.filelist.append(os.path.join(args.dir_demo, f))
        self.filelist.sort()

    def __getitem__(self, idx):
        filename = os.path.splitext(os.path.basename(self.filelist[idx]))[0]
        filepath = self.filelist[idx]
        
        # Check if it's a raw file
        if filepath.endswith('.raw'):
            from utility import read_raw_image, parse_raw_filename
            
            # Try to parse dimensions from filename
            width, height = parse_raw_filename(filepath)
            
            # Use command-line args if dimensions not in filename
            if width is None or height is None:
                width = self.args.raw_width
                height = self.args.raw_height
                
            if width is None or height is None:
                raise ValueError(
                    f"Cannot determine dimensions for raw file: {filepath}\n"
                    f"Either include dimensions in filename (e.g., image_640x480.raw) "
                    f"or use --raw_width and --raw_height arguments"
                )
            
            # Read raw image and get grayscale/RGB based on n_colors
            lr = read_raw_image(filepath, width, height, bit_depth=self.args.raw_bit_depth)
            
            # Add channel dimension if grayscale
            if lr.ndim == 2:
                lr = np.expand_dims(lr, axis=2)
        else:
            lr = imageio.imread(filepath)

        lr, = common.set_channel(lr, n_channels=self.args.n_colors)
        lr_t, = common.np2Tensor(lr, rgb_range=self.args.rgb_range)

        return lr_t, -1, filename

    def __len__(self):
        return len(self.filelist)

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale

