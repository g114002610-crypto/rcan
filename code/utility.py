import numpy as np
import os
import re
import torch
from decimal import Decimal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_raw_image(filepath, width, height, bit_depth=14, dtype=np.uint16):
    """
    Read a raw image file with specified bit depth.
    
    Args:
        filepath: Path to the .raw file
        width: Image width in pixels
        height: Image height in pixels
        bit_depth: Bit depth of the image (default: 14)
        dtype: NumPy data type for reading (default: np.uint16 for 14-bit)
    
    Returns:
        numpy array of shape (height, width) with values scaled to 0-255 range
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw file not found: {filepath}")
    
    # Read raw binary data
    with open(filepath, 'rb') as f:
        raw_data = np.fromfile(f, dtype=dtype)
    
    # Reshape to image dimensions
    expected_size = width * height
    if raw_data.size < expected_size:
        raise ValueError(f"File size mismatch: expected {expected_size} pixels, got {raw_data.size}")
    
    img = raw_data[:expected_size].reshape((height, width))
    
    # Scale from bit_depth range to 8-bit range (0-255)
    # For 14-bit: max value is 2^14 - 1 = 16383
    max_val = (1 << bit_depth) - 1
    img_scaled = (img.astype(np.float32) / max_val * 255.0).astype(np.uint8)
    
    return img_scaled


def parse_raw_filename(filename):
    """
    Parse raw filename to extract width and height.
    Supports formats like: image_WIDTHxHEIGHT.raw or image_WIDTH_HEIGHT.raw
    
    Args:
        filename: Filename or path
    
    Returns:
        tuple: (width, height) or (None, None) if not found
    """
    basename = os.path.basename(filename)
    # Try WIDTHxHEIGHT format (e.g., image_1920x1080.raw)
    match = re.search(r'(\d+)x(\d+)', basename)
    if match:
        return int(match.group(1)), int(match.group(2))
    
    # Try WIDTH_HEIGHT format (e.g., image_1920_1080.raw)
    match = re.search(r'_(\d+)_(\d+)', basename)
    if match:
        return int(match.group(1)), int(match.group(2))
    
    return None, None


def quantize(img, rgb_range):
    """Quantize image to specified range."""
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)


class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if not args.load:
            if not args.save:
                args.save = now
            self.dir = os.path.join('..', 'experiment', args.save)
        else:
            self.dir = os.path.join('..', 'experiment', args.load)
            if os.path.exists(self.dir):
                self.log = torch.load(self.get_path('psnr_log.pt'))
                print('Continue from epoch {}...'.format(len(self.log)))
            else:
                args.load = ''

        if args.reset:
            os.system('rm -rf ' + self.dir)
            args.load = ''

        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.get_path('model'), exist_ok=True)
        for d in args.data_test:
            os.makedirs(self.get_path('results-{}'.format(d)), exist_ok=True)

        open_type = 'a' if os.path.exists(self.get_path('log.txt'))else 'w'
        self.log_file = open(self.get_path('log.txt'), open_type)
        with open(self.get_path('config.txt'), open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

        self.n_processes = 8

    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.get_path('model'), epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)

        self.plot_psnr(epoch)
        trainer.optimizer.save(self.dir)
        torch.save(self.log, self.get_path('psnr_log.pt'))

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.get_path('log.txt'), 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        for idx_data, d in enumerate(self.args.data_test):
            label = 'SR on {}'.format(d)
            fig = plt.figure()
            plt.title(label)
            for idx_scale, scale in enumerate(self.args.scale):
                plt.plot(
                    axis,
                    self.log[:, idx_data, idx_scale].numpy(),
                    label='Scale {}'.format(scale)
                )
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('PSNR')
            plt.grid(True)
            plt.savefig(self.get_path('test_{}.pdf'.format(d)))
            plt.close(fig)

    def begin_background(self):
        self.queue = Queue.Queue()

        def bg_target(queue):
            while True:
                if not queue.empty():
                    filename, tensor = queue.get()
                    if filename is None: break
                    imageio.imwrite(filename, tensor.numpy())
        
        self.process = [
            Process(target=bg_target, args=(self.queue,)) \
            for _ in range(self.n_processes)
        ]
        
        for p in self.process: p.start()

    def end_background(self):
        for _ in range(self.n_processes): self.queue.put((None, None))
        while not self.queue.empty(): time.sleep(1)
        for p in self.process: p.join()

    def save_results(self, dataset, filename, save_list, scale):
        if self.args.save_results:
            filename = self.get_path(
                'results-{}'.format(dataset.dataset.name),
                '{}_x{}_'.format(filename, scale)
            )

            postfix = ('SR', 'LR', 'HR')
            for v, p in zip(save_list, postfix):
                normalized = v[0].mul(255 / self.args.rgb_range)
                tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
                self.queue.put(('{}{}.png'.format(filename, p), tensor_cpu))


import datetime
import imageio
import torch
import time
import queue as Queue
from multiprocessing import Process
