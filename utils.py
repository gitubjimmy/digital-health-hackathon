import io
import sys

import contextlib
import functools

import matplotlib.pyplot as plt
import numpy as np

from config import OUTPUT_FILE_PATH


def img_show(img, filename):
    img = img / 2 + 0.5
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.savefig(filename)


def file_output(line):
    open_method = 'a' if file_output.count else 'w'
    file_output.count += 1
    with open(OUTPUT_FILE_PATH, open_method) as output_file:
        output_file.write(f'{line}\n')

file_output.count = 0


@contextlib.contextmanager
def kill_stderr():  # context manager
    """
    Kills stderr in context.

    Example usage:

    >>> from torchvision.datasets import CIFAR10
    >>> with kill_stderr():
    ...     CIFAR10(root='data', download=True)
    ...
    Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to data/cifar-10-python.tar.gz

    """
    try:
        # make dummy stderr
        killed_stderr = io.StringIO()
        # swap to dummy stderr
        sys.stderr = killed_stderr
        yield killed_stderr
    finally:  # safe-wrap with exception handler
        sys.stderr = sys.__stderr__  # NOTE: __stderr__ - original stderr


def without_stderr(func):  # decorator
    """Decorator that makes function run without stderr output."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with kill_stderr():
            return func(*args, **kwargs)
    return wrapper


def write_to_csv(file_name, array):
    with open(file_name, 'w') as csv_file:
        for row in array:
            row = [str(cell) for cell in row]
            csv_file.write(','.join(row) + '\n')
