import io
import sys

import contextlib
import functools

import matplotlib.pyplot as plt
import numpy as np

from config import OUTPUT_FILE_PATH


def img_show(img, filename: str):
    img = img / 2 + 0.5
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.savefig(filename)


def file_output(data: str):
    open_method = 'a' if file_output.count else 'w'
    lines = data.splitlines()
    with open(OUTPUT_FILE_PATH, open_method) as output_file:
        for line in lines:
            output_file.write(f'{line}\n')
    file_output.count += len(lines)


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
    original_stderr = sys.stderr
    try:
        # make dummy stderr
        killed_stderr = io.StringIO()
        # swap to dummy stderr
        sys.stderr = killed_stderr
        yield killed_stderr
    finally:  # safe-wrap with exception handler
        sys.stderr = original_stderr  # NOTE: __stderr__ - original stderr


def without_stderr(func):  # decorator
    """Decorator that makes function run without stderr output."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with kill_stderr():
            return func(*args, **kwargs)
    return wrapper


def catch_stdout(func):  # decorator
    """Decorator that makes function return stdout output string, instead of original function result."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> str:
        original_stdout = sys.stdout
        try:
            stdout_catcher = io.StringIO()
            sys.stdout = stdout_catcher
            func(*args, **kwargs)
            return stdout_catcher.getvalue()
        finally:
            sys.stdout = original_stdout
    return wrapper
