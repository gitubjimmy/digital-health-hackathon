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


class StdoutCatcher(object):  # context manager: available as str with subclassing

    def __new__(cls, func=None):
        if func is not None:  # as decorator
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with cls() as catcher:
                    func(*args, **kwargs)
                return catcher.getvalue()
            return wrapper  # return not new object but decorator
        result = object.__new__(cls)
        result.__wrapper = io.StringIO()
        result.__prior = None
        return result
    __new__.__text_signature__ = '($cls, func=None, /)'

    def __enter__(self):
        self.__prior = sys.stdout
        sys.stdout = self.__wrapper
        return self

    open = __enter__

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        if any([exc_type, exc_val, exc_tb]):
            sys.stdout = sys.__stdout__
        try:
            sys.stdout = self.__prior
            self.__prior = None
        except BaseException:
            sys.stdout = sys.__stdout__
            raise

    close = __exit__

    def __str__(self):
        return self.__wrapper.getvalue()

    getvalue = __str__

    def __repr__(self):
        value = self.__str__()
        if not value:
            return "<%s object at %s (empty)>" % (type(self).__name__, hex(id(self)))
        return value


catch_stdout = StdoutCatcher  # alias
