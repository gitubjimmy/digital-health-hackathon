import os

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
