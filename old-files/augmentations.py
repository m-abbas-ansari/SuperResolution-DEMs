import cv2
import math
import random
import numpy as np
import glob
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2


def load_numpy_files(npy_files):
    npy_files = npy_files
    demaps = []
    for image_path in demaps:
        try:
            demaps.append(np.load(image_path))
        except:
            print('FAILED TO READ {} IN GRAYSCALE'.format(image_path))
            continue
    return demaps
