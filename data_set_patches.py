"""
Project:    
File:       data_set_patches.py
Created by: louise
On:         10/10/17
At:         6:27 PM
"""
from time import time
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d

def construct_patches(im):
    # Convert from uint8 representation with values between 0 and 255 to
    # a floating point representation with values between 0 and 1.
    im_pil = im / 255.

    # downsample for higher speed
    height, width, _ = im_pil.shape

    # Extract all reference patches from the left half of the image
    print('Extracting reference patches...')
    t0 = time()
    patch_size = (8, 8)
    data = extract_patches_2d(im_pil, patch_size)
    #data = data.reshape(data.shape[0], -1)
    data -= np.mean(data, axis=0)
    data /= np.std(data, axis=0)
    print('done in %.2fs.' % (time() - t0))
    return data
