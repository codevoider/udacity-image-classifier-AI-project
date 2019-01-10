import torch
import numpy as np
from torchvision import datasets, transforms, models

from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sb

import json

def resize_n_crop_to_224(image):
    width, height = image.size
    
    max_dim = 'w' if width > height else 'h'
    
    re_w = min(width, 256) if width >= height else min(256, height)  * width / height
    re_h = min(256, height) if width <= height else min(256, width) * height / width
    
    crop_rect = ((re_w - 224)/2, (re_h - 224)/2, (re_w + 224)/2, (re_h + 224)/2)
    
    return image.resize((int(re_w), int(re_h))).crop(crop_rect)

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''    
    # TODO: Process a PIL image for use in a PyTorch model
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    im = Image.open(image)
    im = resize_n_crop_to_224(im)
    
    np_im = np.array(im)
    
    np_im = ((np_im/256) - mean) / std
    np_im = np.transpose(np_im, (2, 0, 1))
    
    return np_im
