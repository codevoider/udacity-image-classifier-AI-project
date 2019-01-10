import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict

from PIL import Image
import json
import argparse

import image_processing as ip
import model_transfer as mt

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    processed_img = ip.process_image(image_path)
    ts_im = torch.from_numpy(processed_img).float()
    ts_im = ts_im.unsqueeze(0)
    
    model.to(device)
    
    with torch.no_grad():
        outputs = model.forward(ts_im.to(device))
        results = np.exp(outputs).topk(topk)
        
    class_to_idx_dic = {key: value for value, key in model.class_to_idx.items()}
        
    probs = results[0].data.cpu().numpy()[0]
    classes = results[1].data.cpu().numpy()[0]
    
    classes = [class_to_idx_dic[classes[i]] for i in range(classes.size)]
    
    print(probs)
    print(classes)
    
    return probs, classes

# argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('img_path', metavar='img_path', help='an image path to predict')
parser.add_argument('checkpoint', metavar='checkpoint', help='a checkpoint path to load the model')
parser.add_argument('--top_k', type=int, default='1', help='top K')
parser.add_argument('--cat_name', help='a category file path to use')
parser.add_argument('--gpu', nargs='?', default='cpu', help='use GPU if specified')

args = parser.parse_args()

# select GPU or CPU mode
device = None
if(args.gpu == 'cpu'):
    device = 'cpu'
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    if(device == 'cpu'):
        print('There is no GPU in this env.  CPU mode is applied instead.')

# define top K
topk = args.top_k

# define category path ;else default value is applied
cat_to_name_path_arg = args.cat_name
cat_to_name = dict()
cat_to_name_path = 'cat_to_name.json' if (cat_to_name_path_arg == None) else cat_to_name_path_arg

with open(cat_to_name_path, 'r') as f:
    cat_to_name = json.load(f)

# put the image to process
image_path = args.img_path
processed_img = ip.process_image(image_path)

# checkpoint file
checkpoint_path = args.checkpoint

# main logic
# load model from checkpoint file
model_to_predict, optimizer = mt.load_nn_model_from_path(checkpoint_path)

ts_im = torch.from_numpy(processed_img).float()
ts_im = ts_im.unsqueeze(0)

model_to_predict.to(device)
model_to_predict.eval()

with torch.no_grad():
    outputs = model_to_predict.forward(ts_im.to(device))
    results = np.exp(outputs).topk(topk)

class_to_idx_dic = {key: value for value, key in model_to_predict.class_to_idx.items()}

probs = results[0].data.cpu().numpy()[0]
classes = results[1].data.cpu().numpy()[0]

classes = [class_to_idx_dic[classes[i]] for i in range(classes.size)]
cat_name = [cat_to_name[classes[i]] for i in range(len(classes))]


for prob, class_name, i in zip(probs, cat_name, range(len(cat_name))):
    print('Top K #{}: {:27} Prob = {:6.5f}'.format(i + 1, class_name, prob))

# for debugging
# print(probs)
# print(cat_name)
# print(topk)
# print("Mode: {}".format(device))
# print(args.gpu)
# print(cat_to_name_path)

# process_image = ip.process_image('flowers/valid/10/image_07094.jpg')
# python predict.py flowers/valid/10/image_07094.jpg checkpoint1.pth --gpu --top_k 5