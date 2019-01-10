import torch
import numpy as np
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict

import torch
from torch import nn

def load_nn_model_from_path(filepath): 
    checkpoint = torch.load(filepath)
    
    arch_model = 'vgg'
    
    try:
        arch_model = checkpoint['pretrained_arch']
    except:
        print('hello error')
        arch_model = 'vgg'
    
    # model selection
    alexnet = models.alexnet(pretrained = True)
    densenet201 = models.densenet201(pretrained = True)
    vgg16 = models.vgg16(pretrained = True)

    pre_defined_models = {'alexnet': [alexnet, 9216], 'densenet': [densenet201, 1920], 'vgg': [vgg16, 25088]}

    nnet = pre_defined_models[arch_model][0]
    
    classifier_dict = checkpoint['classifier_dict']
    
    nnet.classifier = nn.Sequential(OrderedDict(classifier_dict))
    nnet.classifier_dict = classifier_dict
    nnet.load_state_dict(checkpoint['state_dict'])    
    nnet.class_to_idx = checkpoint['class_to_idx']
    
    optimizer = optim.Adam(nnet.classifier.parameters(), lr = 0.001)
    optimizer.load_state_dict(checkpoint['optim_state_dict'])
    
    return nnet, optimizer

# loaded_model, loaded_optimizer = load_nn_model_from_path('checkpoint_final.pth')

def save_nn_model_by_path(filepath, saved_model, saved_optimizer):
    saved_model.cpu()
    checkpoint = {'classifier_dict': saved_model.classifier_dict,
                  'pretrained_arch': saved_model.pretrained_arch,
                  'class_to_idx': saved_model.class_to_idx,
                  'optim_state_dict': saved_optimizer.state_dict(),
                  'state_dict': saved_model.state_dict()}

    torch.save(checkpoint, filepath)

# save_nn_model_by_path('checkpoint_final.pth', model, loaded_optimizer)