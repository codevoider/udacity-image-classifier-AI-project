import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict

from PIL import Image
import model_transfer as mt

import argparse

# argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('data_dir', metavar='data_dir', help='a directory path for data')
parser.add_argument('--save_dir', metavar='save_dir', help='a directory path to save a checkpoint')
parser.add_argument('--arch', help='a pretrained model to be used (resnet, densenet, vgg)')
parser.add_argument('--learning_rate', type=float, help='learning rate')
parser.add_argument('--hidden_units', type=int, help='hidden units')
parser.add_argument('--epochs', type=int, help='epochs')
parser.add_argument('--gpu', nargs='?', default='cpu', help='use GPU if specified')

args = parser.parse_args()

# reassign arguments
data_dir_root = args.data_dir
save_dir = args.save_dir

arch_model = args.arch if (args.learning_rate != None) else 'vgg'

learning_rate = args.learning_rate if (args.learning_rate != None) else 0.001
hidden_units = args.hidden_units if (args.hidden_units != None) else 256
epochs = args.epochs if (args.epochs != None) else 2

drop_p = 0.1
print_every = 50
steps = 0

# build dataset from data_dir and train dataloader
data_dir = data_dir_root
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

verify_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

# load the datasets with ImageFolder
train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_datasets = datasets.ImageFolder(valid_dir, transform=verify_transforms)

# using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_datasets, batch_size=32)
    
# model selection
alexnet = models.alexnet(pretrained = True)
densenet201 = models.densenet201(pretrained = True)
vgg16 = models.vgg16(pretrained = True)

pre_defined_models = {'alexnet': [alexnet, 9216], 'densenet': [densenet201, 1920], 'vgg': [vgg16, 25088]}

model = pre_defined_models[arch_model][0]
out_feature = pre_defined_models[arch_model][1]

for param in model.parameters():
    param.requires_grad = False
# unused 
# pretained_model = models.__dict__[arch_model](pretrained=True)

# build classifier
classifier_dict = [('fc1', nn.Linear(out_feature, hidden_units)),
                    ('relu1', nn.ReLU()),
                    ('drop1', nn.Dropout(p=drop_p)),
                    ('fc_final', nn.Linear(hidden_units, 102)),
                    ('output', nn.LogSoftmax(dim=1))]

classifier = nn.Sequential(OrderedDict(classifier_dict))
    
model.classifier = classifier

# parameter for saving
model.classifier_dict = classifier_dict
model.pretrained_arch = arch_model
# select GPU or CPU mode
device = None
if(args.gpu == 'cpu'):
    device = 'cpu'
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    if(device == 'cpu'):
        print('There is no GPU in this env.  CPU mode is applied instead.')

model.to(device)
# define optimizer and criterion
optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
criterion = nn.NLLLoss()

# method for validation
def validation(model, validloader, criterion):
    valid_loss = 0
    accuracy = 0
    for images, labels in validloader:
        
        images, labels = images.to(device), labels.to(device)

        output = model.forward(images)
        valid_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return valid_loss, accuracy

print('Training Start...')
print('learning_rate = ', learning_rate)
print('epochs = ', epochs, '\n')

for e in range(epochs):
    running_loss = 0
    model.train()
    for images, labels in trainloader:
        steps += 1
        
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward and backward passes 
        outputs = model.forward(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            # change nn to eval mode for inference
            model.eval()
            
            # turn off gradients for validation
            with torch.no_grad():
                valid_loss, valid_accuracy = validation(model, validloader, criterion)
            
            print("Epoch: {}/{}... ".format(e+1, epochs),
                  "Train Loss: {:.4f}".format(running_loss/print_every),
                  "Validation Loss: {:.4f}".format(valid_loss/len(validloader)),
                  "Validation Accuracy: {:.4f}".format(valid_accuracy/len(validloader)))
            
            running_loss = 0

# save model
if(save_dir != None):
    model.class_to_idx = train_datasets.class_to_idx
    mt.save_nn_model_by_path(save_dir, model, optimizer)
    print('Save Trained Model at Path: ', save_dir)
    
# python train.py flowers --save_dir checkpoint1.pth --arch vgg --learning_rate 0.002 --hidden_units 256 --epochs 1 --gpu