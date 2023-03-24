

import pandas as pd
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F

import torchvision
from torchvision import datasets, transforms, models
import torch
from collections import OrderedDict
from PIL import Image

import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse


"""
Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

Basic usage: python predict.py /path/to/image checkpoint

Options:
Return top KK most likely classes: python predict.py input checkpoint --top_k 3
Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
Use GPU for inference: python predict.py input checkpoint --gpu
"""

CHECKPOINT = "checkpoint.pth"
TOP_K = 3
CAT_TO_NAME = "cat_to_name.json"
GPU = True
IMAGE_PATH = 'flowers/test/19/image_06155.jpg'

    
parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type = str, default = IMAGE_PATH, help = 'Base image directory')
parser.add_argument('--checkpoint', type = str, default = CHECKPOINT, help = 'Model checkpoint path')
parser.add_argument('--top_k', type = int, default = TOP_K, help = 'Top K most likely classes')
parser.add_argument('--cat_to_name', type = str, default = CAT_TO_NAME, help = 'Mapping file category label to name')
parser.add_argument('--gpu', action = "store_true", default = GPU, help = 'Enable GPU')

args = parser.parse_args()




def load_checkpoint(checkpoint_path, map_location):
    """Load a checkpoint and rebuild the model
    
    Args:
        checkpoint_path (str): model checkpoint path
        map_location
    
    Return:
        model (object): model loaded from checkpoint path
    """
    
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    network = checkpoint['network']
    model = getattr(torchvision.models, network)(pretrained = True)

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    model.class_to_idx = checkpoint['class_to_idx']

    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(checkpoint['input_size'], checkpoint['hidden_units'])),
                          ('dropout', nn.Dropout(checkpoint['dropout_prob'])),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(checkpoint['hidden_units'], checkpoint['output_size'])),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    return model


def process_image(image):
    """ Scales, crops, and normalizes a PIL image for a PyTorch model,
       
    Args:
        image (PIL.Image): PIL image to process for use in PyTorch model
    
    Return:
        image_transform (numpy.array): processed image for use in PyTorch model
    """
    
    image = Image.open(image)
    image_transform = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    
    return image_transform(image)
  


def label_mapping(cat_to_name):
    """Load in a mapping from category label to category name
    
    Args:
       data_dir (str) = 'flowers'
    
    Return:
       cat_to_name (dict): dictionary that maps category label to category name
    """
    
    json_file = cat_to_name
    with open(json_file, 'r') as f:
        cat_to_name = json.load(f)
    
    return cat_to_name




def predict(image_path, checkpoint, top_k, cat_to_name):
    """Class prediction with the trained neural network
    
    Args:
        image_path (str): path of the image to classify
        model (object): trained neural network
        top_k (int): number of top classes
    
    Return:
        top_ps (list): top k probabilities
        top_labels (list): : top k labels
        top_f (list): top k label classes/flowers
    """
    # Use GPU if it's available
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'

    model = load_checkpoint(checkpoint, map_location)
    cat_to_name = label_mapping(cat_to_name)
    
    image_processed = process_image(image_path)
    image_processed.unsqueeze_(0)
    
    with torch.no_grad():
        output = model.forward(image_processed)
    
    ps = torch.exp(output)
    top_ps, top_class = ps.topk(top_k)
    
    class_to_idx = {}
    for key, value in model.class_to_idx.items():
        class_to_idx[value] = key
        
    top_labels_np = top_class[0].numpy()
    
    top_labels = []
    for label in top_labels_np:
        top_labels.append(int(class_to_idx[label]))
        
    top_f = [cat_to_name[str(label)] for label in top_labels]
    
    print('Probabilities: ', top_ps)
    print('Categories:    ', top_f)
    
    return top_ps, top_labels, top_f


if __name__ == "__main__":
    
    image_path = args.image_path
    checkpoint = args.checkpoint
    top_k = args.top_k
    cat_to_name = args.cat_to_name
    gpu = args.gpu

    
    predict(image_path, checkpoint, top_k, cat_to_name)