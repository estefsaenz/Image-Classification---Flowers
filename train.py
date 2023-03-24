

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

import json
import argparse

"""
Train a new network on a data set with train.py
Basic usage: python train.py data_directory

Prints out training loss, validation loss, and validation accuracy as the network trains

Options:
Set directory to save checkpoints: python train.py data_dir --save_dir save_directory

Choose architecture: python train.py data_dir --arch "vgg13"

Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20

Use GPU for training: python train.py data_dir --gpu
"""

# Set default values:
DATA_DIR = 'flowers'
SAVE_DIRECTORY = "checkpoint.pth"
ARCHITECTURE = "vgg13"
LEARNING_RATE = 0.001
HIDDEN_UNITS = 1024
EPOCHS = 10
GPU = True


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type = str, default = DATA_DIR, help = 'Base directory')
parser.add_argument('--save_dir', type = str, default = SAVE_DIRECTORY,\
                    help = 'Directory to save the checkpoint of the model')
parser.add_argument('--network', type = str, default = ARCHITECTURE, choices =\
                    ['vgg11', 'vgg13', 'vgg16', 'vgg19'], help = 'Pre-trained VGG architecture')
parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE,\
                    help = 'Learning rate')
parser.add_argument('--hidden_units', type = int, default = HIDDEN_UNITS,\
                    help = 'Number of hidden units')
parser.add_argument('--epochs', type = int, default = EPOCHS,\
                    help = 'Number of epochs')
parser.add_argument('--gpu', action = "store_true", default = GPU, help = 'Enable GPU')

args = parser.parse_args()
    

def get_data_directories(data_dir):
    """Funciton to get the train, validation and test data directories
    
    Args: 
        data_dir (str) = 'flowers'
    
    Return:
        train_directory (str): training data directory
        valid_directory (str): validation data directory
        test_directory (str): testing data directory
    """
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    return train_dir, valid_dir, test_dir


def data_loader(train_dir, valid_dir, test_dir):
    """Load, transform and get data loader for all datasets
    
    Args:
        train_dir (str): training data directory
        valid_dir (str): validation data directory
        test_dir (str): testing data directory
    
    Return:
       train_data (torchvision.datasets.ImageFolder): train dataset
       valid_data (torchvision.datasets.ImageFolder): validation dataset
       test_data (torchvision.datasets.ImageFolder): test dataset
       trainloader (torch.utils.data.DataLoader): train data loader
       validloader (torch.utils.data.DataLoader): validation data loade
       testloader (torch.utils.data.DataLoader): test data loade
    """
    
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(250),
                                           transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomVerticalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    
    valid_transforms = transforms.Compose([transforms.Resize(250),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(250),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])



    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    
    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

    return train_data, valid_data, test_data, trainloader, validloader, testloader


def label_mapping(data_dir):
    """Load in a mapping from category label to category name
    
    Args:
       data_dir (str) = 'flowers'
    
    Return:
       cat_to_name (dict): dictionary that maps category label to category name
    """
    
    json_file = data_dir + '/' + 'cat_to_name.json'
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    return cat_to_name


def load_pretrained_network(network, hidden_units):
    """Load a pre-trained network
    
    Args:
        network (str): pretrained neural network architecture
        hidden_units (int): number of hidden units to use
    
    Return:
       model (object): pretrained neural network with freezed params and adapted classifier
    """
    
    model = getattr(torchvision.models, network)(pretrained = True)

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, hidden_units)),
                          ('dropout', nn.Dropout(0.5)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
        
    model.classifier = classifier
    
    return model


def save_checkpoint(model, train_data, save_dir, network, hidden_units, learning_rate, epocs):
    """Save the model checkpoint to the selected directory
    
    Args:
        save_dir (str): directory to save checkpoint
        network (str): pretrained neural network architecture
        classifier (obj): adapted classifier for network
        hidden_units (int): number of hidden units to used
        learning_rate (float): learning rate used
        epochs (int): number of epochs to used
    
    Return:
        None
    """
    
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'network': network,
                  'input_size': 25088,
                  'hidden_units': hidden_units,
                  'output_size': 102,
                  'batch_size': 32,
                  'learning_rate': learning_rate,
                  'epocs': epocs,
                  'dropout_prob': 0.5,
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx}

    torch.save(checkpoint, save_dir)
    
    
def train(data_dir, save_dir, network, hidden_units, learning_rate, epochs, gpu):
    """Building & training & saving the classifier
    
    Args:
        data_dir (str) = 'flowers'
        save_dir (str): directory to save checkpoint
        network (str): pretrained neural network architecture
        hidden_units (int): number of hidden units to used
        learning_rate (float): learning rate used
        epochs (int): number of epochs to used
        gpu (boolean): boolean to activate use GPU for training
    
    Returns:
       None
    """

    train_dir, valid_dir, test_dir = get_data_directories(data_dir)
    train_data, valid_data, test_data, trainloader, validloader, testloader = data_loader(train_dir, valid_dir, test_dir)
    cat_to_name = label_mapping(data_dir)
    
    print('Download pretrained model')
    model = load_pretrained_network(network, hidden_units)

    # Training phase:
    criterion = nn.NLLLoss() 
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    
    if gpu == True:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device('cpu')
    
    model.to(device);
    
    steps = 0
    running_loss = 0
    print_every = 5
    
    for epoch in range(epochs):
        for inputs, labels in trainloader:     
            steps += 1
            
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
            model.eval()

            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    valid_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))
            
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(validloader):.3f}")
            running_loss = 0
            model.train()


    # Validation on the test set:
    model.eval()
    accuracy = 0
    test_loss = 0


    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            test_loss = test_loss + loss.item()
            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Test accuracy: {accuracy/len(testloader):.3f}")

    # Save model checkpoint
    save_checkpoint(model, train_data, save_dir, network, hidden_units, learning_rate, epochs)                                                                     
    
    
if __name__ == "__main__":
    
    data_dir = args.data_dir
    save_dir = args.save_dir
    network = args.network
    learning_rate = args.learning_rate
    hidden_units = args.hidden_units
    epochs = args.epochs
    gpu = args.gpu 
    
    train(data_dir, save_dir, network, hidden_units, learning_rate, epochs, gpu)
