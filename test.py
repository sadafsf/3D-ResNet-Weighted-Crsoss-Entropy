import argparse
import ast
import os
import torch
import torch.nn as nn 
import torch.backends.cudnn as cudnn
from torch.utils.data import ConcatDataset 
import torchvision
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from torch.optim.lr_scheduler import StepLR
import numpy as np

# from torch.utils.tensorboard import SummaryWriter

from model import generate_model
# from train_lowlevel import 
from utils import  generate_dataloader
import datasets
import transforms
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", default=50, type=int, help="On the contrastive step this will be multiplied by two.")
    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')
    parser.add_argument('--use_cuda', default=False, type=ast.literal_eval, help='If true, cuda is used.')
    # model part    
    parser.add_argument('--sample_duration', default=16, type=int, help='Temporal duration of each video clip')
    parser.add_argument('--sample_size', default=112, type=int, help='Height and width of inputs')
    parser.add_argument('--model_type', default='resnet', type=str, help='so far only resnet')
    parser.add_argument('--model_depth', default=18, type=int, help='Depth of resnet (18 | 50 | 101)')
    parser.add_argument('--resnet_shortcut', default='B', type=str, help='Shortcut type of resnet (A | B)')
    parser.add_argument("--n_classes", default=4, type=int, help=" number of classes ")
    parser.add_argument('--mode', default='score', type=str, help='Mode (score | feature). score outputs class scores. feature outputs features (after global average pooling).')

    parser.add_argument("--epochs", default=100, type=int)

#  learning rate part
    parser.add_argument("--lr", default=0.001, type=float)

    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight Decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--dampening', default=0.0, type=float, help='dampening of SGD')
    parser.add_argument('--nesterov', action='store_true', help='Nesterov momentum')
    parser.set_defaults(nesterov=False)
    args = parser.parse_args()

    return args




args = parse_args()
# device = "cuda" if torch.cuda.is_available() else "cpu"
# args.device = device
# print("Device being used:", device, flush=True)

# Some args stuff to do
torch.manual_seed(args.manual_seed)
if args.use_cuda:
    torch.cuda.manual_seed(args.manual_seed)

if args.nesterov:
    dampening = 0
else:
    dampening = args.dampening

train=datasets.VideoDataset('/cluster/projects/khangroup/Sadaf/DAiSEE/DataSet/train.csv',
                            os.path.join(os.getcwd(), '/cluster/projects/khangroup/Sadaf/DAiSEE/DataSet/TrainFrames'), transform=torchvision.transforms.Compose([transforms.VideoFolderPathToTensor(state='train')]))

val=datasets.VideoDataset( '/cluster/projects/khangroup/Sadaf/DAiSEE/DataSet/validation.csv',
                            os.path.join(os.getcwd(), '/cluster/projects/khangroup/Sadaf/DAiSEE/DataSet/ValidationFrames'), transform=torchvision.transforms.Compose([transforms.VideoFolderPathToTensor(state='train')]))

# combine val+train dataset
data=ConcatDataset([train,val])

# train and test loader - spatial downsampling and trasfromation - temporal downsampling
train_loader, dataset_size_train,weight= generate_dataloader(args.batch_size,
                            '/cluster/projects/khangroup/Sadaf/DAiSEE/DataSet/trainval.csv',
                            os.path.join(os.getcwd(), '/cluster/projects/khangroup/Sadaf/DAiSEE/DataSet/TrainFrames'), pre_process=data)
# dataset_size_train =train_loader.dataset
print(dataset_size_train, flush=True) 
# print(dataset_sizes) 


test_loader,dataset_size_test, _ = generate_dataloader(args.batch_size,
                            '/cluster/projects/khangroup/Sadaf/DAiSEE/DataSet/test.csv',
                            os.path.join(os.getcwd(), '/cluster/projects/khangroup/Sadaf/DAiSEE/DataSet/TestFrames'), pre_process=None)

# dataset_size_test =test_loader.dataset
import argparse
import ast
import os
import torch
import torch.nn as nn 
import torch.backends.cudnn as cudnn
from torch.utils.data import ConcatDataset 
import torchvision
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from torch.optim.lr_scheduler import StepLR
import numpy as np

# from torch.utils.tensorboard import SummaryWriter

from model import generate_model
# from train_lowlevel import 
from utils import  generate_dataloader
import datasets
import transforms
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", default=50, type=int, help="On the contrastive step this will be multiplied by two.")
    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')
    parser.add_argument('--use_cuda', default=False, type=ast.literal_eval, help='If true, cuda is used.')
    # model part    
    parser.add_argument('--sample_duration', default=16, type=int, help='Temporal duration of each video clip')
    parser.add_argument('--sample_size', default=112, type=int, help='Height and width of inputs')
    parser.add_argument('--model_type', default='resnet', type=str, help='so far only resnet')
    parser.add_argument('--model_depth', default=18, type=int, help='Depth of resnet (18 | 50 | 101)')
    parser.add_argument('--resnet_shortcut', default='B', type=str, help='Shortcut type of resnet (A | B)')
    parser.add_argument("--n_classes", default=4, type=int, help=" number of classes ")
    parser.add_argument('--mode', default='score', type=str, help='Mode (score | feature). score outputs class scores. feature outputs features (after global average pooling).')

    parser.add_argument("--epochs", default=100, type=int)

#  learning rate part
    parser.add_argument("--lr", default=0.001, type=float)

    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight Decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--dampening', default=0.0, type=float, help='dampening of SGD')
    parser.add_argument('--nesterov', action='store_true', help='Nesterov momentum')
    parser.set_defaults(nesterov=False)
    args = parser.parse_args()

    return args




args = parse_args()
# device = "cuda" if torch.cuda.is_available() else "cpu"
# args.device = device
# print("Device being used:", device, flush=True)

# Some args stuff to do
torch.manual_seed(args.manual_seed)
if args.use_cuda:
    torch.cuda.manual_seed(args.manual_seed)

if args.nesterov:
    dampening = 0
else:
    dampening = args.dampening

train=datasets.VideoDataset('/cluster/projects/khangroup/Sadaf/DAiSEE/DataSet/train.csv',
                            os.path.join(os.getcwd(), '/cluster/projects/khangroup/Sadaf/DAiSEE/DataSet/TrainFrames'), transform=torchvision.transforms.Compose([transforms.VideoFolderPathToTensor(state='train')]))

val=datasets.VideoDataset( '/cluster/projects/khangroup/Sadaf/DAiSEE/DataSet/validation.csv',
                            os.path.join(os.getcwd(), '/cluster/projects/khangroup/Sadaf/DAiSEE/DataSet/ValidationFrames'), transform=torchvision.transforms.Compose([transforms.VideoFolderPathToTensor(state='train')]))

# combine val+train dataset
data=ConcatDataset([train,val])

# train and test loader - spatial downsampling and trasfromation - temporal downsampling
train_loader, dataset_size_train,weight= generate_dataloader(args.batch_size,
                            '/cluster/projects/khangroup/Sadaf/DAiSEE/DataSet/trainval.csv',
                            os.path.join(os.getcwd(), '/cluster/projects/khangroup/Sadaf/DAiSEE/DataSet/TrainFrames'), pre_process=data)
# dataset_size_train =train_loader.dataset
print(dataset_size_train, flush=True) 
# print(dataset_sizes) 


test_loader,dataset_size_test, _ = generate_dataloader(args.batch_size,
                            '/cluster/projects/khangroup/Sadaf/DAiSEE/DataSet/test.csv',
                            os.path.join(os.getcwd(), '/cluster/projects/khangroup/Sadaf/DAiSEE/DataSet/TestFrames'), pre_process=None)

# dataset_size_test =test_loader.dataset
# print(dataset_size_test, flush=True)
# for idx, batch in enumerate(train_loader):
#     print('Batch index: ', idx, flush=True)
#     print('Batch size: ', batch[0].size(),flush=True)
#     print('Batch label: ', batch[1],flush=True)
#     print('Batch label size: ', batch[1].size(),flush=True)

# for idx, batch in enumerate(test_loader):
#     print('Batch index: ', idx, flush=True)
#     print('Batch size: ', batch[0].size(),flush=True)
#     print('Batch label: ', batch[1],flush=True)
#     print('Batch label size: ', batch[1].size(),flush=True)

model = generate_model(args)
print(model, flush=True)
    
