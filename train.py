
import argparse
import ast
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from loss import ContrastiveLoss
from model import generate_model
import resnet
from utils import adjust_learning_rate, AverageMeter, Logger


def parse_args():
    parser = argparse.ArgumentParser(description='DAD training on Videos')
    parser.add_argument('--feature_dim', default=128, type=int, help='To which dimension will video clip be embedded')
    parser.add_argument('--sample_duration', default=16, type=int, help='Temporal duration of each video clip')
    parser.add_argument('--sample_size', default=112, type=int, help='Height and width of inputs')
    parser.add_argument('--model_type', default='resnet', type=str, help='so far only resnet')
    parser.add_argument('--model_depth', default=18, type=int, help='Depth of resnet (18 | 50 | 101)')
    parser.add_argument('--num_classes', default=4, type=int, help='Number of classes for classification')
    parser.add_argument('--shortcut_type', default='B', type=str, help='Shortcut type of resnet (A | B)')
    parser.add_argument('--pre_train_model', default=True, type=ast.literal_eval, help='Whether use pre-trained model')
    parser.add_argument('--use_cuda', default=True, type=ast.literal_eval, help='If true, cuda is used.')
    parser.add_argument('--train_batch_size', default=3, type=int, help='Batch Size for normal training data')
    # parser.add_argument('--a_train_batch_size', default=25, type=int, help='Batch Size for anormal training data')
    parser.add_argument('--val_batch_size', default=25, type=int, help='Batch Size for validation data')
    parser.add_argument('--learning_rate', default=0.01, type=float,
                        help='Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--dampening', default=0.0, type=float, help='dampening of SGD')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight Decay')
    parser.add_argument('--epochs', default=250, type=int, help='Number of total epochs to run')
    parser.add_argument('--n_threads', default=8, type=int, help='num of workers loading dataset')
    parser.add_argument('--tracking', default=True, type=ast.literal_eval,
                        help='If true, BN uses tracking running stats')
    parser.add_argument('--tau', default=0.1, type=float,
                        help='a temperature parameter that controls the concentration level of the distribution of embedded vectors')
    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')
    # parser.add_argument('--memory_bank_size', default=200, type=int, help='Memory bank size')
    parser.add_argument('--nesterov', action='store_true', help='Nesterov momentum')
    parser.set_defaults(nesterov=False)
    parser.add_argument('--lr_decay', default=100, type=int,
                        help='Number of epochs after which learning rate will be reduced to 1/10 of original value')
    # parser.add_argument('--resume_path', default='', type=str, help='path of previously trained model')
    parser.add_argument('--checkpoint_folder', default='./checkpoints/', type=str, help='folder to store checkpoints')
    parser.add_argument('--log_folder', default='./logs/', type=str, help='folder to store log files')
    parser.add_argument('--log_resume', default=False, type=ast.literal_eval,
                        help='True|False: a flag controlling whether to create a new log file')
    args = parser.parse_args()
    return args


def train_contrastive(train_loader, model, criterion, optimizer, epoch, args,
          batch_logger, epoch_logger):
# def train_contrastive(train_loader, model, criterion, optimizer, epoch, args,
#           batch_logger, epoch_logger, memory_bank=None):

    print('train at epoch {}'.format(epoch))

    model.train()

    losses = AverageMeter()
    for batch, (data, targets) in enumerate(train_loader):
        data = torch.cat(data, dim=0)
        targets = targets.repeat(2)
          # n_vec as well as a_vec are all normalized value
        if args.use_cuda:
            data = data.cuda()
            targets = targets.cuda()
        # ================forward====================
        projections = model.forward_constrative(data)
        loss = criterion(projections, targets)

        # ================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===========update memory bank===============
        # model.eval()
        # n = model(data)
        # n = n.detach()
        # average = torch.mean(n, dim=0, keepdim=True)
        # if len(memory_bank) < args.memory_bank_size:
        #     memory_bank.append(average)
        # else:
        #     memory_bank.pop(0)
        #     memory_bank.append(average)
        # model.train()

        # ===============update meters ===============
        losses.update(loss.data[0], data.size(0))

        # =================logging=====================
        batch_logger.log({
            'epoch': epoch,
            'batch': batch,
            'loss': losses.val,
            'lr': optimizer.param_groups[0]['lr']
        })
        print(
            f'Training Process is running: {epoch}/{args.epochs}  | Batch: {batch} | Loss: {losses.val} ({losses.avg})')
    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'lr': optimizer.param_groups[0]['lr']
    })
    # return memory_bank, losses.avg

def train_crossentropy_no_proj(train_loader, model, criterion, optimizer, epoch, args,
          batch_logger, epoch_logger):

    print('train at epoch {}'.format(epoch))

    model.train()

    losses = AverageMeter()
    accuracies = AverageMeter()

    for batch, (data, targets) in enumerate(train_loader):
        data = torch.cat(data, dim=0)
        targets = targets.repeat(2)
          # n_vec as well as a_vec are all normalized value
        if args.use_cuda:
            data = data.cuda()
            targets = targets.cuda()
        

        # ================forward====================
        projections = model(data)
        loss = criterion(projections, targets)
        # get the code for this 
        acc = calculate_accuracy(projections, targets)

        # ================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===========update memory bank===============
        # model.eval()
        # n = model(data)
        # n = n.detach()
        # average = torch.mean(n, dim=0, keepdim=True)
        # if len(memory_bank) < args.memory_bank_size:
        #     memory_bank.append(average)
        # else:
        #     memory_bank.pop(0)
        #     memory_bank.append(average)
        # model.train()

        # ===============update meters ===============
        losses.update(loss.data[0], data.size(0))
        accuracies.update(acc, data.size(0))
        # prob_meter.update(probs.item(), outs.size(0))

        # =================logging=====================
        val_batch_logger.log({
            'epoch': epoch,
            'batch': batch,
            'loss': losses.val,
            'acc': accuracies.val,
            'lr': optimizer.param_groups[0]['lr']
        })
        print(
            f'Training Process is running: {epoch}/{args.epochs}  | Batch: {batch} | Loss: {losses.val} ({losses.avg})')
    val_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'acc': accuracies.avg,
        'lr': optimizer.param_groups[0]['lr']
    })


def train_crossentropy(train_loader, model, criterion, optimizer, epoch, args,
          batch_logger, epoch_logger):

    print('train at epoch {}'.format(epoch))

    model.train()

    losses = AverageMeter()
    accuracies = AverageMeter()

    for batch, (data, targets) in enumerate(train_loader):
        data = torch.cat(data, dim=0)
        targets = targets.repeat(2)
          # n_vec as well as a_vec are all normalized value
        if args.use_cuda:
            data = data.cuda()
            targets = targets.cuda()
        

        # ================forward====================
        projections = model.forward_projection(data)
        loss = criterion(projections, targets)
        # get the code for this 
        acc = calculate_accuracy(projections, targets)

        # ================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===============update meters ===============
        losses.update(loss.data[0], data.size(0))
        accuracies.update(acc, data.size(0))
        # prob_meter.update(probs.item(), outs.size(0))

        # =================logging=====================
        val_batch_logger.log({
            'epoch': epoch,
            'batch': batch,
            'loss': losses.val,
            'acc': accuracies.val,
            'lr': optimizer.param_groups[0]['lr']
        })
        print(
            f'Training Process is running: {epoch}/{args.epochs}  | Batch: {batch} | Loss: {losses.val} ({losses.avg})')
    val_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'acc': accuracies.avg,
        'lr': optimizer.param_groups[0]['lr']
    })
   

if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.checkpoint_folder):
        os.makedirs(args.checkpoint_folder)
    if not os.path.exists(args.log_folder):
        os.makedirs(args.log_folder)
    torch.manual_seed(args.manual_seed)
    if args.use_cuda:
        torch.cuda.manual_seed(args.manual_seed)
    if args.nesterov:
        dampening = 0
    else:
        dampening = args.dampening

    

    # ADD DATALOADER PART TO IT
    dataloader = get_dataloader(batch_size,
                            'train.csv',
                            os.path.join(os.getcwd(), 'images_train'),
                            'test.csv',
                            os.path.join(os.getcwd(), 'images_test'))
    dataset_sizes = {x: len(dataloader[x].dataset) for x in ['train', 'test']}
    print(dataset_sizes, flush=True)



        "============================================Generating Model============================================")
    # ===============generate new model or pre-trained model===============
    model = generate_model(args)
    optimizer = torch.optim.SGD(model.parameters() , lr=args.learning_rate, momentum=args.momentum,
                                dampening=dampening, weight_decay=args.weight_decay, nesterov=args.nesterov)
    criterion = ContrastiveLoss()
    begin_epoch = 1
    best_auc = 0

    print(
        "==========================================!!!START TRAINING!!!==========================================")
    cudnn.benchmark = True
    batch_logger = Logger(os.path.join(args.log_folder, 'batch.log'), ['epoch', 'batch', 'loss', 'lr'],
                        args.log_resume)
    val_batch_logger = Logger(os.path.join(args.log_folder, 'batch.log'), ['epoch', 'batch', 'loss', 'acc', 'lr'],
                        args.log_resume)
    epoch_logger = Logger(os.path.join(args.log_folder, 'epoch.log'), ['epoch', 'loss', 'lr'],
                        args.log_resume)
    val_logger = Logger(os.path.join(args.log_folder, 'val.log'),
                        ['epoch', 'loss', 'acc', 'lr'], args.log_resume)



