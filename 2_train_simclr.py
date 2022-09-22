import argparse
import torch
from utils.monitoring import make_directories, compute_parameter_grad, get_logger, \
    plot_gif, visualize, plot_img, str2bool, visual_evaluation
from utils.data_loader import load_dataset_exemplar

from torch import optim
import torch.nn as nn
import numpy as np
import random

import os

from model.simclr import ImageEmbedding, ContrastiveLoss
from torch.optim import RMSprop
from utils.custom_loader import Contrastive_augmentation
import torchvision.transforms as tf
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.custom_transform import Binarize_batch, Scale_0_1_batch

parser = argparse.ArgumentParser()
parser.add_argument('--z_size', type=int, default=256)
parser.add_argument('--dataset', type=str, default='omniglot', choices=['omniglot'],
                    metavar='DATASET', help='Dataset choice.')
parser.add_argument('--download_data', type=eval, default=False, choices=[True, False])
parser.add_argument('--dataset_root', type=str, default="/media/data_cifs_lrs/projects/prj_control/data")
parser.add_argument('--input_type', type=str, default='binary',
                    choices=['binary'], help='type of the input')
parser.add_argument('--batch_size', type=int, default=128, metavar='BATCH_SIZE',
                    help='input batch size for training')
parser.add_argument('--learning_rate', type=float, default=1e-3, metavar='LR',
                    help='learning rate of the optimizer')
parser.add_argument('--seed', type=int, default=None, metavar='SEED', help='random seed (None is no seed)')
parser.add_argument('--epoch', type=int, default=100, metavar='EPOCH', help='number of epoch')

parser.add_argument("--input_shape", nargs='+', type=int, default=[1, 50, 50],
                    help='shape of the input [channel, height, width]')
parser.add_argument('-od', '--out_dir', type=str, default='/media/data_cifs/projects/prj_zero_gene/exp/',
                    metavar='OUT_DIR', help='output directory for model snapshots etc.')
parser.add_argument('--debug', default=False, action='store_true', help='debugging flag (do not save the network)')
parser.add_argument('--device', type=str, default='cuda:0', help='cuda device')
parser.add_argument('--strength', type=str, default='normal',  choices=['normal','light','strong'], help='strength of the augmentation')
parser.add_argument('--tag', type=str, default='', help='tag of the experiment')

parser.add_argument('--model_name', type=str, default='simclr', choices=['simclr'],
                    help="type of the model")
parser.add_argument('--auto_param', default=False, action='store_true', help='set all the param automatically')

parser.add_argument('--preload', default=False, action='store_true', help='preload the dataset')

parser.add_argument("--augment", type=str2bool, nargs='?', const=True, default=False, help="data augmentation")
parser.add_argument("--rate_scheduler", type=str2bool, nargs='?', const=True, default=False, help="include a rate scheduler")
parser.add_argument("--exemplar_type", default='prototype', choices=['prototype', 'first', 'shuffle'],
                    metavar='EX_TYPE', help='type of exemplar')

args = parser.parse_args()

args.input_shape = tuple(args.input_shape)

if args.device == 'meso':
    args.device = torch.cuda.current_device()


batch_size_loss = args.batch_size

default_args = parser.parse_args([])
if args.auto_param:
    args = retrieve_param(args, default_args)

if args.seed is not None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)



visual_steps, monitor_steps = 10, 50

args = make_directories(args)
kwargs = {'preload': args.preload}

train_loader, test_loader, args = \
    load_dataset_exemplar(args, shape = args.input_shape, shuffle=True, drop_last=True)


model_embedding = ImageEmbedding(args.z_size).to(args.device)

SimCLR_loss = ContrastiveLoss(batch_size_loss).to(args.device)
optimizer = RMSprop(model_embedding.parameters(), lr=args.learning_rate)

scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

if not args.debug:
    logger = get_logger(args, __file__)
    writer = SummaryWriter(args.snap_dir)
else:
    logger = None
    writer = None

augment = Contrastive_augmentation(train_loader.dataset, target_size=args.input_shape[1:], strength=args.strength)

print(model_embedding)
print('number of parameters : {0:,}'.format(sum(p.numel() for p in model_embedding.parameters())))

best_loss = np.inf

for epoch in range(args.epoch):
    train_loss = 0
    len_dataset = 0
    model_embedding.train()
    for batch_idx, (data, exemplar, label) in enumerate(train_loader):
        exemplar = exemplar.to(args.device)
        data = data.to(args.device)
        X, Y = augment(data)

        X, Y = X.to(args.device), Y.to(args.device)
        embX, projX = model_embedding(X)
        embY, projY = model_embedding(Y)
        loss = SimCLR_loss(projX, projY)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        len_dataset += X.size(0)

        train_loss += loss.item()

        if batch_idx % monitor_steps == 0:
            to_print = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.3f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item()
                       )
            if args.debug:
                print(to_print)
            else:
                logger.info(to_print)

    train_loss /= len(train_loader)

    to_print = '====> Epoch: {} Avg loss: {:.4f}'.format(
        epoch, train_loss)

    if args.debug:
        print(to_print)
    else:
        logger.info(to_print)

    if writer is not None:
        writer.add_scalar("Loss/train", train_loss, epoch)


    model_embedding.eval()
    eval_loss = 0
    for batch_idx, (data, exemplar, label) in enumerate(test_loader):

        exemplar = exemplar.to(args.device)
        data = data.to(args.device)
        X, Y = augment(data)

        #if args.generative_model is not None :
        #    data = data.to(args.device)
        #    exemplar = exemplar.to(args.device)
        #    samples, _, _ = gen_model.generate(exemplar.size(0), exemplar=exemplar, low_memory=True)
        #    X, Y = augment(samples)
        #else:
        #    X, Y = augment(data)
        X, Y = X.to(args.device), Y.to(args.device)
        embX, projX = model_embedding(X)
        embY, projY = model_embedding(Y)
        loss = SimCLR_loss(projX, projY)



        len_dataset += X.size(0)

        eval_loss += loss.item()

    eval_loss /= len(test_loader)

    if args.rate_scheduler:
        scheduler.step(eval_loss)

    to_print = '====> TEST Epoch: {} Avg loss: {:.4f}'.format(
        epoch, eval_loss)

    if not args.debug:
        torch.save(model_embedding.state_dict(), args.snap_dir + '_end.model')

    if args.debug:
        print(to_print)
    else:
        logger.info(to_print)

    if writer is not None:
        writer.add_scalar("Loss/eval", eval_loss, epoch)
        writer.add_scalar("rate", optimizer.param_groups[0]['lr'], epoch)


    if eval_loss < best_loss:
        if not args.debug:
            torch.save(model_embedding.state_dict(), args.snap_dir + '_best.model')
        best_loss = eval_loss





