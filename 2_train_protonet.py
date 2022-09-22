import argparse
import torch
from utils.monitoring import make_directories, compute_parameter_grad, get_logger, \
    plot_gif, visualize, plot_img, str2bool, visual_evaluation
#from evaluation_utils.generative_models import load_reco_model

from utils.data_loader import load_dataset_exemplar

import wandb
from torch import optim
import torch.nn as nn
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from model.few_shot import ProtoNet
#from model.param import retrieve_param
from torch.optim import Adam
from model.few_shot import pairwise_distances, compute_prototypes
#from utils.loading_tools import load_weights, load_net
from utils.custom_transform import Binarize_batch, Scale_0_1_batch

# torch.backends.cudnn.enabled = False


# --dataset mnist --batch_size 28 --z_size 5 --seed 75 --device cuda:1 --debug --lstm_size 400 --epoch 5
# python 1_train_vae.py --dataset omniglot_weak --auto_param --device cuda:2 --model_name vae_stn --exemplar --time_step 80 --tag CondGeneral2c_LstmSize400_TimeStep80_batch_size28_GlobalInit

parser = argparse.ArgumentParser()
parser.add_argument('--z_size', type=int, default=128)
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
parser.add_argument('--epoch', type=int, default=80, metavar='EPOCH', help='number of epoch')

parser.add_argument("--input_shape", nargs='+', type=int, default=[1, 50, 50],
                    help='shape of the input [channel, height, width]')
parser.add_argument('-od', '--out_dir', type=str, default='/media/data_cifs/projects/prj_zero_gene/exp/',
                    metavar='OUT_DIR', help='output directory for model snapshots etc.')
parser.add_argument('--debug', default=False, action='store_true', help='debugging flag (do not save the network)')
#parser.add_argument('--beta_adam', type=float, default=0.9, help='value of the first order beta in adam optimizer')

parser.add_argument('--device', type=str, default='cuda:0', help='cuda device')
parser.add_argument('--tag', type=str, default='', help='tag of the experiment')
#parser.add_argument("--exemplar", type=str2bool, nargs='?', const=True, default=True, help="For conditional VAE")
parser.add_argument('--model_name', type=str, default='proto_net', choices=['proto_net'],
                    help="type of the model ")
parser.add_argument('--auto_param', default=False, action='store_true', help='set all the param automatically')
parser.add_argument('--transform_variation', default=False, action='store_true', help='apply transform only to the image variation')
parser.add_argument("--augment", type=str2bool, nargs='?', const=True, default=False, help="data augmentation")
parser.add_argument("--exemplar_type", default='prototype', choices=['prototype', 'first', 'shuffle'],
                    metavar='EX_TYPE', help='type of exemplar')

parser.add_argument('--preload', default=False, action='store_true', help='preload the dataset')
#parser.add_argument("--shuffle_exemplar", type=str2bool, nargs='?', const=True, default=False, help="shuffle the exemplar")
#parser.add_argument("--rate_scheduler", type=str2bool, nargs='?', const=True, default=False, help="include a rate scheduler")
parser.add_argument('--drop_lr_every', default=20, type=int)
parser.add_argument('--episodes_per_epoch', default=100, type=int)
parser.add_argument('--evaluation_episodes', default=1000, type=int)
parser.add_argument('--distance', default='l2')
parser.add_argument('--n-train', default=1, type=int) #n shots
parser.add_argument('--n-test', default=1, type=int) #n shots
parser.add_argument('--k-train', default=60, type=int) #k ways
parser.add_argument('--k-test', default=20, type=int) #k ways
parser.add_argument('--q-train', default=5, type=int) #query
parser.add_argument('--q-test', default=1, type=int) #query
parser.add_argument('--gene_type', type=str, default='reco', choices=['reco', 'gene'])

args = parser.parse_args()
args.input_shape = tuple(args.input_shape)

hyperparameters = {"device": args.device,
                   "image_size": args.input_shape[-1],
                   "channels": 1,
                   "batch_size": args.batch_size,
                   "learning_rate": args.learning_rate,
                   "epochs": args.epoch}

if args.device == 'meso':
    args.device = torch.cuda.current_device()


default_args = parser.parse_args([])

if args.seed is not None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


if args.debug:
    visual_steps, monitor_steps = 1, 10 # 50
else:
    visual_steps, monitor_steps = 10, 25

args = make_directories(args, 'few_shot')
kwargs = {'preload': args.preload}

#train_loader, test_loader, train_exemplars, test_exemplars, args = load_dataset_exemplar(args, shape=args.input_shape, **kwargs)
train_loader, test_loader, args = load_dataset_exemplar(args, shape=args.input_shape, few_shot=True, **kwargs)

#model = get_few_shot_encoder(args.input_shape[0])
if args.model_name == 'proto_net' and args.input_shape[-1] == 50:
    model = ProtoNet(z_size=args.z_size)
else:
    raise NotImplementedError()
model.to(args.device)

optimizer = Adam(model.parameters(), lr=args.learning_rate)
loss_fn = torch.nn.NLLLoss().cuda()


scheduler = StepLR(optimizer, step_size=args.drop_lr_every, gamma=0.5)

if not args.debug:
    wandb.init(project='prototypical_net_qd', config=hyperparameters)
    wandb.run.name = args.model_signature
    wandb.run.save()
else:
    logger = None
    writer = None

print(model)
print('number of parameters : {0:,}'.format(sum(p.numel() for p in model.parameters())))


best_loss = np.inf
for epoch in range(1, args.epoch+1):
    model.train()
    train_loss, train_accu = 0, 0
    nb_queries = args.q_train * args.k_train
    for batch_idx, (data, exemplar, label) in enumerate(train_loader):
        exemplar = exemplar.to(args.device)
        data = data.to(args.device)
        support = exemplar[:args.n_train*args.k_train]
        label = torch.arange(0, args.k_train, 1 / args.q_train).long().to(args.device)

        query = data[args.n_train*args.k_train:]
        all_label = label

        data_to_pass = torch.cat([support,query], dim=0)
        optimizer.zero_grad()

        features, last_layer = model(data_to_pass)

        support = last_layer[:args.n_train * args.k_train]
        queries = last_layer[args.n_train * args.k_train:]

        prototypes = compute_prototypes(support, args.k_train, args.n_train)
        distances = pairwise_distances(queries, prototypes, args.distance)

        # Calculate log p_{phi} (y = k | x)
        log_p_y = (-distances).log_softmax(dim=1)
        loss = loss_fn(log_p_y, all_label)

        # Prediction probabilities are softmax over distances
        y_pred = (-distances).softmax(dim=1)

        loss.backward()
        optimizer.step()

        pred = y_pred.argmax(dim=1)

        correct = pred.eq(all_label)
        accu = 100*(correct.sum().float()/torch.numel(correct)).item()
        train_accu += accu
        train_loss += loss.item()

        if batch_idx % monitor_steps == 0:
            to_print = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.3f}\t%: {:.2f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item(),
                       accu,
                       )
            #if args.debug:
            print(to_print)
            #else:
            #    logger.info(to_print)

    train_loss /= len(train_loader)
    train_accu /= len(train_loader)

    to_print = '====> Epoch: {} Avg loss: {:.4f} -- Accu {:.2f}'.format(
        epoch, train_loss,
        train_accu)

    model.eval()
    eval_loss, eval_accu = 0, 0
    nb_queries = args.q_test * args.k_test
    for batch_idx, (data, exemplar, label) in enumerate(test_loader):
        data = data.to(args.device)
        exemplar = exemplar.to(args.device)
        support = exemplar[:args.n_test * args.k_test]
        label = torch.arange(0, args.k_test, 1 / args.q_test).long().to(args.device)


        query = data[args.n_test*args.k_test:]
        all_label = label

        data_to_pass = torch.cat([support, query], dim=0)
        features, last_layer = model(data_to_pass)

        support = last_layer[:args.n_test * args.k_test]
        queries = last_layer[args.n_test * args.k_test:]
        prototypes = compute_prototypes(support, args.k_test, args.n_test)
        distances = pairwise_distances(queries, prototypes, args.distance)


        log_p_y = (-distances).log_softmax(dim=1)
        loss = loss_fn(log_p_y, all_label)

        # Prediction probabilities are softmax over distances
        y_pred = (-distances).softmax(dim=1)
        pred = y_pred.argmax(dim=1)

        correct = pred.eq(all_label)
        accu = 100 * (correct.sum().float() / torch.numel(correct)).item()
        eval_accu += accu
        eval_loss += loss.item()
    eval_loss /= len(test_loader)
    eval_accu /= len(test_loader)

    to_print = '====> TEST Epoch: {} Avg loss: {:.4f} -- Avg Accu {:.2f}'.format(
        epoch, eval_loss,
        eval_accu)

    if not args.debug:
        torch.save(model.state_dict(), args.snap_dir + '_end.model')


    print(to_print)
    if not args.debug:
        wandb.log({"loss_training": train_loss,
                   "loss_testing": eval_loss,
                   "accuracy_train": train_accu,
                   "accuracy_test": eval_accu})

    scheduler.step()




