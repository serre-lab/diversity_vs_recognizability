import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"

import argparse
import torch
from utils.monitoring import make_directories, compute_parameter_grad, get_logger, \
    plot_gif, visualize, plot_img, str2bool, visual_evaluation
#from utils.loading_tools import normalise_args
from utils.data_loader import load_dataset_exemplar
from model.network import VaeStn
#from model.network import SeqVaeCondGeneral2b, ConvVaeStn, vae_stn_big2b, SeqVaeCondGeneral2b2,SeqVaeCondGeneral2b3, SeqVaeCondGeneral2b4, SeqVaeCondGeneral2b5
#from model.network import SeqVaeArmageddon, vae_stn_big2bA, SeqVaeCondGeneral2b6, SeqVaeCondGeneral2bb, SeqVaeCondGeneral2bc, SeqVaeCondGeneral2bd, SeqVaeCondGeneral2b_VarSize
from model.losses import loss_vae_bernouilli
from torch import optim
import torch.nn as nn
#from model.param import retrieve_param
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from operator import add
import wandb
import sys



# torch.backends.cudnn.enabled = False


# --dataset mnist --batch_size 28 --z_size 5 --seed 75 --device cuda:1 --debug --lstm_size 400 --epoch 5
# python 1_train_vaestn.py --dataset omniglot_weak --auto_param --device cuda:2 --model_name vae_stn --exemplar --time_step 80 --tag CondGeneral2c_LstmSize400_TimeStep80_batch_size28_GlobalInit

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='omniglot', choices=['omniglot', 'omniglot_weak', 'mnist', 'human_drawing',
                                                                             'human_drawing_and_omniglot', 'quick_draw',
                                                                             'human_drawing_and_quick_draw'],
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
parser.add_argument('--epoch', type=int, default=500, metavar='EPOCH', help='number of epoch')
# parser.add_argument("--add_noise", type=eval, default=False, choices=[True, False])
parser.add_argument("--input_shape", nargs='+', type=int, default=[1, 50, 50],
                    help='shape of the input [channel, height, width]')
parser.add_argument('-od', '--out_dir', type=str, default='/media/data_cifs/projects/prj_zero_gene/exp/',
                    metavar='OUT_DIR', help='output directory for model snapshots etc.')
parser.add_argument('--debug', default=False, action='store_true', help='debugging flag (do not save the network)')
parser.add_argument('--beta_adam', type=float, default=0.9, help='value of the first order beta in adam optimizer')
parser.add_argument('--clip_value', type=float, default=5, help='value of the gradient clipping')

parser.add_argument('--z_size', type=int, default=60)
parser.add_argument('--lstm_size', type=int, default=400, help='size of the hidden state of the LSTM')
parser.add_argument('--time_step', type=int, default=40, help='number of time step of the LSTM')
parser.add_argument('--device', type=str, default='cuda:0', help='cuda device')
parser.add_argument('--tag', type=str, default='', help='tag of the experiment')
parser.add_argument("--exemplar", type=str2bool, nargs='?', const=True, default=True, help="For conditional VAE")
parser.add_argument('--conv_gru', default=False, action='store_true', help='for a convgru in the canvas')
parser.add_argument('--model_name', type=str, default='vae_stn', choices=['vae_stn'],
                    help="type of the model ['vae_stn', 'vae_draw']")
parser.add_argument('--read_size', nargs='+', type=int, default=[1, 15, 15],
                    help='Size of the read operation visual field.')
parser.add_argument('--write_size', nargs='+', type=int, default=[1, 15, 15],
                    help='Size of the write operation visual field.')
parser.add_argument('--auto_param', default=False, action='store_true', help='set all the param automatically')
#parser.add_argument('--hd_classes', nargs='+', type=int, default=[0,1,2,3,4,5,6])
#parser.add_argument('--loading_type', type=str, default='all_classes', choices=['all_classes'], help="type of the model")
parser.add_argument("--augment", type=str2bool, nargs='?', const=True, default=False, help="data augmentation")
parser.add_argument("--use_wandb", default=False, action='store_true', help='use weight and bias for monitoring')
parser.add_argument("--attention_type", type=str, default='stn', choices=['stn', 'gaussian', 'no_attention'])

parser.add_argument('--preload', default=True, action='store_true', help='preload the dataset')
parser.add_argument('--annealing_time', type=int, default=None)
#parser.add_argument("--shuffle_exemplar", type=str2bool, nargs='?', const=True, default=False, help="shuffle the exemplar")
parser.add_argument("--rate_scheduler", type=str2bool, nargs='?', const=True, default=True, help="include a rate scheduler")
parser.add_argument('--beta', type=float, default=1.0, metavar='BETA', help='beta that weight the KL in the vae Loss')
parser.add_argument("--exemplar_type", default='prototype', choices=['prototype', 'first', 'shuffle'],
                    metavar='EX_TYPE', help='type of exemplar')

parser.add_argument("--attention_ratio", type=float, default=1.0, help='zoom of the attention')



#args = normalize_args(args)

args = parser.parse_args()


if args.seed is not None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = False



if args.device == 'meso':

    args.device = torch.cuda.current_device()

args.read_size = tuple(args.read_size)
args.write_size = tuple(args.write_size)
args.input_shape = tuple(args.input_shape)
default_args = parser.parse_args([])


if args.auto_param:
    args = retrieve_param(args, default_args)


if args.debug:
    visual_steps, monitor_steps = 1, 25  # 50
else:
    visual_steps, monitor_steps = 10, 50

beta_list = torch.ones(args.epoch)*args.beta
if args.annealing_time is not None:
    beta_list[:args.annealing_time] = torch.linspace(0.2, args.beta, args.annealing_time)

args = make_directories(args, model_class='vae_stn')
kwargs = {'preload': args.preload}
#train_loader, test_loader, train_exemplars, test_exemplars, args = load_dataset_exemplar(args, shape=args.input_shape, **kwargs)
train_loader, test_loader, args = load_dataset_exemplar(args, shape=args.input_shape, **kwargs)

vae = VaeStn(args).to(args.device)


optimizer = optim.Adam(vae.parameters(), lr=args.learning_rate, betas=(args.beta_adam, 0.999))

if args.rate_scheduler:
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)


if not args.debug:
    logger = get_logger(args, __file__)
    if args.use_wandb:
        wandb.init(project="zero_shot_gene", entity="victorboutin", config=hyperparameter)
        writer = 'wandb'
    else:
        writer = SummaryWriter(args.snap_dir)

else:
    logger = None
    writer = None

my_args = ''
for idx in range(len(sys.argv)):
    my_args += sys.argv[idx] + ' '
if args.debug:
    print(my_args)
    print(vae)
    print('number of parameters : {0:,}'.format(sum(p.numel() for p in vae.parameters())))
else:
    logger.info(vae)
    logger.info('number of parameters : {0:,}'.format(sum(p.numel() for p in vae.parameters())))
    logger.info(my_args)


vis_dict = {"data": None, "reco": None, "reco_seq": None, "bu_att_seq": None,
            "gene_in": None, "gene_seq_in": None, "exemplar_in": None,
            "gene_ood": None, "gene_seq_ood": None, "exemplar_ood": None, "data_test": None}
#to_plot = ['reco', 'reco_seq', 'gene_in', 'gene_seq_in']
to_plot = ['reco', 'gene_in']
if args.exemplar:
    to_plot += ['gene_ood']

best_loss = np.inf

for epoch in range(args.epoch):

    vae.train()
    train_loss, kl_loss, mse_loss, all_grad = 0, 0, 0, 0
    # all_t = 0
    len_dataset = 0
    batch_exemplar = None
    vis_dict["reco"] = None
    vis_dict["data"] = None
    beta = beta_list[epoch]
    for batch_idx, (data, exemplar, label) in enumerate(train_loader):
        data = data.to(args.device)
        if args.exemplar:
            exemplar = exemplar.to(args.device)
        else :
            exemplar = None
        len_dataset += 1
        mus, log_vars, x_hat, _, _, _, _ = vae(data, exemplar=exemplar, low_memory=True)

        loss, mse, kld = loss_vae_bernouilli(x_hat, data, args, mus=mus, log_vars=log_vars, beta=beta)
        #
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(vae.parameters(), max_norm=args.clip_value, norm_type=2)
        grad_norm = compute_parameter_grad(vae)
        if batch_idx == len(train_loader) - 2:
            vis_dict["data"] = data.detach()
            vis_dict["reco"] = x_hat.detach()
            if args.exemplar:
                vis_dict["exemplar_in"] = exemplar.detach()

        # accumulate for epoch monitoring
        train_loss += loss.item()
        mse_loss += mse.item()
        kl_loss += kld.item()
        all_grad += grad_norm


        optimizer.step()

        if batch_idx % monitor_steps == 0:
            to_print = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tMSE: {:.3f}\tKL: {:.3f}\tgrad: {:.3f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       mse.item(),
                       kld.item(),
                       grad_norm)
            if args.debug:
                print(to_print)
            else:
                logger.info(to_print)

    train_loss /= len(train_loader)
    mse_loss /= len(train_loader)
    kl_loss /= len(train_loader)
    all_grad /= len(train_loader)


    to_print = '====> Epoch: {} Avg loss: {:.4f} -- Avg mse: {:.4f} -- Avg kl: {:.4f} -- Avg grad: {:.4f} '.format(
        epoch, train_loss,
        mse_loss,
        kl_loss,
        all_grad)

    if args.debug:
        print(to_print)
    else:
        logger.info(to_print)

    if writer is not None:
        if writer == 'wandb':
            wandb.log({"Loss/train": train_loss,
                       "MSE/train": mse_loss,
                       "KL/train": kl_loss,
                       "Gradient/train": all_grad}, step=epoch)
            wandb.watch(vae)
        else:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("MSE/train", mse_loss, epoch)
            writer.add_scalar("KL/train", kl_loss, epoch)
            writer.add_scalar("Gradient/train", all_grad, epoch)


    ##evaluation
    with torch.no_grad():
        vae.eval()
        eval_loss, eval_kl, eval_mse = 0, 0, 0
        len_dataset = 0
        for batch_idx, (data, exemplar_ood, label) in enumerate(test_loader):
            data = data.to(args.device)
            if args.exemplar:
                exemplar_ood = exemplar_ood.to(args.device)
            else:
                exemplar_ood = None
            len_dataset += data.size(0)
            mus, log_vars, x_hat, _, _, _, _ = vae(data, exemplar=exemplar_ood, low_memory=True)
            loss, mse, kld = loss_vae_bernouilli(x_hat, data, args, mus=mus, log_vars=log_vars)


            #if batch_idx == len(test_loader) - 1:
            if batch_idx == len(test_loader) - 2:
                vis_dict["data_test"] = data.detach()
                vis_dict["reco_test"] = x_hat.detach()
                if args.exemplar:
                    vis_dict["exemplar_ood"] = exemplar_ood.detach()

            eval_loss += loss.item()
            eval_mse += mse.item()
            eval_kl += kld.item()


        eval_loss /= len(test_loader)
        eval_mse /= len(test_loader)
        eval_kl /= len(test_loader)


        if args.rate_scheduler:
            if args.annealing_time is not None:
                if epoch > args.annealing_time:
                    scheduler.step(eval_loss)
            else:
                scheduler.step(eval_loss)




        to_print = '====> TEST Epoch: {} Avg loss: {:.4f} -- Avg mse: {:.4f} -- Avg kl: {:.4f}'.format(
            epoch, eval_loss,
            eval_mse,
            eval_kl)

        if args.debug:
            print(to_print)
        else:
            logger.info(to_print)

        if writer is not None:
            if writer == 'wandb':
                wandb.log({"Loss/eval": eval_loss,
                           "MSE/eval": eval_mse,
                           "KL/eval": eval_kl,
                           "rate": optimizer.param_groups[0]['lr']}, step=epoch)

            else:
                writer.add_scalar("Loss/eval", eval_loss, epoch)
                writer.add_scalar("MSE/eval", eval_mse, epoch)
                writer.add_scalar("KL/eval", eval_kl, epoch)
                writer.add_scalar("rate", optimizer.param_groups[0]['lr'], epoch)
                writer.add_scalar("beta", beta, epoch)
                #writer.add_hparams(hyperparams, {'hparam/eval_loss': eval_loss, 'hparam/train_loss': train_loss}, run_name=args.model_signature)

        if (epoch % visual_steps == 0) or (epoch == args.epoch - 1):
            visual_evaluation(vae, args, exemplar, exemplar_ood, vis_dict, epoch=epoch, best=False, to_plot=to_plot, writer=writer)
        if eval_loss < best_loss:
            visual_evaluation(vae, args, exemplar, exemplar_ood, vis_dict, epoch=epoch, best=True, to_plot=to_plot, writer=writer)
            #visual_evaluation(vae, args, exemplar, exemplar_ood, vis_dict, epoch=epoch, tag='BEST', to_plot=to_plot,
            #                  writer=writer)
            to_print = '====> BEST TEST Avg loss: {:.4f} -- Avg mse: {:.4f} -- Avg kl: {:.4f}'.format(
                eval_loss,
                eval_mse,
                eval_kl)
            if args.debug:
                print(to_print)
            else:
                logger.info(to_print)
            if not args.debug:
                torch.save(vae.state_dict(), args.snap_dir + '_best.model')
            best_loss = eval_loss

if writer == 'wandb':
    wandb.finish()

