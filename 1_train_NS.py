import argparse
import torch
from utils.monitoring import make_directories, compute_parameter_grad, get_logger, \
    plot_gif, visualize, plot_img, str2bool, visual_evaluation
from utils.data_loader import load_dataset_exemplar
from model.NS.model import select_model
from model.NS.util import select_optimizer, count_params, model_kwargs, set_seed
from torch import optim
import torch.nn as nn
import numpy as np
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau
from operator import add
import wandb
import sys


from model.NS.parser import parse_args

parser = parse_args()
parser.add_argument('--download_data', type=eval, default=False, choices=[True, False])
parser.add_argument('--dataset_root', type=str, default="/media/data_cifs_lrs/projects/prj_control/data")
parser.add_argument('--input_type', type=str, default='binary',
                    choices=['binary'], help='type of the input')
parser.add_argument('--batch_size', type=int, default=128, metavar='BATCH_SIZE',
                    help='input batch size for training')
parser.add_argument('--learning_rate', type=float, default=1e-3, metavar='LR',
                    help='learning rate of the optimizer')
parser.add_argument('--epoch', type=int, default=200, metavar='EPOCH', help='number of epoch')
parser.add_argument("--input_shape", nargs='+', type=int, default=[1, 50, 50],
                    help='shape of the input [channel, height, width]')
parser.add_argument('-od', '--out_dir', type=str, default='X',
                    metavar='OUT_DIR', help='output directory for model snapshots etc.')
parser.add_argument('--debug', default=False, action='store_true', help='debugging flag (do not save the network)')
parser.add_argument('--beta_adam', type=float, default=0.9, help='value of the first order beta in adam optimizer')
parser.add_argument('--clip_value', type=float, default=5, help='value of the gradient clipping')
parser.add_argument("--exemplar", type=str2bool, nargs='?', const=True, default=True, help="For conditional VAE")
parser.add_argument('--model_name', type=str, default='ns', choices=['ns', 'hfsgm', 'chfsgm_multi', 'tns', 'cns', 'ctns'],
                    help="type of the model ['ns', 'hfsgm', 'chfsgm_multi', 'tns', 'cns', 'ctns]")
parser.add_argument('--preload', default=True, action='store_true', help='preload the dataset')
parser.add_argument('--beta', type=float, default=1.0, metavar='BETA', help='beta that weight the KL in the vae Loss')
parser.add_argument("--exemplar_type", default='prototype', choices=['prototype', 'first', 'shuffle'],
                    metavar='EX_TYPE', help='type of exemplar')

args = parser.parse_args()


if args.device == 'meso':
    args.device = torch.cuda.current_device()


default_args = parser.parse_args([])

if args.seed is not None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

if args.debug:
    visual_steps, monitor_steps = 1, 25
else:
    visual_steps, monitor_steps = 100, 200

args = make_directories(args)
kwargs = {'preload': args.preload}

train_loader, test_loader, args = load_dataset_exemplar(args, shape=args.input_shape, drop_last=True, **kwargs)


vae = select_model(args)(**model_kwargs(args))
vae = vae.to(args.device)

# We use the standard Neural Statistician architecture in our experiments, 
# which can be found at model/NS/model/NS/ns.py

# Majority of the code is adopted from https://github.com/georgosgeorgos/hierarchical-few-shot-generative-models
# and https://github.com/comRamona/Neural-Statistician


optimizer = optim.Adam(vae.parameters(), lr=args.learning_rate,  weight_decay=args.weight_decay)
scheduler = ReduceLROnPlateau(
    optimizer,
    mode="max",
    patience=args.patience,
    factor=args.lr_step,
    min_lr=args.lr_min,
)
if not args.debug:
    logger = get_logger(args, __file__)
    writer = None
else:
    logger = None
    writer = None

print(vae)
to_print = 'number of parameters : {0:,}'.format(sum(p.numel() for p in vae.parameters()))
print(to_print)
logger.info(to_print)
# exit()

vis_dict = {"data": None, "reco": None, "reco_seq": None, "bu_att_seq": None,
        "gene_in": None, "gene_seq_in": None, "exemplar_in": None,
        "gene_ood": None, "gene_seq_ood": None, "exemplar_ood": None, "data_test": None}
to_plot = ['reco', 'gene_in']
if args.exemplar:
    to_plot += ['gene_ood']

best_loss = np.inf

def lr_f(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

from utils.custom_transform import Binarize_batch, Scale_0_1_batch
scale_01, binarize = Scale_0_1_batch(), Binarize_batch(binary_threshold=0.5)

#from torchvision.utils import save_image
#def save_test_grid(inputs, samples, save_path, n=30):
#    size = 50
#    inputs = 1 - inputs.cpu().data.view(-1, 5, 1, size, size)[:n]
#    reconstructions = samples.cpu().data.view(-1, 5, 1, size, size)[:n]
#    images = torch.cat((inputs, reconstructions), dim=1).view(-1, 1, size, size)
#    save_image(images, save_path, nrow=n)
#    return images

size = args.input_shape[-1]

print("Beta = ", args.beta)

for epoch in range(args.epoch):

    vae.train()
    train_loss, kl_loss, mse_loss, all_grad, train_vlb = 0, 0, 0, 0, 0
    len_dataset = 0
    batch_exemplar = None
    vis_dict["reco"] = None
    vis_dict["data"] = None
    for batch_idx, data in enumerate(train_loader):

        x=data
        x = x.to(args.device).float()
        out = vae.step(x, 
                        args.alpha, 
                        optimizer, 
                        args.clip_gradients, 
                        args.free_bits,
                        args.beta)
        loss = out["loss"]
        mse = out["mse"]
        kl_c = out["kl_c"] 
        kl_z = out["kl_z"]
        kld = kl_c + kl_z
        x_hat = out["x_rec"]
        vlb = out["vlb"]
        x_hat = x_hat[:, :1, :, :, :].reshape((-1, 1, size, size))

        exemplar = data[:, 1:2, :, :, :].to(args.device).reshape((-1, 1, 50, 50)).float()

        # monitor visualization
        if batch_idx == len(train_loader) - 2:
            data = data[:, :1, :, :, :].to(args.device).reshape((-1, 1, 50, 50)).float()
            vis_dict["data"] = data.detach()
            if args.model_name == 'vae_draw':
                vis_dict["reco"] = x_hat.probs.detach()
            else:
                vis_dict["reco"] = x_hat.detach()
            if args.exemplar:
                vis_dict["exemplar_in"] = exemplar.detach()


        # accumulate for epoch monitoring
        train_loss += loss.item()
        mse_loss += mse.item()
        kl_loss += kld.item()

        train_vlb += vlb.item()


        if batch_idx % monitor_steps == 0:
            to_print = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tMSE: {:.3f}\tVLB: {:.3f}\tKL_C: {:.3f}\tKL_Z: {:.3f}\talpha: {:.4f}\tlr: {:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    mse.item(),
                    vlb.item(),
                    kl_c.item(),
                    kl_z.item(),
                    args.alpha, 
                    lr_f(optimizer))
            if args.debug:
                print(to_print)
            else:
                logger.info(to_print)

    train_loss /= len(train_loader)
    mse_loss /= len(train_loader)
    kl_loss /= len(train_loader)
    all_grad /= len(train_loader)

    args.alpha *= args.alpha_step

    to_print = '====> Epoch: {} Avg loss: {:.4f} -- Avg mse: {:.4f} -- Avg kl: {:.4f} -- Avg grad: {:.4f} -- alpha:{:.2f} -- lr:{:.6f} '.format(
        epoch, train_loss,
        mse_loss,
        kl_loss,
        all_grad, 
        args.alpha, 
        lr_f(optimizer))

    if args.debug:
        print(to_print)
    else:
        logger.info(to_print)


    #evaluation
    with torch.no_grad():
        vae.eval()
        eval_loss, eval_kl, eval_mse, eval_vlb = 0, 0, 0, 0
        len_dataset = 0
        for batch_idx, data in enumerate(test_loader):
            x_test = data
            x_test = x_test.to(args.device).float()
            out = vae.forward(x_test)
            mse = out["mse"]
            x_hat = out["xp"]
            x_hat = x_hat[:, :1, :, :, :].reshape((-1, 1, 50, 50))
            out = vae.loss(out)
            loss = out["loss"]
            kld = out["kl_c"] + out["kl_z"]
            vlb_test = out["vlb"]

            exemplar_ood = data[:, 1:2, :, :, :].to(args.device).reshape((-1, 1, 50, 50))
            
            if batch_idx == len(test_loader) - 2:
                data = data[:, :1, :, :, :].to(args.device).reshape((-1, 1, 50, 50)).float()
                vis_dict["data_test"] = data.detach()
                if args.model_name == 'vae_draw':
                    vis_dict["reco_test"] = x_hat.probs.detach()
                else:
                    vis_dict["reco_test"] =x_hat.detach()
                if args.exemplar:
                    exemplar_ood = exemplar_ood.to(args.device).float()
                    vis_dict["exemplar_ood"] = exemplar_ood.detach()

            eval_loss += loss.item()
            eval_mse += mse.item()
            eval_kl += kld.item()
            eval_vlb += vlb_test.item()

        eval_loss /= len(test_loader)
        eval_mse /= len(test_loader)
        eval_kl /= len(test_loader)
        eval_vlb /= len(test_loader)

        scheduler.step(eval_vlb)


        to_print = '====> TEST Epoch: {} Avg loss: {:.4f} -- Avg mse: {:.4f} -- Avg kl: {:.4f} -- alpha:{:.2f} -- lr:{:.6f}'.format(
            epoch, eval_loss,
            eval_mse,
            eval_kl, 
            args.alpha, 
            lr_f(optimizer))

        if args.debug:
            print(to_print)
        else:
            logger.info(to_print)

        if (epoch % visual_steps == 0) or (epoch == args.epoch - 1):
            visual_evaluation(vae, args, exemplar.to(args.device), exemplar_ood.float(), vis_dict, epoch=epoch, best=False, to_plot=to_plot, writer=writer)
        if eval_loss < best_loss:
            visual_evaluation(vae, args, exemplar.to(args.device), exemplar_ood.float(), vis_dict, epoch=epoch, best=True, to_plot=to_plot, writer=writer)
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

            filename = args.snap_dir  + f"{args.sample_size}-shot.png"
            path = filename
            with torch.no_grad():
                samples = vae.conditional_sample_cqL(x_test)["xp"]
            conditional_samples = save_test_grid(x_test, samples, path)

            filename = args.snap_dir + f"{args.sample_size}-shot_REC.png"
            path = filename
            with torch.no_grad():
                samples = vae.reconstruction(x_test)
            conditional_samples = save_test_grid(x_test, samples, path)

