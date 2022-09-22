import torchvision
import matplotlib.pyplot as plt
import numpy as np
import datetime
import os
import torch
import logging
import matplotlib.animation as animation
import argparse
#import wandb
import numpy as np


def create_logger(logpath, filepath, displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    return logger


def get_logger(args, file):
    logger = create_logger(logpath=os.path.join(args.snap_dir, 'logs'), filepath=os.path.abspath(file))
    logger.info(args)
    logger.info('PID {}'.format(os.getpid()))
    torch.save(args, args.snap_dir + 'param.config')
    return logger


#def compute_parameter_grad(model):

#    batch_norm = []
#    for name, p in model.named_parameters():
#        norm_grad = p.grad.norm().cpu().numpy()
#        batch_norm.append(norm_grad)
#    return np.mean(batch_norm)


def compute_parameter_grad(model):
    grad_norm = 0
    for name, p in model.named_parameters():
        grad_norm += p.grad.norm().item() if p.grad is not None else 0
    return grad_norm


def make_directories(args, model_class=None):

    full_model_name = args.model_name
    if hasattr(args, 'attention_type'):
        full_model_name += '_' + args.attention_type
    args.model_signature = full_model_name
    data_time = str(datetime.datetime.now())[0:19].replace(' ', '_')

    args.model_signature += '_' + data_time.replace(':', '_')
    if hasattr(args, 'exemplar'):
        if args.exemplar:
            args.model_signature += '_' + 'exVAE'
    if hasattr(args, 'z_size'):
        args.model_signature += '_' + 'z{0}'.format(args.z_size)
    if hasattr(args, 'hidden_size') and hasattr(args, 'k'):
        args.model_signature += '_' + 'hid{0}_k{1}'.format(args.hidden_size, args.k)
    if hasattr(args, 'hidden_prior') and hasattr(args, 'num_layer_prior'):
        args.model_signature += '_' + 'hid_p{0}_layer_p{1}'.format(args.hidden_prior, args.num_layer_prior)
    if hasattr(args, 'time_step'):
        args.model_signature += '_' + 'T{0}'.format(args.time_step)
    if hasattr(args, 'read_size'):
        args.model_signature += '_' + 'rs{0}'.format(args.read_size[-1])
    if hasattr(args, 'write_size'):
        args.model_signature += '_' + 'rs{0}'.format(args.write_size[-1])
    if hasattr(args, 'lstm_size'):
        args.model_signature += '_' + 'lstm{0}'.format(args.lstm_size)
    if hasattr(args, 'beta'):
        args.model_signature += '_' + 'beta{0}'.format(args.beta)
    if hasattr(args, 'order'):
         args.model_signature += '_' + 'order{0}'.format(args.order)
    if hasattr(args, 'size_factor'):
        args.model_signature += '_' + '{0}sf'.format(args.size_factor)
    if args.model_name == 'vae_stn_var':
        if hasattr(args, 'attention_ratio'):
            args.model_signature += '_' + 'attn_ratio{0}'.format(args.attention_ratio)
    if hasattr(args, 'strength'):
        args.model_signature += '_' + 'str={0}'.format(args.strength)
    if hasattr(args, 'annealing_time'):
        if args.annealing_time is not None:
            args.model_signature += '_' + 'BetaAnneal{}'.format(args.annealing_time)
    if hasattr(args, 'shuffle_exemplar'):
        if args.shuffle_exemplar:
            args.model_signature += '_se'
    if hasattr(args, 'rate_scheduler'):
        if args.rate_scheduler:
            args.model_signature += '_rc'
    if hasattr(args, 'embedding_size'):
        args.model_signature += '_' + 'emb_sz={0}'.format(args.embedding_size)
    if model_class == 'pixel_cnn' and hasattr(args, 'latent_size'):
        args.model_signature += '_latent_size=[{0},{1}]'.format(args.latent_size[0], args.latent_size[1])

    if args.tag != '':
        args.model_signature += '_' + args.tag

    if model_class is None:
        snapshots_path = os.path.join(args.out_dir, args.dataset, full_model_name)
    else:
        snapshots_path = os.path.join(args.out_dir, args.dataset, model_class)

    args.snap_dir = snapshots_path + '/' + args.model_signature + '/'

    if args.model_name in ['ns', 'tns', 'hfsgm']:
        args.snap_dir = snapshots_path + '/' + args.model_signature + '_' + str(args.c_dim) + '_' + str(args.z_dim)  + '_' + str(args.hidden_dim)
        if args.model_name == 'tns':
            args.snap_dir += '_' + str(args.n_head)
        args.snap_dir += '/'

    if not args.debug:
        os.makedirs(snapshots_path, exist_ok=True)
        os.makedirs(args.snap_dir, exist_ok=True)
        args.fig_dir = args.snap_dir + 'fig/'
        os.makedirs(args.fig_dir, exist_ok=True)
    else:
        args.fig_dir = None
    return args


def retrieve_vqvae_directories(args):

    snapshots_path = os.path.join(args.out_dir, args.dataset, 'vqvae')
    args.snap_dir = snapshots_path + '/' + args.vqvae_model + '/'

    if not args.debug:
        os.makedirs(snapshots_path, exist_ok=True)
        os.makedirs(args.snap_dir, exist_ok=True)
        args.fig_dir = args.snap_dir + 'fig/'
        os.makedirs(args.fig_dir, exist_ok=True)
    else:
        args.fig_dir = None
    return args

def show(img, title=None, saving_path=None, figsize=(8, 8), dpi=100):
    npimg = img.numpy()
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    plt.axis('off')
    if title is not None:
        plt.title(title)
    if saving_path is None:
        plt.show()
    else:
        plt.savefig(saving_path + '/' + title + '.png')
    plt.close()

def show_with_ax(img, ax, title=None, saving_path=None):
    npimg = img.numpy()
    ax.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    ax.set_axis_off()
    if title is not None:
        ax.set_title(title)

def generate_img_grid(data, nrow=4, ncol=8, padding=2, normalize=True, pad_value=0):
    nb_image = nrow * ncol
    data_to_plot = torchvision.utils.make_grid(data[:nb_image], nrow=ncol, padding=padding, normalize=normalize,
                                               pad_value=pad_value)
    return data_to_plot.detach().cpu()

def plot_img(data, nrow=4, ncol=8, padding=2, normalize=True, saving_path=None, title=None, pad_value=0, figsize=(8, 8), dpi=100, scale_each=False):
    nb_image = nrow * ncol
    data_to_plot = torchvision.utils.make_grid(data[:nb_image], nrow=ncol, padding=padding, normalize=normalize,
                                               pad_value=pad_value, scale_each=scale_each)
    show(data_to_plot.detach().cpu(), saving_path=saving_path, title=title, figsize=figsize, dpi=dpi)


def make_grid(data, nrow=4, ncol=8, padding=2, normalize=True, pad_value=0, scale_each=False):
    nb_image = nrow * ncol
    data_to_plot = torchvision.utils.make_grid(data[:nb_image], nrow=ncol, padding=padding, normalize=normalize,
                                               pad_value=pad_value, scale_each=scale_each)
    return data_to_plot

def plot_gif(data, nrow=8, ncol=4, padding=2, normalize=True, saving_path=None, title=None, pad_value=0, figsize=(8,8)):
    time_step = data.size(1)
    if title is not None:
        plt.title(title)
    #fig = plt.figure()  # initialise la figure
    fig, ax = plt.subplots(figsize=figsize)
    plt.axis('off')
    nb_image = nrow*ncol
    data_to_plot = torchvision.utils.make_grid(data[:nb_image, 0, :, :, :], nrow=ncol, padding=padding,
                                               normalize=normalize, pad_value=pad_value)
    im = plt.imshow(np.transpose(data_to_plot.detach().cpu(), (1, 2, 0)), interpolation='nearest')

    def animate(i):
        nex_data = torchvision.utils.make_grid(data[:nb_image, i, :, :, :], nrow=nrow, padding=padding,
                                               normalize=normalize, pad_value=pad_value)
        im.set_array(np.transpose(nex_data.detach().cpu(), (1, 2, 0)))
        return [im]

    if title is not None:
        fig.suptitle(title)

    ani = animation.FuncAnimation(fig, animate, frames=time_step, blit=False, interval=10, repeat=False)

    if title is not None and saving_path is not None:
        ani.save(saving_path + '/' + title + '.gif', fps=10, writer='imagemagick')

    plt.clf()

def visualize(model, vis_dict, args, epoch, best=False, to_plot=None, nrow=8, ncol=8, writer=None):

    epoch_tag = '_ep={0:0.0f}'.format(epoch)
    if best:
        epoch_tag = 'BEST'
        best_tag = '_BEST'
    else :
        best_tag='_END'
    #if tag ==
    #if tag is not None:
    #    if isinstance(epoch, int):
    #        epoch_tag = '_ep={0:0.0f}'.format(epoch)
    #    elif isinstance(epoch, str):
    #        epoch_tag = '_' + epoch
    #else:
    #    epoch_tag = ''

    if to_plot is None:
        to_plot = ["bu_att_seq", "reco_seq", "reco", "gene_in", "gene_seq_in", "gene_ood", "gene_seq_ood"]
    model.eval()
    if vis_dict["reco_seq"] is not None:
        seq_len = (model.time_steps+1)*vis_dict["reco_seq"].size(0)
        size_seq_image = (seq_len, args.input_shape[-3], args.input_shape[-2], args.input_shape[-1])

    if vis_dict["bu_att_seq"] is not None and "bu_att_seq" in to_plot:
        bu_att_seq = vis_dict["bu_att_seq"]

        plot_img(bu_att_seq.view(seq_len, 1, model.imagette_size[-2], model.imagette_size[-1]),
                 saving_path=args.fig_dir,
                 title='bu_att_seq' + epoch_tag,
                 nrow=nrow, ncol=model.time_steps+1)

    if vis_dict["reco_seq"] is not None and "reco_seq" in to_plot:
        all_reco = vis_dict["reco_seq"]
        if writer is not None:
            to_writer = make_grid(all_reco.view(size_seq_image), nrow=nrow, ncol=model.time_steps+1)
            if writer == 'wandb':
                wandb.log({'Reco_Sequential' + best_tag: wandb.Image(to_writer)}, step=epoch)
            else:
                writer.add_image('Reco_Sequential' + best_tag, to_writer, epoch)

        plot_img(all_reco.view(size_seq_image),
                 saving_path=args.fig_dir,
                 title='reco_seq' + epoch_tag,
                 nrow=nrow, ncol=model.time_steps+1)

    # if vis_dict["gene_seq_in"] is not None and "gene_seq_in" in to_plot:
    #    gene_seq_in = vis_dict["gene_seq_in"]
    #    if vis_dict["exemplar_in"] is not None

    # if vis_dict["gene_seq_ood"] is not None and "gene_seq_ood" in to_plot:
    #    gene_seq_ood = vis_dict["gene_seq_ood"].view(size_seq_image)

    if vis_dict["reco"] is not None and "reco" in to_plot:
        reco = vis_dict["reco"].view_as(vis_dict["data"])
        nrow_reco = 10
        ncol_reco = 16
        cat_data = torch.cat([vis_dict["data"][0:ncol_reco], reco[0:ncol_reco]], dim=0)
        for i in range(1, nrow_reco//2):
            cat_data = torch.cat([cat_data, vis_dict["data"][i * ncol_reco:(i + 1) * ncol_reco],
                                  reco[i * ncol_reco:(i + 1) * ncol_reco]], dim=0)

        #cat_data = torch.cat([vis_dict["data"][0:ncol], reco[0:ncol], vis_dict["data"][ncol:2*ncol], reco[ncol:2*ncol]],
        #                     dim=0)
        if writer is not None:
            to_writer = make_grid(cat_data, nrow=nrow_reco, ncol=ncol_reco)
            if writer == 'wandb':
                wandb.log({'Reco_end' + best_tag: wandb.Image(to_writer)}, step=epoch)
            else:
                writer.add_image('Reco_end' + best_tag, to_writer, epoch)
        plot_img(cat_data,
                 saving_path=args.fig_dir,
                 title='reco_ep' + epoch_tag,
                 nrow=nrow_reco, ncol=ncol_reco)

    if vis_dict["reco_test"] is not None and "reco_test" in to_plot:
        reco = vis_dict["reco_test"].view_as(vis_dict["data_test"])
        nrow_reco = 10
        ncol_reco = 16
        cat_data = torch.cat([vis_dict["data_test"][0:ncol_reco], reco[0:ncol_reco]], dim=0)
        for i in range(1, nrow_reco//2):
            cat_data = torch.cat([cat_data, vis_dict["data_test"][i * ncol_reco:(i + 1) * ncol_reco],
                                  reco[i * ncol_reco:(i + 1) * ncol_reco]], dim=0)

        #cat_data = torch.cat([vis_dict["data"][0:ncol], reco[0:ncol], vis_dict["data"][ncol:2*ncol], reco[ncol:2*ncol]],
        #                     dim=0)
        if writer is not None:
            to_writer = make_grid(cat_data, nrow=nrow_reco, ncol=ncol_reco)
            if writer == 'wandb':
                wandb.log({'Reco_test' + best_tag: wandb.Image(to_writer)}, step=epoch)
            else:
                writer.add_image('Reco_test' + best_tag, to_writer, epoch)
        plot_img(cat_data,
                 saving_path=args.fig_dir,
                 title='reco_test_ep' + epoch_tag,
                 nrow=nrow_reco, ncol=ncol_reco)

    if vis_dict["gene_in"] is not None and "gene_in" in to_plot:
        gene_in = vis_dict["gene_in"]
        if vis_dict["exemplar_in"] is not None:
            exemplar = vis_dict["exemplar_in"][:16]
            cat_data = torch.cat([exemplar, gene_in], dim=0)
            nrow, ncol = 11, 16
        else:
            cat_data = gene_in
        if writer is not None:
            to_writer = make_grid(cat_data, nrow=nrow, ncol=ncol)
            if writer == 'wandb':
                wandb.log({'Gene_IID' + best_tag: wandb.Image(to_writer)}, step=epoch)
            else:
                writer.add_image('Gene_IID' + best_tag , to_writer, epoch)

        plot_img(cat_data,
                 saving_path=args.fig_dir,
                 title='gene_in' + epoch_tag,
                 nrow=nrow, ncol=ncol)

    if vis_dict["gene_ood"] is not None and "gene_ood" in to_plot:
        gene_ood = vis_dict["gene_ood"]
        if vis_dict["exemplar_ood"] is not None:
            exemplar = vis_dict["exemplar_ood"][:16]
            cat_data = torch.cat([exemplar, gene_ood], dim=0)
            nrow, ncol = 11, 16
        else:
            cat_data = gene_ood

        if writer is not None:
            to_writer = make_grid(cat_data, nrow=nrow, ncol=ncol)
            if writer == 'wandb':
                wandb.log({'Gene_OOD' + best_tag: wandb.Image(to_writer)}, step=epoch)
            else:
                writer.add_image('Gene_OOD' + best_tag, to_writer, epoch)
        plot_img(cat_data,
                 saving_path=args.fig_dir,
                 title='gene_ood' + epoch_tag,
                 nrow=nrow, ncol=ncol)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'T', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'F', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def visual_evaluation(model, args, exemplar, exemplar_ood, vis_dict, epoch=0, best=False, to_plot=[], writer=None):
    model.eval()

    if args.exemplar:
        exemplar_in = vis_dict['exemplar_in'][0:16, :, :, :].repeat(10, 1, 1, 1)
        exemplar_ood = vis_dict['exemplar_ood'][0:16, :, :, :].repeat(10, 1, 1, 1)
        n_samples = exemplar_in.size(0)
    else :
        exemplar_ood, exemplar_in = None, None
        n_samples = 10*11

    if 'gene_in' in to_plot:
        sampled_in, _, gene_seq_in = model.generate(n_samples, exemplar=exemplar_in)
        vis_dict["gene_in"], vis_dict["gene_seq_in"] = sampled_in, gene_seq_in

    if (exemplar_ood is not None) and ('gene_ood' in to_plot):
        sampled_ood, _, gene_seq_ood = model.generate(n_samples, exemplar=exemplar_ood)
        vis_dict["gene_ood"], vis_dict["gene_seq_ood"] = sampled_ood, gene_seq_ood
    visualize(model, vis_dict, args, epoch, best=best, to_plot=to_plot, writer=writer)

    if not args.debug:
        torch.save(model.state_dict(), args.snap_dir + '_end.model')

    if epoch == args.epoch - 1:
        if vis_dict["reco_seq"] is not None :
            plot_gif(vis_dict["reco_seq"], saving_path=args.fig_dir, title='anim_reco')
        if vis_dict["gene_seq_in"] is not None:
            plot_gif(vis_dict["gene_seq_in"], saving_path=args.fig_dir, title='anim_gene')


def visual_evaluation_vqvae(prior_model, vqvae_model, args, vqvae_args, exemplar, exemplar_ood, vis_dict, epoch=0, best=False, to_plot=[], writer=None):
    prior_model.eval()
    vqvae_model.eval()

    if args.exemplar:
        exemplar_in = vis_dict['exemplar_in'][0:16, :, :, :].repeat(10, 1, 1, 1)
        exemplar_ood = vis_dict['exemplar_ood'][0:16, :, :, :].repeat(10, 1, 1, 1)
        n_samples = exemplar_in.size(0)
    else :
        exemplar_ood, exemplar_in = None, None
        n_samples = 10*11

    with torch.no_grad():
        if 'gene_in' in to_plot:
            if args.exemplar:
                if args.model_name in ['pixel_cnn', 'pixel_cnn3', 'pixel_snail', 'pixel_snail_small', 'vae_prior']:
                    if vqvae_args.exemplar:
                        latent_exemplar = vqvae_model.encode(exemplar_in, exemplar=exemplar_in)
                    else:
                        latent_exemplar = vqvae_model.encode(exemplar_in)
                elif args.model_name == 'pixel_cnn2':
                    latent_exemplar = exemplar_in
                elif args.model_name in ['pixel_snail', 'pixel_snail_small']:
                    logits, _ = prior(latents, latent_exemplar)
                elif args.model_name == 'vae_prior':
                    logits, _, _ = prior(latents, latent_exemplar)
            else :
                latent_exemplar = None
            #generated_latent = prior_model.generate(args, n_samples, exemplar=exemplar_in, shape=(12, 12))
            generated_latent = prior_model.generate(args, n_samples, exemplar=latent_exemplar, shape=args.latent_size)
            sampled_in = vqvae_model.decode(generated_latent)
            #sampled_in, _, gene_seq_in = model.generate(n_samples, exemplar=exemplar_in)
            vis_dict["gene_in"] = sampled_in

    with torch.no_grad():
        if (exemplar_ood is not None) and ('gene_ood' in to_plot):
            if args.exemplar:
                if args.model_name in ['pixel_cnn', 'pixel_cnn3', 'pixel_snail','pixel_snail_small', 'vae_prior']:
                    if vqvae_args.exemplar:
                        latent_exemplar = vqvae_model.encode(exemplar_ood, exemplar=exemplar_ood)
                    else:
                        latent_exemplar = vqvae_model.encode(exemplar_ood)
                elif args.model_name == 'pixel_cnn2':
                    latent_exemplar = exemplar_ood
                elif args.model_name in ['pixel_snail', 'pixel_snail_small']:
                    logits, _ = prior(latents, latent_exemplar)
                elif args.model_name == 'vae_prior':
                    logits, _, _ = prior(latents, latent_exemplar)
            else:
                latent_exemplar = None
            #generated_latent = prior_model.generate(args, n_samples, exemplar=exemplar_ood, shape=(12, 12))
            generated_latent = prior_model.generate(args, n_samples, exemplar=latent_exemplar, shape=args.latent_size)
            sampled_ood = vqvae_model.decode(generated_latent)
            vis_dict["gene_ood"] = sampled_ood
    visualize(prior_model, vis_dict, args, epoch, best=best, to_plot=to_plot, writer=writer)

    if not args.debug:
        torch.save(prior_model.state_dict(), args.snap_dir + '_prior_end.model')
