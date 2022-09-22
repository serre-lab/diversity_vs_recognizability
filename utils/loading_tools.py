import os
import torch
from model.network import VaeStn

def load_weights(out_dir, model_name=None, mode='end', weight=True):

    if model_name is not None:
        path_to_load = os.path.join(out_dir, model_name)
    else:
        path_to_load = out_dir
    path_args = os.path.join(path_to_load, 'param.config')
    loaded_args = torch.load(path_args)
    if mode == 'end':
        path_weights = os.path.join(path_to_load, '_end.model')
    elif mode == 'best':
        path_weights = os.path.join(path_to_load, '_best.model')

    if weight == True:
        loaded_weights = torch.load(path_weights, map_location='cpu')
    else:
        loaded_weights = None

    return loaded_args, loaded_weights

def load_net(args):
    if args.model_name == 'vae_stn':
        model = VaeStn(args)
    elif args.model_name in ["ns", "tns", "ctns", "cns", "hfsgm"]:
        from model.NS.model import select_model
        from model.NS.util import select_optimizer, count_params, model_kwargs, set_seed
        model = select_model(args)(**model_kwargs(args))
    elif args.model_name in ['dagan']:
        if args.architecture == 'UResNet':
            from model.DAGAN.generator import Generator
            model = Generator(dim=50, channels=1, dropout_rate=0.5, z_dim=args.z_size)
        elif args.architecture == 'ResNet':
            from model.DAGAN.generator import ResNetGenerator
            model = ResNetGenerator(dim=50, channels=1, dropout_rate=0.5, z_dim=args.z_size)
        else:
             raise ValueError(args.architecture + " not defined. Choose from {ResNet, UResNet}")

    else:
        raise NotImplementedError()
    return model