import torch
import os
from model.few_shot import ProtoNet
from model.simclr import ImageEmbedding
from utils.loading_tools import load_weights, load_net


def load_embedding(model_name='', path_to_embedding='', device='cpu', image_size=50):
    if path_to_embedding == '':
        def to_embedding(x):
            return x
    else:
        model_args, model_weights = load_weights(path_to_embedding,
                                                 model_name,
                                                 mode='best')

        if model_args.model_name == 'proto_net':
            embedding_model = ProtoNet(model_args.z_size).to(device)
            embedding_model.load_state_dict(model_weights)
            embedding_model.eval()
            def to_embedding(x):
                features, last_layer = embedding_model(x)
                return features


        elif model_args.model_name == "simclr":
            embedding_model = ImageEmbedding(model_args.z_size).to(device)
            embedding_model.load_state_dict(model_weights)
            embedding_model.eval()

            def to_embedding(x):
                features, last_layer = embedding_model(x)
                return features

        else:
            raise NotImplementedError

    return to_embedding


def load_generative_model(path_to_model, model_name, device, mode='best'):

    if path_to_model == '':
        ## this is the human model
        def generate(image, exemplar, w=1):
            return image
    else:
        vae_args, vae_weights = load_weights(path_to_model,
                                         model_name,
                                         mode=mode)

        if vae_args.model_name == 'vae_stn':

            vae_args.device = device
            generative_model = load_net(vae_args).to(device)
            generative_model.load_state_dict(vae_weights)
            generative_model.eval()

            def generate(image, exemplar, w=1):
                sampled_in, _, _ = generative_model.generate(exemplar.size(0), exemplar=exemplar, low_memory=True)

                return sampled_in

        elif vae_args.model_name in ["ns", "cns", "tns", "ctns", "hfsgm"]:
            vae_args.device = device
            generative_model = load_net(vae_args).to(device)
            generative_model.load_state_dict(vae_weights)
            generative_model.eval()

            def generate(image, exemplar=None, w=1):
                sampled_in, _, _ = generative_model.generate(exemplar.size(0), exemplar=exemplar)
                return sampled_in

        elif vae_args.model_name in ['dagan']:
            vae_args.device = device
            gan_model = load_net(vae_args).to(device)
            gan_model.load_state_dict(vae_weights)
            gan_model.eval()

            def generate(image, exemplar, w=1):
                z = torch.randn((exemplar.size(0), gan_model.z_dim)).to(device)
                image = gan_model(exemplar, z)
                return image
        else:
            raise NotImplementedError()
    return generate

def interclass_variability(type='l2'):

    if type == 'cosine':
        def measure(features, proto=None):
            center = features.mean(dim=0, keepdim=True)
            output = f.cosine_similarity(features, center, dim=1, eps=0)
            output = torch.sum(1 - output) / (output.size(0) - 1)
            return output

    elif type == 'l2':
        def measure(features, proto=None):
            output = features.std(dim=0, unbiased=True)
            return output.mean()

    else :
        raise NotImplementedError()
    return measure