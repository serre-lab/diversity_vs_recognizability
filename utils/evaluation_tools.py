import torch
import os
from model.few_shot import ProtoNet, compute_prototypes, pairwise_distances
from model.simclr import ImageEmbedding
from utils.loading_tools import load_weights, load_net
import math
import torch.nn as nn
from collections import OrderedDict




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


def generate_evaluation_task(args, batch_stop= None, nb_test=500, nb_class=150, seed=44):
    if seed is not None:
        torch.manual_seed(seed)

    all_label = torch.arange(nb_class)
    if batch_stop is not None:
        all_label = all_label[:batch_stop]

    fixed_task = []

    label_extended = all_label.repeat(nb_test)
    range1 = (len(all_label) // args.k_test) * args.k_test
    nb_range_1 = int(math.ceil(label_extended.size(0) / range1))

    for i_r_1 in range(nb_range_1):
        if i_r_1 == nb_range_1 - 1:
            int_lab = label_extended[i_r_1 * range1:]
        else:
            int_lab = label_extended[i_r_1 * range1: (i_r_1 + 1) * range1]

        int_lab = int_lab[torch.randperm(int_lab.size(0))]
        if int_lab.size(0) % args.k_test != 0:
            raise Exception("int lab not divisible size ({0}) by k_way ({1})".format(int_lab.size(0), args.k_test))
        nb_range_2 = int_lab.size(0) // args.k_test

        for i_r_2 in range(nb_range_2):
            fixed_task.append(int_lab[i_r_2 * args.k_test: (i_r_2 + 1) * args.k_test].numpy())

    return fixed_task


def classifier_prediction(path_to_classifier, classifier_name, device, args):
        #model, model_name, device, args

    few_shot_args, few_shot_weights = load_weights(
            path_to_classifier,
            classifier_name,
            mode='best')

    if (few_shot_args.model_name == 'proto_net'):

        few_shot_model = ProtoNet(z_size=few_shot_args.z_size, num_input_channels=1).to(device)
        check_args = (args.k_test == few_shot_args.k_test) and \
                     (args.n_test == few_shot_args.n_test) and \
                     (args.q_test == few_shot_args.q_test)
        if not check_args:
            raise NameError('the dataset args and the network args are not the same')
        few_shot_model.load_state_dict(few_shot_weights)
        few_shot_model.eval()

        def predict(image):
            _, metric_layer = few_shot_model(image)
            support = metric_layer[:few_shot_args.n_test * few_shot_args.k_test]
            queries = metric_layer[few_shot_args.n_test * few_shot_args.k_test:]
            prototypes = compute_prototypes(support, few_shot_args.k_test, few_shot_args.n_test)
            distances = pairwise_distances(queries, prototypes, few_shot_args.distance)
            y_pred = (-distances).softmax(dim=1)
            return y_pred, distances

    else:
        raise NotImplementedError()

    return predict