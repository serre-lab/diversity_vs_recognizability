import torch
import argparse

import json
from utils.data_loader import load_dataset_exemplar
from utils.monitoring import str2bool, plot_img
#from few_shot.proto import compute_prototypes

from torch.utils.data import DataLoader
#from model.network import SeqVaeCondGeneral2b
import matplotlib.pyplot as plt
import torch.nn.functional as f
import torchvision
import numpy as np
from utils.custom_loader import OmniglotDataset
import math
from utils.evaluation_tools import load_generative_model, generate_evaluation_task, classifier_prediction

import os

from utils.custom_transform import Binarize, Invert, Scale_0_1, Dilate, ScaleCenter
import torchvision.transforms as tforms
from utils.custom_loader import NShotTaskSampler
from utils.custom_transform import Binarize_batch, Scale_0_1_batch


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='omniglot', choices=['omniglot_weak', 'human_drawing', 'omniglot'],
                    metavar='DATASET', help='Dataset choice.')
#parser.add_argument('--out_dir', type=str, default='/media/data_cifs/projects/prj_zero_gene/exp/',
#                    metavar='OUT_DIR', help='output directory for model saving etc.')
parser.add_argument('--device', type=str, default='cuda:0', help='cuda device')

parser.add_argument('--input_type', type=str, default='binary',
                    choices=['binary'], help='type of the input')
parser.add_argument('--learning_rate', type=float, default=1e-3, metavar='LR',
                    help='learning rate of the optimizer')
parser.add_argument('--seed', type=int, default=None, metavar='SEED', help='random seed (None is no seed)')
parser.add_argument('--epoch', type=int, default=200, metavar='EPOCH', help='number of epoch')
parser.add_argument('--debug', default=False, action='store_true', help='debugging flag (do not save the network)')
parser.add_argument('--tag', type=str, default='', help='tag of the experiment')
parser.add_argument("--input_shape", nargs='+', type=int, default=[1, 48, 48],
                    help='shape of the input [channel, height, width]')
parser.add_argument('--download_data', type=eval, default=False, choices=[True, False])
parser.add_argument('--dataset_root', type=str, default="/media/data_cifs_lrs/projects/prj_control/data")
parser.add_argument("--augment", type=str2bool, nargs='?', const=True, default=False, help="data augmentation")
parser.add_argument('--scale_center', default=False, action='store_true', help='rescale and recenter the dataset')
parser.add_argument('--preload', default=True, action='store_true', help='preload the dataset')
#parser.add_argument('--episode_per_epoch', default=100, type=int)
parser.add_argument('--model_name', type=str, default='proto_net', choices=['proto_net'],
                    help="type of the model ['vae_stn', 'vae_draw']")
parser.add_argument("--exemplar_type", default='prototype', choices=['prototype', 'first', 'shuffle'],
                    metavar='EX_TYPE', help='type of exemplar')
parser.add_argument('--n-train', default=1, type=int) #n shots
parser.add_argument('--n-test', default=1, type=int) #n shots
parser.add_argument('--k-train', default=60, type=int) #k ways
parser.add_argument('--k-test', default=20, type=int) #k ways
parser.add_argument('--q-train', default=5, type=int) #query
parser.add_argument('--q-test', default=1, type=int) #query
args = parser.parse_args()

batch_stop = None
nb_class = 150
nb_test = 500
kwargs = {'preload': args.preload}
scale_01, binarize = Scale_0_1_batch(), Binarize_batch(binary_threshold=0.5)
equalize_statistics = True

evaluation_task = generate_evaluation_task(args, batch_stop= batch_stop, nb_test=nb_test, nb_class=nb_class)
args.episodes_per_epoch = len(evaluation_task)
train_loader, test_loader, args = load_dataset_exemplar(args, shape=args.input_shape, few_shot=True,
                                                        fixed_tasks=evaluation_task, **kwargs)

resize_func = torchvision.transforms.Resize(50)

with open('config.json') as json_file:
    data = json.load(json_file)
experiment = data['exp1']

root_path = './experiment/'
saving_name = root_path + 'accu_diffusion_5'

results = []
all_label = torch.arange(nb_class)
## can be heavily speed up by increasing the batch size

classify_function= []
generation_function = []
results = []

with torch.no_grad():
    for idx_exp, each_exp in enumerate(experiment):
        result_on_exp = {'signature': ''}
        result_on_exp['accuracy'] = torch.zeros(len(all_label))
        result_on_exp['data'] = []
        result_on_exp['logits'] = []
        result_on_exp['labels'] = []

        # result_on_exp = {'': ''}
        for idx, keys in enumerate(each_exp.keys()):
            if keys in ['model', 'classifier']:
                result_on_exp['signature'] += each_exp[keys]
                if idx != len(each_exp.keys()) - 1:
                    result_on_exp['signature'] += '_'
        result_on_exp['classifier_name'] = each_exp['classifier_name']
        result_on_exp['model_name'] = each_exp['model_name']
        if 'w' in each_exp:
            result_on_exp['model_name'] += '_w={0}'.format(each_exp['w'])
            result_on_exp['w'] = each_exp['w']
        result_on_exp['path_to_model'] = each_exp['path_to_model']
        result_on_exp['path_to_classifier'] = each_exp['path_to_classifier']

        if 'mode' in each_exp.keys():
            result_on_exp['mode'] = each_exp['mode']
        else:
            result_on_exp['mode'] = 'best'
        classify_function.append(classifier_prediction(path_to_classifier=each_exp['path_to_classifier'],
                                                       classifier_name=each_exp['classifier_name'], device=args.device,
                                                       args=args))
        generation_function.append(load_generative_model(path_to_model=each_exp['path_to_model'], model_name=each_exp['model_name'],
                                                    device=args.device, mode=result_on_exp['mode']))
        results.append(result_on_exp)

    target = torch.arange(args.k_test).to(args.device)
    for idx, (data, exemplar, label) in enumerate(test_loader):
        if idx % 50 == 0:
            print("{}/{}".format(idx+1, len(test_loader)))
        data = data.to(args.device)
        label = label.to(args.device)
        exemplar = exemplar.to(args.device)
        for idx_exp, each_exp in enumerate(results):
            if 'w' in each_exp:
                w = each_exp['w']
            else:
                w = 1
            data_generated = generation_function[idx_exp](data, exemplar, w=w)
            data_generated = data_generated.to(args.device)
            data_generated = resize_func(data_generated)
            exemplar_generated = resize_func(exemplar)
            if equalize_statistics:
                data_generated = binarize(scale_01(data_generated))
                exemplar_generated = binarize(scale_01(exemplar_generated))
            #result_on_exp['data'].append(data)
            data_for_classifier = torch.cat([exemplar_generated[:args.n_test * args.k_test], data_generated[args.n_test * args.k_test:]],
                                            dim=0)
            y_pred, logits = classify_function[idx_exp](data_for_classifier)
            each_exp['logits'].append(logits.cpu())
            each_exp['labels'].append(label[args.n_test * args.k_test:].cpu())
            pred = y_pred.argmax(dim=1)
            correct = pred.eq(target)
            correct_label = label[args.n_test * args.k_test:][correct]
            each_exp['accuracy'][correct_label] += 1

    for idx_exp, each_exp in enumerate(results):
        each_exp['accuracy'] /= nb_test
        each_exp['logits'] = torch.stack(each_exp['logits'], dim=0)
        each_exp['labels'] = torch.stack(each_exp['labels'], dim=0)

    if saving_name != '':
        torch.save(results, saving_name + '.pkl')

