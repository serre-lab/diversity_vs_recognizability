import torch
import argparse

from utils.data_loader import load_dataset_exemplar
from utils.monitoring import str2bool, plot_img
import json

from utils.evaluation_tools import load_embedding, load_generative_model, interclass_variability

#from utils.loading_tools import load_weights, load_net
from utils.custom_transform import Binarize_batch, Scale_0_1_batch
import torch.nn.functional as f
import torchvision
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='omniglot', choices=['omniglot'],
                    metavar='DATASET', help='Dataset choice.')
parser.add_argument('--device', type=str, default='cuda:0', help='cuda device')
parser.add_argument('--input_type', type=str, default='binary',
                    choices=['binary'], help='type of the input')
parser.add_argument("--input_shape", nargs='+', type=int, default=[1, 50, 50],
                    help='shape of the input [channel, height, width]')
parser.add_argument('--download_data', type=eval, default=False, choices=[True, False])
parser.add_argument('--dataset_root', type=str, default="/media/data_cifs_lrs/projects/prj_control/data")
parser.add_argument("--augment", type=str2bool, nargs='?', const=True, default=False, help="data augmentation")
parser.add_argument('--preload', default=True, action='store_true', help='preload the dataset')
parser.add_argument("--exemplar_type", default='prototype', choices=['prototype', 'first', 'shuffle'],
                    metavar='EX_TYPE', help='type of exemplar')
args = parser.parse_args()


with open('config.json') as json_file:
    data = json.load(json_file)
experiment = data['exp1']

args.batch_size = 20
kwargs = {'preload': args.preload}
scale_01, binarize = Scale_0_1_batch(), Binarize_batch(binary_threshold=0.5)
equalize_statistics = True
train_loader, test_loader, args = load_dataset_exemplar(args, shape=args.input_shape, shuffle=False, **kwargs)
resize_func = torchvision.transforms.Resize(50)


root_path = './experiment/'
saving_name = root_path + 'exp1'

results = []


all_label = []
all_non_zeros = []

with torch.no_grad():

    for idx_exp, each_exp in enumerate(experiment):

        result_on_exp = {'signature': ''}
        result_on_exp['features'] = []
        result_on_exp['creativity'] = []
        result_on_exp['data'] = []
        result_on_exp['prototype'] = []

        for idx, keys in enumerate(each_exp.keys()):
            if keys in ['model', 'embedding', 'distance']:
                result_on_exp['signature'] += each_exp[keys]
                if idx != len(each_exp.keys()) - 1:
                    result_on_exp['signature'] += '_'

        result_on_exp['embedding_model'] = each_exp['embedding_model']
        result_on_exp['model_name'] = each_exp['model_name']
        if 'w' in each_exp:
            result_on_exp['model_name'] += '_w={0}'.format(each_exp['w'])
        print(result_on_exp['model_name'])
        result_on_exp['path_to_model'] = each_exp['path_to_model']
        result_on_exp['path_to_embedding'] = each_exp['path_to_embedding']
        if 'mode' in each_exp.keys():
            result_on_exp['mode'] = each_exp['mode']
        else:
            result_on_exp['mode'] = 'best'

        embedding_function = load_embedding(model_name= each_exp['embedding_model'], path_to_embedding=each_exp['path_to_embedding'], device=args.device, image_size=args.input_shape[-1])
        variability_function = interclass_variability(type=each_exp['distance'])
        generation_function = load_generative_model(path_to_model=each_exp['path_to_model'], model_name=each_exp['model_name'],
                                                    device=args.device, mode=result_on_exp['mode'])


        for batch_idx, (data, exemplar, label) in enumerate(test_loader):
            if label[0] != label[-1]:
                raise Exception('all the label in the batch are not the same')

            if idx_exp == 0:
                all_label.append(label)
                all_non_zeros.append(data.sum() / data.numel())

            data = data.to(args.device)
            exemplar = exemplar.to(args.device)

            if 'w' in each_exp:
                w = each_exp['w']
            else:
                w = 1
            data = generation_function(data, exemplar, w=w).to(args.device)
            resized_data = resize_func(data)
            resized_exemplar = resize_func(exemplar)
            if equalize_statistics:
                resized_data = binarize(scale_01(resized_data))
                resized_exemplar = binarize(scale_01(resized_exemplar))
            result_on_exp['data'].append(resized_data)
            result_on_exp['prototype'].append(resized_exemplar)
            features_data = embedding_function(resized_data)
            features_exemplar = embedding_function(resized_exemplar)
            result_on_exp['features'].append(features_data.view(features_data.size(0), -1).detach())

            measure = variability_function(features_data, proto=features_exemplar).item()

            result_on_exp['creativity'].append(measure)

        result_on_exp['data'] = torch.cat(result_on_exp['data'], dim=0).cpu()
        result_on_exp['prototype'] = torch.cat(result_on_exp['prototype'], dim=0).cpu()

        result_on_exp['features'] = torch.stack(result_on_exp['features'], dim=0).cpu()
        result_on_exp['creativity'] = torch.tensor(result_on_exp['creativity']).cpu()

        if idx_exp == 0:

            all_label = torch.cat(all_label, dim=0)
            all_non_zeros = torch.tensor(all_non_zeros)

        results.append(result_on_exp)

if saving_name != '':
    torch.save(results, saving_name + '.pkl')
