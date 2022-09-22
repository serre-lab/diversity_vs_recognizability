import torch
import argparse

from utils.data_loader import load_dataset_exemplar
from utils.monitoring import str2bool, plot_img
from few_shot.proto import compute_prototypes

from torch.utils.data import DataLoader
from model.network import SeqVaeCondGeneral2b
import matplotlib.pyplot as plt
import torch.nn.functional as f
import torchvision

from few_shot.models import get_few_shot_encoder
from few_shot.utils_few_shot import pairwise_distances
import matplotlib.image as mpimg
import numpy as np
from utils.custom_loader import OmniglotDataset
import math
from evaluation_utils.decision_boundary import classifier_prediction, generate_evaluation_task
from evaluation_utils.generative_models import load_generative_model
#from evaluation_utils.decision_boundary import generator_prediction
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import matplotlib.image as mpimg

import cv2
import os
import scipy.stats as stats
from utils.custom_transform import Binarize, Invert, Scale_0_1, Dilate, ScaleCenter
import torchvision.transforms as tforms
from few_shot.core import NShotTaskSampler
from utils.custom_transform import Binarize_batch, Scale_0_1_batch
from exp_list_accu import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='omniglot', choices=['omniglot_weak', 'human_drawing', 'omniglot'],
                    metavar='DATASET', help='Dataset choice.')
parser.add_argument('--out_dir', type=str, default='/media/data_cifs/projects/prj_zero_gene/exp/',
                    metavar='OUT_DIR', help='output directory for model saving etc.')
parser.add_argument('--device', type=str, default='cuda:0', help='cuda device')
#parser.add_argument('--batch_size', type=int, default=128, metavar='BATCH_SIZE',
#                    help='input batch size for training')

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
### BETA Experiment
"""
experiment = [
{'model': 'human', 'model_name': '',
    'classifier': 'prototypical_net', 'classifier_name': 'proto_net_2022-02-23_10_27_41_z256_Omniglot_vae_stn13_6betas_Qfixed'},
]
"""

#experiment = experiment_control#experiment_steps_r3#experiment_param_r3#experiment_beta_1annealing#experiment_wdw_1#experiment_z_1#experiment_beta_2#experiment_steps_r2#experiment_param_r2#experiment_beta_1 #experiment_param
#experiment = experiment_z_4#global_plot#experiment_z_2#global_plot#test_vqvae#experiment_beta_3#experiment_wdw_20_lstm500
#experiment = experiment_control

#experiment = experiment_beta_3f#experiment_z_3#experiment_steps_r7#experiment_beta_5#experiment_vqvae_lbda_and_beta#experiment_vqvae_k1
#experiment = global_plot
#experiment = experiment_steps_r3_f#experiment_lstm_r1#experiment_z_7
#experiment = test_debug_vqvae#experiment_lstm_r1#experiment_z_7
#experiment = experiment_diffusion_model_4
experiment = experiment_diffusion_model_5
root_path = './experiment/'
#root_path = '/media/data_cifs/projects/prj_zero_gene/neurips2022/results/control_experiment/'

#saving_name = root_path + 'accu_step_exp'
#saving_name = root_path + 'accu_beta1_exp'
#saving_name = root_path + 'accu_param_r2_exp'
#saving_name = root_path + 'accu_step_r2_exp'
#saving_name = root_path + 'accu_beta2_exp'
#saving_name = root_path + 'accu_z1_exp'
#saving_name = root_path + 'accu_wdw1_exp'
#saving_name = root_path + 'accu_beta1_anneal_exp'
#saving_name = root_path + 'accu_param_r3_exp'
#saving_name = root_path + 'accu_step_r3_exp'
#saving_name = root_path + 'accu_ctrl_exp'
#saving_name = root_path + 'experiment_wdw_20_lstm500'
#saving_name = root_path + 'accu_beta3_exp'
#saving_name = root_path + 'accu_vqvae'
#saving_name = root_path + 'accu_global_plot_logits'
#saving_name = root_path + 'accu_z2_exp'
#saving_name = root_path + 'accu_z4_exp'
#saving_name = root_path + 'accu_ctrl_exp'
#saving_name = root_path + 'accu_vqvae_k1_exp'
#saving_name = root_path + 'accu_vqvae_lbda_beta_1_exp'
#saving_name = root_path + 'accu_beta5_exp'
#saving_name = root_path + 'accu_step_r7_exp'
#saving_name = root_path + 'accu_z3_exp'
#saving_name = root_path + 'accu_beta1f_exp'
#saving_name = root_path + 'accu_beta2f_exp'
#saving_name = root_path + 'accu_beta3f_exp'
#saving_name = root_path + 'accu_global_exp'
#saving_name = root_path + 'accu_z7_exp'
#saving_name = root_path + 'accu_lstm_r1_exp'
#saving_name = root_path + 'accu_step_r3f_exp'
#saving_name = root_path + 'accu_diffusion_4'
saving_name = root_path + 'accu_diffusion_5'
### control_experiment


print_mean = True
show_histogram = False

plot_comparison = False
plot_accuracy = False
visualize_linear = False

compare_rank = False

#saving_name = './experiment/accu_beta_exp'
#saving_name = './experiment/accu_varying_window_size'
#saving_name = ''


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
        #classify_function.append(classifier_prediction(model=each_exp['classifier'],
        #                                          model_name=each_exp['classifier_name'], device=args.device,
        #                                          args=args))
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


"""
with torch.no_grad():
    for idx_exp, each_exp in enumerate(experiment):
        result_on_exp = {'signature': ''}
        result_on_exp['accuracy'] = torch.zeros(len(all_label))
        result_on_exp['data'] = []

        # result_on_exp = {'': ''}
        for idx, keys in enumerate(each_exp.keys()):
            if keys in ['model', 'classifier']:
                result_on_exp['signature'] += each_exp[keys]
                if idx != len(each_exp.keys()) - 1:
                    result_on_exp['signature'] += '_'
        result_on_exp['classifier_name'] = each_exp['classifier_name']
        result_on_exp['model_name'] = each_exp['model_name']

        classify_function = classifier_prediction(model = each_exp['classifier'], model_name = each_exp['classifier_name'], device = args.device, args=args)
        generation_function = load_generative_model(model = each_exp['model'], model_name = each_exp['model_name'],
                                                    device=args.device)

        target = torch.arange(args.k_test).to(args.device)
        for idx, (data, exemplar, label) in enumerate(test_loader):
            data= data.to(args.device)
            label = label.to(args.device)
            exemplar = exemplar.to(args.device)

            data = generation_function(data, exemplar)

            if equalize_statistics:
                data = binarize(scale_01(data))
            result_on_exp['data'].append(data)
            data_for_classifier = torch.cat([exemplar[:args.n_test*args.k_test], data[args.n_test*args.k_test:]], dim=0)
            y_pred = classify_function(data_for_classifier)
            pred = y_pred.argmax(dim=1)
            correct = pred.eq(target)
            correct_label = label[args.n_test * args.k_test:][correct]
            result_on_exp['accuracy'][correct_label] += 1
        result_on_exp['data'] = torch.cat(result_on_exp['data'], dim=0)
        result_on_exp['accuracy'] /= nb_test

        results.append(result_on_exp)
"""
#generate_function = {gen_model: generator_prediction(gen_model, n_shot, k_way, args) for gen_model in generator_list}

#class_accuracy = {gen_model: {classif: torch.zeros(len(all_label)) for classif in classifier_list} for gen_model in generator_list}


#class_accuracy = {classif: {gen_model: torch.zeros(len(all_label)) for gen_model in generator_list} for classif in classifier_list}


"""
plt.plot(torch.arange(50), class_accuracy_0['prototypical_net']['vae_stn'][:50], color='red')
plt.plot(torch.arange(50), class_accuracy_1['prototypical_net']['vae_stn'][:50], color='blue')
plt.show()

fig, ax = plt.subplots(figsize=(5, 5))

accu1 = class_accuracy_0['prototypical_net']['vae_stn'][:50]
accu2 = class_accuracy_1['prototypical_net']['vae_stn'][:50]
ax.scatter(accu1, accu2 , color='blue')
#ax.set_xlabel('% : {0}'.format(generative_model_1))
#ax.set_ylabel('% : {0}'.format(generative_model_2))
a, b, r_value, p_value, std_err = stats.linregress(accu1, accu2)


ax.plot(accu1, a*accu1+b, color='red')
print(r'{0:0.2f}x + {1:0.2f} : $R^2$={2:0.5f}'.format(a, b, r_value**2))

"""
if print_mean:
    for idx_exp, each_exp in enumerate(results):
        to_print = each_exp['accuracy'].mean()
        print("{0} - {1} - {2} -- Mean = {3:0.4f}".format(each_exp['signature'], each_exp['model_name'], each_exp['classifier_name'], to_print))

if show_histogram:
    for idx_exp, each_exp in enumerate(results):
        plt.hist(each_exp['accuracy'].numpy(), density=True, bins=30)
        plt.ylabel('Probability')
        plt.xlabel('classification accuracy')
        plt.title('accuracy distribution for {0} \n {1}'.format(each_exp['signature'], each_exp['classifier_name']))
        plt.show()


if plot_accuracy:
    fig, ax = plt.subplots(figsize=(5, 5))
    for each_exp in results:
        ax.plot(torch.arange(nb_class), each_exp['accuracy'].cpu(), label=each_exp['signature']+each_exp['classifier_name'])
    ax.set_xlabel('label')
    ax.set_ylabel('classification accuracy')
    #ax.set_title('Classification accuracy with {0}'.format(classifier_net))
    plt.show()

if plot_comparison:
    for idx1, each_exp1 in enumerate(results):
        for idx2, each_exp2 in enumerate(results):
            if idx2 > idx1 :
                fig, ax = plt.subplots(figsize=(5, 5))
                min_val = each_exp1['accuracy'].min().item()
                max_val = each_exp2['accuracy'].max().item()
                range = max_val - min_val
                ax.scatter(each_exp1['accuracy'], each_exp2['accuracy'], color='blue')
                ax.set_xlabel('% : {0} \n gen_model : {1} \n cl_model : {2}'.format(each_exp1['signature'],
                                                                                               each_exp1['model_name'],
                                                                                               each_exp1[
                                                                                                   'classifier_name']),
                              fontsize=8)
                ax.set_ylabel('% : {0} \n gen_model : {1} \n cl_model : {2}'.format(each_exp2['signature'],
                                                                                               each_exp2['model_name'],
                                                                                               each_exp2[
                                                                                                   'classifier_name']),
                              fontsize=8)
                a, b, r_value, p_value, std_err = stats.linregress(each_exp1['accuracy'],
                                                                   each_exp2['accuracy'])
                ax.plot(each_exp1['accuracy'], a * each_exp1['accuracy'] + b, color='red')
                ax.annotate(r'{0:0.2f}x + {1:0.2f} : $R^2$={2:0.5f} p={3:0.2e}'.format(a, b, r_value ** 2, p_value),
                            (min_val, max_val))
                plt.show()

if compare_rank:
    for idx1, each_exp1 in enumerate(results):
        for idx2, each_exp2 in enumerate(results):
            if idx2 > idx1 :
                corr, p_value = stats.spearmanr(each_exp1['accuracy'], each_exp2['accuracy'])
                print('{0}-{1}-{2}  vs {3}-{4}-{5} : {6:0.2f} p_value {7:0.2e}'.format(each_exp1['signature'], each_exp1['model_name'], each_exp1['classifier_name'],
                                                                              each_exp2['signature'], each_exp2['model_name'], each_exp2['classifier_name'],
                                                                              corr, p_value))

"""
if visualize_linear:
    for idx_exp, each_exp in enumerate(results):
        sorted_accuracy, _ = torch.sort(each_exp['accuracy'])
        fig, ax = plt.subplots(figsize=(25, 5))
        nb_point = 30
        max_value = sorted_accuracy[-1]
        min_value = sorted_accuracy[0]
        distance_list = torch.linspace(min_value, max_value, nb_point)

        range = max_value - min_value
        ax.set_xlim(min_value - 0.1*range, max_value + 0.1*range)
        ax.set_ylim(-0.1, 0.1)

        if nb_point % 2 == 0:
            offset = 0
        else :
            offset = 1
        y_val = [-0.05, 0.05] * ((nb_point // 2)+offset)

        new_idx = 0
        idx_dis = 0
        all_x_val = []
        old_val = np.inf
        for dis in distance_list:
            idx = torch.abs(each_exp['accuracy'] - dis).argmin()
            lab = torch.unique_consecutive(all_label)[idx]
            val = each_exp['accuracy'][idx]
            if val != old_val:
                all_x_val.append(val.item())
                filter = all_label == lab
                if filter.sum()!=20:
                    raise Exception('all the label in the batch are not the same')
                #img_to_plot = all_data[filter, :, :, :].view(20, 1, 50, 50)
                img_to_plot = each_exp['data'][filter, :, :, :].view(20, 1, 50, 50).cpu()
                image_to_plot = torchvision.utils.make_grid(img_to_plot[0:20], nrow=4, padding=2, normalize=True,
                                                       pad_value=0)
                image_to_plot = np.transpose(image_to_plot, (1, 2, 0))
                image_box = OffsetImage(image_to_plot, zoom=0.4)
                ab = AnnotationBbox(image_box, (val, y_val[idx_dis]), frameon=False)
                ax.add_artist(ab)
                idx_dis += 1
                old_val = val
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_position('center')
        ax.axes.get_yaxis().set_visible(False)
        ax.set_title('{0} \n gen_model : {1} \n emb_model : {2}'.format(each_exp['signature'], each_exp['model_name'], each_exp['classifier_name']))

        plt.xticks(all_x_val)
        plt.draw()
        plt.show()
"""
"""
if visualize_linear:
    for classifier_net in class_accuracy.keys():
        for idx_keys_1, generative_model_1 in enumerate(class_accuracy[classifier_net].keys()):
            fig, ax = plt.subplots(figsize=(25, 5))
            nb_point = 10
            max_value = class_accuracy[classifier_net][generative_model_1].max()
            min_value = class_accuracy[classifier_net][generative_model_1].min()
            distance_list = torch.linspace(min_value, max_value, nb_point)
            range = max_value - min_value
            ax.set_xlim(min_value - 0.1 * range, max_value + 0.1 * range)
            ax.set_ylim(-0.1, 0.1)

            if nb_point % 2 == 0:
                offset = 0
            else :
                offset = 1
            y_val = [-0.05, 0.05] * ((nb_point // 2)+offset)

            new_idx = 0
            idx_dis = 0
            all_x_val = []
            old_val = np.inf
            for dis in distance_list:
                # print(dis)
                idx = torch.abs(class_accuracy[classifier_net][generative_model_1] - dis).argmin()
                val = class_accuracy[classifier_net][generative_model_1][idx]
                if val != old_val:
                    # if val != all_x_val[-1]:
                    all_x_val.append(val.item())
                    # print(val)


                    filter = eval.label == idx
                    img_to_plot = eval.data[filter, :, :, :].view(20, 1, 105, 105).float()
                    img_to_plot = f.interpolate(img_to_plot, (50, 50))
                    image_to_plot = torchvision.utils.make_grid(img_to_plot[0:18], nrow=3, padding=2, normalize=True,
                                                                pad_value=0)
                    image_to_plot = np.transpose(image_to_plot, (1, 2, 0))
                    image_box = OffsetImage(image_to_plot, zoom=0.4)
                    ab = AnnotationBbox(image_box, (val, y_val[idx_dis]), frameon=False)
                    ax.add_artist(ab)
                    idx_dis += 1
                    old_val = val
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_position('center')
            ax.axes.get_yaxis().set_visible(False)
            ax.set_title('{0} with {1}'.format(generative_model_1, classifier_net))
            plt.xticks(all_x_val)
            plt.draw()
            plt.show()
"""




