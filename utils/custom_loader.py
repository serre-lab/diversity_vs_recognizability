from torchvision.datasets.vision import VisionDataset
from torch.utils.data import Dataset
from typing import Any, Callable, Optional, Tuple
from torchvision import transforms
import torch
import os
from PIL import Image
import numpy as np
#from .exemplar_quickdraw import exemplar_quickdraw_idx
import pandas as pd
from tqdm import tqdm
import torchvision.transforms as tforms
from utils.custom_transform import Binarize, Scale_0_1
#from utils.quickdraw_label import quickdraw_train_label, quickdraw_test_label, quickdraw_exemplar_idx, labels_to_idx
import torchvision.transforms.functional as TF
import random

class Contrastive_augmentation(object):
    def __init__(self, ds: Dataset, target_size, strength='normal'):
        super().__init__()
        self.ds = ds
        self.target_size = target_size


        #self.randomize = transforms.Compose([
        #    transforms.RandomChoice([
        #        transforms.RandomResizedCrop(target_size, scale=(0.75, 1.33), ratio=(0.8, 1.2)), ## normal
                #transforms.RandomResizedCrop(target_size, scale=(0.6, 1.5), ratio=(0.6, 1.4)), ## severe
                #transforms.RandomResizedCrop(target_size, scale=(0.9, 1.1), ratio=(0.9, 1.1)),  ## soft

        #        transforms.RandomAffine((-15, 15), scale=(0.75, 1.33), translate=(0.1, 0.1), shear=(-10, 10),
        #                                fillcolor=256), ## normal
                #transforms.RandomAffine((-45, 45), scale=(0.6, 1.5), translate=(0.4, 0.4), shear=(-30, 30),
                #                        fillcolor=256), ## severe
                #transforms.RandomAffine((-10, 10), scale=(0.9, 1.1), translate=(0.1, 0.1), shear=(-5, 5),
                #                        fillcolor=256),  ## soft

        #        transforms.RandomPerspective(distortion_scale=0.5, p=0.5, fill=256) ## normal
                #transforms.RandomPerspective(distortion_scale=0.8, p=0.5, fill=256) ## severe
                #transforms.RandomPerspective(distortion_scale=0.25, p=0.5, fill=256)  ## soft

        #    ])
        #])
        if strength == 'normal':
            self.tr_augment = transforms.Compose([
                transforms.RandomChoice([
                    transforms.RandomResizedCrop(self.target_size, scale=(0.75, 1.33), ratio=(0.8, 1.2)),
                    transforms.RandomAffine((-15, 15), scale=(0.75, 1.33), translate=(0.1, 0.1), shear=(-10, 10),
                                            fillcolor=0),
                    transforms.RandomPerspective(distortion_scale=0.5, p=0.5, fill=0)
                ])
            ])

        elif strength == 'light':
            self.tr_augment = transforms.Compose([
                transforms.RandomChoice([
                    transforms.RandomResizedCrop(self.target_size, scale=(0.90, 1.1), ratio=(0.9, 1.1)),
                    transforms.RandomAffine((-7, 7), scale=(0.90, 1.1), translate=(0.05, 0.05), shear=(-5, 5),
                                            fillcolor=0),
                    transforms.RandomPerspective(distortion_scale=0.25, p=0.5, fill=0)
                ])
            ])
        elif strength == 'strong':
            self.tr_augment = transforms.Compose([
                transforms.RandomChoice([
                    transforms.RandomResizedCrop(self.target_size, scale=(0.5, 1.5), ratio=(0.6, 1.4)),
                    transforms.RandomAffine((-30, 30), scale=(0.5, 1.5), translate=(0.2, 0.2), shear=(-20, 20),
                                            fillcolor=0),
                    transforms.RandomPerspective(distortion_scale=0.75, p=0.5, fill=0)
                ])
            ])

        self.tranform_special = transforms.Compose([
            transforms.ToTensor(),
            Scale_0_1(),
            Binarize(binary_threshold=0.5),

        ])

    def __len__(self):
        return len(self.ds)

    def __call__(self, input_image):
        input_image = input_image.cpu()
        all_image_1 = torch.zeros_like(input_image)
        all_image_2 = torch.zeros_like(input_image)
        for idx_image in range(input_image.size(0)):
            image = tforms.functional.to_pil_image(input_image[idx_image], mode="L")
            #image = Image.fromarray(tensor[input_image, 0, :, :].cpu().numpy(), mode = 'L')
            image_tr_1 = self.tr_augment(image)
            image_tr_2 = self.tr_augment(image)
            image_tr_1 = self.tranform_special(image_tr_1)
            image_tr_2 = self.tranform_special(image_tr_2)
            all_image_1[idx_image, 0, :, :] = image_tr_1.to(input_image.device)
            all_image_2[idx_image, 0, :, :] = image_tr_2.to(input_image.device)
        return all_image_1, all_image_2

class OmniglotDataset(Dataset):
    def __init__(self, root, split, transform=None, exemplar_transform=None, target_tranform=None, preloading=False, exemplar_type='first'):
        """Dataset class representing Omniglot dataset

        # Arguments:
            subset: Whether the dataset represents the background or evaluation set
        """
        if split not in ('background', 'evaluation', 'weak_background',
                         'weak_evaluation'):
            raise(ValueError, 'split must be one of (background, evaluation)')
        self.transform = transform
        self.exemplar_transform = exemplar_transform
        self.target_transform = target_tranform
        #self.exemplar = exemplar
        #self.shuffle_exemplar = shuffle_exemplar
        self.exemplar_type = exemplar_type
        self.preloading = preloading

        self.split = split
        self.root = root

        df_path = os.path.join(self.root, 'preload_df.pkl')

        if os.path.exists(df_path) and self.preloading:
            self.df = pd.read_pickle(df_path)
            #self.df_train = pd.DataFrame(self.index_subset(self.root, 'background'))
        else:
            self.df_train = pd.DataFrame(self.index_subset(self.root, 'background'))
            self.df_test = pd.DataFrame(self.index_subset(self.root, 'evaluation'))
            self.df = pd.concat([self.df_train, self.df_test], ignore_index=True)
            if self.preloading:
                self.df.to_pickle(df_path)


        # Index of dataframe has direct correspondence to item in dataset



        # Convert arbitrary class names of dataset to ordered 0-(num_speakers - 1) integers

        if self.split == 'background':
            self.df = self.df[self.df['subset'] == 'background']
        elif self.split == 'evaluation':
            self.df = self.df[self.df['subset'] == 'evaluation']
        elif self.split == 'weak_evaluation':
            character_list = ['character01', 'character02', 'character03']
            self.df = self.df[self.df['character_number'].isin(character_list)]
        elif self.split == 'weak_background':
            character_list = ['character01', 'character02', 'character03']
            self.df = self.df[~self.df['character_number'].isin(character_list)]

        self.unique_characters = sorted(self.df['class_name'].unique())

        self.class_name_to_id = {self.unique_characters[i]: i for i in range(self.num_classes())}
        self.df = self.df.assign(class_id=self.df['class_name'].apply(lambda c: self.class_name_to_id[c]))

        self.df = self.df.reset_index(drop=True)
        self.df = self.df.assign(id=self.df.index.values)

        """
        if self.exemplar:
            if self.shuffle_exemplar:
                self.df_exemplar = self.df
            else:
                self.df_exemplar = self.df[self.df['sample_number'].isin(['1'])]
        """

        if self.exemplar_type == 'first':
            self.df_exemplar = self.df[self.df['sample_number'].isin(['1'])]
            a=1
        elif self.exemplar_type == 'shuffle':
            self.df_exemplar = self.df
        elif self.exemplar_type == 'prototype':
            if self.split == 'weak_evaluation':
                path_to_proto = os.path.join(self.root, 'te_proto_weak.pkl')
                flag = torch.load(path_to_proto)
                flag2 = torch.arange(flag.size(0))[flag == 1]
                self.df_exemplar = self.df[self.df['id'].isin(flag2)]
                a=1
            elif self.split == 'weak_background':
                path_to_proto = os.path.join(self.root, 'tr_proto_weak.pkl')
                flag = torch.load(path_to_proto)
                flag2 = torch.arange(flag.size(0))[flag == 1]
                self.df_exemplar = self.df[self.df['id'].isin(flag2)]


            else :
                Exception('prototype exemplar implemented only for weak split for now')
            #self.df_exemplar = self.df[self.df['id'].isin(id_list)]


        # Create dicts
        self.datasetid_to_filepath = self.df.to_dict()['filepath']
        self.datasetid_to_class_id = self.df.to_dict()['class_id']

        if self.preloading:
            self.data, self.label = self.preload()

    def preload(self):
        all_images, all_classes = [], []

        path_preload = os.path.join(self.root, 'preload_'+ self.split )
        path_preload += '.pkl'
        if not os.path.exists(path_preload):
            progress_bar = tqdm(total=len(self.df))
            for idx_image in range(len(self.df)):
                image = Image.open(self.datasetid_to_filepath[idx_image], mode="r").convert("L")
                character_class = self.datasetid_to_class_id[idx_image]

                all_images.append(tforms.functional.pil_to_tensor(image))
                all_classes.append(character_class)
                progress_bar.update(1)
            progress_bar.close()
            all_images = torch.stack(all_images, dim=0)
            all_classes = torch.tensor(all_classes)
            torch.save([all_images, all_classes], path_preload)
        else:
            [all_images, all_classes] = torch.load(path_preload)

        return all_images, all_classes

    def __getitem__(self, item):
        if self.preloading:
            image = tforms.functional.to_pil_image(self.data[item], mode="L")
            character_class = self.label[item].item()
        else:
            image = Image.open(self.datasetid_to_filepath[item], mode="r").convert("L")
            character_class = self.datasetid_to_class_id[item]

        #if self.exemplar:
        #if self.shuffle_exemplar:
        if self.exemplar_type == 'shuffle':
            item_exemplar = self.df_exemplar[self.df_exemplar['class_id'] == character_class].sample(1).index.values[0]
        elif self.exemplar_type == 'first':
            item_exemplar = self.df_exemplar[self.df_exemplar['class_id'] == character_class].index.values[0]
        elif self.exemplar_type == 'prototype':
            item_exemplar = self.df_exemplar[self.df_exemplar['class_id'] == character_class].index.values[0]
        if self.preloading:
            image_exemplar = tforms.functional.to_pil_image(self.data[item_exemplar], mode="L")
        else:
            image_exemplar = Image.open(self.datasetid_to_filepath[item_exemplar], mode="r").convert("L")
            #

        if self.transform:
            image = self.transform(image)

        if self.exemplar_transform :#and self.exemplar:
            exemplar = self.exemplar_transform(image_exemplar)
        else :
            exemplar = image_exemplar
        #else:
        #    exemplar = None

        if self.target_transform:
            character_class = self.target_transform(character_class)

        return image, exemplar, character_class


    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.df['class_name'].unique())

    @staticmethod
    def index_subset(dir_data, subset):
        """Index a subset by looping through all of its files and recording relevant information.

        # Arguments
            subset: Name of the subset

        # Returns
            A list of dicts containing information about all the image files in a particular subset of the
            Omniglot dataset dataset
        """
        images = []
        print('Indexing {}...'.format(subset))
        # Quick first pass to find total for tqdm bar
        subset_len = 0
        for root, folders, files in os.walk(dir_data + '/images_{}/'.format(subset)):
            subset_len += len([f for f in files if f.endswith('.png')])

        progress_bar = tqdm(total=subset_len)
        for root, folders, files in os.walk(dir_data + '/images_{}/'.format(subset)):
            if len(files) == 0:
                continue
            sample_number = 0
            alphabet = root.split('/')[-2]
            character_number = root.split('/')[-1]
            class_name = '{}.{}'.format(alphabet, root.split('/')[-1])


            for f in files:
                sample_number += 1
                progress_bar.update(1)
                images.append({
                    'subset': subset,
                    'alphabet': alphabet,
                    'character_number': character_number,
                    'class_name': class_name,
                    'sample_number': sample_number,
                    'filepath': os.path.join(root, f)
                })

        progress_bar.close()
        return images

class OmniglotSetsDatasetNS(Dataset):
    def __init__(self, data, label, split, sample_size, transform):
        self.data = data 
        self.label = label
        self.split = split
        self.sample_size = sample_size
        self.transform = transform

        print(self.split)
        print(self.data.shape, self.label.shape)
        self.init_sets()

    def init_sets(self):
        sets, set_labels = self.make_sets(self.data.numpy(), self.label.numpy())
        
 
        if self.split in ['weak_background', 'train', 'val', 'test', 'weak_evaluation']:
            if self.transform:
                sets = self.augment_sets(sets)
        sets = np.random.binomial(1, p=sets, size=sets.shape).astype(np.float32)  
        # (batch_size, sample_size, xdim)

        sets = sets.reshape(-1, self.sample_size, 1, 50, 50)
        self.n = len(sets)
        self.data = {
        'inputs': sets,
        'targets': set_labels
        }

    def augment_sets(self, sets):
        """
        Augment training sets.
        """

        from skimage.transform import rotate

        sets = sets.reshape(-1, self.sample_size, 105, 105)
        n_sets = len(sets)
        new_sets = np.zeros((n_sets, self.sample_size, 50, 50))

        for s in range(n_sets):
            for item in range(self.sample_size):
                new_sets[s, item] = self.transform(Image.fromarray(sets[s, item]))

        if self.split in ['weak_evaluation', 'test']:

            return new_sets.reshape(n_sets, self.sample_size, 50*50)

        augmented = np.copy(new_sets)

        for i in range(self.sample_size-1):
            
            augmented = augmented.reshape(-1, self.sample_size, 50, 50)
            n_sets = len(augmented)
            
            for s in range(n_sets):
                flip_horizontal = np.random.choice([0, 1])
                flip_vertical = np.random.choice([0, 1])
                if flip_horizontal:
                    augmented[s] = augmented[s, :, :, ::-1]
                if flip_vertical:
                    augmented[s] = augmented[s, :, ::-1, :]

            for s in range(n_sets):
                angle = np.random.uniform(0, 360)
                for item in range(self.sample_size):
                    augmented[s, item] = rotate(augmented[s, item], angle, cval=0)
            
            augmented = augmented.reshape(n_sets, self.sample_size, 50*50)

            if i == 0:
                augmented_ = np.concatenate([augmented, new_sets.reshape(n_sets, self.sample_size, 50*50)])
            else:
                augmented_ = np.concatenate([augmented_, augmented.reshape(n_sets, self.sample_size, 50*50)])

        return augmented_

    @staticmethod
    def one_hot(dense_labels, num_classes):
        num_labels = len(dense_labels)
        offset = np.arange(num_labels) * num_classes
        one_hot_labels = np.zeros((num_labels, num_classes))
        one_hot_labels.flat[offset + dense_labels.ravel()] = 1
        return one_hot_labels

    def make_sets(self, images, labels):
        """
        Create sets of arbitrary size between 1 and 20. 
        The sets are composed of one class.
        """
        
        num_classes = np.max(labels) + 1
        labels = self.one_hot(labels, num_classes)
        
        n = len(images)
        perm = np.random.permutation(n)
        images = images[perm]
        labels = labels[perm]
        
        # init sets
        image_sets = []
        label_sets = []

        for j in range(num_classes):

            label = labels[:, j].astype(bool)
            num_instances_per_class = np.sum(label)
            # if num instances less than what we want (30 > 20 Omniglot max 20)
            if num_instances_per_class < self.sample_size:
                pass
            else:
                # check if sample_size is a multiple of num_instances
                remainder = num_instances_per_class % self.sample_size
                # select all images with a certain label
                image_set = images[label]
                if remainder > 0:
                    # remove samples from image_sets
                    image_set = image_set[:-remainder]
                # collect sets
                image_sets.append(image_set)
                # for Omniglot k should be 20
                k = len(image_set)
                # select only elements with certain label
                label_set = labels[label]
                # then select (k/sample_size) times the same label
                label_set = label_set[:int(k / self.sample_size)]
                label_sets.append(label_set)

        x = np.concatenate(image_sets, axis=0).reshape(-1, self.sample_size, 105*105)
        y = np.concatenate(label_sets, axis=0)
        if np.max(x) > 1:
            x = np.divide(x, 255, out=x, casting="unsafe")
            # x /= 255

        perm = np.random.permutation(len(x))
        x = x[perm]
        y = y[perm]
        return x, y

    def __getitem__(self, item, lbl=None):

        samples = self.data['inputs'][item]

        # if self.transform:
        #     samples = self.transform(samples)

        # if self.split in ['train', 'val'] and self.augment:
        #     targets = np.zeros(samples.shape)
        # else:
        #     targets = self.data['targets'][item] 
        # if lbl:
        #     return samples, targets

        return samples

    def __len__(self):
        return self.n