import torch.utils.data as data_utils
from .custom_loader import OmniglotDataset, OmniglotSetsDatasetNS#, OmniglotGeneral, OmniglotWeakFast
from .custom_transform import Binarize, Invert, Scale_0_1, Dilate, ScaleCenter
import torchvision.transforms as tforms
from torchvision.datasets import MNIST
import os
import PIL
#from simclr.data_utils import PretrainingDatasetWrapper
#from few_shot.core import NShotTaskSampler
import torch


def load_omniglot(args, type='weak', **kwargs):
    tr_eval = [tforms.Resize(args.input_shape[1:])]
    tr_train = [tforms.Resize(args.input_shape[1:])]
    if args.augment:
        tr_train += [tforms.RandomAffine((-45, 45))]
    tr_eval += [tforms.ToTensor(), Scale_0_1(), Invert()]
    tr_train += [tforms.ToTensor(), Scale_0_1(), Invert()]
    if args.input_type == 'binary':
        tr_eval += [Binarize(binary_threshold=0.5)]
        tr_train += [Binarize(binary_threshold=0.5)]

    tr_eval = tforms.Compose(tr_eval)
    tr_train = tforms.Compose(tr_train)
    dir_data = os.path.join(args.dataset_root, "omniglot", "omniglot-py")


    if type == 'weak':
        split_tr, split_te = 'weak_background', 'weak_evaluation'
    elif type == 'strong':
        split_tr, split_te = 'background', 'evaluation'
    else :
        raise NotImplementedError()

    train = OmniglotDataset(dir_data, split=split_tr, transform=tr_train,
                            preloading=args.preload, exemplar_transform=tr_eval, exemplar_type=args.exemplar_type)
    eval = OmniglotDataset(dir_data, split=split_te, transform=tr_eval, exemplar_transform=tr_eval,
                           preloading=args.preload, exemplar_type=args.exemplar_type)

    if hasattr(args, 'model_name'):
        if args.model_name in ['ns', 'tns', 'hfsgm']:
            train = OmniglotSetsDatasetNS(train.data, train.label, split=split_tr, sample_size=args.sample_size, transform=tr_train)
            eval = OmniglotSetsDatasetNS(eval.data, eval.label, split=split_te, sample_size=args.sample_size, transform=tr_eval)
    #if args.contrastive:
    #    train = PretrainingDatasetWrapper(train, target_size=args.input_shape[1:], additional_transforms=tr_train)
    #    eval = PretrainingDatasetWrapper(eval, target_size=args.input_shape[1:], additional_transforms=tr_eval)
    #train_exemplars = OmniglotDataset(dir_data, split='weak_background', exemplar=True, transform = tr_train, preloading=kwargs['preload'])
    #eval_exemplars = OmniglotDataset(dir_data, split='weak_evaluation', exemplar=True, transform=tr_eval, preloading=kwargs['preload'])


    #return train, eval, train_exemplars, eval_exemplars, args
    return train, eval, None, None, args




def load_dataset_exemplar(args, shape=None, shuffle=True, drop_last=False, few_shot=False,
                          fixed_tasks=None,**kwargs):

    #if args.dataset == 'omniglot_weak':
    #    train_set, test_set, train_set_exemplars, test_set_exemplars, args = load_omniglot_weak(args, shape=shape, **kwargs)
    #elif args.dataset == 'omniglot_gneneral':
    #    train_set, test_set, train_set_exemplars, test_set_exemplars, args = load_omniglot_general(args, shape=shape,
    #                                                                                                **kwargs)
    if args.dataset == 'omniglot':
        train_set, test_set, train_set_exemplars, test_set_exemplars, args = load_omniglot(args, shape=shape, type='weak',
                                                                                                    **kwargs )
    elif args.dataset == 'strong_omniglot':
        train_set, test_set, train_set_exemplars, test_set_exemplars, args = load_omniglot(args, shape=shape, type='strong',
                                                                                                **kwargs)
    else:
        raise Exception('Wrong name of the dataset!')

    if few_shot and args.model_name == 'proto_net':

        train_loader = data_utils.DataLoader(train_set,
                                             batch_sampler=NShotTaskSampler(train_set, args.episodes_per_epoch,
                                                                            args.n_train, args.k_train, args.q_train,
                                                                            fixed_tasks=fixed_tasks)
                                             )
        test_loader = data_utils.DataLoader(test_set,
                                            batch_sampler=NShotTaskSampler(test_set, args.episodes_per_epoch,
                                                                           args.n_test, args.k_test, args.q_test,
                                                                           fixed_tasks=fixed_tasks)
                                            )
    elif few_shot and args.model_name == 'maml':
        train_loader = data_utils.DataLoader(train_set,
                                             batch_sampler=NShotTaskSampler(train_set, args.epoch_len,
                                                                            args.n, args.k, args.q,
                                                                            num_tasks=args.meta_batch_size,
                                                                            fixed_tasks=fixed_tasks)
                                             )
        test_loader = data_utils.DataLoader(test_set,
                                            batch_sampler=NShotTaskSampler(test_set, args.eval_batches,
                                                                           args.n, args.k, args.q,
                                                                           num_tasks=args.meta_batch_size,
                                                                           fixed_tasks=fixed_tasks)
                                            )

    else:
        g = torch.Generator()
        g.manual_seed(0)
        train_loader = data_utils.DataLoader(train_set, batch_size=args.batch_size, shuffle=shuffle, drop_last=drop_last, generator=g)

        test_loader = data_utils.DataLoader(test_set, batch_size=args.batch_size, shuffle=shuffle, drop_last=drop_last)
    """
    if train_set_exemplars is not None:
        train_exemplars = data_utils.DataLoader(train_set_exemplars, batch_size=len(train_set_exemplars), shuffle=False)
    else:
        train_exemplars = None
    if test_set_exemplars is not None:
        test_exemplars = data_utils.DataLoader(test_set_exemplars, batch_size=len(test_set_exemplars), shuffle=False)
    else:
        test_exemplars = None
    """
    #return train_loader, test_loader, train_exemplars, test_exemplars, args
    return train_loader, test_loader, args

