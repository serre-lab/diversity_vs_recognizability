import argparse
import torch

# In parser bool arguments evaluate always to True used as an argument.
# this means that if the default is True and want --feature False
# it will evaluate to True.
# use action="store_true" - it sets the default to False and then --feature
# evaluate to True
# use bool or str2bool for things you want True by default
# (https://stackoverflow.com/questions/8203622/argparse-store-false-if-unspecified)
# use store_true for all the rest


def parse_args():
    # command line args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name", type=str, help="readable name for run",
    )
    parser.add_argument(
        "--model", type=str, help="select model {fsgm, lns, ns}", default='ns'
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/home/data",
        help="location of formatted Omniglot data",
    )
    parser.add_argument(
        "--mnist-data-dir",
        type=str,
        default="/home/data/mnist",
        help="location of MNIST data (required for few shot learning)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/gigi/ns_output",
        help="output directory for checkpoints and figures",
    )

    parser.add_argument(
        "--tag", type=str, default="", help="readable tag for interesting runs",
    )

    # dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default="omniglot",
        help="select dataset (omniglot_ns, omniglot, minimagenet, fc100)",
    )
    # optmizer
    parser.add_argument(
        "--optimizer", type=str, default="adam", help="optimizer {adam, adamw, sgd}",
    )

    parser.add_argument(
        "--scheduler", type=str, default="step", help="scheduler {plateau, step}",
    )

    # likelihood
    parser.add_argument(
        "--likelihood",
        type=str,
        default="binary",
        help="select likelihood (binary, discretized_normal, discretized_mix_logistic)",
    )

    parser.add_argument(
        "--binarize",
        action="store_true",
        help="binarize datasets (omniglot_ns, omniglot)",
    )

    parser.add_argument(
        "--augment", action="store_true", help="augment dataset (omniglot_ns)",
    )

    parser.add_argument(
        "--num-workers", type=int, default=1, help="number workers",
    )
    parser.add_argument(
        "--drop-last", action="store_true", help="drop last partial batch",
    )

    parser.add_argument(
        "--controller-steps", type=int, default=3, help="number of controller steps",
    )

    # useful in cases of multi-sets - next step
    parser.add_argument(
        "--num-classes",
        type=int,
        default=1,
        help="number classes in conditioning dataset (n ways)",
    )
    
    parser.add_argument(
        "--n-head", type=int, default=4, help="heads for attention",
    )

    parser.add_argument(
        "--attn-drop", type=float, default=0.1, help="dropout on attention",
    )

    parser.add_argument(
        "--resid-drop", type=float, default=0.1, help="dropout residual",
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="L2 regularization hyperparameter",
    )

    parser.add_argument(
        "--free-bits",
        action="store_true",
        help="free bits for KL over cL. If positive, it's applied (around 0.5).",
    )

    parser.add_argument(
        "--number-modes-context", type=int, default=2, help="",
    )

    # model hyperparameters
    parser.add_argument(
        "--ch-enc",
        type=int,
        default=64,
        help="base channels for encoder (default: 64)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="batch size (of datasets) for training (default: 32)",
    )
    parser.add_argument(
        "--batch-num",
        type=int,
        default=100,
        help="number of batches for training (default: 100)",
    )
    parser.add_argument(
        "--batch-num-test",
        type=int,
        default=10,
        help="number of batches for training (default: 100)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5,
        help="number of samples per dataset (default: 5)",
    )
    parser.add_argument(
        "--sample-size-test",
        type=int,
        default=5,
        help="number of samples per dataset in test (default: 5)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=3,
        help="number of layers inside residual block (default: 5)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="dimension of hidden layers in modules outside statistic network "
        "(default: 256)",
    )
    parser.add_argument(
        "--hidden-dim-c",
        type=int,
        default=256,
        help="dimension of hidden layers in statistic network (default: 256)",
    )
    parser.add_argument(
        "--n-stochastic",
        type=int,
        default=1,
        help="number of z variables in hierarchy",
    )
    parser.add_argument(
        "--c-dim", type=int, default=512, help="dimension of c variables"
    )
    parser.add_argument(
        "--z-dim", type=int, default=32, help="dimension of z variables"
    )
    parser.add_argument(
        "--h-dim", type=int, default=32, help="dimension of h variables"
    )
    parser.add_argument(
        "--print-parameters",
        action="store_true",
        help="whether to print all trainable parameters for sanity check ",
    )

    parser.add_argument(
        "--parallel_mode",
        action="store_true",
        help="use data parallelism on gpu",
    )

    parser.add_argument(
        "--str-enc",
        type=str,
        default="32-4,16-4,8-2,4-2,2-2,1-2",
        help="(res, group) for encoder.",
    )

    parser.add_argument(
        "--str-gen-z",
        type=str,
        default="8-2,4-2,2-2,1-2",
        help="(res, group) for prior/posterior z.",
    )

    parser.add_argument(
        "--str-gen-c",
        type=str,
        default="8-2,4-2,2-2,1-2",
        help="(res, group) for prior/posterior c.",
    )

    parser.add_argument(
        "--str-dec",
        type=str,
        default="32-4,16-4,8-2",
        help="(res, group) for decoder.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="learning rate for Adam optimizer (default: 1e-3).",
    )
    parser.add_argument(
        "--lr-step", type=float, default=0.5, help="step reducing learning rate.",
    )

    parser.add_argument(
        "--lr-min", type=float, default=0, help="step reducing learning rate.",
    )

    parser.add_argument(
        "--adjust-lr", action="store_true", help="adjust learning rate.",
    )

    parser.add_argument(
        "--randomize-sample-size",
        action="store_true",
        help="randomize sample size in [5, 10, 20].",
    )

    parser.add_argument(
        "--entropy", action="store_true", help="add entropy regularization.",
    )

    parser.add_argument(
        "--adjust-metric",
        type=str,
        default="vlb",
        help="choose the metric to adjust the learning rate.",
    )

    parser.add_argument(
        "--patience", type=int, default=10, help="number of epochs for plateau",
    )

    parser.add_argument(
        "--epochs", type=int, default=1000, help="number of epochs for training",
    )
    parser.add_argument(
        "--alpha", type=int, default=1, help="weighting loss hyperparameter",
    )
    parser.add_argument(
        "--alpha-step",
        type=float,
        default=0.5,
        help="step reducing weighting loss hyperparameter",
    )

    parser.add_argument(
        "--dropout-sample", action="store_true", help="dropout context set",
    )
    parser.add_argument(
        "--dropout", action="store_true", help="dropout on resnet",
    )
    parser.add_argument(
        "--batch-norm", type=bool, default=True, help="batch norm in resnet",
    )

    parser.add_argument(
        "--pixelcnn-mode", action="store_true", help="use pixelcnn decoder",
    )

    parser.add_argument(
        "--aggregation-mode", type=str, default="mean", help="aggregation",
    )

    # vis and log
    parser.add_argument(
        "--viz-interval",
        type=int,
        default=50,
        help="number of epochs between visualizing context space",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=50,
        help="number of epochs between saving model",
    )
    parser.add_argument(
        "--clip-gradients",
        type=bool,
        default=True,
        help="whether to clip gradients to range [-0.5, 0.5] " "(default: True)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="GPU support",
    )
    parser.add_argument(
        "--download", action="store_true", help="Download data",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="random seed",
    )

    parser.add_argument(
        "--dry-run", action="store_true", help="run training without logs, vis, ckpt",
    )

    parser.add_argument(
        "--ladder", action="store_true", help="ladder formulation",
    )

    parser.add_argument(
        "--is-vis", action="store_true", help="samples and visualizations during training",
    )

    # args = parser.parse_args()
    # args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return parser
