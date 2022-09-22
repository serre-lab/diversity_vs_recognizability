import torch
import os
import torch.optim as optim
import numpy as np
import random
from utils.monitoring import make_directories, compute_parameter_grad, get_logger, \
    plot_gif, visualize, plot_img, str2bool, visual_evaluation

from utils.data_loader import load_omniglot

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from model.DAGAN.parser import get_dagan_args
from model.DAGAN.dagan_trainer import DaganTrainer
from model.DAGAN.discriminator import Discriminator
from model.DAGAN.generator import Generator, ResNetGenerator


parser = get_dagan_args()
parser.add_argument('--download_data', type=eval, default=False, choices=[True, False])
parser.add_argument('--dataset_root', type=str, default="/media/data_cifs_lrs/projects/prj_control/data")
parser.add_argument('--input_type', type=str, default='not-binary',
                    choices=['binary', 'not-binary'], help='type of the input')
parser.add_argument("--input_shape", nargs='+', type=int, default=[1, 50, 50],
                    help='shape of the input [channel, height, width]')
parser.add_argument('--device', type=str, default='cuda:0', help='cuda device')
parser.add_argument("--exemplar", type=str2bool, nargs='?', const=True, default=False, help="For conditional VAE")
parser.add_argument('--preload', default=True, action='store_true', help='preload the dataset')
parser.add_argument("--augment", type=str2bool, nargs='?', const=True, default=False, help="data augmentation")
parser.add_argument("--exemplar_type", default='prototype', choices=['prototype', 'first', 'shuffle'],
                    metavar='EX_TYPE', help='type of exemplar')
parser.add_argument('--model_name', type=str, default='dagan', choices=['dagan', 'ns', 'hfsgm', 'chfsgm_multi', 'tns', 'cns', 'ctns'],
                    help="type of the model ['ns', 'hfsgm', 'chfsgm_multi', 'tns', 'cns', 'ctns]")
parser.add_argument('--debug', default=False, action='store_true', help='debugging flag (do not save the network)')
parser.add_argument('-od', '--out_dir', type=str, default='X',
                    metavar='OUT_DIR', help='output directory for model snapshots etc.')

# Load input args
args = parser.parse_args()

# To maintain reproducibility
if args.seed is not None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

args = make_directories(args)

in_channels = 1
img_size = args.input_shape[-1]
num_training_classes = args.num_training_classes
num_val_classes = args.num_val_classes
batch_size = args.batch_size
epochs = args.epochs
dropout_rate = args.dropout_rate
max_pixel_value = args.max_pixel_value
should_display_generations = not args.suppress_generations

final_generator_path = args.snap_dir  + f"{args.final_model_path}"
args.save_image_path = args.snap_dir

# Input sanity checks
final_generator_dir = os.path.dirname(final_generator_path) or os.getcwd()
if not os.access(final_generator_dir, os.W_OK):
    raise ValueError(final_generator_path + " is not a valid filepath.")

if args.architecture == 'UResNet':
    g = Generator(dim=img_size, channels=in_channels, dropout_rate=dropout_rate, z_dim=args.z_size)
elif args.architecture == 'ResNet':
    g = ResNetGenerator(dim=img_size, channels=in_channels, dropout_rate=dropout_rate, z_dim=args.z_size)
else:
    raise ValueError(args.architecture + " not defined. Choose from {ResNet, UResNet}")
d = Discriminator(dim=img_size, channels=in_channels * 2, dropout_rate=dropout_rate, z_dim=args.z_size)

if not args.debug:
    logger = get_logger(args, __file__)

device = args.device

kwargs = {'preload': args.preload}
train_dataset, val_dataset, _, _, args = load_omniglot(args, shape=args.input_shape, **kwargs)
num_channels = 1

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=False)


g_opt = optim.Adam(g.parameters(), lr=0.0001, betas=(0.0, 0.9))
d_opt = optim.Adam(d.parameters(), lr=0.0001, betas=(0.0, 0.9))

val_data_list = []

for _, (x1, _, _) in enumerate(val_dataloader):
    val_data_list.append(x1)

flat_val_data = torch.cat(val_data_list, dim=0)
display_transform = None


trainer = DaganTrainer(
    generator=g,
    discriminator=d,
    gen_optimizer=g_opt,
    dis_optimizer=d_opt,
    batch_size=batch_size,
    device=device,
    critic_iterations=args.c_iter,
    print_every=75,
    num_tracking_images=10,
    display_transform=display_transform,
    should_display_generations=should_display_generations,
    args = args
)

trainer.train(data_loader=train_dataloader, epochs=epochs, val_images=flat_val_data)

# Save final generator model
torch.save(trainer.g.state_dict(), final_generator_path)