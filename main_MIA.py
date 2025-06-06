# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
import model_training,model_training_paral,model_training_paral_pruning
from datasets_torch import *
from utils import setup_logger

import argparse

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# training setting ()
parser.add_argument('--arch', default="vgg11_bn", type=str, help='please type save_file name for the testing purpose')
parser.add_argument('--cutlayer', default=4, type=int, help='number of layers in local')
parser.add_argument('--batch_size', default=128, type=int, help='training batch size')
parser.add_argument('--filename', required=True, type=str, help='please type save_file name for the testing purpose')
parser.add_argument('--folder', default="saves", type=str, help='please type folder name for the testing purpose')
parser.add_argument('--num_client', default=1, type=int, help='number of client')
parser.add_argument('--num_epochs', default=100, type=int, help='number of epochs')
parser.add_argument('--learning_rate', default=0.01, type=float, help='Learning Rate for server-side model')
parser.add_argument('--lambd', default=1.0, type=float, help='kl_weight')
parser.add_argument('--dataset_portion', default=1.0, type=float, help='dataset_portion')
parser.add_argument('--client_sample_ratio', default=1.0, type=float, help='client_sample_ratio')
parser.add_argument('--noniid', default=1.0, type=float, help='noniid_ratio, if = 0.1, meaning 1 out of 10 class per client')
parser.add_argument('--local_lr', default=-1, type=float, help='Learning Rate for client-side model')
parser.add_argument('--dataset', default="cifar10", type=str, help='number of classes for the testing dataset')
parser.add_argument('--scheme', default="V2_epoch", type=str, help='the name of the scheme, either V3 or others')

#regularization setting ()
parser.add_argument('--regularization', default="None", type=str, help='apply regularization in multi-client training.')
parser.add_argument('--regularization_strength', default=0, type=float, help='regularization_strength of regularization in multi-client training.')
parser.add_argument('--var_threshold', default=0.1, type=float, help='regularization threshold for the variance score.')
parser.add_argument('--AT_regularization', default="None", type=str, help='apply regularization in multi-client training.')
parser.add_argument('--AT_regularization_strength', default=0, type=float, help='regularization_strength of regularization in multi-client training.')
parser.add_argument('--log_entropy', default=0, type=float, help='regularization_strength of regularization in multi-client training.')
parser.add_argument('--ssim_threshold', default=0.0, type=float, help='regularization threshold for the SSIM score.')
parser.add_argument('--gan_AE_type', default="custom", type=str, help='the name of the AE used in GAN_adv, option: custom, simple, simplest')
parser.add_argument('--gan_loss_type', default="SSIM", type=str, help='loss type of training defensive decoder: SSIM or MSE')
parser.add_argument('--bottleneck_option', default="None", type=str, help='set bottleneck option')
parser.add_argument('--optimize_computation', default=1, type=int, help='set interval N to optimize_computation')
parser.add_argument('--decoder_sync', action='store_true', default=False, help='if True, we sync decoder')

#training dataset setting ()
parser.add_argument('--load_from_checkpoint', action='store_true', default=False, help='if True, we load_from_checkpoint')
parser.add_argument('--load_from_checkpoint_server', action='store_true', default=False, help='if True, we load_from_checkpoint for server-side model')
parser.add_argument('--transfer_source_task', default="cifar100", type=str, help='the name of the transfer_source_task, option: cifar10, cifar100')
parser.add_argument('--finetune_freeze_bn', action='store_true', default=False, help='if True, we finetune_freeze_bn')
parser.add_argument('--save_more_checkpoints', action='store_true', default=False, help='if True, we save_more_checkpoints')
parser.add_argument('--initialize_different', action='store_true', default=False, help='if True, we initialze differently for different client')


#training randomseed setting ()
parser.add_argument('--random_seed', default=123, type=int, help='random_seed')

args = parser.parse_args()

random_seed = args.random_seed
torch.manual_seed(random_seed)
np.random.seed(random_seed)
batch_size = args.batch_size
cutting_layer = args.cutlayer
num_client = args.num_client
save_dir_name = "./{}/{}".format(args.folder, args.filename)
mi = model_training_paral_pruning.MIA_train(args.arch, cutting_layer, batch_size,lambd=args.lambd, n_epochs = args.num_epochs, scheme = args.scheme,
                 num_client = num_client, dataset=args.dataset, save_dir=save_dir_name,random_seed=random_seed,
                 regularization_option=args.regularization, regularization_strength = args.regularization_strength, AT_regularization_option=args.AT_regularization, AT_regularization_strength = args.AT_regularization_strength, log_entropy=args.log_entropy,
                 initialize_different=args.initialize_different, learning_rate = args.learning_rate, local_lr = args.local_lr, gan_AE_type = args.gan_AE_type, 
                 load_from_checkpoint = args.load_from_checkpoint, bottleneck_option = args.bottleneck_option, 
                 optimize_computation = args.optimize_computation, decoder_sync = args.decoder_sync, 
                 finetune_freeze_bn = args.finetune_freeze_bn, gan_loss_type=args.gan_loss_type, ssim_threshold = args.ssim_threshold,var_threshold = args.var_threshold,
                 source_task = args.transfer_source_task, load_from_checkpoint_server = args.load_from_checkpoint_server, save_more_checkpoints = args.save_more_checkpoints,
                 dataset_portion = args.dataset_portion, noniid = args.noniid, client_sample_ratio = args.client_sample_ratio)
mi.logger.debug(str(args))

log_frequency = 500

LOG = mi(verbose=True, progress_bar=True, log_frequency=log_frequency)


# %%
