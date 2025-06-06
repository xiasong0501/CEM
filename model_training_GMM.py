import torch
import numpy as np
import torch.nn as nn
from torch.serialization import save
import architectures_torch as architectures
from utils import setup_logger, accuracy, AverageMeter, WarmUpLR, apply_transform_test, apply_transform, TV, l2loss, dist_corr, get_PSNR
from utils import freeze_model_bn, average_weights, DistanceCorrelationLoss, spurious_loss, prune_top_n_percent_left, dropout_defense, prune_defense
from thop import profile
import logging
from torch.autograd import Variable
from model_architectures.resnet_cifar import ResNet20, ResNet32
from model_architectures.mobilenetv2 import MobileNetV2
from model_architectures.vgg import vgg11, vgg13, vgg11_bn, vgg13_bn
import pytorch_ssim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from datetime import datetime
import os
import time
from shutil import rmtree
from GMM import fit_gmm_torch
from sklearn.manifold import TSNE
from datasets_torch import get_cifar100_trainloader, get_cifar100_testloader, get_cifar10_trainloader, \
    get_cifar10_testloader, get_mnist_bothloader, get_facescrub_bothloader, get_SVHN_trainloader, get_SVHN_testloader, get_fmnist_bothloader, get_tinyimagenet_bothloader

def init_weights(m): # weight initialization
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None:
            m.bias.data.zero_()
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None:
            m.bias.data.zero_()

def denormalize(x, dataset): # normalize a zero mean, std = 1 to range [0, 1]
    
    if dataset == "mnist" or dataset == "fmnist":
        return torch.clamp((x + 1)/2, 0, 1)
    elif dataset == "cifar10":
        std = [0.247, 0.243, 0.261]
        mean = [0.4914, 0.4822, 0.4465]
    elif dataset == "cifar100":
        std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
        mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
    elif dataset == "imagenet":
        std = [0.229, 0.224, 0.225]
        mean = [0.485, 0.456, 0.406]
    elif dataset == "facescrub":
        std = (0.2058, 0.2275, 0.2098)
        mean = (0.5708, 0.5905, 0.4272)
    elif dataset == "svhn":
        std = (0.1189, 0.1377, 0.1784)
        mean = (0.3522, 0.4004, 0.4463)
    # 3, H, W, B
    tensor = x.clone().permute(1, 2, 3, 0)
    for t, m, s in zip(range(tensor.size(0)), mean, std):
        tensor[t] = (tensor[t]).mul_(s).add_(m)
    # B, 3, H, W
    return torch.clamp(tensor, 0, 1).permute(3, 0, 1, 2)

def test_denorm(): # test function for denorm
    CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
    ])
    cifar10_training = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_train)
    cifar10_training_loader_iter = iter(DataLoader(cifar10_training, shuffle=False, num_workers=1, batch_size=128))
    transform_orig = transforms.Compose([
        transforms.ToTensor()
    ])
    cifar10_original = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_orig)
    cifar10_original_loader_iter = iter(DataLoader(cifar10_original, shuffle=False, num_workers=1, batch_size=128))
    images, _ = next(cifar10_training_loader_iter)
    orig_image, _  = next(cifar10_original_loader_iter)
    recovered_image = denormalize(images, "cifar100")
    return torch.isclose(orig_image, recovered_image)

def save_images(input_imgs, output_imgs, epoch, path, offset=0, batch_size=64): # saved image from tensor to jpg
    """
    """
    input_prefix = "inp_"
    output_prefix = "out_"
    out_folder = "{}/{}".format(path, epoch)
    out_folder = os.path.abspath(out_folder)
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)
    for img_idx in range(output_imgs.shape[0]):
        inp_img_path = "{}/{}{}.jpg".format(out_folder, input_prefix, offset * batch_size + img_idx)
        out_img_path = "{}/{}{}.jpg".format(out_folder, output_prefix, offset * batch_size + img_idx)

        if input_imgs is not None:
            save_image(input_imgs[img_idx], inp_img_path)
        if output_imgs is not None:
            save_image(output_imgs[img_idx], out_img_path)
            



class MIA_train: # main class for every thing

    def __init__(self, arch, cutting_layer, batch_size, n_epochs,lambd=1, scheme="V2_epoch", num_client=1, dataset="cifar10",
                 logger=None, save_dir=None, regularization_option="None", regularization_strength=0,
                 collude_use_public=False, initialize_different=False, learning_rate=0.1, local_lr = -1,
                 gan_AE_type="custom", random_seed=123, client_sample_ratio = 1.0,
                 load_from_checkpoint = False, bottleneck_option="None", measure_option=False,
                 optimize_computation=1, decoder_sync = False, bhtsne_option = False, gan_loss_type = "SSIM", attack_confidence_score = False,
                 ssim_threshold = 0.0, finetune_freeze_bn = False, load_from_checkpoint_server = False, source_task = "cifar100", 
                 save_activation_tensor = False, save_more_checkpoints = False, dataset_portion = 1.0, noniid = 1.0):
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        self.arch = arch
        self.bhtsne = bhtsne_option
        self.batch_size = batch_size
        self.lr = learning_rate
        self.finetune_freeze_bn = finetune_freeze_bn

        if local_lr == -1: # if local_lr is not set
            self.local_lr = self.lr
        else:
            self.local_lr = local_lr
        self.lambd=lambd
        self.n_epochs = n_epochs
        self.measure_option = measure_option
        self.optimize_computation = optimize_computation
       # self.client_sample_ratio = client_sample_ratio
       # self.dataset_portion = dataset_portion
       # self.noniid_ratio = noniid
        self.save_more_checkpoints = save_more_checkpoints

        # setup save folder
        if save_dir is None:
            self.save_dir = "./saves/{}/".format(datetime.today().strftime('%m%d%H%M'))
        else:
            self.save_dir = str(save_dir) + "/"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # setup tensorboard
        tensorboard_path = str(save_dir) + "/tensorboard"
        if not os.path.isdir(tensorboard_path):
            os.makedirs(tensorboard_path)
        self.writer = SummaryWriter(log_dir=tensorboard_path)
        
        self.save_activation_tensor = save_activation_tensor

        # setup logger
        model_log_file = self.save_dir + '/MIA.log'
        if logger is not None:
            self.logger = logger
        else:
            self.logger = setup_logger('{}_logger'.format(str(save_dir)), model_log_file, level=logging.DEBUG)
        
        self.warm = 1
        self.scheme = scheme

        # migrate old naming:
        # if self.scheme == "V1" or self.scheme == "V2" or self.scheme == "V3" or self.scheme == "V4":
            # self.scheme = self.scheme + "_batch"

        # self.num_client = num_client
        self.num_client=num_client
        self.dataset = dataset
        self.call_resume = False

        self.load_from_checkpoint = load_from_checkpoint
        self.load_from_checkpoint_server = load_from_checkpoint_server
        self.source_task = source_task
        self.cutting_layer = cutting_layer

        if self.cutting_layer == 0:
            self.logger.debug("Centralized Learning Scheme:")
        if "resnet20" in arch:
            self.logger.debug("Split Learning Scheme: Overall Cutting_layer {}/10".format(self.cutting_layer))
        if "vgg11" in arch:
            self.logger.debug("Split Learning Scheme: Overall Cutting_layer {}/13".format(self.cutting_layer))
        if "mobilenetv2" in arch:
            self.logger.debug("Split Learning Scheme: Overall Cutting_layer {}/9".format(self.cutting_layer))
        
        # self.confidence_score = attack_confidence_score
        # self.collude_use_public = collude_use_public
        # self.initialize_different = initialize_different
        
        if "C" in bottleneck_option or "S" in bottleneck_option:
            self.adds_bottleneck = True
            self.bottleneck_option = bottleneck_option
        else:
            self.adds_bottleneck = False
            self.bottleneck_option = bottleneck_option
        
        # self.decoder_sync = decoder_sync

        ''' Activation Defense ''' ##this is kind of defense method
        self.regularization_option = regularization_option

        # If strength is 0.0, then there is no regularization applied, train normally.
        self.regularization_strength = regularization_strength
        if self.regularization_strength == 0.0:
            self.regularization_option = "None"

        # setup nopeek regularizer
        if "nopeek" in self.regularization_option:
            self.nopeek = True
        else:
            self.nopeek = False

        self.alpha1 = regularization_strength  # set to 0.1 # 1000 in Official NoteBook https://github.com/tremblerz/nopeek/blob/master/noPeekCifar10%20(1)-Copy2.ipynb


        # setup gan_adv regularizer
        ## this is used to trian a robust model
        self.gan_AE_activation = "sigmoid"
        self.gan_AE_type = gan_AE_type
        self.gan_loss_type = gan_loss_type
        self.gan_decay = 0.2
        self.alpha2 = regularization_strength  # set to 1~10
        self.pretrain_epoch = 100

        self.ssim_threshold = ssim_threshold
        if "gan_adv" in self.regularization_option:
            self.gan_regularizer = True
            if "step" in self.regularization_option:
                try:
                    self.gan_num_step = int(self.regularization_option.split("step")[-1])
                except:
                    print("Auto extract step fail, geting default value 3")
                    self.gan_num_step = 3
            else:
                self.gan_num_step = 3
            if "noise" in self.regularization_option:
                self.gan_noise = True
            else:
                self.gan_noise = False
        else:
            self.gan_regularizer = False
            self.gan_noise = False
            self.gan_num_step = 1

        # setup local dp (noise-injection defense)
        if "local_dp" in self.regularization_option:
            self.local_DP = True
        else:
            self.local_DP = False

        self.dp_epsilon = regularization_strength

        if "dropout" in self.regularization_option:
            self.dropout_defense = True
            try: 
                self.dropout_ratio = float(self.regularization_option.split("dropout")[1].split("_")[0])
            except:
                self.dropout_ratio = regularization_strength
                print("Auto extract dropout ratio fail, use regularization_strength input as dropout ratio")
        else:
            self.dropout_defense = False
            self.dropout_ratio = regularization_strength
        
        if "topkprune" in self.regularization_option:
            self.topkprune = True
            try: 
                self.topkprune_ratio = float(self.regularization_option.split("topkprune")[1].split("_")[0])
            except:
                self.topkprune_ratio = regularization_strength
                print("Auto extract topkprune ratio fail, use regularization_strength input as topkprune ratio")
        else:
            self.topkprune = False
            self.topkprune_ratio = regularization_strength
        
        ''' Activation Defense (end)'''


        # client sampling: dividing datasets to actual number of clients, self.num_clients is fake num of clients for ease of simulation.
        # multiplier = 1/self.client_sample_ratio #100
        # actual_num_users = int(multiplier * self.num_client)
        # self.
        actual_num_users = 1
        self.collude_use_public=False
        num_workers=8
        # setup dataset
        if self.dataset == "cifar10":
            self.client_dataloader, self.mem_trainloader, self.mem_testloader = get_cifar10_trainloader(batch_size=self.batch_size,
                                                                                                        num_workers=num_workers,
                                                                                                        shuffle=True,
                                                                                                        num_client=actual_num_users,
                                                                                                        collude_use_public=self.collude_use_public
                                                                                                        )
            self.pub_dataloader, self.nomem_trainloader, self.nomem_testloader = get_cifar10_testloader(batch_size=self.batch_size,
                                                                                                        num_workers=num_workers,
                                                                                                        shuffle=False)
            self.orig_class = 10
        elif self.dataset == "cifar100":
            self.client_dataloader, self.mem_trainloader, self.mem_testloader = get_cifar100_trainloader(batch_size=self.batch_size,
                                                                                                         num_workers=num_workers,
                                                                                                         shuffle=True,
                                                                                                         num_client=actual_num_users,
                                                                                                         collude_use_public=self.collude_use_public)
            self.pub_dataloader, self.nomem_trainloader, self.nomem_testloader = get_cifar100_testloader(batch_size=self.batch_size,
                                                                                                         num_workers=num_workers,
                                                                                                         shuffle=False)
            self.orig_class = 100

        elif self.dataset == "svhn":
            self.client_dataloader, self.mem_trainloader, self.mem_testloader = get_SVHN_trainloader(batch_size=self.batch_size,
                                                                                                         num_workers=num_workers,
                                                                                                         shuffle=True,
                                                                                                         num_client=actual_num_users,
                                                                                                         collude_use_public=self.collude_use_public)
            self.pub_dataloader, self.nomem_trainloader, self.nomem_testloader = get_SVHN_testloader(batch_size=self.batch_size,
                                                                                                         num_workers=num_workers,
                                                                                                         shuffle=False)
            self.orig_class = 10

        elif self.dataset == "facescrub":
            self.client_dataloader, self.pub_dataloader = get_facescrub_bothloader(batch_size=self.batch_size, 
                                                                                num_workers=num_workers,
                                                                                shuffle=True,
                                                                                num_client=actual_num_users,
                                                                                collude_use_public=self.collude_use_public)
            self.orig_class = 530
        elif self.dataset == "tinyimagenet":
            self.client_dataloader, self.pub_dataloader = get_tinyimagenet_bothloader(batch_size=self.batch_size, 
                                                                                num_workers=num_workers,
                                                                                shuffle=True,
                                                                                num_client=actual_num_users,
                                                                                collude_use_public=self.collude_use_public)
            self.orig_class = 200
        elif self.dataset == "mnist":
            self.client_dataloader, self.pub_dataloader = get_mnist_bothloader(batch_size=self.batch_size, 
                                                                                num_workers=num_workers,
                                                                                shuffle=True,
                                                                                num_client=actual_num_users,
                                                                                collude_use_public=self.collude_use_public)
            self.orig_class = 10
        elif self.dataset == "fmnist":
            self.client_dataloader, self.pub_dataloader = get_fmnist_bothloader(batch_size=self.batch_size, 
                                                                                num_workers=num_workers,
                                                                                shuffle=True,
                                                                                num_client=actual_num_users,
                                                                                collude_use_public=self.collude_use_public)
            self.orig_class = 10
        else:
            raise ("Dataset {} is not supported!".format(self.dataset))
        self.num_class = self.orig_class
        self.num_batches = len(self.client_dataloader[0])
        print("Total number of batches per epoch for each client is ", self.num_batches)

        self.model = None


        # Initialze all client, server side models.

        self.initialize_different = False

        if arch == "resnet20":
            model = ResNet20(cutting_layer, self.logger, num_client=self.num_client, num_class=self.num_class,
                             initialize_different=self.initialize_different, adds_bottleneck=self.adds_bottleneck, bottleneck_option = self.bottleneck_option)
        elif arch == "resnet32":
            model = ResNet32(cutting_layer, self.logger, num_client=self.num_client, num_class=self.num_class,
                             initialize_different=self.initialize_different, adds_bottleneck=self.adds_bottleneck, bottleneck_option = self.bottleneck_option)
        elif arch == "vgg13":
            model = vgg13(cutting_layer, self.logger, num_client=self.num_client, num_class=self.num_class,
                          initialize_different=self.initialize_different, adds_bottleneck=self.adds_bottleneck, bottleneck_option = self.bottleneck_option)
        elif arch == "vgg11":
            model = vgg11(cutting_layer, self.logger, num_client=self.num_client, num_class=self.num_class,
                          initialize_different=self.initialize_different, adds_bottleneck=self.adds_bottleneck, bottleneck_option = self.bottleneck_option)
        elif arch == "vgg13_bn":
            model = vgg13_bn(cutting_layer, self.logger, num_client=self.num_client, num_class=self.num_class,
                             initialize_different=self.initialize_different, adds_bottleneck=self.adds_bottleneck, bottleneck_option = self.bottleneck_option)
        elif arch == "vgg11_bn":
            model = vgg11_bn(cutting_layer, self.logger, num_client=self.num_client, num_class=self.num_class,
                             initialize_different=self.initialize_different, adds_bottleneck=self.adds_bottleneck, bottleneck_option = self.bottleneck_option)
        elif arch == "mobilenetv2":
            model = MobileNetV2(cutting_layer, self.logger, num_client=self.num_client, num_class=self.num_class,
                             initialize_different=self.initialize_different, adds_bottleneck=self.adds_bottleneck, bottleneck_option = self.bottleneck_option)
        else:
            raise ("No such architecture!")
        self.model = model

        self.f = model.local_list[0]
        self.f_tail = model.cloud
        self.classifier = model.classifier
        self.f.cuda()
        self.f_tail.cuda()
        self.classifier.cuda()
        self.params = list(self.f_tail.parameters()) + list(self.classifier.parameters())
        self.local_params = []
        if cutting_layer > 0:
            self.local_params.append(self.f.parameters())

    


        # setup optimizers
        self.optimizer = torch.optim.SGD(self.params, lr=self.lr, momentum=0.9, weight_decay=5e-4)
        milestones = [60, 120, 180]


        self.local_optimizer_list = []
        self.train_local_scheduler_list = []
        self.warmup_local_scheduler_list = []
        for i in range(len(self.local_params)):
            self.local_optimizer_list.append(torch.optim.SGD(list(self.local_params[i]), lr=self.local_lr, momentum=0.9, weight_decay=5e-4))
            self.train_local_scheduler_list.append(torch.optim.lr_scheduler.MultiStepLR(self.local_optimizer_list[i], milestones=milestones,
                                                                    gamma=0.2))  # learning rate decay
            self.warmup_local_scheduler_list.append(WarmUpLR(self.local_optimizer_list[i], self.num_batches * self.warm))

        self.train_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones,
                                                                    gamma=0.2)  # learning rate decay
        self.warmup_scheduler = WarmUpLR(self.optimizer, self.num_batches * self.warm)
        
        # Set up GAN_ADV, used to training the encoder
        self.local_AE_list = []
        self.gan_params = []
        if self.gan_regularizer:
            feature_size = self.model.get_smashed_data_size()

            if self.gan_AE_type == "custom":
                self.local_AE_list.append(
                    architectures.custom_AE(input_nc=feature_size[1], output_nc=3, input_dim=feature_size[2],
                                            output_dim=32, activation=self.gan_AE_activation))
            elif "conv_normN" in self.gan_AE_type:
                try:
                    afterfix = self.gan_AE_type.split("conv_normN")[1]
                    N = int(afterfix.split("C")[0])
                    internal_C = int(afterfix.split("C")[1])
                except:
                    print("auto extract N from conv_normN failed, set N to default 0")
                    N = 0
                    internal_C = 64
                self.local_AE_list.append(architectures.conv_normN_AE(N = N, internal_nc = internal_C, input_nc=feature_size[1], output_nc=3,
                                                         input_dim=feature_size[2], output_dim=32,
                                                         activation=self.gan_AE_activation))
            elif "res_normN" in self.gan_AE_type:
                try:
                    afterfix = self.gan_AE_type.split("res_normN")[1]
                    N = int(afterfix.split("C")[0])
                    internal_C = int(afterfix.split("C")[1])
                except:
                    print("auto extract N from res_normN failed, set N to default 0")
                    N = 0
                    internal_C = 64
                self.local_AE_list.append(architectures.res_normN_AE(N = N, internal_nc = internal_C, input_nc=feature_size[1], output_nc=3,
                                                         input_dim=feature_size[2], output_dim=32,
                                                         activation=self.gan_AE_activation))
            else:
                raise ("No such GAN AE type.")
            self.gan_params.append(self.local_AE_list[i].parameters())
            self.local_AE_list[i].apply(init_weights)
            self.local_AE_list[i].cuda()
            
            self.gan_optimizer_list = []
            self.gan_scheduler_list = []
            milestones = [60, 120, 160, 200]


            for i in range(len(self.gan_params)):
                self.gan_optimizer_list.append(torch.optim.Adam(list(self.gan_params[i]), lr=1e-3))
                self.gan_scheduler_list.append(torch.optim.lr_scheduler.MultiStepLR(self.gan_optimizer_list[i], milestones=milestones,
                                                                      gamma=self.gan_decay))  # learning rate decay
            
    def optimizer_step(self, set_client = False, client_id = 0):
        
        for i in range(len(self.local_optimizer_list)):
            self.local_optimizer_list[i].step()
        self.optimizer.step()

    def optimizer_zero_grad(self):
        self.optimizer.zero_grad()
        for i in range(len(self.local_optimizer_list)):
            self.local_optimizer_list[i].zero_grad()

    def scheduler_step(self, epoch = 0, warmup = False):
        if warmup:
            self.warmup_scheduler.step()
            for i in range(len(self.warmup_local_scheduler_list)):
                self.warmup_local_scheduler_list[i].step()
        else:
            self.train_scheduler.step(epoch)
            for i in range(len(self.train_local_scheduler_list)):
                self.train_local_scheduler_list[i].step(epoch)

    def gan_scheduler_step(self, epoch = 0):
        for i in range(len(self.gan_scheduler_list)):
            self.gan_scheduler_list[i].step(epoch)


    
    def kmeans_cuda(self,X, num_clusters,centroids,random_ini_centers, num_iterations=10, tol=1e-4):
        N, D = X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]
        X_flat = X.reshape(N, D)  # flatten

        # random initialize 
        
        if random_ini_centers :
            print('randomized selected centroids')
        centroids = X_flat[torch.randperm(N)[:num_clusters]].clone()
        # if torch.isnan(centroids).any() > 0:
        #     centroids = X_flat[torch.randperm(N)[:num_clusters]].clone()
        # else:            
        #     # print(torch.isnan(centroids))
        #     if len(centroids) > num_clusters:
        #         indices = torch.randperm(len(centroids))[:num_clusters]
        #         centroids = centroids[indices]
        #     elif len(centroids) < num_clusters:
        #         indices = torch.randint(low=0, high=len(centroids), size=(num_clusters,))
        #         centroids = centroids[indices]
                # else:
                #     # 如果 n == k，直接使用原 tensor
                #     result_tensor = tensor
        labels = torch.zeros(N, dtype=torch.long, device=X.device)

        for _ in range(num_iterations):
            #calculate distance
            distances = torch.cdist(X_flat, centroids)
            new_labels = distances.argmin(dim=1)

            unique_cluster_assignments = torch.unique(new_labels)
            if torch.equal(labels, new_labels):
                break
            labels = new_labels
            unique_cluster_assignments = torch.unique(labels)
            
            
            for i in unique_cluster_assignments:
                centroids[i] = X_flat[labels == i].mean(dim=0)

        # calculate variance
        # cluster_variances = torch.tensor([((X_flat[labels == i] - centroids[i])**2).mean() for i in range(num_clusters)], device=X.device)
        
        cluster_variances = torch.tensor([((X_flat[labels == i] - centroids[i])**2).mean() for i in unique_cluster_assignments], device=X_flat.device)
        average_variance = cluster_variances.mean().item()
        if torch.isnan(cluster_variances.mean()).any():
            print(cluster_variances,unique_cluster_assignments,centroids.mean()) 

        return centroids, average_variance

    def compute_class_means(self, features, labels, unique_labels,centroids_list):
        class_means = []
        # intra_class_mse = 0.0
        unique_labels=unique_labels.cpu().numpy()
        labels=labels.cpu().numpy()
        label_it = 0
        for i in unique_labels:
            
            centroids = centroids_list[i]

            

            num_clusters = centroids.size(0)

            class_mask = (labels == i)
            class_features = features[class_mask.squeeze(), :]

            class_mean = class_features.mean(dim=0)
            
            N, D = class_features.shape[0], class_features.shape[1] *class_features.shape[2] * class_features.shape[3]
            class_features_flat = class_features.reshape(N, D)  # flatten

            distances = torch.cdist(class_features_flat, centroids).detach().cpu().numpy()  
            
            cluster_assignments = np.argmin(distances, axis=1)
            unique_cluster_assignments = np.unique(cluster_assignments)
            cluster_variances=[]
            # average_variance=torch.tensor(0).cuda()
            num=0
            for j in unique_cluster_assignments:
            #     # print([cluster_assignments == j], sum((cluster_assignments == j)), sum(class_mask))
                # cluster_assignments_np=np.array(cluster_assignments)
                indice_cluster=cluster_assignments == j
                # print(j)
                # print(indice_cluster,cluster_assignments_np)
                weight=sum(indice_cluster)/sum(class_mask)
                variances=torch.mean(((class_features_flat[indice_cluster] - centroids[j])**2),dim=0).cuda()
                # sorted_indices = torch.argsort(variances, descending=True)
            #     # 取出前10个最大的值和它们的索引
            #     # top_indices = sorted_indices[:10]
            #     # top_values = variances[top_indices]
                # top_values, top_indices = torch.topk(variances, 100)
                reg_variances = (variances+self.regularization_strength**2) #/self.regularization_strength**2
                mutual_infor= torch.log(reg_variances+0.000001)
                # reg_mutual_infor=F.relu(mutual_infor-torch.log(torch.tensor(1.25))).mean()
                mean_mutual_infor = mutual_infor.mean()*torch.tensor(weight)
                # variances= torch.mean(top10_values,dim=0)
                if num==0:
                    average_variance=mean_mutual_infor
                else:
                    average_variance+=mean_mutual_infor
                num+=1
                # cluster_variances.append(variances) 

                # print(((class_features_flat[cluster_assignments == j] - centroids[j])**2).shape)
                # covariance_matrix = torch.matmul(centered_data.T, centered_data) / sum((cluster_assignments == j))
                # # cluster_variances_mean = torch.mean(cluster_variances,0)  
                # covariance_matrix = covariance_matrix + self.regularization_strength* torch.eye(covariance_matrix.size(0)).cuda()
                # print(covariance_matrix)
                # det = torch.det(covariance_matrix+0.01)
                # print('det value is:', det)
                # mutal_info= torch.log(covariance_matrix)
            # print(variances.shape,weight,variances)
            # cluster_variances= torch.stack(cluster_variances)
            # print(cluster_variances.shape)

            # cluster_variances = torch.stack([((class_features_flat[cluster_assignments == i] - centroids[i])**2).mean() for i in unique_cluster_assignments])
            # average_variance = cluster_variances.mean()
            # print(f"class_features_flat.requires_grad: {class_features_flat.requires_grad}")
            # print(f"average_variance.requires_grad: {average_variance.requires_grad}")
            if label_it==0:
                intra_class_mse=average_variance
            else:
                intra_class_mse+=average_variance
            class_means.append(class_mean)
            label_it+=1
        intra_class_mse /= len(unique_labels)
        class_means = torch.stack(class_means)  # Shape: [num_classes, 128, 8, 8]
        class_mean_overall = class_means.mean(dim=0)  # 全局均值
        inter_class_mse = F.mse_loss(class_means, class_mean_overall.expand_as(class_means))
        loss = intra_class_mse #- 0.5*inter_class_mse

        # print(loss)
        # print(centroids_list)
        return loss,intra_class_mse

    '''Main training function, the communication between client/server is implicit to keep a fast training speed'''
    def train_target_step(self, x_private, label_private, adding_noise,random_ini_centers,centroids_list,client_id=0):
        self.f_tail.train()
        self.classifier.train()
        self.f.train()
        x_private = x_private.cuda()
        label_private = label_private.cuda()

        # Freeze batchnorm parameter of the client-side model.
        if self.load_from_checkpoint and self.finetune_freeze_bn:
            freeze_model_bn(self.f)


        z_private = self.f(x_private)
        unique_labels = torch.unique(label_private)
        if not random_ini_centers:
            rob_loss,intra_class_mse = self.compute_class_means(z_private, label_private, unique_labels, centroids_list)
        else:
            rob_loss,intra_class_mse=torch.tensor(0.0),torch.tensor(0.0)
        # assert 1==0, print(x_private.shape,label_private.shape,unique_values)
        # Final Prediction Logits (complete forward pass)
        if "Gaussian" in self.regularization_option: # and adding_noise:
            if not random_ini_centers:
                if intra_class_mse<1:
                    sigma = self.regularization_strength#*intra_class_mse
                else:
                    sigma = self.regularization_strength
            else:
                sigma = self.regularization_strength
            noise = sigma * torch.randn_like(z_private).cuda()
            z_private_n =z_private + noise
        else:
            z_private_n=z_private
        # Perform various activation defenses, default no defense
        if self.local_DP:
            if "laplace" in self.regularization_option:
                noise = torch.from_numpy(
                    np.random.laplace(loc=0, scale=1 / self.dp_epsilon, size=z_private.size())).cuda()
                z_private = z_private + noise.detach().float()
            else:  # apply gaussian noise
                delta = 10e-5
                sigma = np.sqrt(2 * np.log(1.25 / delta)) * 1 / self.dp_epsilon
                noise = sigma * torch.randn_like(z_private).cuda()
                z_private = z_private + noise.detach().float()
        if self.dropout_defense:
            z_private = dropout_defense(z_private, self.dropout_ratio)
        if self.topkprune:
            z_private = prune_defense(z_private, self.topkprune_ratio)
        if self.gan_noise:
            epsilon = self.alpha2
            
            self.local_AE_list[client_id].eval()
            fake_act = z_private.clone()
            grad = torch.zeros_like(z_private).cuda()
            fake_act = torch.autograd.Variable(fake_act.cuda(), requires_grad=True)
            x_recon = self.local_AE_list[client_id](fake_act)
            x_private = denormalize(x_private, self.dataset)
            
            if self.gan_loss_type == "SSIM":
                ssim_loss = pytorch_ssim.SSIM()
                loss = ssim_loss(x_recon, x_private)
                loss.backward()
                grad -= torch.sign(fake_act.grad)
            elif self.gan_loss_type == "MSE":
                mse_loss = torch.nn.MSELoss()
                loss = mse_loss(x_recon, x_private)
                loss.backward()
                grad += torch.sign(fake_act.grad)  
            z_private = z_private - grad.detach() * epsilon

        output = self.f_tail(z_private_n)

        if "mobilenetv2" in self.arch:
            output = F.avg_pool2d(output, 4)
            output = output.view(output.size(0), -1)
            output = self.classifier(output)
        elif self.arch == "resnet20" or self.arch == "resnet32":
            output = F.avg_pool2d(output, 8)
            output = output.view(output.size(0), -1)
            output = self.classifier(output)
        else:
            output = output.view(output.size(0), -1)
            output = self.classifier(output)
        
        criterion = torch.nn.CrossEntropyLoss()

        f_loss = criterion(output, label_private)

        if not random_ini_centers:
            total_loss = f_loss#+2*rob_loss
        else:    
            total_loss = f_loss#+0*rob_loss


        # perform nopeek regularization
        if self.nopeek:
            #
            if "ttitcombe" in self.regularization_option:
                dc = DistanceCorrelationLoss()
                dist_corr_loss = self.alpha1 * dc(x_private, z_private)
            else:
                dist_corr_loss = self.alpha1 * dist_corr(x_private, z_private).sum()

            total_loss = total_loss + dist_corr_loss
        
        # perform our proposed attacker-aware training
        if self.gan_regularizer and not self.gan_noise:
            self.local_AE_list[client_id].eval()
            output_image = self.local_AE_list[client_id](z_private)
            
            x_private = denormalize(x_private, self.dataset)
            
            if self.gan_loss_type == "SSIM":
                ssim_loss = pytorch_ssim.SSIM()
                ssim_term = ssim_loss(output_image, x_private)
                
                if self.ssim_threshold > 0.0:
                    if ssim_term > self.ssim_threshold:
                        gan_loss = self.alpha2 * (ssim_term - self.ssim_threshold) # Let SSIM approaches 0.4 to avoid overfitting
                    else:
                        gan_loss = 0.0 # Let SSIM approaches 0.4 to avoid overfitting
                else:
                    gan_loss = self.alpha2 * ssim_term  
            elif self.gan_loss_type == "MSE":
                mse_loss = torch.nn.MSELoss()
                mse_term = mse_loss(output_image, x_private)
                gan_loss = - self.alpha2 * mse_term  
            total_loss = total_loss + gan_loss
            

       
        if not random_ini_centers and self.lambd>0:
            # print(rob_loss)
            rob_loss.backward(retain_graph=True)
            encoder_gradients = {name: param.grad.clone() for name, param in self.f.named_parameters()}
            # optimizer.zero_grad()
            self.optimizer_zero_grad()

        total_loss.backward()
        if not random_ini_centers and self.lambd>0:
            for name, param in self.f.named_parameters():
                param.grad += self.lambd*encoder_gradients[name]
            # print('Nonekl' in self.regularization_option)
            # print(self.regularization_option)
            # print('consider kl loss')
        total_losses = total_loss.detach().cpu().numpy()
        f_losses = f_loss.detach().cpu().numpy()
        del total_loss, f_loss

        return intra_class_mse, f_losses, z_private

    # Main function for validation accuracy, is also used to get statistics
    def validate_target(self, client_id=0):
        """
        Run evaluation
        """
        # batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        val_loader = self.pub_dataloader

        # switch to evaluate mode
        if client_id == 0:
            self.f.eval()
        elif client_id == 1:
            self.c.eval()
        elif client_id > 1:
            self.model.local_list[client_id].eval()
        self.f_tail.eval()
        self.classifier.eval()
        criterion = nn.CrossEntropyLoss()

        activation_0 = {}

        def get_activation_0(name):
            def hook(model, input, output):
                activation_0[name] = output.detach()

            return hook
            # with torch.no_grad():

            #     count = 0
            #     for name, m in self.model.cloud.named_modules():
            #         if attack_from_later_layer == count:
            #             m.register_forward_hook(get_activation_4("ACT-{}".format(name)))
            #             valid_key = "ACT-{}".format(name)
            #             break
            #         count += 1
            #     output = self.model.cloud(ir)

            # ir = activation_4[valid_key]

        # for name, m in self.model.local_list[client_id].named_modules():
        #     m.register_forward_hook(get_activation_0("ACT-client-{}-{}".format(name, str(m).split("(")[0])))

        # for name, m in self.f_tail.named_modules():
        #     m.register_forward_hook(get_activation_0("ACT-server-{}-{}".format(name, str(m).split("(")[0])))


        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()
            activation_0 = {}
            # compute output
            with torch.no_grad():

                output = self.model.local_list[client_id](input)

                # code for save the activation of cutlayer
                

                if self.bhtsne:
                    self.save_activation_bhtsne(output, target, client_id)
                    exit()
                if "Gaussian" in self.regularization_option:
                    sigma = self.regularization_strength
                    noise = sigma * torch.randn_like(output).cuda()
                    output += noise
                '''Optional, Test validation performance with local_DP/dropout (apply DP during query)'''
                if self.local_DP:
                    if "laplace" in self.regularization_option:
                        noise = torch.from_numpy(
                            np.random.laplace(loc=0, scale=1 / self.dp_epsilon, size=output.size())).cuda()
                    else:  # apply gaussian noise
                        delta = 10e-5
                        sigma = np.sqrt(2 * np.log(1.25 / delta)) * 1 / self.dp_epsilon
                        noise = sigma * torch.randn_like(output).cuda()
                    output += noise
                if self.dropout_defense:
                    output = dropout_defense(output, self.dropout_ratio)
                if self.topkprune:
                    output = prune_defense(output, self.topkprune_ratio)
            
            '''Optional, Test validation performance with gan_noise (apply gan_noise during query)'''
            if self.gan_noise:
                epsilon = self.alpha2
                
                self.local_AE_list[client_id].eval()
                fake_act = output.clone()
                grad = torch.zeros_like(output).cuda()
                fake_act = torch.autograd.Variable(fake_act.cuda(), requires_grad=True)
                x_recon = self.local_AE_list[client_id](fake_act)
                
                input = denormalize(input, self.dataset)

                if self.gan_loss_type == "SSIM":
                    ssim_loss = pytorch_ssim.SSIM()
                    loss = ssim_loss(x_recon, input)
                    loss.backward()
                    grad -= torch.sign(fake_act.grad)
                elif self.gan_loss_type == "MSE":
                    mse_loss = torch.nn.MSELoss()
                    loss = mse_loss(x_recon, input)
                    loss.backward()
                    grad += torch.sign(fake_act.grad) 

                output = output - grad.detach() * epsilon
            
            with torch.no_grad():
                output = self.f_tail(output)

                if "mobilenetv2" in self.arch:
                    output = F.avg_pool2d(output, 4)
                    output = output.view(output.size(0), -1)
                    output = self.classifier(output)
                elif self.arch == "resnet20" or self.arch == "resnet32":
                    output = F.avg_pool2d(output, 8)
                    output = output.view(output.size(0), -1)
                    output = self.classifier(output)
                else:
                    output = output.view(output.size(0), -1)
                    output = self.classifier(output)
                loss = criterion(output, target)


            # Get statistics of server/client's per-layer activation
            # if i == 0:
            #     try:
            #         if not os.path.exists(self.save_dir):
            #             os.makedirs(self.save_dir)

            #         # setup tensorboard
            #         if self.save_activation_tensor:
            #             save_tensor_path = self.save_dir + "/saved_tensors"
            #             if not os.path.isdir(save_tensor_path):
            #                 os.makedirs(save_tensor_path)
            #         for key, value in activation_0.items():
            #             if "client" in key:
            #                 self.writer.add_histogram("local_act/{}".format(key), value.clone().cpu().data.numpy(), i)
            #                 if self.save_activation_tensor:
            #                     np.save(save_tensor_path + "/{}_{}.npy".format(key, i), value.clone().cpu().data.numpy())
            #             if "server" in key:
            #                 self.writer.add_histogram("server_act/{}".format(key), value.clone().cpu().data.numpy(), i)
            #                 if self.save_activation_tensor:
            #                     np.save(save_tensor_path + "/{}_{}.npy".format(key, i), value.clone().cpu().data.numpy())
                    
            #         for name, m in self.model.local_list[client_id].named_modules():
            #             handle = m.register_forward_hook(get_activation_0("ACT-client-{}-{}".format(name, str(m).split("(")[0])))
            #             handle.remove()
            #         for name, m in self.f_tail.named_modules():
            #             handle = m.register_forward_hook(get_activation_0("ACT-server-{}-{}".format(name, str(m).split("(")[0])))
            #             handle.remove()
            #     except:
            #         print("something went wrong adding histogram, ignore it..")

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            # prec1 = accuracy(output.data, target, compress_V4shadowlabel=self.V4shadowlabel, num_client=self.num_client)[0] #If V4shadowlabel is activated, add one extra step to process output back to orig_class
            prec1 = accuracy(output.data, target)[
                0]  # If V4shadowlabel is activated, add one extra step to process output back to orig_class
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time

            if i % 50 == 0:
                self.logger.debug('Epoch {ep}\t''Test (client-{0}):\t'
                                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    client_id, ep=i, loss=losses,
                    top1=top1))
        # for name, param in self.model.local_list[client_id].named_parameters():
        #     self.writer.add_histogram("local_params/{}".format(name), param.clone().cpu().data.numpy(), 1)
        # for name, param in self.model.cloud.named_parameters():
        #     self.writer.add_histogram("server_params/{}".format(name), param.clone().cpu().data.numpy(), 1)
        self.logger.debug(' * Prec@1 {top1.avg:.3f}'
                          .format(top1=top1))

        return top1.avg, losses.avg

    # auto complete model's name, since we have many
    def infer_path_list(self, path_to_infer):
        split_list = path_to_infer.split("checkpoint_f")
        first_part = split_list[0]
        second_part = split_list[1]
        model_path_list = []
        for i in range(self.num_client):
            if i == 0:
                model_path_list.append(path_to_infer)
            elif i == 1:
                model_path_list.append(first_part + "checkpoint_c" + second_part)
            else:
                model_path_list.append(first_part + "checkpoint_local{}".format(i) + second_part)

        return model_path_list

    # resume all client and server model from checkpoint
    def resume(self, model_path_f=None):
        if model_path_f is None:
            try:
                if "V" in self.scheme:
                    checkpoint = torch.load(self.save_dir + "checkpoint_f_{}.tar".format(self.n_epochs))
                    model_path_list = self.infer_path_list(self.save_dir + "checkpoint_f_{}.tar".format(self.n_epochs))
                else:
                    checkpoint = torch.load(self.save_dir + "checkpoint_{}.tar".format(self.n_epochs))
                    # model_path_list = self.infer_path_list(self.save_dir + "checkpoint_200.tar")
            except:
                print("No valid Checkpoint Found!")
                return
        else:
            if "V" in self.scheme:
                model_path_list = self.infer_path_list(model_path_f)

        if "V" in self.scheme:
            for i in range(self.num_client):
                print("load client {}'s local".format(i))
                checkpoint_i = torch.load(model_path_list[i])
                self.model.local_list[i].cuda()
                self.model.local_list[i].load_state_dict(checkpoint_i, strict = False)
        else:
            checkpoint = torch.load(model_path_f)
            self.model.cuda()
            self.model.load_state_dict(checkpoint, strict = False)
            self.f = self.model.local
            self.f.cuda()

        try:
            self.call_resume = True
            print("load cloud")
            checkpoint = torch.load(self.save_dir + "checkpoint_cloud_{}.tar".format(self.n_epochs))
            self.f_tail.cuda()
            self.f_tail.load_state_dict(checkpoint, strict = False)
            print("load classifier")
            checkpoint = torch.load(self.save_dir + "checkpoint_classifier_{}.tar".format(self.n_epochs))
            self.classifier.cuda()
            self.classifier.load_state_dict(checkpoint, strict = False)
        except:
            print("might be old style saving, load entire model")
            checkpoint = torch.load(model_path_f)
            self.model.cuda()
            self.model.load_state_dict(checkpoint, strict = False)
            self.call_resume = True
            self.f = self.model.local
            self.f.cuda()
            self.f_tail = self.model.cloud
            self.f_tail.cuda()
            self.classifier = self.model.classifier
            self.classifier.cuda()
    
    # client-side model sync
    def sync_client(self):
        # update global weights
        global_weights = average_weights(self.model.local_list)

        # update global weights
        for i in range(self.num_client):
            self.model.local_list[i].load_state_dict(global_weights)

    # decoder sync
    def sync_decoder(self):
        # update global weights
        global_weights = average_weights(self.local_AE_list)

        # update global weights
        for i in range(self.num_client):
            self.local_AE_list[i].load_state_dict(global_weights)

    # train local inversion model
    def gan_train_step(self, input_images, client_id, loss_type="SSIM"):
        device = next(self.model.local_list[client_id].parameters()).device

        input_images = input_images.to(device)

        self.model.local_list[client_id].eval()

        z_private = self.model.local_list[client_id](input_images)

        self.local_AE_list[client_id].train()

        x_private, z_private = Variable(input_images).to(device), Variable(z_private)

        x_private = denormalize(x_private, self.dataset)

        if self.gan_noise:
            epsilon = self.alpha2
            
            self.local_AE_list[client_id].eval()
            fake_act = z_private.clone()
            grad = torch.zeros_like(z_private).cuda()
            fake_act = torch.autograd.Variable(fake_act.cuda(), requires_grad=True)
            x_recon = self.local_AE_list[client_id](fake_act)
            if loss_type == "SSIM":
                ssim_loss = pytorch_ssim.SSIM()
                loss = ssim_loss(x_recon, x_private)
                loss.backward()
                grad -= torch.sign(fake_act.grad)
            elif loss_type == "MSE":
                MSE_loss = torch.nn.MSELoss()
                loss = MSE_loss(x_recon, x_private)
                loss.backward()
                grad += torch.sign(fake_act.grad)
            else:
                raise ("No such loss_type for gan train step")
            
            z_private = z_private - grad.detach() * epsilon
            
            self.local_AE_list[client_id].train()

        output = self.local_AE_list[client_id](z_private.detach())

        if loss_type == "SSIM":
            ssim_loss = pytorch_ssim.SSIM()
            loss = -ssim_loss(output, x_private)
        elif loss_type == "MSE":
            MSE_loss = torch.nn.MSELoss()
            loss = MSE_loss(output, x_private)
        else:
            raise ("No such loss_type for gan train step")
        for i in range(len(self.gan_optimizer_list)):
            self.gan_optimizer_list[i].zero_grad()

        loss.backward()

        for i in range(len(self.gan_optimizer_list)):
            self.gan_optimizer_list[i].step()

        losses = loss.detach().cpu().numpy()
        del loss

        return losses


    # Main function for controlling training and testing, soul of ResSFL
    def __call__(self, log_frequency=100, verbose=False, progress_bar=True):
        

        self.logger.debug("Model's smashed-data size is {}".format(str(self.model.get_smashed_data_size())))
        
        best_avg_accu = 0.0
        if not self.call_resume:
            LOG = np.zeros((self.n_epochs * self.num_batches, self.num_client))
            client_iterator_list = []
            for client_id in range(self.num_client):
                client_iterator_list.append(iter(self.client_dataloader[client_id]))

            #load pre-train models
            if self.load_from_checkpoint:
                checkpoint_dir = "./pretrained_models/{}_cutlayer_{}_bottleneck_{}_dataset_{}/".format(self.arch, self.cutting_layer, self.bottleneck_option, self.source_task)
                try:
                    checkpoint_i = torch.load(checkpoint_dir + "checkpoint_f_best.tar")
                except:
                    print("No valid Checkpoint Found!")
                    return
                    
                self.model.cuda()
                self.model.local.load_state_dict(checkpoint_i)
                self.f = self.model.local
                self.f.cuda()
                
                load_classfier = False
                if self.load_from_checkpoint_server:
                    print("load cloud")
                    checkpoint = torch.load(checkpoint_dir + "checkpoint_cloud_best.tar")
                    self.f_tail.cuda()
                    self.f_tail.load_state_dict(checkpoint)
                if load_classfier:
                    print("load classifier")
                    checkpoint = torch.load(checkpoint_dir + "checkpoint_classifier_best.tar")
                    self.classifier.cuda()
                    self.classifier.load_state_dict(checkpoint)

            if self.gan_regularizer:
                self.pre_GAN_train(30, range(self.num_client))


            self.logger.debug("Real Train Phase: done by all clients, for total {} epochs".format(self.n_epochs))

            if self.save_more_checkpoints:
                epoch_save_list = [1, 2 ,5 ,10 ,20 ,50 ,100]
            else:
                epoch_save_list = []
            # If optimize_computation, set GAN updating frequency to 1/5.
            ssim_log = 0.
            
            interval = self.optimize_computation
            self.logger.debug("GAN training interval N (once every N step) is set to {}!".format(interval))
            
            adding_noise=False
            centroids_list= [torch.tensor(float('nan')) for _ in range(self.num_class)]
            #Main Training
            lambd_start= self.lambd 
            lambd_end=lambd_start*2
            for epoch in range(1, self.n_epochs+1):
                ep_start_time = time.time() 
                if epoch > self.warm:
                    self.scheduler_step(epoch)
                    if self.gan_regularizer:
                        self.gan_scheduler_step(epoch)
                
                if epoch > 0.3*self.n_epochs:
                    print('start to adding noise')
                    adding_noise=True


                self.logger.debug("Train in {} style".format(self.scheme))
                print("adding noise:",adding_noise)
                Z_all = []
                label_all = [] 
                if epoch ==1:
                    random_ini_centers = True
                else: 
                    random_ini_centers = False
                train_loss_list=[]
                f_loss_list= []
                if "epoch" in self.scheme:
                    for batch in range(self.num_batches):

                        # shuffle_client_list = range(self.num_client)
                        for client_id in range(self.num_client):
                            batch_data_read_start_time = time.time()

                            # try:
                            images, labels = next(client_iterator_list[client_id])
                            if images.size(0) != self.batch_size:
                                client_iterator_list[client_id] = iter(self.client_dataloader[client_id])
                                images, labels = next(client_iterator_list[client_id])
                            # except StopIteration:
                            #     client_iterator_list[client_id] = iter(self.client_dataloader[idxs_users[client_id]])
                            #     images, labels = next(client_iterator_list[client_id])

                            batch_data_read_time = time.time() 

                            # Train the AE decoder if self.gan_regularizer is enabled:
                            if self.gan_regularizer and batch % interval == 0:
                                for i in range(self.gan_num_step): #equals 1 or 3 
                                    ssim_log = -self.gan_train_step(images, client_id, loss_type=self.gan_loss_type)  # orig_epoch_gan_train

                            self.optimizer_zero_grad()
                            
                            # Train step (client/server)
                            
                            train_loss, f_loss, z_private = self.train_target_step(images, labels, adding_noise,random_ini_centers,centroids_list,client_id)

                            train_loss_list.append(torch.tensor(train_loss))
                            f_loss_list.append(torch.tensor(f_loss))
                            self.optimizer_step()
                            
                            # Logging
                            # LOG[batch, client_id] = train_loss
                            model_train_time= time.time()
                            if verbose and (batch==0 or batch==self.num_batches-1):
                                train_loss_mean=torch.stack(train_loss_list).mean()
                                f_loss_mean=torch.stack(f_loss_list).mean()
                                self.logger.debug(
                                    "log--[{}/{}][{}/{}][client-{}] train loss: {:1.4f} cross-entropy loss: {:1.4f}".format(
                                        epoch, self.n_epochs, batch, self.num_batches, client_id, train_loss_mean, f_loss_mean))
                                if self.gan_regularizer:
                                    self.logger.debug(
                                        "log--[{}/{}][{}/{}][client-{}] Adversarial Loss of local AE: {:1.4f}".format(epoch,
                                                                                                                self.n_epochs,
                                                                                                                batch,
                                                                                                                self.num_batches,
                                                                                                                client_id,
                                                                                                                ssim_log))
                            if batch == 0:
                                self.writer.add_scalar('train_loss/client-{}/total'.format(client_id), train_loss,
                                                        epoch)
                                self.writer.add_scalar('train_loss/client-{}/cross_entropy'.format(client_id), f_loss,
                                                        epoch)
                            # if batch%20 ==0:
                                # print(f"data_readtime:{batch_data_read_time - batch_data_read_start_time} s",f"model_infer_time:{model_train_time - batch_data_read_time} s")   
                # print(f"train_one_ep_time:{model_train_time - ep_start_time} s")
                    for batch in range(self.num_batches):
                        with torch.no_grad():
                            for client_id in range(self.num_client):
                                # try:
                                images, labels = next(client_iterator_list[client_id])
                                if images.size(0) != self.batch_size:
                                    client_iterator_list[client_id] = iter(self.client_dataloader[client_id])
                                    images, labels = next(client_iterator_list[client_id])
                                self.f.eval()
                                x_private = images.cuda()
                                label_private = labels.cuda()
                                z_private = self.f(x_private)
                                Z_all.append(z_private.cpu())
                                label_all.append(labels.cpu()) 
                    
                Z_all = torch.cat(Z_all, dim=0).cuda()
                label_all = torch.cat(label_all, dim=0).cuda()
                # print(Z_all.shape,label_all.shape)
                num_clusters=10
                gmm_params = fit_gmm_torch(Z_all, label_all, self.num_class, num_clusters)
                for class_label in range(self.num_class):
                    
                    centroids=centroids_list[class_label].detach().clone()
                    class_features = Z_all[label_all == class_label].detach().clone()
                    # print(class_label,len(class_features),len(label_all[label_all == class_label]))
                    if class_features.size(0) > num_clusters:
                        # print(class_features.size(0))
                        centroids, average_variance=self.kmeans_cuda(class_features, num_clusters,centroids,random_ini_centers, num_iterations=60, tol=1e-4)  # 
                        centroids_list[class_label] = centroids.clone()
                    print(abs(centroids_list[class_label]).mean())
                    del class_features,centroids,average_variance

                if (epoch-1)%20 ==0:
                    Z_visual=Z_all[0:10000].detach().cpu()
                    label_visual=label_all[0:10000].detach().cpu()
                    Z_visual = Z_visual.view(len(Z_visual), -1).detach().cpu()

                    mask = (label_visual >= 0) & (label_visual <= 2)
                    Z_visual = Z_visual[mask]
                    label_visual = label_visual[mask]

                    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42)
                    reduced_features = tsne.fit_transform(Z_visual)
                    visual_dir = self.save_dir + '/visualize'
                    os.makedirs(visual_dir, exist_ok=True)
                    plt.figure(figsize=(10, 8))
                    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=label_visual.cpu(), cmap='viridis')
                    plt.colorbar(scatter)
                    plt.xlabel('t-SNE Component 1')
                    plt.ylabel('t-SNE Component 2')
                    plt.title('t-SNE of n*8*8*8 features')
                    file_name=visual_dir+'/'+str(epoch)+'.png'
                    # 保存图像到 visual 文件夹
                    plt.savefig(file_name)
                    print(file_name)
                del Z_all,label_all
                # torch.cuda.empty_cache()
                # label_all = [] 
                

                # kmeans_cuda(self,X, num_clusters,centroids, num_iterations=10, tol=1e-4):
                
                # V1/V2 synchronization
                if self.scheme == "V1_epoch" or self.scheme == "V2_epoch":
                    self.sync_client()
                    if self.gan_regularizer and self.decoder_sync:
                        self.sync_decoder()

                # Step the warmup scheduler
                if epoch <= self.warm:
                    self.scheduler_step(warmup=True)


                # Validate and get average accu among clients
                avg_accu = 0
                val_start_time= time.time()
                for client_id in range(self.num_client):
                    accu, loss = self.validate_target(client_id=client_id)
                    self.writer.add_scalar('valid_loss/client-{}/cross_entropy'.format(client_id), loss, epoch)
                    avg_accu += accu
                avg_accu = avg_accu / self.num_client
                val_time=time.time()
                # print(f"val_one_ep_time:{val_time-val_start_time} s")
                # Save the best model
                if avg_accu > best_avg_accu:
                    self.save_model(epoch, is_best=True)
                    best_avg_accu = avg_accu
                    best_rob_loss= train_loss_mean
                if epoch==160:
                    best_avg_accu=0
                
                # if epoch ==150 or epoch ==170:
                # self.lambd = lambd_end + 0.5 * (lambd_start - lambd_end) * (1 + np.cos(np.pi * epoch / self.n_epochs))
                print('lambd value is:', self.lambd) 
                # Save Model regularly
                if epoch % 50 == 0 or epoch == self.n_epochs or epoch in epoch_save_list:  # save model
                    self.save_model(epoch)

        if not self.call_resume:
            self.logger.debug("Best Average Validation Accuracy is {}".format(best_avg_accu))
        else:
            LOG = None
            avg_accu = 0
            for client_id in range(self.num_client):
                accu, loss = self.validate_target(client_id=client_id)
                avg_accu += accu
            avg_accu = avg_accu / self.num_client
            self.logger.debug("Best Average Validation Accuracy is {}".format(avg_accu))
        return LOG

    def save_model(self, epoch, is_best=False):
        if is_best:
            epoch = "best"

        if "V" in self.scheme:
            torch.save(self.f.state_dict(), self.save_dir + 'checkpoint_f_{}.tar'.format(epoch))
            if self.num_client > 1:
                torch.save(self.c.state_dict(), self.save_dir + 'checkpoint_c_{}.tar'.format(epoch))
            torch.save(self.f_tail.state_dict(), self.save_dir + 'checkpoint_cloud_{}.tar'.format(epoch))
            torch.save(self.classifier.state_dict(), self.save_dir + 'checkpoint_classifier_{}.tar'.format(epoch))
            if self.num_client > 2:
                for i in range(2, self.num_client):
                    torch.save(self.model.local_list[i].state_dict(),
                                self.save_dir + 'checkpoint_local{}_{}.tar'.format(i, epoch))
        else:
            torch.save(self.model.state_dict(), self.save_dir + 'checkpoint_{}.tar'.format(epoch))
            torch.save(self.f_tail.state_dict(), self.save_dir + 'checkpoint_cloud_{}.tar'.format(epoch))
            torch.save(self.classifier.state_dict(), self.save_dir + 'checkpoint_classifier_{}.tar'.format(epoch))