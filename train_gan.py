import os
import time
import torch.nn.functional as F
from utils import setup_logger, accuracy, get_PSNR
import GMI.utils as utils
import torch
from GMI.utils import save_tensor_images, init_dataloader, load_json
from torch.autograd import grad
from GMI.discri import DGWGAN, DGWGAN32
from GMI.generator import Generator, GeneratorMNIST,GeneratorCIFAR
from argparse import ArgumentParser
from datasets_torch import get_cifar100_trainloader, get_cifar100_testloader, get_cifar10_trainloader, \
    get_cifar10_testloader, get_mnist_bothloader, get_facescrub_bothloader, get_SVHN_trainloader, get_SVHN_testloader, get_fmnist_bothloader, get_tinyimagenet_bothloader,get_imagenet_bothloader
from model_architectures.vgg import vgg11, vgg13, vgg11_bn, vgg13_bn,vgg11_bn_sgm
import architectures_torch as architectures
import pytorch_ssim
from model_architectures.resnet_cifar import ResNet20, ResNet32
from model_architectures.resnet_imagenet import Imagenet_ResNet20
def freeze(net):
    for p in net.parameters():
        p.requires_grad_(False)


def unfreeze(net):
    for p in net.parameters():
        p.requires_grad_(True)


def gradient_penalty(x, y):
    # interpolation
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    alpha = torch.rand(shape).cuda()
    z = x + alpha * (y - x)
    z = z.cuda()
    z.requires_grad = True

    o = DG(z)
    g = grad(o, z, grad_outputs=torch.ones(o.size()).cuda(), create_graph=True)[0].view(z.size(0), -1)
    gp = ((g.norm(p=2, dim=1) - 1) ** 2).mean()

    return gp

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
    elif dataset == "tinyimagenet":
        mean = (0.5141, 0.5775, 0.3985)
        std = (0.2927, 0.2570, 0.1434)
    # 3, H, W, B
    tensor = x.clone().permute(1, 2, 3, 0)
    for t, m, s in zip(range(tensor.size(0)), mean, std):
        tensor[t] = (tensor[t]).mul_(s).add_(m)
    # B, 3, H, W
    return torch.clamp(tensor, 0, 1).permute(3, 0, 1, 2)


def normalize(x, dataset):
    if dataset == "mnist" or dataset == "fmnist":
        return 2 * x - 1
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
    elif dataset == "tinyimagenet":
        mean = (0.5141, 0.5775, 0.3985)
        std = (0.2927, 0.2570, 0.1434)

    # 3, H, W, B
    tensor = x.clone().permute(1, 2, 3, 0)
    for t, m, s in zip(range(tensor.size(0)), mean, std):
        tensor[t] = (tensor[t] - m).div_(s)
    # B, 3, H, W
    return tensor.permute(3, 0, 1, 2)

if __name__ == "__main__":
    parser = ArgumentParser(description='Step1: train GAN')
    parser.add_argument('--dataset', default='cifar10', help='cifar10 | facescrub') 
    parser.add_argument('--subfolder', default='Aresult_model', help='cifar10 | facescrub') 
    parser.add_argument('--arch', default='vgg11_bn_sgm')
    parser.add_argument('--adds_bottleneck', default=True)
    parser.add_argument('--double_local_layer', default=False)
    parser.add_argument('--bottleneck_option', default='noRELU_C8S1')
    parser.add_argument('--AT_regularization', default='gan_adv_step1')
    # parser.add_argument('--lambd', default='32')
    parser.add_argument('--log_entropy', default=1)
    parser.add_argument('--var_threshold', default=0.125)
    parser.add_argument('--pretrain', default="False")
    parser.add_argument('--lambd', default=0) # variable
    parser.add_argument('--regularization_strength',default=0.01,type=float) #variable
    parser.add_argument('--num_epochs', default=240)
    parser.add_argument('--AT_regularization_strength', default=0.3)
    parser.add_argument('--learning_rate', default=0.05)
    parser.add_argument('--mse_weight', default=1500)

    args = parser.parse_args()
    path = f"new_saves/"+args.dataset+"/"+args.subfolder+'/'+f"{args.AT_regularization}_infocons_sgm_lg{args.log_entropy}_thre{args.var_threshold}"
    filename = (f"pretrain_{args.pretrain}_lambd_{args.lambd}_noise_{args.regularization_strength}_epoch_{args.num_epochs}_"
            f"bottleneck_{args.bottleneck_option}_log_{args.log_entropy}_ATstrength_{args.AT_regularization_strength}_"
            f"lr_{args.learning_rate}_varthres_{args.var_threshold}")
    if args.dataset=='facescrub'or args.dataset=='tinyimagenet':
        ssim_threshold=0.6
        batch_size=256
        path = f"new_saves/"+args.dataset+"/"+args.subfolder+'/'+f"{args.AT_regularization}_infocons_sgm_lg{args.log_entropy}_thre{args.var_threshold}_{batch_size}_ganthre{ssim_threshold}"
        filename = (f"pretrain_{args.pretrain}_lambd_{args.lambd}_noise_{args.regularization_strength}_epoch_{args.num_epochs}_"
                f"bottleneck_{args.bottleneck_option}_log_{args.log_entropy}_ATstrength_{args.AT_regularization_strength}_"
                f"lr_{args.learning_rate}")        
    if args.dataset=='cifar100':
        filename = (f"pretrain_{args.pretrain}_lambd_{args.lambd}_noise_{args.regularization_strength}_epoch_{args.num_epochs}_"
                f"bottleneck_{args.bottleneck_option}_log_{args.log_entropy}_ATstrength_{args.AT_regularization_strength}_"
                f"lr_{args.learning_rate}")     
    ckpt_path=path+'/'+filename+'/'
    # print(path+'/'+filename) 
    folder= args.dataset+"/"+ args.subfolder+'/'+f"{args.AT_regularization}_infocons_sgm_lg{args.log_entropy}_thre{args.var_threshold}"
    ############################# mkdirs ##############################
    save_model_dir = f"GMI/result/models_{args.dataset}_gan"+'/'+folder+'/'+str(args.mse_weight)+'_'+filename
    log_file_path = os.path.join(save_model_dir, 'log.txt')
    os.makedirs(save_model_dir, exist_ok=True)
    with open(log_file_path, 'w') as f:
        f.write("Log initialized. Writing training and validation losses...\n")
    
    save_img_dir = f"GMI/result/imgs_{args.dataset}_gan"+'/'+folder+'/'+str(args.mse_weight)+'_'+filename
    os.makedirs(save_img_dir, exist_ok=True)
    ############################# mkdirs ##############################

    file = "./GMI/config/" + args.dataset + ".json"
    loaded_args = load_json(json_file=file)
    file_path = loaded_args['dataset']['img_path']
    model_name = loaded_args['dataset']['model_name']
    lr = loaded_args[model_name]['lr']
    batch_size = int(loaded_args[model_name]['batch_size'])
    z_dim = loaded_args[model_name]['z_dim'] # this term should be decided by the encoder

    epochs = loaded_args[model_name]['epochs']
    n_critic = loaded_args[model_name]['n_critic']

    print("---------------------Training [%s]------------------------------" % model_name)
    utils.print_params(loaded_args["dataset"], loaded_args[model_name])
    upsize=False
    if "pruning" in args.AT_regularization:
        args.double_local_layer = True
    else:
        args.double_local_layer = False
    if "SCA" in  args.AT_regularization:
        SCA = True
    else:
        SCA = False   
    # dataset, dataloader = init_dataloader(loaded_args, file_path, batch_size, mode="gan")
    if args.dataset == "cifar10":
        orig_class = 10
        feature_size=8
        _, train_single_loader, train_single_loader_irg = get_cifar10_trainloader(batch_size=batch_size, num_workers=4, shuffle=True)
        val_loader, val_train_loader, val_test_loader = get_cifar10_testloader(batch_size=batch_size, num_workers=4, shuffle=True)
    elif args.dataset == "cifar100":
        orig_class = 100
        feature_size=8
        _, train_single_loader, train_single_loader_irg = get_cifar100_trainloader(batch_size=batch_size, num_workers=4, shuffle=True)
        val_loader, val_train_loader, val_test_loader = get_cifar100_testloader(batch_size=batch_size, num_workers=4, shuffle=True)
    elif args.dataset == "facescrub":
        orig_class = 530
        input_nc=8
        input_dim=48
        feature_size=12
        _, val_loader,train_single_loader,val_train_loader,val_test_loader = get_facescrub_bothloader(batch_size=batch_size, num_workers=4, shuffle=False)
    elif args.dataset == "tinyimagenet":
        orig_class = 200
        feature_size=16
        _, val_loader,train_single_loader,val_train_loader,val_test_loader  = get_tinyimagenet_bothloader(batch_size=batch_size, num_workers=4, shuffle=False)
    elif args.dataset == "imagenet":
        orig_class = 1000
        _, val_loader,train_single_loader,val_train_loader,val_test_loader = get_imagenet_bothloader(batch_size=batch_size, num_workers=4, shuffle=False)
        input_nc=8
        input_dim=64  

    if args.arch == "vgg11_bn_sgm":
        model = vgg11_bn_sgm(3, None, num_client=1, num_class=orig_class,
                             initialize_different=False, adds_bottleneck=args.adds_bottleneck, bottleneck_option = args.bottleneck_option, double_local_layer=args.double_local_layer,upsize=upsize,SCA=SCA,feature_size=feature_size)   
    elif args.arch == "resnet20":
        model = ResNet20(5, None, num_client=1, num_class=orig_class,
                            initialize_different=False, adds_bottleneck=args.adds_bottleneck, bottleneck_option = args.bottleneck_option, double_local_layer=args.double_local_layer,upsize=upsize,SCA=SCA)

       
    try:
        checkpoint_i = torch.load(ckpt_path + "checkpoint_f_best.tar")
    except:
        print("No valid Checkpoint Found in:",ckpt_path)
        
        
    model.cuda()
    if "pruning" in args.AT_regularization:
        model.local_list[0]=checkpoint_i
        f = model.local_list[0].cuda()
        print(f)
    else:
         f=model.local_list[0].cuda()
         f.load_state_dict(checkpoint_i, strict = False)
    # f=model.local_list[0].cuda()
    # f.load_state_dict(checkpoint_i, strict = False)
    print('load the weight from defensed model')
    f_tail = model.cloud
    classifier = model.classifier
    print("load cloud")
    checkpoint = torch.load(ckpt_path + "checkpoint_cloud_best.tar")
    f_tail.cuda()
    f_tail.load_state_dict(checkpoint, strict = False)

    print("load classifier")
    checkpoint = torch.load(ckpt_path + "checkpoint_classifier_best.tar")

    classifier.cuda()
    classifier.load_state_dict(checkpoint, strict = False)
 



    for images, labels in val_loader:
        sample_image = images[0]
        sample_image=sample_image.cuda()
        images=images.cuda()
        # print("Sample image shape:", sample_image.shape)
        recons_dim=sample_image.shape[-1]
        if sample_image.shape[-1]>63:
            upsize=True
        else:
            upsize=False
        # print(sample_image.shape[-1])
        with torch.no_grad():
            model.eval()
            f.eval()
            encoder = f.cuda()
            z=encoder(images.cuda())#.view(images.size(0),-1)   
            sigma = args.regularization_strength
            noise = sigma * torch.randn_like(z).cuda()
            z += noise
            output = f_tail(z)
            if args.arch == "resnet20" or args.arch == "resnet32":
            # output = F.avg_pool2d(output, 8)
                output = F.adaptive_avg_pool2d(output,(1,1))
                output = output.view(output.size(0), -1)
                output = classifier(output)
            else:
                output = output.view(output.size(0), -1)
                output = classifier(output)
        prec1 = accuracy(output.data, labels.cuda())[
                0]
        print('the classification acc is',prec1)
        feature_size=z.shape
        dim = feature_size
        print('the dimension of the feature is:', dim)
        break

    
    if args.dataset == 'celeba':
        G = Generator(z_dim).cuda()
        DG = DGWGAN(3).cuda()
    else:
        # G = GeneratorCIFAR(in_dim=dim,dim=64).cuda()
        G = architectures.res_normN_AE(N = 8, internal_nc = 64, input_nc=feature_size[1], output_nc=3,
                                                         input_dim=feature_size[2], output_dim=sample_image.shape[-1],
                                                         activation= "sigmoid").cuda()
        # print((feature_size[2]*(sample_image.shape[-1]/32)))
        # print(G.in_dim)
        # print(sample_image.shape[-1])
        DG = DGWGAN32().cuda()
        G_path= ckpt_path+'/MIA_attack_0to0/model.pt'
        checkpoint_G = torch.load(G_path)
        G.load_state_dict(checkpoint_G, strict = False)
        print('load weight from the trained inversion model')
    # G = torch.nn.DataParallel(G).cuda()
    # DG = torch.nn.DataParallel(DG).cuda()

    # 0.004
    dg_optimizer = torch.optim.Adam(DG.parameters(), lr=lr, betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(G.parameters(), lr=lr*0.1, betas=(0.5, 0.999))
    dg_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(dg_optimizer, T_max=300)
    g_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(g_optimizer, T_max=300)
    step = 0
    
    for epoch in range(epochs):
        start = time.time()
        mse_loss_list = []
        ssim_loss_list = []
        psnr_loss_list = []
        mean_mse_loss=0
        mean_ssim_loss=0
        mean_psnr_loss=0
        val_mse_loss_list=[]
        val_ssim_loss_list = []
        val_psnr_loss_list = []
        mean_val_mse_loss=0
        mean_val_ssim_loss=0
        mean_val_psnr_loss=0
        for i, (imgs, target) in enumerate(train_single_loader):
            step += 1
            imgs = imgs.cuda()
            bs = imgs.size(0)

            freeze(G)
            unfreeze(DG)
            ssim_loss = pytorch_ssim.SSIM()

            # z = torch.randn(bs, z_dim).cuda()
            with torch.no_grad():
                z=encoder(imgs)#.view(imgs.size(0),-1)
                sigma = args.regularization_strength
                noise = sigma * torch.randn_like(z).cuda()
                # noise = 10000*sigma * torch.randn_like(save_activation).cuda()
                z += noise
            G.eval()
            f_imgs = G(z)
            # print(f_imgs.shape)

            de_imgs=denormalize(imgs,args.dataset)
            criterion = torch.nn.MSELoss().cuda()
            # f_nor_imgs= normalize(f_imgs,args.dataset)
            mse = criterion(f_imgs,de_imgs)
            ssim_l = ssim_loss(f_imgs,de_imgs)
            psnr_l = get_PSNR(f_imgs,de_imgs)
            if i%20==0:
                print('the mse loss of geneated img is:',criterion(f_imgs,de_imgs),ssim_l,psnr_l)
            r_logit = DG(de_imgs)
            f_logit = DG(f_imgs)

            wd = r_logit.mean() - f_logit.mean()  # Wasserstein-1 Distance
            gp = gradient_penalty(de_imgs.data, f_imgs.data)
            dg_loss = - wd + gp * 10.0

            dg_optimizer.zero_grad()
            dg_loss.backward()
            dg_optimizer.step()
            
            # g_loss=0
            mse_loss_list.append(mse.mean())
            ssim_loss_list.append(ssim_l.mean())
            psnr_loss_list.append(psnr_l.mean())
            # train G
            if step % n_critic == 0:
                freeze(DG)
                unfreeze(G)
                G.train()
                with torch.no_grad():
                    model.eval()
                    encoder = model.local_list[0].cuda()
                    z=encoder(imgs)#.view(imgs.size(0),-1)
                    sigma = args.regularization_strength
                    noise = sigma * torch.randn_like(z).cuda()
                    # noise = 10000*sigma * torch.randn_like(save_activation).cuda()
                    z += noise
                f_imgs = G(z)
                f_nor_imgs= normalize(f_imgs,args.dataset)
                logit_mse = criterion(f_imgs,de_imgs)
                logit_dg = DG(f_imgs)
                # print(logit_dg.mean(),logit_mse.mean())
                
                # calculate g_loss
                g_loss = - 1*logit_dg.mean()+args.mse_weight*logit_mse.mean()

                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()
        dg_scheduler.step()        
        g_scheduler.step()
        end = time.time()
        mean_mse_loss = torch.mean(torch.stack(mse_loss_list), dim=0)
        mean_ssim_loss = torch.mean(torch.stack(ssim_loss_list), dim=0)
        mean_psnr_loss = torch.mean(torch.stack(psnr_loss_list), dim=0)
        interval = end - start
        if (epoch) % 10 == 0:
            with torch.no_grad():
                model.eval()
                encoder = model.local_list[0].cuda()
                z=encoder(imgs)#.view(imgs.size(0),-1)
                sigma = args.regularization_strength
                noise = sigma * torch.randn_like(z).cuda()
                # noise = 10000*sigma * torch.randn_like(save_activation).cuda()
                z += noise
                fake_image = G(z)
            # fake_image = denormalize(fake_image, args.dataset)
            save_tensor_images(fake_image.detach(), os.path.join(save_img_dir, "result_image_{}.png".format(epoch + 1)),
                               nrow=8)
            de_imgs=denormalize(imgs,args.dataset)
            save_tensor_images(de_imgs.detach(), os.path.join(save_img_dir, "ori_image_{}.png".format(epoch + 1)),
                               nrow=8)
            for i, (imgs, target) in enumerate(val_loader):
                with torch.no_grad():
                    imgs=imgs.cuda()
                    target=target.cuda()
                    model.eval()
                    encoder = model.local_list[0].cuda()
                    z=encoder(imgs)#.view(imgs.size(0),-1)
                    sigma = args.regularization_strength
                    noise = sigma * torch.randn_like(z).cuda()
                    # noise = 10000*sigma * torch.randn_like(save_activation).cuda()
                    z += noise
                    de_imgs=denormalize(imgs,args.dataset)
                    fake_image = G(z)
                criterion = torch.nn.MSELoss().cuda()
                mse = criterion(fake_image,de_imgs)
                ssim_l = ssim_loss(fake_image,de_imgs)
                psnr_l = get_PSNR(fake_image,de_imgs)
                val_mse_loss_list.append(mse.mean())
                ssim_loss_list.append(ssim_l.mean())
                psnr_loss_list.append(psnr_l.mean())
            mean_val_mse_loss = torch.mean(torch.stack(val_mse_loss_list), dim=0)
            mean_val_ssim_loss = torch.mean(torch.stack(ssim_loss_list), dim=0)
            mean_val_psnr_loss = torch.mean(torch.stack(psnr_loss_list), dim=0)
            print("Epoch:%d \tTime:%.2f\tval_mse_loss:%.4f\tval_ssim_loss:%.4f\tval_psnr_loss:%.4f" % (epoch, interval, mean_val_mse_loss, mean_val_ssim_loss,mean_val_psnr_loss))
            with open(log_file_path, 'a') as f:
                f.write("Epoch:%d \tTime:%.2f\tval_mse_loss:%.4f\tval_ssim_loss:%.4f\tval_psnr_loss:%.4f\n" % 
                    (epoch, interval, mean_val_mse_loss, mean_val_ssim_loss, mean_val_psnr_loss))
        print("Epoch:%d \tTime:%.2f\tD_loss:%.2f\tG_loss:%.2f\tmse_loss_train:%.4f" % (epoch, interval, dg_loss, g_loss,mean_mse_loss),mean_ssim_loss,mean_psnr_loss)
        with open(log_file_path, 'a') as f:
            f.write("Epoch:%d \tTime:%.2f\tD_loss:%.2f\tG_loss:%.2f\tmse_loss_train:%.4f\tssim_loss_train:%.4f\tpsnr_loss_train:%.4f\n" % 
                (epoch, interval, dg_loss, g_loss, mean_mse_loss, mean_ssim_loss, mean_psnr_loss))
        
        if epoch + 1 >= 100:
            print('saving weights file')
            torch.save({'state_dict': G.state_dict()},
                       os.path.join(save_model_dir, f"{args.dataset}_G.tar"))
            torch.save({'state_dict': DG.state_dict()},
                       os.path.join(save_model_dir, f"{args.dataset}_D.tar"))
