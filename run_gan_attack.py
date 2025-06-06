import os
import re
import subprocess

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# define the fold of your saved models
dataset='cifar100' #tinyimagenet;facescrub;cifar10;
subfolder='Aresult_model'
AT_regularizations=['gan_adv_step1_pruning191'] #gan_adv_step1_pruning190
log_entropy=1
if dataset=='tinyimagenet':
    arch='resnet20'
    var_threshold=0.0625
else:
    arch='vgg11_bn_sgm'
    var_threshold=0.125


for AT_regularization in AT_regularizations:
    target_directory = "./saves/"+dataset+'/'+subfolder+'/'+f"{AT_regularization}_infocons_sgm_lg{log_entropy}_thre{var_threshold}"
    pattern = r"pretrain_(?P<pretrain>\w+)_lambd_(?P<lambd>[\d\.]+)_noise_(?P<regularization_strength>[\d\.]+)_epoch_(?P<num_epochs>\d+)_bottleneck_(?P<bottleneck_option>\w+)_log_(?P<log_entropy>\w+)_ATstrength_(?P<AT_regularization_strength>[\d\.]+)_lr_(?P<learning_rate>[\d\.]+)_varthres_(?P<var_threshold>[\d\.]+)"
    if dataset=='facescrub' or dataset=='tinyimagenet':
        ssim_threshold=0.6
        batch_size=256
        target_directory="./new_saves/"+dataset+'/'+subfolder+'/'+f"{AT_regularization}_infocons_sgm_lg{log_entropy}_thre{var_threshold}_{batch_size}_ganthre{ssim_threshold}"
        pattern = r"pretrain_(?P<pretrain>\w+)_lambd_(?P<lambd>[\d\.]+)_noise_(?P<regularization_strength>[\d\.]+)_epoch_(?P<num_epochs>\d+)_bottleneck_(?P<bottleneck_option>\w+)_log_(?P<log_entropy>\w+)_ATstrength_(?P<AT_regularization_strength>[\d\.]+)_lr_(?P<learning_rate>[\d\.]+)"
    if dataset=='cifar100':
        # target_directory="./new_saves/"+dataset+'/'+subfolder+'/'+f"{AT_regularization}_infocons_sgm_lg{log_entropy}_thre{var_threshold}_{batch_size}_ganthre{ssim_threshold}"
        pattern = r"pretrain_(?P<pretrain>\w+)_lambd_(?P<lambd>[\d\.]+)_noise_(?P<regularization_strength>[\d\.]+)_epoch_(?P<num_epochs>\d+)_bottleneck_(?P<bottleneck_option>\w+)_log_(?P<log_entropy>\w+)_ATstrength_(?P<AT_regularization_strength>[\d\.]+)_lr_(?P<learning_rate>[\d\.]+)"
    print(target_directory)
    
    
# evaluate the inversion robustness of the saved models in the target directory
    for folder_name in os.listdir(target_directory):
        print(folder_name)
        folder_path = os.path.join(target_directory, folder_name)

        if os.path.isdir(folder_path):

            match = re.match(pattern, folder_name)
            print(match)
            if match:

                    pretrain = match.group("pretrain")
                    lambd = match.group("lambd")
                    regularization_strength = match.group("regularization_strength")
                    num_epochs = match.group("num_epochs")
                    bottleneck_option = match.group("bottleneck_option")
                    log_entropy = match.group("log_entropy")
                    AT_regularization_strength = match.group("AT_regularization_strength")
                    learning_rate = match.group("learning_rate")
                    # var_threshold = match.group("var_threshold")
                    args = [                
                                    "python", "train_gan.py", 
                                    "--dataset", dataset, 
                                    "--subfolder",subfolder,
                                    "--arch", arch, 
                                    '--AT_regularization', AT_regularization,
                                    "--bottleneck_option", bottleneck_option, 
                                    "--AT_regularization_strength", AT_regularization_strength,
                                    "--lambd",lambd, 
                                    "--regularization_strength", regularization_strength,
                                    "--log_entropy",log_entropy, 
                                    "--var_threshold",str(var_threshold), 
                                    "--pretrain", pretrain, 
                                    "--num_epochs", num_epochs, 
                                    "--learning_rate",learning_rate, 
                                    ]
                    print(args)
                    subprocess.run(args)