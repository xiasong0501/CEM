#!/bin/bash
# cd "$(dirname "$0")"
# cd ../../
GPU_id=0
arch=vgg11_bn_sgm ##bn means batchnorm and sgm means sigmoid. For Imagenet, please change to resnet20
batch_size=128
random_seed=125
cutlayer_list="3"
num_client=1

AT_regularization=gan_adv_step1_pruning190 #"gan_adv_step1;dropout0.2;gan_adv_step1_pruning180"
AT_regularization_strength=1
ssim_threshold=0.5
train_gan_AE_type=res_normN4C64
gan_loss_type=SSIM

dataset_list="cifar100" # "svhn facescrub mnist"
scheme=V2_epoch
random_seed_list="125"
#Extra argement (store_true): --collude_use_public, --initialize_different  --collude_not_regularize  --collude_not_regularize --num_client_regularize ${num_client_regularize}

regularization='Gaussian_kl' #'Gaussian_Nonekl'
var_threshold=0.125
learning_rate=0.05
local_lr=-1
num_epochs=240
regularization_strength_list="0.01 0.025"
lambd_list="0 16" #
log_entropy=1
folder_name="new_saves/cifar100/${AT_regularization}_infocons_sgm_lg${log_entropy}_thre${var_threshold}_${batch_size}_ganthre${ssim_threshold}_new"
bottleneck_option_list="noRELU_C8S1" #"noRELU_C8S1"
pretrain="False"
for dataset in $dataset_list; do
        for lambd in $lambd_list; do
                for regularization_strength in $regularization_strength_list; do
                        for cutlayer in $cutlayer_list; do
                                for bottleneck_option in $bottleneck_option_list; do

                                        filename=pretrain_${pretrain}_lambd_${lambd}_noise_${regularization_strength}_epoch_${num_epochs}_bottleneck_${bottleneck_option}_log_${log_entropy}_ATstrength_${AT_regularization_strength}_lr_${learning_rate}
                                       
                                        
                                        #model_training

                                        if [ "$pretrain" = "True" ]; then
                                                num_epochs=80
                                                learning_rate=0.0001
                                                CUDA_VISIBLE_DEVICES=${GPU_id} python main_MIA_xs.py --arch=${arch}  --cutlayer=$cutlayer --batch_size=${batch_size} \
                                                --filename=$filename --num_client=$num_client --num_epochs=$num_epochs \
                                                --dataset=$dataset --scheme=$scheme --regularization=${regularization} --regularization_strength=${regularization_strength} --log_entropy=${log_entropy} --AT_regularization=${AT_regularization} --AT_regularization_strength=${AT_regularization_strength}\
                                                --random_seed=$random_seed --learning_rate=$learning_rate --lambd=${lambd}  --gan_AE_type ${train_gan_AE_type} --gan_loss_type ${gan_loss_type}\
                                                --local_lr $local_lr --bottleneck_option ${bottleneck_option} --folder ${folder_name} --ssim_threshold ${ssim_threshold} --var_threshold ${var_threshold} --load_from_checkpoint --load_from_checkpoint_server
                                        else
                                                num_epochs=240
                                                learning_rate=0.05
                                                CUDA_VISIBLE_DEVICES=${GPU_id} python main_MIA_xs.py --arch=${arch}  --cutlayer=$cutlayer --batch_size=${batch_size} \
                                                --filename=$filename --num_client=$num_client --num_epochs=$num_epochs \
                                                --dataset=$dataset --scheme=$scheme --regularization=${regularization} --regularization_strength=${regularization_strength} --log_entropy=${log_entropy} --AT_regularization=${AT_regularization} --AT_regularization_strength=${AT_regularization_strength}\
                                                --random_seed=$random_seed --learning_rate=$learning_rate --lambd=$lambd  --gan_AE_type ${train_gan_AE_type} --gan_loss_type ${gan_loss_type}\
                                                --local_lr $local_lr --bottleneck_option ${bottleneck_option} --folder ${folder_name} --ssim_threshold ${ssim_threshold} --var_threshold ${var_threshold}
                                        fi

                                        # model inversion attack 
                                        target_client=0
                                        attack_scheme=MIA
                                        attack_epochs=50
                                        average_time=1
                                        internal_C=64
                                        N=8
                                        test_gan_AE_type=res_normN${N}C${internal_C}
  
                                        CUDA_VISIBLE_DEVICES=${GPU_id} python main_test_MIA.py --arch=${arch}  --cutlayer=$cutlayer --batch_size=${batch_size} \
                                                --filename=$filename --num_client=$num_client --num_epochs=$num_epochs \
                                                --dataset=$dataset --scheme=$scheme --regularization=${regularization} --regularization_strength=${regularization_strength} --log_entropy=${log_entropy} --AT_regularization=${AT_regularization} --AT_regularization_strength=${AT_regularization_strength}\
                                                --random_seed=$random_seed --gan_AE_type ${train_gan_AE_type} --gan_loss_type ${gan_loss_type}\
                                                --attack_epochs=$attack_epochs --bottleneck_option ${bottleneck_option} --folder ${folder_name} --var_threshold ${var_threshold}\
                                                --average_time=$average_time --gan_AE_type ${test_gan_AE_type} --test_best
                                                                                
                                done
                        done
                done
        done
done
