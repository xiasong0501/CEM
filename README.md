## Official code for "Theoretical Insights in Model Inversion Robustness and Conditional Entropy Maximization for Collaborative Inference Systems" in CVPR 2025.

#### Requirement:

` pip install -r requirements.txt`

#### Dataset:

For convenience, we put the dataset utilized in our experiment in this [link](https://drive.google.com/drive/folders/18TB_UHVkmHP65IaOMH3gb7VUMI7mMhHs?usp=sharing). Please put the four folders in the root folder.

```
1. CIFAR10
2. CIFAR100
3. Facescrub
4. TinyImagenet
```

#### Trained Model

Download our trained models: please download our trained models via this [link](https://drive.google.com/drive/folders/1ZWizVdgBW117Yf1VWPD6V4LyTJ0yFG1M?usp=sharing)

#### Conduct the experiments

1. Train the classifier and run decoing-based MIA attack (for inference only, please comment the training part)
   `bash run_exp.sh`
2. Run the GAN-based MIA attack
   `python run_gan_attack.py`

#### Important hyperparameters in **run_exp.sh** file

1. arch:  `vgg11_bn_sgm` (this for CIFAR10, 100, and facescrub, bn means batchnorm and sgm means Sigmoid activation) and `resnet20` (for Tinyimagenet)
2. cutlayer_list: which layer to split the encoder and decoder;
3. AT_regularization: The defense method for split learning; `gan_adv` for adversarial training, `dropout` for dropout defense, `topkprune` for topkpruning, `pruning` for PATROL. For example, `gan_adv_step1_pruning180` means using adversarial training with PATROL pruning at 180 epochs.
4. AT_regularization_strength: The weight for the defense method
5. dataset_list: dataset for experiment: `cifar10, cifar100, facescrub and Tinyimagenet`
6. regularization: `Gaussian` means adding the noise corruption mentioned in CEM
7. regularization_strength_list: the variance of the noise corruption, can be a list like: `"0.01 0.025 0.05 0.1 0.15"`
8. lambd_list: weight for CEM regularization strength (default 8 or 16); 0 without using CEM
9. log_entropy: for log optimization in the  loss function.
10. bottleneck_option_list means adding the bottleneck layer after the encoder

## Cite the work:

```
@inproceedings{xia2025theoretical,
  title={Theoretical Insights in Model Inversion Robustness and Conditional Entropy Maximization for Collaborative Inference Systems},
  author={Xia, Song and Yu, Yi and Yang, Wenhan and Ding, Meiwen and Chen, Zhuo and Duan, Ling-Yu and Kot, Alex C and Jiang, Xudong},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={8753--8763},
  year={2025}
}
```
