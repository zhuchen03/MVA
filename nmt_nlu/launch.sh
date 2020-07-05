#!/usr/bin/env bash

# CIFAR10
# SGD
python train_cifar.py --model resnet18 --gpu 0 --optim sgd --lr 2e-1 --weight-decay 3e-4 --lr-scheduler cosine --eta-min 2e-6 --seed 1234

# Adam, which is equivalent to beta_min=0.999 for MAdam
python train_cifar.py --model resnet18 --gpu 0 --optim madam --lr 3e-3 --weight-decay 0.2 --use-adamw --lr-scheduler cosine --eta-min 2e-6  --beta2-min 0.999 --beta2 0.999 --beta1 0.9  --amsgrad  --seed 1234

# LaProp
python train_cifar.py --model resnet18 --gpu 0 --optim lamadam --lr 1e-3 --weight-decay 0.4 --use-adamw --lr-scheduler cosine --eta-min 2e-6  --beta2-min 0.999 --beta2 0.999 --beta1 0.9  --amsgrad  --seed 1234


# MAdam
python train_cifar.py --model resnet18 --gpu 0 --optim madam --lr 8e-3 --weight-decay 0.05 --use-adamw --lr-scheduler cosine --eta-min 2e-6  --beta2-min 0.5 --beta2 0.999 --beta1 0.9  --amsgrad  --seed 1234

# LaMAdam
python train_cifar.py --model resnet18 --gpu 0 --optim lamadam --lr 6e-3 --weight-decay 0.05 --use-adamw --lr-scheduler cosine --eta-min 2e-6  --beta2-min 0.5 --beta2 0.999 --beta1 0.9  --amsgrad  --seed 1234


# CIFAR100
# SGD
python train_cifar.py --dataset cifar100 --model resnet18 --gpu 0 --optim sgd --lr 3e-2 --weight-decay 2e-3 --lr-scheduler cosine --eta-min 2e-6 --seed 1234

# Adam, which is equivalent to beta_min=0.999 for MAdam
python train_cifar.py --dataset cifar100 --model resnet18 --gpu 0 --optim madam --lr 2e-3 --weight-decay 0.4 --use-adamw --lr-scheduler cosine --eta-min 2e-6  --beta2-min 0.999 --beta2 0.999 --beta1 0.9  --amsgrad  --seed 1234

# LaProp
python train_cifar.py --dataset cifar100 --model resnet18 --gpu 0 --optim lamadam --lr 5e-4 --weight-decay 1 --use-adamw --lr-scheduler cosine --eta-min 2e-6  --beta2-min 0.999 --beta2 0.999 --beta1 0.9  --amsgrad  --seed 1234


# MAdam
python train_cifar.py --dataset cifar100 --model resnet18 --gpu 0 --optim madam --lr 4e-3 --weight-decay 0.2 --use-adamw --lr-scheduler cosine --eta-min 2e-6  --beta2-min 0.5 --beta2 0.999 --beta1 0.9  --amsgrad  --seed 1234

# LaMAdam
python train_cifar.py --dataset cifar100 --model resnet18 --gpu 0 --optim lamadam --lr 3e-3 --weight-decay 0.2 --use-adamw --lr-scheduler cosine --eta-min 2e-6  --beta2-min 0.5 --beta2 0.999 --beta1 0.9  --amsgrad  --seed 1234
