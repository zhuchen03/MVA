"""Train CIFAR10 with PyTorch."""
from __future__ import print_function
from comet_ml import Experiment

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from optim_adam import MAdam
from optim_laprop import LaMadam
import numpy as np
import random
import time


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--model', default='resnet', type=str, help='model')
    parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum term')
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam coefficients beta_1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam coefficients beta_2')
    parser.add_argument('--beta2-min', default=0.5, type=float, help="min beta2 for adaptive methods")
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--weight-decay', default=0.05, type=float,
                        help='weight decay for optimizers')
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--comet', default=False, action="store_true")
    parser.add_argument('--gpu', default="0", type=str)
    parser.add_argument('--amsgrad', default=False, action="store_true")
    parser.add_argument('--nesterov', default=False, action="store_true")
    parser.add_argument('--use-adamw', default=False, action="store_true")
    parser.add_argument('--use-adabound', default=False, action="store_true",
                        help="extra choice for incofporating adabound")
    parser.add_argument('--lr-scheduler', default="cosine", type=str)
    parser.add_argument('--eta-min', default=2e-6, type=float)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--adam-eps', default=1e-8, type=float)
    parser.add_argument('--lap-eps', default=1e-15, type=float)
    parser.add_argument('--tag', default="cifar", type=str)
    parser.add_argument('--expname', default="", type=str)
    parser.add_argument('--dataset', default="cifar10", type=str)
    return parser


def build_dataset(args):
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if args.dataset.lower() == "cifar10":
        dset = torchvision.datasets.CIFAR10
    else:
        dset = torchvision.datasets.CIFAR100

    trainset = dset(root='./data', train=True, download=True,
                                            transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True,
                                               num_workers=1)

    testset = dset(root='./data', train=False, download=True,
                                           transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=1)

    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return train_loader, test_loader


def build_model(args, device, ckpt=None):
    print('==> Building model..')
    n_cls = 10 if args.dataset.lower() == "cifar10" else 100
    net = {
        'resnet34': ResNet34,
        'resnet18': ResNet18,
    }[args.model](ncls=n_cls)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if ckpt:
        net.load_state_dict(ckpt['net'])

    return net


def create_optimizer(args, model_params):
    if args.optim == 'sgd':
        return optim.SGD(model_params, args.lr, momentum=args.momentum,
                         weight_decay=args.weight_decay, nesterov=args.nesterov)
    elif args.optim == 'sysadam':
        return optim.Adam(model_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay, amsgrad=args.amsgrad)
    elif args.optim.lower() == 'madam':
        return MAdam(model_params, lr=args.lr, weight_decay=args.weight_decay, beta1=args.beta1,
                           beta2_range=(args.beta2_min, args.beta2), eps=args.adam_eps,
                           adamw=args.use_adamw, amsgrad=args.amsgrad)
    elif args.optim == "lamadam":
        return LaMadam(model_params, lr=args.lr,
                        momentum=args.beta1, beta=args.beta2, beta_min=args.beta2_min,
                        weight_decay=args.weight_decay, use_adamw=args.use_adamw,
                        eps=args.lap_eps, amsgrad=args.amsgrad)


def train(net, epoch, device, data_loader, optimizer, criterion, scheduler, experiment=None):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
            scheduler.step()

        train_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 50 == 0 or batch_idx == len(data_loader) - 1:
            print("{}, Epoch {}, {}/{}, LR: {:.2e} idx {},\t train loss {:.2e}, train acc {:.2f}" \
                  .format(time.strftime('%Y-%m-%d-%H:%M:%S'),
                          epoch, batch_idx, len(data_loader), optimizer.param_groups[0]['lr'], batch_idx, train_loss / total,
                          100 * correct / float(total)
                          ))

            if experiment is not None:
                ret_names = ['update_min', 'update_max', 'update_mean']
                rets = getattr(optimizer, "update_size", [None, None, None])
                for name, val in zip(ret_names, rets):
                    if val is not None:
                        experiment.log_metric(name, val, epoch * len(data_loader) + batch_idx)


    if experiment is not None:
        experiment.log_metric("TrainLoss", train_loss / total, epoch)
        experiment.log_metric("TrainAcc", 100 * correct / float(total), epoch)
        experiment.log_metric('LearningRate', optimizer.param_groups[0]['lr'])

    accuracy = 100. * correct / total
    print('train acc %.3f' % accuracy)

    return accuracy


def test(net, device, data_loader, criterion, epoch, experiment=None):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()


    accuracy = 100. * correct / total
    print(' test acc %.3f' % accuracy)

    if experiment is not None:
        experiment.log_metric("TestLoss", test_loss / total, epoch)
        experiment.log_metric("TestAcc", 100 * correct / float(total), epoch)

    return accuracy


def main():

    parser = get_parser()
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cudnn.deterministic = True

    exp_name = args.expname
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.comet:
        experiment = Experiment(api_key="",
                            project_name="{}".format(args.dataset), workspace="default",
                            parse_args=True
                            )
        experiment.set_name(exp_name)
        experiment.add_tag(args.tag)
    else:
        experiment = None

    print(args)

    train_loader, test_loader = build_dataset(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ckpt_name = "ckpt.pth"
    ckpt = None
    best_acc = 0
    start_epoch = -1

    net = build_model(args, device, ckpt=ckpt)
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(args, net.parameters())
    if args.lr_scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1,
                                          last_epoch=start_epoch)
    elif args.lr_scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader),
                                                            eta_min=args.lr * 1e-4 if args.eta_min is None else args.eta_min)
    elif args.lr_scheduler == "multistep":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

    train_accuracies = []
    test_accuracies = []

    for epoch in range(start_epoch + 1, args.epochs):
        if isinstance(scheduler, optim.lr_scheduler.StepLR) or isinstance(scheduler, optim.lr_scheduler.MultiStepLR):
            scheduler.step()
        train_acc = train(net, epoch, device, train_loader, optimizer, criterion, scheduler, experiment)
        test_acc = test(net, device, test_loader, criterion, epoch, experiment)

        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        # Save checkpoint.
        if test_acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': test_acc,
                'epoch': epoch,
            }

            torch.save(state, os.path.join('checkpoint', ckpt_name))
            best_acc = test_acc

        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        if not os.path.isdir('curve'):
            os.mkdir('curve')
        torch.save({'train_acc': train_accuracies, 'test_acc': test_accuracies},
                   os.path.join('curve', ckpt_name))
        if experiment is not None:
            experiment.log_metric("BestTestAcc", best_acc, epoch)


if __name__ == '__main__':
    main()
