'''Train Adversarially Robust Models with Feature Scattering'''
from __future__ import print_function
import time
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

from torch.autograd.gradcheck import zero_gradients
import copy
from torch.autograd import Variable
from PIL import Image

import os
import argparse
import datetime

from tqdm import tqdm
from models import *

import utils
from utils import softCrossEntropy
from utils import one_hot_tensor
#from attack_methods import Attack_FeaScatter
from attack_methods import *

torch.set_printoptions(threshold=10000)
np.set_printoptions(threshold=np.inf)

parser = argparse.ArgumentParser(description='Feature Scatterring Training')

# add type keyword to registries
parser.register('type', 'bool', utils.str2bool)

parser.add_argument('--resume',
                    '-r',
                    action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--adv_mode',
                    default='feature_scatter',
                    type=str,
                    help='adv_mode (feature_scatter)')
parser.add_argument('--model_dir', type=str, help='model path')
parser.add_argument('--init_model_pass',
                    default='-1',
                    type=str,
                    help='init model pass (-1: from scratch; K: checkpoint-K)')
parser.add_argument('--max_epoch',
                    default=200,
                    type=int,
                    help='max number of epochs')
parser.add_argument('--save_epochs', default=5, type=int, help='save period')
parser.add_argument('--decay_epoch1',
                    default=60,
                    type=int,
                    help='learning rate decay epoch one')
parser.add_argument('--decay_epoch2',
                    default=90,
                    type=int,
                    help='learning rate decay point two')
parser.add_argument('--decay_rate',
                    default=0.1,
                    type=float,
                    help='learning rate decay rate')
parser.add_argument('--batch_size_train',
                    default=128,
                    type=int,
                    help='batch size for training')
parser.add_argument('--batch_size_test',
                    default=128,
                    type=int,
                    help='batch size for training')
parser.add_argument('--momentum',
                    default=0.9,
                    type=float,
                    help='momentum (1-tf.momentum)')
parser.add_argument('--weight_decay',
                    default=2e-4,
                    type=float,
                    help='weight decay')
parser.add_argument('--log_step', default=200, type=int, help='log_step')

# number of classes and image size will be updated below based on the dataset
parser.add_argument('--num_classes', default=10, type=int, help='num classes')
parser.add_argument('--image_size', default=32, type=int, help='image size')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset')  # concat cascade
parser.add_argument('--ls_factor', default=0.0, type=float,
                    help='ls_factor')
parser.add_argument('--ls_factor_2', default=0.0, type=float,
                    help='ls_factor_2')
parser.add_argument('--vertex_r', default=2.0, type=float,
                    help='vertex_r')


args = parser.parse_args()

if args.dataset == 'cifar10':
    print('------------cifar10---------')
    args.num_classes = 10
    args.image_size = 32
if args.dataset == 'cifar100':
    print('----------cifar100---------')
    args.num_classes = 100
    args.image_size = 32
if args.dataset == 'svhn':
    print('------------svhn10---------')
    args.num_classes = 10
    args.image_size = 32
if args.dataset == 'mnist':
    print('------------mnist---------')
    args.num_classes = 10
    args.image_size = 28
if args.dataset == 'cifar_aug':
    print('------------cifar aug---------')
    args.num_classes = 10
    args.image_size = 32

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0

# Data
print('==> Preparing data..')

if args.dataset == 'cifar10' or args.dataset == 'cifar100' or args.dataset == 'cifar_aug':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [-1 1]
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [-1 1]
    ])
elif args.dataset == 'svhn':
    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [-1 1]
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [-1 1]
    ])

elif args.dataset == 'mnist':
    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [-1 1]
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [-1 1]
    ])

if args.dataset == 'cifar10':
    trainset = torchvision.datasets.CIFAR10(root='./data',
                                            train=True,
                                            download=True,
                                            transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data',
                                           train=False,
                                           download=True,
                                           transform=transform_test)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
               'ship', 'truck')
elif args.dataset == 'cifar100':
    trainset = torchvision.datasets.CIFAR100(root='./data',
                                             train=True,
                                             download=True,
                                             transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data',
                                            train=False,
                                            download=True,
                                            transform=transform_test)
elif args.dataset == 'cifar_aug':
    trainset = torchvision.datasets.CIFAR10(root='./data',
                                            train=True,
                                            download=True,
                                            transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data',
                                           train=False,
                                           download=True,
                                           transform=transform_test)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
               'ship', 'truck')
    trainset_aug = torchvision.datasets.CIFAR100(root='./data',
                                             train=True,
                                             download=True,
                                             transform=transform_train)

elif args.dataset == 'svhn':
    trainset = torchvision.datasets.SVHN(root='./data',
                                         split='train',
                                         download=True,
                                         transform=transform_train)
    testset = torchvision.datasets.SVHN(root='./data',
                                        split='test',
                                        download=True,
                                        transform=transform_test)

elif args.dataset == 'mnist':
    trainset = torchvision.datasets.MNIST(root='./data',
                                         train=True,
                                         download=True,
                                         transform=transform_train)
    testset = torchvision.datasets.MNIST(root='./data',
                                        train=False,
                                        download=True,
                                        transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=args.batch_size_train,
                                          shuffle=True,
                                          num_workers=2)
if 'aug' in args.dataset:
    trainloader_aug = torch.utils.data.DataLoader(trainset_aug,
                                          batch_size=args.batch_size_train,
                                          shuffle=True,
                                          num_workers=2)

testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=args.batch_size_test,
                                         shuffle=False,
                                         num_workers=2)

print('==> Building model..')

if 'cifar' in args.dataset or args.dataset == 'svhn':
    print('---wide resenet-----')
    basic_net = WideResNet(depth=28,
                           num_classes=args.num_classes,
                           widen_factor=10)
elif args.dataset == 'mnist':
    print('--smallnet--')
    basic_net = SmallNet()



def print_para(net):
    for name, param in net.named_parameters():
        if param.requires_grad:
            print(name)
            print(param.data)
        break


basic_net = basic_net.to(device)

# config for feature scatter

if args.dataset=='mnist':

    # mnist

    config_pgd_train = {
        'train': True,
        'targeted': False,
        'epsilon': 76.5 / 255 * 2,
        'num_steps': 40,
        'step_size': 2.55 / 255 * 2,
        'random_start': True,
        'ls_factor': args.ls_factor,
        'ls_factor_2': args.ls_factor_2,
        'alpha1': 1.0,
        'alpha2': 1.0,
        'r': args.vertex_r
    }

    config_lip_reg = {
        'train': True,
        'targeted': False,
        'epsilon': 76.5 / 255 * 2,
        'num_steps': 1,
        'step_size': 76.5 / 255 * 2,
        'random_start': True,
        'ls_factor': args.ls_factor,
        'ls_factor_2': args.ls_factor_2,
        'alpha1': 1.0,
        'alpha2': 1.0,
        'r': args.vertex_r
    }

    config_feature_scatter = {
        'train': True,
        'epsilon': 76.5 / 255 * 2,
        'num_steps': 1,
        'step_size': 76.5 / 255 * 2,
        'random_start': True,
        'ls_factor': args.ls_factor,
    }

    config_vertex = {
        'train': True,
        'epsilon': 76.5 / 255 * 2,
        'num_steps': 1,
        'step_size': 76.5 / 255 * 2,
        'random_start': True,
        'ls_factor': args.ls_factor,
        'ls_factor_2': args.ls_factor_2,
        'alpha1': 1.0,
        'alpha2': 1.0,
        'r': args.vertex_r
    }

else:

    config_pgd_train = {
    'train': True,
    'targeted': False,
    'epsilon': 8.0 / 255 * 2,
    'num_steps': 1,
    'step_size': 8.0 / 255 * 2,
    'ls_factor': args.ls_factor,
    'random_start': True
     }

    config_feature_scatter = {
    'train': True,
    'epsilon': 8.0 / 255 * 2,
    'num_steps': 1,
    'step_size': 8.0 / 255 * 2,
    'random_start': True,
    'ls_factor': args.ls_factor,
    }

    config_lip_reg = {
    'train': True,
    'epsilon': 8.0 / 255 * 2,
    'num_steps': 1,
    'step_size': 8.0 / 255 * 2,
    'random_start': True,
    'ls_factor': args.ls_factor,
    }

    config_trades = {
        'train': True,
        'epsilon': 8.0 / 255 * 2,
        'num_steps': 1,
        'step_size': 8.0 / 255 * 2,
        'random_start': True,
        'ls_factor': args.ls_factor,
        'lam': 6.0
    }

    config_vertex = {
        'train': True,
        'epsilon': 8.0 / 255 * 2,
        'num_steps': 1,
        'step_size': 10.0 / 255 * 2,
        'random_start': True,
        'ls_factor': args.ls_factor,
        'ls_factor_2': args.ls_factor_2,
        'alpha1': 1.0,
        'alpha2': 1.0,
        'r': args.vertex_r
    }

    config_linear = {
        'train': True,
        'targeted': False,
        'epsilon': 8.0 / 255 * 2,
        'num_steps': 1,
        'step_size': 8.0 / 255 * 2,
        'ls_factor': args.ls_factor,
        'random_start': True
    }






config_vertex_pgd = {
    'train': True,

}

config_pgd = {
    'train': False,
    'targeted': False,
    'epsilon': 8.0 / 255 * 2,
    'num_steps': 20,
    'step_size': 2.0 / 255 * 2,
    'random_start': True
    }



if args.adv_mode.lower() == 'feature_scatter':
    print('-----Feature Scatter mode -----')
    net = Attack_FeaScatter(basic_net, config_feature_scatter)
    test_net = Attack_PGD(basic_net, config_pgd)
elif args.adv_mode.lower() == 'madry':
    print('-----Default Madry mode -----')
    net = Attack_PGD_Train(basic_net, config_pgd_train)
elif args.adv_mode.lower() == 'vertex':
    print('-----Vertex mode -----')
    net = Attack_vertex(basic_net, config_vertex)
    test_net = Attack_PGD(basic_net, config_pgd)
elif args.adv_mode.lower() == 'vertex_pgd':
    print('-----Vertex PGD mode -----')
    net = Attack_vertex_pgd(basic_net, config_pgd_train)
elif args.adv_mode.lower() == 'natural':
    print('-----Natural mode -----')
    net = Attack_None_Train(basic_net, config_vertex_pgd)
elif args.adv_mode.lower() == 'linear':
    print('-----Linear mode -----')
    net = Attack_local_linear(basic_net, config_linear)
elif args.adv_mode.lower() == 'lip_reg':
    print('-----Lipshitz reg mode -----')
    net = Attack_Lipshitz_reg(basic_net, config_lip_reg)
elif args.adv_mode.lower() == 'trades':
    print('-----Trades mode -----')
    net = Attack_Trades(basic_net, config_trades)
else:
    print('-----OTHER_ALGO mode -----')
    raise NotImplementedError("Please implement this algorithm first!")

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

optimizer = optim.SGD(net.parameters(),
                      lr=args.lr,
                      momentum=args.momentum,
                      weight_decay=args.weight_decay)

if args.resume and args.init_model_pass != '-1':
    # Load checkpoint.
    '''
    print('==> Resuming from checkpoint..')
    f_path_latest = os.path.join(args.model_dir, 'latest')
    f_path = os.path.join(args.model_dir,
                          ('checkpoint-%s' % args.init_model_pass))
    if not os.path.isdir(args.model_dir):
        print('train from scratch: no checkpoint directory or file found')
    elif args.init_model_pass == 'latest' and os.path.isfile(f_path_latest):
        checkpoint = torch.load(f_path_latest)
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch'] + 1
        print('resuming from epoch %s in latest' % start_epoch)
    elif os.path.isfile(f_path):
        checkpoint = torch.load(f_path)
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch'] + 1
        print('resuming from epoch %s' % (start_epoch - 1))
    elif not os.path.isfile(f_path) or not os.path.isfile(f_path_latest):
        print('train from scratch: no checkpoint directory or file found')
    '''
    print('==> Resuming from checkpoint..')
    f_path_latest = os.path.join(args.model_dir, 'latest')
    f_path = os.path.join(args.model_dir,
                          ('checkpoint-%s' % args.init_model_pass))
    if not os.path.isdir(args.model_dir):
        print(args.model_dir)
        print('train from scratch: no checkpoint directory or file found')
    elif args.init_model_pass == 'latest' and os.path.isfile(
            f_path_latest):
        checkpoint = torch.load(f_path_latest)
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        print('resuming from epoch %s in latest' % start_epoch)
    elif os.path.isfile(f_path):
        checkpoint = torch.load(f_path)
        net.load_state_dict(checkpoint['net'])
        # start_epoch = checkpoint['epoch']
        start_epoch = args.init_model_pass
        print('resuming from epoch %s' % start_epoch)
        start_epoch = int(args.init_model_pass)
    elif not os.path.isfile(f_path) or not os.path.isfile(f_path_latest):
        print('train from scratch: no checkpoint directory or file found')

soft_xent_loss = softCrossEntropy()


def train_fun(epoch, net):
    print('\nEpoch: %d' % epoch)
    net.train()

    train_loss = 0
    correct = 0
    total = 0

    # update learning rate
    if epoch < args.decay_epoch1:
        lr = args.lr
    elif epoch < args.decay_epoch2:
        lr = args.lr * args.decay_rate
    else:
        lr = args.lr * args.decay_rate * args.decay_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    def get_acc(outputs, targets):
        _, predicted = outputs.max(1)
        total = targets.size(0)
        correct = predicted.eq(targets).sum().item()
        acc = 1.0 * correct / total
        return acc



    iterator = tqdm(trainloader, ncols=0, leave=False)
    # iterator = trainloader


    for batch_idx, (inputs, targets) in enumerate(iterator):
    # for tuples in enumerate(iterator):
        start_time = time.time()
        if args.dataset == 'cifar_aug':

            inputs_aug, targets_aug = next(iter(trainloader_aug))
            indices = np.random.permutation(targets_aug.size()[0])
            inputs_aug = inputs_aug[indices]
            inputs_orig, targets_orig = inputs.detach(), targets.detach()

            inputs[:args.batch_size_train // 5] = inputs_aug[:args.batch_size_train // 5]
            # targets = np.eye(args.batch_size_train)[targets]
            targets = one_hot_tensor(targets, 10, device)
            targets[:args.batch_size_train // 5, :] = 0.1



        inputs, targets = inputs.to(device), targets.to(device)

        adv_acc = 0

        optimizer.zero_grad()

        # forward feature_scatter
        if (args.adv_mode.lower() == 'feature_scatter' or args.adv_mode.lower() == 'lip_reg' or
            args.adv_mode.lower() == 'trades'):
            outputs, loss_fs, flag_out, _, diff_loss = net(inputs.detach(), targets)
            loss = loss_fs
            optimizer.zero_grad()
        elif args.adv_mode.lower() == 'madry':
            # forward madry
            outputs, _, _, pert_inputs, pert_i, y_train = net(inputs, targets)
            loss = soft_xent_loss(outputs, y_train)
            # loss = soft_xent_loss(outputs * 0.5, y_train) # temperturing
            #loss = F.cross_entropy(outputs, targets)
            optimizer.zero_grad()

        elif args.adv_mode.lower() == 'vertex':
            # forward vertex
            outputs, _, _, _, _, y_vertex = net(inputs, targets)
            # outputs, _, _, _, _, y_vertex = net(inputs, targets, epoch = (epoch+1) / args.max_epoch)
            loss = soft_xent_loss(outputs, y_vertex)
            optimizer.zero_grad()
        elif args.adv_mode.lower() == 'vertex_pgd':
            # forward vertex
            outputs, _, _, _, _, y_vertex = net(inputs, targets)
            loss = soft_xent_loss(outputs, y_vertex)
            optimizer.zero_grad()
        elif args.adv_mode.lower() == 'natural':
            # forward vertex
            outputs, _, _, _, _ = net(inputs, targets)
            # loss = F.cross_entropy(basic_net(inputs.detach())[0], targets)
            loss = F.cross_entropy(outputs, targets)
            optimizer.zero_grad()
        elif args.adv_mode.lower() == 'linear':
            # forward vertex
            outputs, _, _, x_train, _ = net(inputs, targets)
            # net(inputs, targets)
            # outputs = basic_net(inputs.detach())[0]
            # loss = F.cross_entropy(outputs, targets.detach())
            outputs, loss_fs, flag_out, _, diff_loss = net(inputs.detach(), targets)
            loss = loss_fs
            optimizer.zero_grad()
        else:
            print('no adv_mode')
            loss = None

        loss.backward()

        optimizer.step()

        train_loss = loss.item()

        duration = time.time() - start_time

        if batch_idx % args.log_step == 0:
            if args.dataset == 'cifar_aug':
                inputs, targets = inputs_orig.to(device), targets_orig.to(device)
            if adv_acc == 0:
                adv_acc = get_acc(outputs, targets)
            iterator.set_description(str(adv_acc))

            nat_outputs, _, _, _, _ = net(inputs, targets, attack=False)

            nat_acc = get_acc(nat_outputs, targets)

            print(
                "epoch %d, step %d, lr %.4f, duration %.2f, training nat acc %.2f, training adv acc %.2f, training adv loss %.4f"
                % (epoch, batch_idx, lr, duration, 100 * nat_acc,
                   100 * adv_acc, train_loss))

    if epoch % 10 == 0:
        print('Saving..')
        f_path = os.path.join(args.model_dir, ('checkpoint-%s' % epoch))
        state = {
            'net': net.state_dict(),
            # 'optimizer': optimizer.state_dict()
        }
        if not os.path.isdir(args.model_dir):
            os.mkdir(args.model_dir)
        torch.save(state, f_path)

    if epoch >= 0:
        print('Saving latest @ epoch %s..' % (epoch))
        f_path = os.path.join(args.model_dir, 'latest')
        state = {
            'net': net.state_dict(),
            'epoch': epoch,
            'optimizer': optimizer.state_dict()
        }
        if not os.path.isdir(args.model_dir):
            os.mkdir(args.model_dir)
        torch.save(state, f_path)

    '''

    if epoch >= 200 and epoch % 5 == 0:

        net.eval()
        test_net.eval()
        correct = 0
        total = 0

        iterator = testloader
        for batch_idx, (inputs, targets) in enumerate(iterator):
            start_time = time.time()
            inputs, targets = inputs.to(device), targets.to(device)

            outputs, _, _, pert_inputs, pert_i = test_net(inputs, targets)

            _, predicted = outputs.max(1)
            batch_size = targets.size(0)
            total += batch_size
            correct_num = predicted.eq(targets).sum().item()
            correct += correct_num


        acc = 100. * correct / total
        print('Val acc:', acc)
    '''




for epoch in range(start_epoch, args.max_epoch):
    train_fun(epoch, net)
