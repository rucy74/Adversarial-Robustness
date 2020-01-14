from __future__ import print_function
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import sys
import datetime

from tqdm import tqdm
from models import *
import utils


from attack_methods import *

from utils import softCrossEntropy

parser = argparse.ArgumentParser(
    description='Feature Scattering Adversarial Training')

parser.register('type', 'bool', utils.str2bool)

parser.add_argument('--resume',
                    '-r',
                    action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--attack', default=True, type='bool', help='attack')
parser.add_argument('--model_dir', type=str, help='model path')
parser.add_argument('--init_model_pass',
                    default='-1',
                    type=str,
                    help='init model pass')

parser.add_argument('--attack_method',
                    default='pgd',
                    type=str,
                    help='adv_mode (natural, pdg or cw)')
parser.add_argument('--attack_method_list', type=str)

parser.add_argument('--log_step', default=7, type=int, help='log_step')

# dataset dependent
parser.add_argument('--num_classes', default=10, type=int, help='num classes')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset')  # concat cascade
parser.add_argument('--batch_size_test',
                    default=100,
                    type=int,
                    help='batch size for testing')
parser.add_argument('--image_size', default=32, type=int, help='image size')

args = parser.parse_args()

if args.dataset == 'cifar10':
    print('------------cifar10---------')
    args.num_classes = 10
    args.image_size = 32
elif args.dataset == 'cifar100':
    print('----------cifar100---------')
    args.num_classes = 100
    args.image_size = 32
if args.dataset == 'svhn':
    print('------------svhn10---------')
    args.num_classes = 10
    args.image_size = 32
elif args.dataset == 'mnist':
    print('----------mnist---------')
    args.num_classes = 10
    args.image_size = 28

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0

# Data
print('==> Preparing data..')

if args.dataset == 'cifar10' or args.dataset == 'cifar100':
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [-1 1]
    ])
elif args.dataset == 'svhn':
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [-1 1]
    ])
elif args.dataset == 'mnist':
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [-1 1]
    ])

if args.dataset == 'cifar10':
    testset = torchvision.datasets.CIFAR10(root='./data',
                                           train=False,
                                           download=True,
                                           transform=transform_test)
elif args.dataset == 'cifar100':
    testset = torchvision.datasets.CIFAR100(root='./data',
                                            train=False,
                                            download=True,
                                            transform=transform_test)

elif args.dataset == 'svhn':
    testset = torchvision.datasets.SVHN(root='./data',
                                        split='test',
                                        download=True,
                                        transform=transform_test)
elif args.dataset == 'mnist':
    testset = torchvision.datasets.MNIST(root='./data',
                                         train=False,
                                         download=True,
                                         transform=transform_test)

testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=args.batch_size_test,
                                         shuffle=False,
                                         num_workers=2)

print('==> Building model..')
if args.dataset == 'cifar10' or args.dataset == 'cifar100' or args.dataset == 'svhn':
    print('---wide resenet-----')
    basic_net = WideResNet(depth=28,
                           num_classes=args.num_classes,
                           widen_factor=10)
elif args.dataset == 'mnist':
    print('--smallnet--')
    basic_net = SmallNet()

basic_net = basic_net.to(device)

# configs
config_natural = {'train': False}


if args.dataset != 'mnist':
    config_pgd = {
    'train': False,
    'targeted': False,
    'epsilon': 8.0 / 255 * 2,
    'num_steps': 10,
    'step_size': 2.0 / 255 * 2,
    'random_start': True
    }

    config_fgsm = {
        'train': False,
        'targeted': False,
        'epsilon': 8.0 / 255 * 2,
        'num_steps': 1,
        'step_size': 8.0 / 255 * 2,
        'random_start': True
    }

    config_linear = {
        'train': False,
        'targeted': False,
        'epsilon': 8.0 / 255 * 2,
        'num_steps': 10,
        'step_size': 2.0 / 255 * 2,
        'random_start': True
    }

    config_linear_v2 = {
        'train': False,
        'targeted': False,
        'epsilon': 8.0 / 255 * 2,
        'num_steps': 10,
        'step_size': 2.0 / 255 * 2,
        'random_start': True
    }

    config_feature_scatter = {
        'train': False,
        'targeted': False,
        'epsilon': 8.0 / 255 * 2,
        'num_steps': 10,
        'step_size': 2.0 / 255 * 2,
        'random_start': True
    }

    config_cos = {
        'train': False,
        'targeted': False,
        'epsilon': 8.0 / 255 * 2,
        'num_steps': 50,
        'step_size': 1.0 / 255 * 2,
        'random_start': True
    }

    config_reverse = {
        'train': False,
        'targeted': False,
        'epsilon': 8.0 / 255 * 2,
        'num_steps': 10,
        'step_size': 2.0 / 255 * 2,
        'random_start': True
    }

else:
    # mnist

    config_fgsm = {
        'train': False,
        'targeted': False,
        'epsilon': 76.5 / 255 * 2,
        'num_steps': 1,
        'step_size': 76.5 / 255 * 2,
        'random_start': True
    }

    config_pgd = {
    'train': True,
    'targeted': False,
    'epsilon': 76.5 / 255 * 2,
    'num_steps': 40,
    'step_size': 2.55 / 255 * 2,
    'random_start': True
    }

    config_linear = {
        'train': False,
        'targeted': False,
        'epsilon': 76.5 / 255 * 2,
        'num_steps': 100,
        'step_size': 76.5 / 255 * 2,
        'random_start': True
    }

config_cw = {
    'train': False,
    'targeted': False,
    'epsilon': 8.0 / 255 * 2,
    'num_steps': 20,
    'step_size': 2.0 / 255 * 2,
    'random_start': True,
    'confidence': 0

}

config_pgd_100 = {
    'train': False,
    'targeted': False,
    'epsilon': 8.0 / 255 * 2,
    'num_steps': 100,
    'step_size': 2.0 / 255 * 2,
    'random_start': True
}

config_cw_100 = {
    'train': False,
    'targeted': False,
    'epsilon': 8.0 / 255 * 2,
    'num_steps': 100,
    'step_size': 2.0 / 255 * 2,
    'random_start': True,
    'confidence': 0

}

def test(epoch, net):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    xce = 0.



    iterator = tqdm(testloader, ncols=0, leave=False)
    # x_adv = torch.load('x_adv.pt')['x_adv']
    # print(x_adv.size())
    # i = -1
    for batch_idx, (inputs, targets) in enumerate(iterator):
        # i += 1
        start_time = time.time()
        inputs, targets = inputs.to(device), targets.to(device)
        pert_inputs = inputs.detach()
        # pert_inputs, targets = x_adv[i*args.batch_size_test:np.minimum((i+1)*args.batch_size_test, 10000)].to(device), targets.to(device)

        outputs, _, _, pert_inputs, pert_i = net(pert_inputs,
                                                 targets,
                                                 batch_idx=batch_idx)

        xce_batch = torch.sum(-utils.one_hot_tensor(targets, 10, device)*F.log_softmax(outputs)).item()


        loss = criterion(outputs, targets)
        test_loss += loss.item()

        duration = time.time() - start_time

        _, predicted = outputs.max(1)
        batch_size = targets.size(0)
        total += batch_size
        correct_num = predicted.eq(targets).sum().item()
        correct += correct_num
        iterator.set_description(
            str(predicted.eq(targets).sum().item() / targets.size(0)))

        xce += xce_batch

        if batch_idx % args.log_step == 0:
            print(
                "step %d, duration %.2f, test  acc %.2f, avg-acc %.2f, loss %.2f"
                % (batch_idx, duration, 100. * correct_num / batch_size,
                   100. * correct / total, test_loss / total))

    acc = 100. * correct / total
    print('Val acc:', acc)
    xce = xce / total
    print('xce : ',xce)
    return acc


attack_list = args.attack_method_list.split('-')
attack_num = len(attack_list)

for attack_idx in range(attack_num):

    args.attack_method = attack_list[attack_idx]

    if args.attack_method == 'natural':
        print('-----natural non-adv mode -----')
        # config is only dummy, not actually used
        net = Attack_None(basic_net, config_natural)
    elif args.attack_method == 'fgsm':
        print('-----FGSM adv mode -----')
        net = Attack_PGD(basic_net, config_fgsm)
    elif args.attack_method == 'pgd':
        print('-----PGD adv mode -----')
        net = Attack_PGD(basic_net, config_pgd)
    elif args.attack_method == 'cw':
        print('-----CW adv mode -----')
        net = Attack_CW(basic_net, config_cw)
    elif args.attack_method == 'pgd100':
        print('-----PGD adv mode -----')
        net = Attack_PGD(basic_net, config_pgd_100)
    elif args.attack_method == 'cw100':
        print('-----CW adv mode -----')
        net = Attack_CW(basic_net, config_cw_100)
    elif args.attack_method == 'linear':
        print('-----Local linear adv mode -----')
        net = Attack_local_linear(basic_net, config_linear)
    elif args.attack_method == 'linear_v2':
        print('-----Local linear V2 adv mode -----')
        net = Attack_local_linear_v2(basic_net, config_linear_v2)
    elif args.attack_method == 'cos':
        print('-----Cos adv mode -----')
        net = Attack_Cos(basic_net, config_cos)
    elif args.attack_method == 'reverse':
        print('-----Reverse adv mode -----')
        net = Attack_Reverse(basic_net, config_reverse)
    elif args.attack_method == 'feature_scatter':
        print('-----Feature scatter adv mode -----')
        net = Attack_FeaScatter_test(basic_net, config_feature_scatter)
    else:
        raise Exception(
            'Should be a valid attack method. The specified attack method is: {}'
            .format(args.attack_method))

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume and args.init_model_pass != '-1':
        # Load checkpoint.
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
            #start_epoch = checkpoint['epoch']
            start_epoch = f_path
            print('resuming from epoch %s' % start_epoch)
        elif not os.path.isfile(f_path) or not os.path.isfile(f_path_latest):
            print('train from scratch: no checkpoint directory or file found')

    criterion = nn.CrossEntropyLoss()

    test(0, net)
