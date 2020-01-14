import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import *
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable
import utils
import math

from utils import *
import ot
import pickle

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Attack_None(nn.Module):
    def __init__(self, basic_net, config):
        super(Attack_None, self).__init__()
        self.train_flag = True if 'train' not in config.keys(
        ) else config['train']
        self.basic_net = basic_net
        print('Attack_None')
        print(config)

    def forward(self, inputs, targets, attack=None, batch_idx=-1):
        if self.train_flag:
            self.basic_net.train()
        else:
            self.basic_net.eval()
        outputs, _ = self.basic_net(inputs)
        return outputs, None, True, inputs.detach(), inputs.detach()

class Attack_None_Train(nn.Module):
    def __init__(self, basic_net, config):
        super(Attack_None_Train, self).__init__()
        self.train_flag = True if 'train' not in config.keys(
        ) else config['train']
        self.basic_net = basic_net
        print('Attack_None_train')
        print(config)

    def forward(self, inputs, targets, attack=None, batch_idx=-1):
        if self.train_flag:
            self.basic_net.train()
        else:
            self.basic_net.eval()
        outputs, _ = self.basic_net(inputs)
        return outputs, None, True, inputs.detach(), inputs.detach()

class Attack_PGD(nn.Module):
    # Back-propogate
    def __init__(self, basic_net, config, attack_net=None):
        super(Attack_PGD, self).__init__()
        self.basic_net = basic_net
        self.attack_net = attack_net
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        self.train_flag = True if 'train' not in config.keys(
        ) else config['train']

        self.box_type = 'white' if 'box_type' not in config.keys(
        ) else config['box_type']

        print('Attack_PGD')
        print(config)

    def forward(self,
                inputs,
                targets,
                attack=True,
                targeted_label=-1,
                batch_idx=0):

        if not attack:
            outputs = self.basic_net(inputs)[0]
            return outputs, None, True, inputs.detach(), inputs.detach()

        if self.box_type == 'white':
            aux_net = pickle.loads(pickle.dumps(self.basic_net))
        elif self.box_type == 'black':
            assert self.attack_net is not None, "should provide an additional net in black-box case"
            aux_net = pickle.loads(pickle.dumps(self.basic_net))
        aux_net.eval()
        logits_pred_nat = aux_net(inputs)[0]
        targets_prob = F.softmax(logits_pred_nat.float(), dim=1)

        num_classes = targets_prob.size(1)

        loss_fun = torch.nn.CrossEntropyLoss(reduction='none')

        outputs = aux_net(inputs)[0]
        targets_prob = F.softmax(outputs.float(), dim=1)
        y_tensor_adv = targets
        step_sign = 1.0

        x = inputs.detach()
        if self.rand:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        x_org = x.detach()
        loss_array = np.zeros((inputs.size(0), self.num_steps))

        for i in range(self.num_steps):
            x.requires_grad_()
            zero_gradients(x)
            if x.grad is not None:
                x.grad.data.fill_(0)
            aux_net.eval()
            logits = aux_net(x)[0]
            loss = loss_fun(logits, y_tensor_adv)
            loss = loss.mean()
            aux_net.zero_grad()
            loss.backward()

            x_adv = x.data + step_sign * self.step_size * torch.sign(
                x.grad.data)
            x_adv = torch.min(torch.max(x_adv, inputs - self.epsilon),
                              inputs + self.epsilon)
            x_adv = torch.clamp(x_adv, -1.0, 1.0)
            x = Variable(x_adv)

        if self.train_flag:
            self.basic_net.train()
        else:
            self.basic_net.eval()

        logits_pert = self.basic_net(x.detach())[0]

        return logits_pert, targets_prob.detach(), True, x.detach(
        ), x_org.detach()

class Attack_Reverse(nn.Module):
    # Back-propogate
    def __init__(self, basic_net, config, attack_net=None):
        super(Attack_Reverse, self).__init__()
        self.basic_net = basic_net
        self.attack_net = attack_net
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        self.train_flag = True if 'train' not in config.keys(
        ) else config['train']

        self.box_type = 'white' if 'box_type' not in config.keys(
        ) else config['box_type']

        print('Attack_Reverse')
        print(config)

    def forward(self,
                inputs,
                targets,
                attack=True,
                targeted_label=-1,
                batch_idx=0):

        if not attack:
            outputs = self.basic_net(inputs)[0]
            return outputs, None, True, inputs.detach(), inputs.detach()

        if self.box_type == 'white':
            aux_net = pickle.loads(pickle.dumps(self.basic_net))
        elif self.box_type == 'black':
            assert self.attack_net is not None, "should provide an additional net in black-box case"
            aux_net = pickle.loads(pickle.dumps(self.basic_net))
        aux_net.eval()
        logits_pred_nat = aux_net(inputs)[0]
        targets_prob = F.softmax(logits_pred_nat.float(), dim=1)

        num_classes = targets_prob.size(1)

        # loss_fun = torch.nn.CrossEntropyLoss(reduction='none')
        loss_fun = softCrossEntropy()

        outputs = aux_net(inputs)[0]
        targets_prob = F.softmax(outputs.float(), dim=1)
        y_tensor_adv = targets
        y_gt = one_hot_tensor(y_tensor_adv, 10, None)
        y_fgsm = label_changing(y_gt, 10, 2.)
        # y_fgsm = label_reverse(y_gt, 10, 99999.)
        step_sign = 1.0

        x = inputs.detach()
        if self.rand:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        x_org = x.detach()
        loss_array = np.zeros((inputs.size(0), self.num_steps))

        for i in range(self.num_steps):
            x.requires_grad_()
            zero_gradients(x)
            if x.grad is not None:
                x.grad.data.fill_(0)
            aux_net.eval()
            logits = aux_net(x)[0]
            loss = loss_fun(logits, y_fgsm)
            loss = loss.mean()
            loss_ = loss_fun(logits, F.softmax(logits.float(), dim=1)).mean()
            loss = loss_ #- loss
            aux_net.zero_grad()
            loss.backward()

            x_adv = x.data + step_sign * self.step_size * torch.sign(
                x.grad.data)
            x_adv = torch.min(torch.max(x_adv, inputs - self.epsilon),
                              inputs + self.epsilon)
            x_adv = torch.clamp(x_adv, -1.0, 1.0)
            x = Variable(x_adv)

        if self.train_flag:
            self.basic_net.train()
        else:
            self.basic_net.eval()

        logits_pert = self.basic_net(x.detach())[0]

        return logits_pert, targets_prob.detach(), True, x.detach(
        ), x_org.detach()

class Attack_Cos(nn.Module):
    # Back-propogate
    def __init__(self, basic_net, config, attack_net=None):
        super(Attack_Cos, self).__init__()
        self.basic_net = basic_net
        self.attack_net = attack_net
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        self.train_flag = True if 'train' not in config.keys(
        ) else config['train']

        self.box_type = 'white' if 'box_type' not in config.keys(
        ) else config['box_type']

        print('Attack_cos')
        print(config)

    def forward(self,
                inputs,
                targets,
                attack=True,
                targeted_label=-1,
                batch_idx=0):

        if not attack:
            outputs = self.basic_net(inputs)[0]
            return outputs, None, True, inputs.detach(), inputs.detach()

        if self.box_type == 'white':
            aux_net = pickle.loads(pickle.dumps(self.basic_net))
        elif self.box_type == 'black':
            assert self.attack_net is not None, "should provide an additional net in black-box case"
            aux_net = pickle.loads(pickle.dumps(self.basic_net))
        aux_net.eval()
        logits_pred_nat = aux_net(inputs)[0]
        targets_prob = F.softmax(logits_pred_nat.float(), dim=1)

        num_classes = targets_prob.size(1)

        # loss_fun = torch.nn.CrossEntropyLoss(reduction='none')

        outputs = aux_net(inputs)[0]
        targets_prob = F.softmax(outputs.float(), dim=1)
        y_tensor_adv = targets
        step_sign = 1.0

        x = inputs.detach()
        if self.rand:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        x_org = x.detach()
        loss_array = np.zeros((inputs.size(0), self.num_steps))

        for i in range(self.num_steps):
            x.requires_grad_()
            zero_gradients(x)
            if x.grad is not None:
                x.grad.data.fill_(0)
            aux_net.eval()
            logits = aux_net(x)[0]
            # loss = loss_fun(logits, y_tensor_adv)
            loss = ot.pair_cos_dist(logits_pred_nat.detach(), logits).mean()
            # loss = loss.mean()
            aux_net.zero_grad()
            loss.backward()

            x_adv = x.data + step_sign * self.step_size * torch.sign(
                x.grad.data)
            x_adv = torch.min(torch.max(x_adv, inputs - self.epsilon),
                              inputs + self.epsilon)
            x_adv = torch.clamp(x_adv, -1.0, 1.0)
            x = Variable(x_adv)

        if self.train_flag:
            self.basic_net.train()
        else:
            self.basic_net.eval()

        logits_pert = self.basic_net(x.detach())[0]

        return logits_pert, targets_prob.detach(), True, x.detach(
        ), x_org.detach()



class Attack_local_linear(nn.Module):
    # Back-propogate
    def __init__(self, basic_net, config, attack_net=None):
        super(Attack_local_linear, self).__init__()
        self.basic_net = basic_net
        self.attack_net = attack_net
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        self.train_flag = True if 'train' not in config.keys(
        ) else config['train']

        self.box_type = 'white' if 'box_type' not in config.keys(
        ) else config['box_type']
        self.ls_factor = 0.1 if 'ls_factor' not in config.keys(
        ) else config['ls_factor']

        print('Attack_local_linear')
        print(config)

    def forward(self,
                inputs,
                targets,
                attack=True,
                targeted_label=-1,
                batch_idx=0):

        if not attack:
            outputs = self.basic_net(inputs)[0]
            return outputs, None, True, inputs.detach(), inputs.detach()

        if self.box_type == 'white':
            aux_net = pickle.loads(pickle.dumps(self.basic_net))
        elif self.box_type == 'black':
            assert self.attack_net is not None, "should provide an additional net in black-box case"
            aux_net = pickle.loads(pickle.dumps(self.basic_net))
        aux_net.eval()
        logits_pred_nat = aux_net(inputs)[0]
        targets_prob = F.softmax(logits_pred_nat.float(), dim=1)

        num_classes = targets_prob.size(1)

        inputs.requires_grad_(True)
        zero_gradients(inputs)
        if inputs.grad is not None:
            inputs.grad.data.fill_(0)
        outputs = aux_net(inputs)[0]
        targets_prob = F.softmax(outputs.float(), dim=1)
        # targets_value, _ = torch.max(targets_prob, dim=1)
        # logits_value, _  = torch.max(outputs.float(), dim=1)
        # torch.sum(targets_value).backward()
        # inputs_grad = inputs.grad.data
        # zero_gradients(inputs_grad)
        # inputs.requires_grad_(False)
        # aux_net.zero_grad()

        y_tensor_adv = targets
        step_sign = 1.0

        x = inputs.detach()
        x_org = x.detach()
        if self.rand:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        delta = x - x_org
        x_org.requires_grad_(False)
        loss_array = np.zeros((inputs.size(0), self.num_steps))
        # criterion_kl = nn.KLDivLoss(reduction='mean')
        y_gt = one_hot_tensor(targets, num_classes, device)

        loss_ce = softCrossEntropy()
        criterion_kl = nn.KLDivLoss(size_average=False)
        x_adv = x_org.data

        for i in range(self.num_steps ):
            delta.requires_grad_()
            zero_gradients(delta)
            if delta.grad is not None:
                delta.grad.data.fill_(0)
            aux_net.eval()
            logits = aux_net(x_org + delta)[0]
            loss_kl = criterion_kl(F.log_softmax(logits, dim=1), targets_prob.detach())
            # x_adv_logits, _ = torch.max(logits.float(), dim=1)
            # x_adv_value, _ = torch.max(F.softmax(logits.float(), dim=1), dim=1)
            # loss = x_adv_value - targets_value.detach() - torch.diag(torch.mm(delta.view(delta.size()[0], -1), torch.t(inputs_grad.detach().view(inputs.size()[0], -1))))
            # loss = torch.norm(x_adv_logits - logits_value.detach() - torch.diag(
            #     torch.mm(delta.view(delta.size()[0], -1), torch.t(inputs_grad.detach().view(inputs.size()[0], -1)))), dim=-1)
            # loss = torch.norm(x_adv_value - targets_value.detach() - torch.diag(
            #     torch.mm(delta.view(delta.size()[0], -1), torch.t(inputs_grad.detach().view(inputs.size()[0], -1)))),
            #                   dim=-1)
            # print(loss.mean(), loss_kl)

            # loss = 0.02*loss.mean() + loss_kl
            # loss = loss_kl
            loss = loss_kl.mean()
            aux_net.zero_grad()
            loss.backward()

            delta = delta.data + step_sign * self.step_size * torch.sign(delta.grad.data)
            delta = torch.clamp(delta, -self.epsilon, self.epsilon)
            x_adv = x_org + delta
            x_adv = torch.clamp(x_adv, -1.0, 1.0)
            delta = Variable(x_adv - x_org)

        aux_net.zero_grad()

        x = Variable(x_adv)
        zero_gradients(delta)
        zero_gradients(x)

        if self.train_flag:
            self.basic_net.train()
        else:
            self.basic_net.eval()

        logits_pert = self.basic_net(x.detach())[0]
        self.basic_net.zero_grad()

        y_sm = utils.label_smoothing(y_gt, y_gt.size(1), self.ls_factor)

        adv_loss = loss_ce(logits_pert, y_sm.detach())

        return logits_pert, adv_loss, True, x.detach(), None


        # return logits_pert, targets_prob.detach(), True, x.detach(
        # ), x_org.detach()

class Attack_local_linear_v2(nn.Module):
    # Back-propogate
    def __init__(self, basic_net, config, attack_net=None):
        super(Attack_local_linear_v2, self).__init__()
        self.basic_net = basic_net
        self.attack_net = attack_net
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        self.train_flag = True if 'train' not in config.keys(
        ) else config['train']

        self.box_type = 'white' if 'box_type' not in config.keys(
        ) else config['box_type']

        print('Attack_local_linear_v2')
        print(config)

    def forward(self,
                inputs,
                targets,
                attack=True,
                targeted_label=-1,
                batch_idx=0):

        if not attack:
            outputs = self.basic_net(inputs)[0]
            return outputs, None, True, inputs.detach(), inputs.detach()

        if self.box_type == 'white':
            aux_net = pickle.loads(pickle.dumps(self.basic_net))
        elif self.box_type == 'black':
            assert self.attack_net is not None, "should provide an additional net in black-box case"
            aux_net = pickle.loads(pickle.dumps(self.basic_net))
        aux_net.eval()
        logits_pred_nat = aux_net(inputs)[0]
        targets_prob = F.softmax(logits_pred_nat.float(), dim=1)

        num_classes = targets_prob.size(1)

        inputs.requires_grad_(True)
        zero_gradients(inputs)
        if inputs.grad is not None:
            inputs.grad.data.fill_(0)
        outputs = aux_net(inputs)[0]
        targets_prob = F.softmax(outputs.float(), dim=1)
        targets_value, _ = torch.max(targets_prob, dim=1)
        logits_value, _  = torch.max(outputs.float(), dim=1)
        torch.sum(targets_value).backward()
        inputs_grad = inputs.grad.data
        zero_gradients(inputs_grad)
        inputs.requires_grad_(False)
        aux_net.zero_grad()

        y_tensor_adv = targets
        step_sign = 1.0

        x = inputs.detach()
        x_org = x.detach()
        if self.rand:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        delta = x - x_org
        x_org.requires_grad_(False)
        loss_array = np.zeros((inputs.size(0), self.num_steps))

        x_adv = x_org.data

        for i in range(self.num_steps ):
            delta.requires_grad_()
            zero_gradients(delta)
            if delta.grad is not None:
                delta.grad.data.fill_(0)
            aux_net.eval()
            logits = aux_net(x_org)[0]
            logits_plus = aux_net(x_org + delta)[0]
            # logits_minus = aux_net(x_org - delta)[0]
            # loss = torch.abs(torch.norm(logits_plus - logits, dim=-1) - torch.norm(logits_minus - logits, dim=-1))
            loss = torch.norm(logits_plus - logits, dim=-1) #- torch.norm(logits_minus - logits, dim=-1)

            # plus_dist = ot.pair_cos_dist(logits_plus, logits)
            # minus_dist = ot.pair_cos_dist(logits_minus, logits)
            # loss = torch.abs(plus_dist - minus_dist.detach().data).abs()

            # loss = torch.norm(logits_plus - logits_minus, dim=-1)
            loss = loss.mean()
            aux_net.zero_grad()
            loss.backward()

            delta = delta.data + step_sign * self.step_size * torch.sign(delta.grad.data)
            delta = torch.clamp(delta, -self.epsilon, self.epsilon)
            x_adv = x_org + delta
            x_adv = torch.clamp(x_adv, -1.0, 1.0)
            delta = Variable(x_adv - x_org)

        aux_net.zero_grad()

        x = Variable(x_adv)
        zero_gradients(delta)
        zero_gradients(x)

        if self.train_flag:
            self.basic_net.train()
        else:
            self.basic_net.eval()

        logits_pert = self.basic_net(x.detach())[0]


        return logits_pert, targets_prob.detach(), True, x.detach(
        ), x_org.detach()
'''
class Attack_PGD_Train(nn.Module):
    # Back-propogate
    def __init__(self, basic_net, config, attack_net=None):
        super(Attack_PGD_Train, self).__init__()
        self.basic_net = basic_net
        self.attack_net = attack_net
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        self.train_flag = True if 'train' not in config.keys(
        ) else config['train']

        self.box_type = 'white' if 'box_type' not in config.keys(
        ) else config['box_type']

        self.ls_factor = 0.1 if 'ls_factor' not in config.keys(
        ) else config['ls_factor']

        print(config)

    def forward(self,
                inputs,
                targets,
                attack=True,
                targeted_label=-1,
                batch_idx=0):

        if not attack:
            outputs = self.basic_net(inputs)[0]
            return outputs, None, True, inputs.detach(), inputs.detach()

        if self.box_type == 'white':
            aux_net = pickle.loads(pickle.dumps(self.basic_net))
        elif self.box_type == 'black':
            assert self.attack_net is not None, "should provide an additional net in black-box case"
            aux_net = pickle.loads(pickle.dumps(self.basic_net))
        aux_net.eval()
        logits_pred_nat = aux_net(inputs)[0]
        targets_prob = F.softmax(logits_pred_nat.float(), dim=1)

        num_classes = targets_prob.size(1)

        loss_fun = torch.nn.CrossEntropyLoss(reduction='none')

        outputs = aux_net(inputs)[0]
        targets_prob = F.softmax(outputs.float(), dim=1)
        y_tensor_adv = targets
        step_sign = 1.0

        x = inputs.detach()
        if self.rand:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        x_org = x.detach()
        loss_array = np.zeros((inputs.size(0), self.num_steps))

        for i in range(self.num_steps):
            grad = torch.zeros_like(x)
            for j in range(5):
                x.requires_grad_()
                zero_gradients(x)
                if x.grad is not None:
                    x.grad.data.fill_(0)
                aux_net.eval()
                rand_noise = torch.zeros_like(x).normal_(0, 0.1 * self.step_size)
                x = rand_noise
                logits = aux_net(x)[0]
                loss = loss_fun(logits, y_tensor_adv)
                loss = loss.mean()
                aux_net.zero_grad()
                loss.backward()
                grad += x.grad.data
                x -= rand_noise

            x_adv = x.data + step_sign * self.step_size * torch.sign(
                x.grad.data)
            x_adv = torch.min(torch.max(x_adv, inputs - self.epsilon),
                              inputs + self.epsilon)
            x_adv = torch.clamp(x_adv, -1.0, 1.0)
            x = Variable(x_adv)

        if self.train_flag:
            self.basic_net.train()
        else:
            self.basic_net.eval()

        logits_pert = self.basic_net(x.detach())[0]
        y_gt = one_hot_tensor(targets, num_classes, device)
        y_train = utils.label_smoothing(y_gt, y_gt.size(1), self.ls_factor)

        return logits_pert, targets_prob.detach(), True, x.detach(
        ), x_org.detach(), y_train

        #return logits_pert, targets_prob.detach(), True, x.detach(
        #), x_org.detach()
'''

class Attack_PGD_Train(nn.Module):
    # Back-propogate
    def __init__(self, basic_net, config, attack_net=None):
        super(Attack_PGD_Train, self).__init__()
        self.basic_net = basic_net
        self.attack_net = attack_net
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        self.train_flag = True if 'train' not in config.keys(
        ) else config['train']

        self.box_type = 'white' if 'box_type' not in config.keys(
        ) else config['box_type']

        self.ls_factor = 0.1 if 'ls_factor' not in config.keys(
        ) else config['ls_factor']

        print('Attack_PGD_Train')
        print(config)

    def forward(self,
                inputs,
                targets,
                attack=True,
                targeted_label=-1,
                batch_idx=0):

        if not attack:
            outputs = self.basic_net(inputs)[0]
            return outputs, None, True, inputs.detach(), inputs.detach()

        if self.box_type == 'white':
            aux_net = pickle.loads(pickle.dumps(self.basic_net))
        elif self.box_type == 'black':
            assert self.attack_net is not None, "should provide an additional net in black-box case"
            aux_net = pickle.loads(pickle.dumps(self.basic_net))
        aux_net.eval()
        logits_pred_nat = aux_net(inputs)[0]
        targets_prob = F.softmax(logits_pred_nat.float(), dim=1)

        num_classes = targets_prob.size(1)

        loss_fun = torch.nn.CrossEntropyLoss(reduction='none')

        outputs = aux_net(inputs)[0]
        targets_prob = F.softmax(outputs.float(), dim=1)
        y_tensor_adv = targets
        step_sign = 1.0

        x = inputs.detach()
        if self.rand:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        x_org = x.detach()
        loss_array = np.zeros((inputs.size(0), self.num_steps))

        for i in range(self.num_steps):
            x.requires_grad_()
            zero_gradients(x)
            if x.grad is not None:
                x.grad.data.fill_(0)
            aux_net.eval()
            logits = aux_net(x)[0]
            loss = loss_fun(logits, y_tensor_adv)
            loss = loss.mean()
            aux_net.zero_grad()
            loss.backward()

            x_adv = x.data + step_sign * self.step_size * torch.sign(x.grad.data)
            x_adv = torch.min(torch.max(x_adv, inputs - self.epsilon),
                              inputs + self.epsilon)
            x_adv = torch.clamp(x_adv, -1.0, 1.0)
            x = Variable(x_adv)

        if self.train_flag:
            self.basic_net.train()
        else:
            self.basic_net.eval()

        logits_pert = self.basic_net(x.detach())[0]
        y_gt = one_hot_tensor(targets, num_classes, device)
        y_train = utils.label_smoothing(y_gt, y_gt.size(1), self.ls_factor)


        return logits_pert, targets_prob.detach(), True, x.detach(
        ), x_org.detach(), y_train

class Attack_Trades(nn.Module):
    # Back-propogate
    def __init__(self, basic_net, config, attack_net=None):
        super(Attack_Trades, self).__init__()
        self.basic_net = basic_net
        self.attack_net = attack_net
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        self.train_flag = True if 'train' not in config.keys(
        ) else config['train']

        self.box_type = 'white' if 'box_type' not in config.keys(
        ) else config['box_type']

        self.ls_factor = 0.1 if 'ls_factor' not in config.keys(
        ) else config['ls_factor']

        self.lam = 6.0 if 'lam' not in config.keys(
        ) else config['lam']

        print('Attack_Trades')
        print(config)

    def forward(self,
                inputs,
                targets,
                attack=True,
                targeted_label=-1,
                batch_idx=0):

        if not attack:
            outputs = self.basic_net(inputs)[0]
            return outputs, None, True, inputs.detach(), inputs.detach()

        if self.box_type == 'white':
            aux_net = pickle.loads(pickle.dumps(self.basic_net))
        elif self.box_type == 'black':
            assert self.attack_net is not None, "should provide an additional net in black-box case"
            aux_net = pickle.loads(pickle.dumps(self.basic_net))
        aux_net.eval()
        logits_pred_nat = aux_net(inputs)[0]
        targets_prob = F.softmax(logits_pred_nat.float(), dim=1)

        num_classes = targets_prob.size(1)

        loss_fun = torch.nn.CrossEntropyLoss(reduction='none')
        loss_kl = nn.KLDivLoss(size_average=False)
        loss_ce = softCrossEntropy()

        outputs = aux_net(inputs)[0]
        targets_prob = F.softmax(outputs.float(), dim=1)
        y_tensor_adv = targets
        step_sign = 1.0

        x = inputs.detach()
        x_org = x.detach()
        if self.rand:
            #x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
            x = x + 0.002 * torch.randn(x.shape).cuda().detach()
        loss_array = np.zeros((inputs.size(0), self.num_steps))

        for i in range(self.num_steps):
            x.requires_grad_()
            zero_gradients(x)
            if x.grad is not None:
                x.grad.data.fill_(0)
            aux_net.eval()
            logits = aux_net(x)[0]
            loss = loss_kl(F.log_softmax(logits, dim=1), F.softmax(logits_pred_nat, dim=1))
            # loss = loss.mean()
            aux_net.zero_grad()
            loss.backward()

            x_adv = x.data + step_sign * self.step_size * torch.sign(x.grad.data)
            x_adv = torch.min(torch.max(x_adv, inputs - self.epsilon),
                              inputs + self.epsilon)
            x_adv = torch.clamp(x_adv, -1.0, 1.0)
            x = Variable(x_adv)

        if self.train_flag:
            self.basic_net.train()
        else:
            self.basic_net.eval()

        logits_pert = self.basic_net(x.detach())[0]
        logits_nat = self.basic_net(x_org.detach())[0]
        y_gt = one_hot_tensor(targets, num_classes, device)
        y_train = utils.label_smoothing(y_gt, y_gt.size(1), self.ls_factor)

        clean_loss = loss_ce(logits_nat, y_train.detach())
        kl_loss = loss_kl(F.log_softmax(logits_pert, dim=1), F.softmax(logits_nat, dim=1)) / x.shape[0]
        loss = clean_loss + self.lam * kl_loss

        return logits_pert, loss, True, x.detach(), True

class Attack_CW(nn.Module):
    # Back-propogate
    def __init__(self, basic_net, config, attack_net=None):
        super(Attack_CW, self).__init__()
        self.basic_net = basic_net
        self.attack_net = attack_net
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        self.train_flag = True if 'train' not in config.keys(
        ) else config['train']

        self.box_type = 'white' if 'box_type' not in config.keys(
        ) else config['box_type']
        self.confidence = 50 if 'confidence' not in config.keys(
        ) else config['confidence']

        print('Attack_CW')
        print(config)


    def forward(self,
                inputs,
                targets,
                attack=True,
                targeted_label=-1,
                batch_idx=0):

        if not attack:
            outputs = self.basic_net(inputs)[0]
            return outputs, None, True, inputs.detach(), inputs.detach()

        if self.box_type == 'white':
            aux_net = pickle.loads(pickle.dumps(self.basic_net))
        elif self.box_type == 'black':
            assert self.attack_net is not None, "should provide an additional net in black-box case"
            aux_net = pickle.loads(pickle.dumps(self.basic_net))
        aux_net.eval()
        logits_pred_nat = aux_net(inputs)[0]
        targets_prob = F.softmax(logits_pred_nat.float(), dim=1)

        num_classes = targets_prob.size(1)

        loss_fun = torch.nn.CrossEntropyLoss(reduction='none')

        outputs = aux_net(inputs)[0]
        targets_prob = F.softmax(outputs.float(), dim=1)
        y_tensor_adv = targets
        step_sign = 1.0

        x = inputs.detach()
        if self.rand:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        x_org = x.detach()
        loss_array = np.zeros((inputs.size(0), self.num_steps))

        x.requires_grad_()
        zero_gradients(x)
        if x.grad is not None:
            x.grad.data.fill_(0)


        for i in range(self.num_steps):
            x.requires_grad_()
            zero_gradients(x)
            if x.grad is not None:
                x.grad.data.fill_(0)
            aux_net.eval()
            logits = aux_net(x)[0]

            ## cw linf loss ##

            label_mask = one_hot_tensor(targets, num_classes, device)
            correct_logit = torch.sum(label_mask * logits, dim=1)
            wrong_logit, _ = torch.max((torch.ones_like(label_mask).cuda().float() - label_mask) * logits - 1e4*label_mask, dim=1)
            loss = -F.relu(correct_logit - wrong_logit + self.confidence*(torch.ones_like(wrong_logit).cuda().float()))
            loss = loss.mean()
            aux_net.zero_grad()
            loss.backward()

            x_adv = x.data + step_sign * self.step_size * torch.sign(
                x.grad.data)
            x_adv = torch.min(torch.max(x_adv, inputs - self.epsilon),
                              inputs + self.epsilon)
            x_adv = torch.clamp(x_adv, -1.0, 1.0)
            x = Variable(x_adv)

        if self.train_flag:
            self.basic_net.train()
        else:
            self.basic_net.eval()

        logits_pert = self.basic_net(x.detach())[0]

        return logits_pert, targets_prob.detach(), True, x.detach(
        ), x_org.detach()


class Attack_vertex_pgd(nn.Module):
    # Back-propogate
    def __init__(self, basic_net, config, attack_net=None):
        super(Attack_vertex_pgd, self).__init__()
        self.basic_net = basic_net
        self.attack_net = attack_net
        self.rand = config['random_start']
        self.step_size = config['step_size']#
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        self.train_flag = True if 'train' not in config.keys(
        ) else config['train']

        self.box_type = 'white' if 'box_type' not in config.keys(
        ) else config['box_type']
        self.ls_factor = 0.1 if 'ls_factor' not in config.keys(
        ) else config['ls_factor']
        self.ls_factor_2 = 0.0 if 'ls_factor_2' not in config.keys(
        ) else config['ls_factor_2']

        self.alpha1 = 1.0 if 'alpha1' not in config.keys(
        ) else config['alpha1']
        self.alpha2 = 1.0 if 'alpha2' not in config.keys(
        ) else config['alpha2']
        self.r = 2.0 if 'r' not in config.keys(
        ) else config['r']

        print('Attack_vertex_pgd')
        print(config)


    def forward(self,
                inputs,
                targets,
                attack=True,
                targeted_label=-1,
                batch_idx=0):

        if not attack:
            outputs = self.basic_net(inputs)[0]
            return outputs, None, True, inputs.detach(), inputs.detach()

        if self.box_type == 'white':
            aux_net = pickle.loads(pickle.dumps(self.basic_net))
        elif self.box_type == 'black':
            assert self.attack_net is not None, "should provide an additional net in black-box case"
            aux_net = pickle.loads(pickle.dumps(self.basic_net))
        aux_net.eval()
        logits_pred_nat = aux_net(inputs)[0]
        targets_prob = F.softmax(logits_pred_nat.float(), dim=1)

        num_classes = targets_prob.size(1)

        loss_fun = torch.nn.CrossEntropyLoss(reduction='none')

        outputs = aux_net(inputs)[0]
        targets_prob = F.softmax(outputs.float(), dim=1)
        y_tensor_adv = targets
        step_sign = 1.0

        x = inputs.detach()
        if self.rand:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        x_org = x.detach()
        loss_array = np.zeros((inputs.size(0), self.num_steps))

        for i in range(self.num_steps):
            x.requires_grad_()
            zero_gradients(x)
            if x.grad is not None:
                x.grad.data.fill_(0)
            aux_net.eval()
            logits = aux_net(x)[0]
            loss = loss_fun(logits, y_tensor_adv)
            loss = loss.mean()
            aux_net.zero_grad()
            loss.backward()

            x_adv = x.data + step_sign * self.step_size * torch.sign(
                x.grad.data)
            x_adv = torch.min(torch.max(x_adv, inputs - self.epsilon),
                              inputs + self.epsilon)
            x_adv = torch.clamp(x_adv, -1.0, 1.0)
            x = Variable(x_adv)

        batch_size = x_org.size(0)
        perturb = (x - x_org) * self.r
        x_vertex = x_org + perturb
        # x_vertex = x
        x_vertex = torch.clamp(x_vertex, -1.0, 1.0)
        #y_org = torch.eye(10)[targets].cuda().float()  # cifar10
        y_gt = one_hot_tensor(targets, num_classes, device)
        y_org = utils.label_smoothing(y_gt, y_gt.size(1), self.ls_factor)
        # print(y_org)
        # y_vertex = torch.ones_like(y_org).cuda().float() * (1. / float(10))
        y_vertex = utils.label_smoothing(y_gt, y_gt.size(1), self.ls_factor_2)
        # print(y_vertex)

        # weight = torch.ones(batch_size)
        # weight = distribution.sample(weight)
        weight = torch.from_numpy(np.random.beta(self.alpha1, self.alpha2, [batch_size])).cuda().float()
        x_weight = weight.view(batch_size, 1, 1, 1)
        y_weight = weight.view(batch_size, 1)
        x_vertex = x_org * x_weight + x_vertex * (torch.ones_like(x_weight).cuda().float() - x_weight)
        y_vertex = y_org * y_weight + y_vertex * (torch.ones_like(y_weight).cuda().float() - y_weight)

        x_vertex = Variable(x_vertex)
        y_vertex = Variable(y_vertex)

        if self.train_flag:
            self.basic_net.train()
        else:
            self.basic_net.eval()

        logits_pert = self.basic_net(x_vertex.detach())[0]

        if self.train_flag:
            self.basic_net.train()
        else:
            self.basic_net.eval()

        return logits_pert, targets_prob.detach(), True, x.detach(
        ), x_org.detach(), y_vertex

#'''

class Attack_Lipshitz_reg(nn.Module):
    def __init__(self, basic_net, config, attack_net=None):
        super(Attack_Lipshitz_reg, self).__init__()
        self.basic_net = basic_net
        self.attack_net = attack_net
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        self.train_flag = True if 'train' not in config.keys(
        ) else config['train']
        self.box_type = 'white' if 'box_type' not in config.keys(
        ) else config['box_type']
        self.ls_factor = 0.1 if 'ls_factor' not in config.keys(
        ) else config['ls_factor']
        self.reg_weight = 1.0 if 'reg_weight' not in config.keys(
        ) else config['reg_weight']

        print('Attack_Lipshitz_reg_for_cifar')
        print(config)


    def forward(self,
                inputs,
                targets,
                attack=True,
                targeted_label=-1,
                batch_idx=0):

        if not attack:
            outputs, _ = self.basic_net(inputs)
            return outputs, None, True, inputs.detach(), inputs.detach()

        if self.box_type == 'white':
            aux_net = pickle.loads(pickle.dumps(self.basic_net))
        elif self.box_type == 'black':
            assert self.attack_net is not None, "should provide an additional net in black-box case"
            aux_net = pickle.loads(pickle.dumps(self.basic_net))

        aux_net.eval()
        batch_size = inputs.size(0)
        m = batch_size
        n = batch_size

        logits = aux_net(inputs)[0]
        num_classes = logits.size(1)

        outputs = aux_net(inputs)[0]
        targets_prob = F.softmax(outputs.float(), dim=1)
        y_tensor_adv = targets
        step_sign = 1.0

        x = inputs.detach()

        x_org = x.detach()
        x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)

        if self.train_flag:
            self.basic_net.train()
        else:
            self.basic_net.eval()

        logits_pred_nat, fea_nat = aux_net(inputs)

        num_classes = logits_pred_nat.size(1)
        y_gt = one_hot_tensor(targets, num_classes, device)

        loss_ce = softCrossEntropy()
        loss_fun = torch.nn.CrossEntropyLoss(reduction='none')

        iter_num = self.num_steps

        # y_sm = utils.label_smoothing(y_gt, y_gt.size(1), 0.5)
        #y_sm = utils.label_changing(y_gt, y_gt.size(1), 0.5)

        for i in range(iter_num):
            x.requires_grad_()
            zero_gradients(x)
            if x.grad is not None:
                x.grad.data.fill_(0)
            aux_net.eval()
            logits = aux_net(x)[0]
            # loss = loss_ce(logits, y_sm.detach())
            loss = loss_fun(logits, y_tensor_adv)
            loss = loss.mean()
            aux_net.zero_grad()
            loss.backward()

            x_adv = x.data + step_sign * self.step_size * torch.sign(
                x.grad.data)
            x_adv = torch.min(torch.max(x_adv, inputs - self.epsilon),
                              inputs + self.epsilon)
            x_adv = torch.clamp(x_adv, -1.0, 1.0)
            x = Variable(x_adv)        

        logits_pred, fea = self.basic_net(x)
        self.basic_net.zero_grad()

        y_sm = utils.label_smoothing(y_gt, y_gt.size(1), self.ls_factor)

        adv_loss = loss_ce(logits_pred, y_sm.detach())

        # method  1                

        # delta = (x.detach() - x_org.detach()) * 0.25
        # 
        # desc_diff = torch.norm(self.basic_net(x_org - delta)[0] - logits_pred_nat.detach().data, dim=-1)
        # asc_diff = torch.norm(self.basic_net(x_org + delta)[0] - logits_pred_nat.detach().data, dim=-1)
        # 
        # diff_loss = torch.abs(asc_diff - desc_diff).mean()        
        # 
        # adv_loss += diff_loss * 0.1
        
        

        # method 2
        #print('is it able to train?')

        delta = (x.detach() - x_org.detach()) * 1.0

        # desc_diff = torch.norm(self.basic_net(x_org - delta)[0] - self.basic_net(x_org)[0], dim=-1)
        # asc_diff = torch.norm(self.basic_net(x_org + delta)[0] - self.basic_net(x_org)[0], dim=-1)
        # diff_loss = torch.abs(asc_diff - desc_diff).mean()
        
        # alp
        # alp_loss = torch.norm(self.basic_net(x_org + delta)[0] - self.basic_net(x_org)[0], dim=-1)
        # diff_loss = alp_loss.mean()
        
        #kl
        # criterion_kl = nn.KLDivLoss(size_average=False)
        # kl_loss = criterion_kl(F.log_softmax(self.basic_net(x_org + delta)[0], dim=1),
        #                        F.softmax(self.basic_net(x_org)[0], dim=1))
        # diff_loss = kl_loss.mean()

        # adv_loss += diff_loss * 1.0

        diff_loss = None

        return logits_pred, adv_loss, True, x.detach(), diff_loss
'''

class Attack_Lipshitz_reg(nn.Module):
    def __init__(self, basic_net, config, attack_net=None):
        super(Attack_Lipshitz_reg, self).__init__()
        self.basic_net = basic_net
        self.attack_net = attack_net
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        self.train_flag = True if 'train' not in config.keys(
        ) else config['train']
        self.box_type = 'white' if 'box_type' not in config.keys(
        ) else config['box_type']
        self.ls_factor = 0.1 if 'ls_factor' not in config.keys(
        ) else config['ls_factor']
        self.reg_weight = 1.0 if 'reg_weight' not in config.keys(
        ) else config['reg_weight']

        print('Attack_Lipshitz_reg_for_mnist')
        print(config)
        

    def forward(self,
                inputs,
                targets,
                attack=True,
                targeted_label=-1,
                batch_idx=0):

        if not attack:
            outputs, _ = self.basic_net(inputs)
            return outputs, None, True, inputs.detach(), inputs.detach()

        if self.box_type == 'white':
            aux_net = pickle.loads(pickle.dumps(self.basic_net))
        elif self.box_type == 'black':
            assert self.attack_net is not None, "should provide an additional net in black-box case"
            aux_net = pickle.loads(pickle.dumps(self.basic_net))

        aux_net.eval()
        batch_size = inputs.size(0)
        m = batch_size
        n = batch_size

        logits = aux_net(inputs)[0]
        num_classes = logits.size(1)

        outputs = aux_net(inputs)[0]
        targets_prob = F.softmax(outputs.float(), dim=1)
        y_tensor_adv = targets
        step_sign = 1.0

        x = inputs.detach()

        x_org = x.detach()
        x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)

        if self.train_flag:
            self.basic_net.train()
        else:
            self.basic_net.eval()

        logits_pred_nat, fea_nat = aux_net(inputs)

        num_classes = logits_pred_nat.size(1)
        y_gt = one_hot_tensor(targets, num_classes, device)

        loss_ce = softCrossEntropy()

        iter_num = self.num_steps

        for i in range(iter_num):
            x.requires_grad_()
            zero_gradients(x)
            if x.grad is not None:
                x.grad.data.fill_(0)

            logits_pred, fea = aux_net(x)

            ot_loss = ot.sinkhorn_loss_joint_IPOT(1, 0.00, logits_pred_nat,
                                                  logits_pred, None, None,
                                                  0.01, m, n)
            aux_net.zero_grad()
            adv_loss = ot_loss
            adv_loss.backward()

            x_adv = x.data + self.step_size * torch.sign(x.grad.data)
            x_adv = torch.min(torch.max(x_adv, inputs - self.epsilon),
                              inputs + self.epsilon)
            x_adv = torch.clamp(x_adv, -1.0, 1.0)
            x = Variable(x_adv)

            logits_pred, fea = self.basic_net(x)
            self.basic_net.zero_grad()

        y_sm = utils.label_smoothing(y_gt, y_gt.size(1), self.ls_factor)


        adv_loss = loss_ce(logits_pred, y_sm.detach())

        # method 1

        # delta = (x.detach() - x_org.detach()) * 0.25
        #
        # logits_nat = self.basic_net(x_org.detach())[0]
        #
        # desc_diff = torch.norm(self.basic_net(x_org - delta)[0] - logits_nat.detach().data, dim=-1)
        # asc_diff = torch.norm(self.basic_net(x_org + delta)[0] - logits_nat.detach().data, dim=-1)
        #
        # diff_loss = torch.abs(asc_diff - desc_diff).mean()
        #
        # adv_loss += diff_loss * 0.01

        # method 2

        delta = (x.detach() - x_org.detach()) * 1.0

        desc_diff = torch.norm(self.basic_net(x_org - delta)[0] - self.basic_net(x_org.detach())[0], dim=-1)
        asc_diff = torch.norm(self.basic_net(x_org + delta)[0] - self.basic_net(x_org.detach())[0], dim=-1)

        diff_loss = torch.abs(asc_diff - desc_diff).mean()

        adv_loss += diff_loss * 0.001

        return logits_pred, adv_loss, True, x.detach(), x_org.detach()
'''




class Attack_FeaScatter(nn.Module):
    def __init__(self, basic_net, config, attack_net=None):
        super(Attack_FeaScatter, self).__init__()
        self.basic_net = basic_net
        self.attack_net = attack_net
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        self.train_flag = True if 'train' not in config.keys(
        ) else config['train']
        self.box_type = 'white' if 'box_type' not in config.keys(
        ) else config['box_type']
        self.ls_factor = 0.1 if 'ls_factor' not in config.keys(
        ) else config['ls_factor']

        print('Attack_FeaScatter')
        print(config)


    def forward(self,
                inputs,
                targets,
                attack=True,
                targeted_label=-1,
                batch_idx=0):

        if not attack:
            outputs, _ = self.basic_net(inputs)
            return outputs, None, True, inputs.detach(), inputs.detach()

        if self.box_type == 'white':
            aux_net = pickle.loads(pickle.dumps(self.basic_net))
        elif self.box_type == 'black':
            assert self.attack_net is not None, "should provide an additional net in black-box case"
            aux_net = pickle.loads(pickle.dumps(self.basic_net))

        aux_net.eval()
        batch_size = inputs.size(0)
        m = batch_size
        n = batch_size

        logits = aux_net(inputs)[0]
        num_classes = logits.size(1)

        outputs = aux_net(inputs)[0]
        targets_prob = F.softmax(outputs.float(), dim=1)
        y_tensor_adv = targets
        step_sign = 1.0

        x = inputs.detach()

        x_org = x.detach()
        x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)

        if self.train_flag:
            self.basic_net.train()
        else:
            self.basic_net.eval()

        logits_pred_nat, fea_nat = aux_net(inputs)

        num_classes = logits_pred_nat.size(1)
        y_gt = one_hot_tensor(targets, num_classes, device)

        loss_ce = softCrossEntropy()

        iter_num = self.num_steps

        for i in range(iter_num):
            x.requires_grad_()
            zero_gradients(x)
            if x.grad is not None:
                x.grad.data.fill_(0)

            logits_pred, fea = aux_net(x)

            #ot_loss = ot.sinkhorn_loss_joint_IPOT(1, 0.00, logits_pred_nat,
            #                                      logits_pred, None, None,
            #                                      0.01, m, n)

            ot_loss = ot.sinkhorn_loss_joint_IPOT(1, 0.00, logits_pred_nat,
                                                  logits_pred, None, None,
                                                  0.01, m, n)

            aux_net.zero_grad()
            adv_loss = ot_loss
            adv_loss.backward(retain_graph=True)
            x_adv = x.data + self.step_size * torch.sign(x.grad.data)
            x_adv = torch.min(torch.max(x_adv, inputs - self.epsilon),
                              inputs + self.epsilon)
            x_adv = torch.clamp(x_adv, -1.0, 1.0)
            x = Variable(x_adv)

            logits_pred, fea = self.basic_net(x)
            self.basic_net.zero_grad()

            y_sm = utils.label_smoothing(y_gt, y_gt.size(1), self.ls_factor)
            #print(y_sm.detach())

            adv_loss = loss_ce(logits_pred, y_sm.detach())

        ## label penalty added ##

        # label_penalty = loss_ce(logits_pred, y_gt)
        # adv_loss -= 0.1 * label_penalty

        ## clean loss added ##

        # clean_loss = loss_ce(logits_pred_nat, y_sm)
        # adv_loss = 1.0 * adv_loss + 1.0 * clean_loss

        return logits_pred, adv_loss, True, x.detach(), x_org.detach()

class Attack_FeaScatter_test(nn.Module):
    def __init__(self, basic_net, config, attack_net=None):
        super(Attack_FeaScatter_test, self).__init__()
        self.basic_net = basic_net
        self.attack_net = attack_net
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        self.train_flag = True if 'train' not in config.keys(
        ) else config['train']
        self.box_type = 'white' if 'box_type' not in config.keys(
        ) else config['box_type']
        self.ls_factor = 0.1 if 'ls_factor' not in config.keys(
        ) else config['ls_factor']

        print('Attack_FeaScatter_test')
        print(config)


    def forward(self,
                inputs,
                targets,
                attack=True,
                targeted_label=-1,
                batch_idx=0):

        if not attack:
            outputs, _ = self.basic_net(inputs)
            return outputs, None, True, inputs.detach(), inputs.detach()

        if self.box_type == 'white':
            aux_net = pickle.loads(pickle.dumps(self.basic_net))
        elif self.box_type == 'black':
            assert self.attack_net is not None, "should provide an additional net in black-box case"
            aux_net = pickle.loads(pickle.dumps(self.basic_net))

        aux_net.eval()
        batch_size = inputs.size(0)
        m = batch_size
        n = batch_size

        logits = aux_net(inputs)[0]
        num_classes = logits.size(1)

        outputs = aux_net(inputs)[0]
        targets_prob = F.softmax(outputs.float(), dim=1)
        y_tensor_adv = targets
        step_sign = 1.0

        x = inputs.detach()

        x_org = x.detach()
        x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)

        if self.train_flag:
            self.basic_net.train()
        else:
            self.basic_net.eval()

        logits_pred_nat, fea_nat = aux_net(inputs)

        num_classes = logits_pred_nat.size(1)
        y_gt = one_hot_tensor(targets, num_classes, device)

        loss_ce = softCrossEntropy()

        iter_num = self.num_steps

        for i in range(iter_num):
            x.requires_grad_()
            zero_gradients(x)
            if x.grad is not None:
                x.grad.data.fill_(0)

            logits_pred, fea = aux_net(x)

            #ot_loss = ot.sinkhorn_loss_joint_IPOT(1, 0.00, logits_pred_nat,
            #                                      logits_pred, None, None,
            #                                      0.01, m, n)

            ot_loss = ot.sinkhorn_loss_joint_IPOT(1, 0.00, logits_pred_nat,
                                                  logits_pred, None, None,
                                                  0.01, m, n)

            aux_net.zero_grad()
            adv_loss = ot_loss
            adv_loss.backward(retain_graph=True)
            x_adv = x.data + self.step_size * torch.sign(x.grad.data)
            x_adv = torch.min(torch.max(x_adv, inputs - self.epsilon),
                              inputs + self.epsilon)
            x_adv = torch.clamp(x_adv, -1.0, 1.0)
            x = Variable(x_adv)

            logits_pred, fea = self.basic_net(x)
            self.basic_net.zero_grad()

            y_sm = utils.label_smoothing(y_gt, y_gt.size(1), self.ls_factor)
            #print(y_sm.detach())

            adv_loss = loss_ce(logits_pred, y_sm.detach())

        ## label penalty added ##

        # label_penalty = loss_ce(logits_pred, y_gt)
        # adv_loss -= 0.1 * label_penalty

        ## clean loss added ##

        # clean_loss = loss_ce(logits_pred_nat, y_sm)
        # adv_loss = 6.0 * adv_loss + 1.0 * clean_loss

        return logits_pred, targets_prob.detach(), True, x.detach(
        ), x_org.detach()

        # return logits_pred, adv_loss, True, x.detach(), x_org.detach()

class Attack_vertex(nn.Module):
    def __init__(self, basic_net, config, attack_net=None):
        super(Attack_vertex, self).__init__()
        self.basic_net = basic_net
        self.attack_net = attack_net
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        self.train_flag = True if 'train' not in config.keys(
        ) else config['train']
        self.box_type = 'white' if 'box_type' not in config.keys(
        ) else config['box_type']
        self.ls_factor = 0.1 if 'ls_factor' not in config.keys(
        ) else config['ls_factor']
        self.ls_factor_2 = 0.0 if 'ls_factor_2' not in config.keys(
        ) else config['ls_factor_2']

        self.alpha1 = 1.0 if 'alpha1' not in config.keys(
        ) else config['alpha1']
        self.alpha2 = 1.0 if 'alpha2' not in config.keys(
        ) else config['alpha2']
        self.r = 2.0 if 'r' not in config.keys(
        ) else config['r']

        print('Attack_vertex')
        print(config)


    def forward(self,
                inputs,
                targets,
                attack=True,
                targeted_label=-1,
                batch_idx=0,
                epoch = None
                ):

        if not attack:
            outputs, _ = self.basic_net(inputs)
            return outputs, None, True, inputs.detach(), inputs.detach()

        if self.box_type == 'white':
            aux_net = pickle.loads(pickle.dumps(self.basic_net))
        elif self.box_type == 'black':
            assert self.attack_net is not None, "should provide an additional net in black-box case"
            aux_net = pickle.loads(pickle.dumps(self.basic_net))

        aux_net.eval()
        batch_size = inputs.size(0)
        m = batch_size
        n = batch_size

        logits = aux_net(inputs)[0]
        num_classes = logits.size(1)

        outputs = aux_net(inputs)[0]
        targets_prob = F.softmax(outputs.float(), dim=1)
        y_tensor_adv = targets
        step_sign = 1.0

        x = inputs.detach()

        x_org = x.detach()
        x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)

        if self.train_flag:
            self.basic_net.train()
        else:
            self.basic_net.eval()

        logits_pred_nat, fea_nat = aux_net(inputs)

        num_classes = logits_pred_nat.size(1)
        y_gt = one_hot_tensor(targets, num_classes, device)

        loss_ce = softCrossEntropy()

        iter_num = self.num_steps

        for i in range(iter_num):
            x.requires_grad_()
            zero_gradients(x)
            if x.grad is not None:
                x.grad.data.fill_(0)

            logits_pred, fea = aux_net(x)

            ot_loss = ot.sinkhorn_loss_joint_IPOT(1, 0.00, logits_pred_nat,
                                                  logits_pred, None, None,
                                                  0.01, m, n)

            aux_net.zero_grad()
            adv_loss = ot_loss
            adv_loss.backward(retain_graph=True)
            x_adv = x.data + self.step_size * torch.sign(x.grad.data)
            x_adv = torch.min(torch.max(x_adv, inputs - self.epsilon),
                              inputs + self.epsilon)
            x_adv = torch.clamp(x_adv, -1.0, 1.0)
            x = Variable(x_adv)

            #logits_pred, fea = self.basic_net(x)
            #self.basic_net.zero_grad()

        batch_size = x_org.size(0)
        # if epoch is None:
        #     perturb = (x - x_org) * self.r
        # else:
        #     perturb = (x - x_org) * self.r * epoch
        # print(self.r * epoch)
        perturb = (x - x_org) * self.r
        # perturb = (x - x_org) * 0.5
        x_vertex = x_org + perturb
        # perturb = (x - x_org) * 0.5
        # x = x_org + perturb

        # x_vertex = x
        x_vertex = torch.clamp(x_vertex, -1.0, 1.0)

        #y_org = torch.eye(10)[targets].cuda().float()  # cifar10
        y_gt = one_hot_tensor(targets, num_classes, device)
        y_org = utils.label_smoothing(y_gt, y_gt.size(1), self.ls_factor)
        # print(y_org)
        # y_vertex = torch.ones_like(y_org).cuda().float() * (1. / float(10))
        y_vertex = utils.label_smoothing(y_gt, y_gt.size(1), self.ls_factor_2)
        # print(y_vertex)

        # weight = torch.ones(batch_size)
        # weight = distribution.sample(weight)
        weight = torch.from_numpy(np.random.beta(self.alpha1, self.alpha2, [batch_size])).cuda().float()
        x_weight = weight.view(batch_size, 1, 1, 1)
        y_weight = weight.view(batch_size, 1)
        x_vertex = x_org * x_weight + x_vertex * (torch.ones_like(x_weight).cuda().float() - x_weight)
        # x_vertex = x * x_weight + x_vertex * (torch.ones_like(x_weight).cuda().float() - x_weight)
        y_vertex = y_org * y_weight + y_vertex * (torch.ones_like(y_weight).cuda().float() - y_weight)

        x_vertex = Variable(x_vertex)
        y_vertex = Variable(y_vertex)

        if self.train_flag:
            self.basic_net.train()
        else:
            self.basic_net.eval()

        logits_pert = self.basic_net(x_vertex.detach())[0]

        return logits_pert, targets_prob.detach(), True, x.detach(
        ), x_org.detach(), y_vertex

        #return logits_pred, adv_loss, True, x.detach(), x_org.detach()