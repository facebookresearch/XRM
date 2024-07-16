# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torchvision
from transformers import BertForSequenceClassification, AdamW
from torch.optim.lr_scheduler import LambdaLR


def get_network(hparams):

    if hparams['net_type'] == 'mlp':
        lin1 = torch.nn.Linear(2 * 14 * 14, 390)
        lin2 = torch.nn.Linear(390, 390)
        lin3 = torch.nn.Linear(390, 2)
        lin3.bias.data *= 0
        lin3.weight.data *= 0
        for lin in [lin1, lin2, lin3]:
            torch.nn.init.xavier_uniform_(lin.weight)
            torch.nn.init.zeros_(lin.bias)
        net = torch.nn.Sequential(
            lin1, torch.nn.ReLU(True),
            lin2, torch.nn.ReLU(True),
            lin3)

    elif hparams['net_type'] == 'resnet':

        net = torchvision.models.resnet.resnet50(pretrained=True)
        fc = torch.nn.Linear(net.fc.in_features, hparams['num_y'])
        fc.bias.data *= 0
        fc.weight.data *= 0
        if hparams['precompute_features']:
            net = fc
        else:
            net.fc = fc

    elif hparams['net_type'] == 'bert':

        net = BertWrapper(
            BertForSequenceClassification.from_pretrained(
                'bert-base-uncased', num_labels=hparams['num_y']))
        net.zero_grad()
        net.net.classifier.bias.data *= 0.0
        net.net.classifier.weight.data *= 0.0

        if hparams['quick_run']:
            fc = torch.nn.Linear(
                net.net.classifier.in_features, hparams['num_y'])
            fc.bias.data *= 0
            fc.weight.data *= 0
            net = fc

    return net


def get_optim(hparams, net):

    if hparams['net_type'] == 'mlp':

        opt = torch.optim.Adam(
            net.parameters(),
            hparams['lr'],
            weight_decay=hparams['weight_decay'])
        opt.lr_scheduler = None
        return opt

    if hparams['net_type'] == 'resnet':

        opt = torch.optim.SGD(
            net.parameters(),
            hparams['lr'],
            momentum=0.9,
            weight_decay=hparams['weight_decay'])
        opt.lr_scheduler = None
        return opt

    if hparams['net_type'] == 'bert':

        no_decay = ["bias", "LayerNorm.weight"]
        decay_params = []
        nodecay_params = []
        for n, p in net.named_parameters():
            if any(nd in n for nd in no_decay):
                decay_params.append(p)
            else:
                nodecay_params.append(p)

        opt_grouped_parameters = [
            {
                "params": decay_params,
                "weight_decay": hparams['weight_decay'],
            },
            {
                "params": nodecay_params,
                "weight_decay": 0.0,
            },
        ]
        opt = AdamW(
            opt_grouped_parameters,
            lr=hparams['lr'],
            eps=1e-8)

        def lr_lambda(current_step):
            warmup = hparams["num_step"] // 3 + 1
            tot = warmup + hparams["num_step"]
            if current_step < warmup:
                return 1.0 - current_step / warmup
            else:
                return 1.0 - (current_step - warmup) / (tot - warmup)
        opt.lr_scheduler = LambdaLR(opt, lr_lambda)

        return opt


class BertWrapper(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(
            input_ids=x[:, :, 0],
            attention_mask=x[:, :, 1],
            token_type_ids=x[:, :, 2]).logits


class TwinNets(torch.nn.Module):
    def __init__(self, hparams, net_a, net_b):
        super().__init__()
        self.device = hparams["device"]
        self.hparams = hparams
        self.net_a = net_a
        self.net_b = net_b

    def forward(self, x):
        return torch.cat([self.net_a(x)[..., None],
                          self.net_b(x)[..., None]], -1)
