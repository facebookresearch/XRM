# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torch.nn.functional import cross_entropy, softmax, one_hot
import torch.autograd as autograd


def get_algorithm(hparams, net, optim):

    # 'RWG' and 'SUBG' are both ERM but they differ in how they balance batches
    if hparams['algorithm_name'] in ['ERM', 'RWG', 'SUBG']:
        return ERM(hparams, net, optim)
    elif hparams['algorithm_name'] == 'GroupDRO':
        return GroupDRO(hparams, net, optim)
    elif hparams['algorithm_name'] == 'IRM':
        return IRM(hparams, net, optim)
    elif hparams['algorithm_name'] == 'LC':
        return LC(hparams, net, optim)


class ERM:
    def __init__(self, hparams, net, optim):
        self.device = hparams["device"]
        self.hparams = hparams
        self.net = net.to(self.device)
        self.optim = optim

    def get_loss(self, y_hat, y, m=None):
        return cross_entropy(y_hat, y.view(-1).long())

    def update(self, batch):
        _, x, y, m = batch
        x = x.to(self.device)
        y = y.to(self.device)
        m = m.to(self.device)
        self.optim.zero_grad(set_to_none=True)
        loss = self.get_loss(self.net(x), y, m)
        loss.backward()
        self.optim.step()
        if self.optim.lr_scheduler is not None:
            self.optim.lr_scheduler.step()

    def predict(self, x):
        return self.net(x)

    def evaluate(self, loader):
        self.net.eval()
        i_s = []
        ys = []
        y_hats = []
        ms = []
        with torch.no_grad():
            for batch in loader:
                i, x, y, m = batch
                x = x.to(self.device)
                i_s += [i]
                ys += [y.to(self.device)]
                y_hats += [self.predict(x)]
                ms += [m.to(self.device)]
        i_s = torch.cat(i_s)
        sorted_indices = torch.argsort(i_s)
        ys = torch.cat(ys)[sorted_indices].view(-1)
        y_hats = torch.cat(y_hats)[sorted_indices]
        ms = torch.cat(ms)[sorted_indices].view(-1)
        self.net.train()
        return ys, y_hats, ms


class GroupDRO(ERM):
    def __init__(self, hparams, net, optim):
        super(GroupDRO, self).__init__(hparams, net, optim)
        self.eta = hparams['eta']
        self.q = torch.ones(hparams['num_y'] * hparams['num_m'])

    def get_loss(self, y_hat, y, m=None):
        grp_losses = torch.zeros(len(self.q))
        g = (self.hparams['num_m'] * y + m).view(-1)
        losses = cross_entropy(y_hat, y.view(-1).long(), reduction='none')
        for i in g.unique().int():
            grp_losses[i] = losses[g == i].mean()
            self.q[i] *= (self.eta * grp_losses[i]).exp().item()

        self.q /= self.q.sum()
        return (self.q * grp_losses).sum()


class LC(ERM):

    def predict(self, x):
        out = self.net(x)
        if self.net.training:
            return out
        return out[..., :out.size(-1)//2]

    def gce(self, logits, target, tau=0.8):
        probs = softmax(logits, dim=1)
        pred = probs.argmax(1)
        target_one_hot = one_hot(target, num_classes=logits.size(1)).float()
        loss = (1 - (probs ** tau) * target_one_hot).sum(dim=1) / tau
        
        return sum([loss[pred.eq(t)].mean() for t in target.unique()])

    def get_m_y_prior(self, y, m_prob=None):
        g = 2 * y + m_prob.argmax(1)
        m_prob_at_y = m_prob[range(len(y)), y][:, None]
        m_y_prior = torch.tensor([m_prob_at_y[g.eq(g_i)].mean() for g_i in range(4)])
        m_y_prior = torch.nan_to_num(m_y_prior)
        m_y_prior = m_y_prior.reshape(2, 2).T.cuda()
        m_y_prior = m_y_prior / (m_y_prior.sum(1, keepdim=True) + 1e-4)
        return m_y_prior

    def get_loss(self, out, y, m=None):
        y_hat = out[..., :out.size(-1)//2]

        loss = 0
        if self.hparams['use_true_m']:
            m_prob = one_hot(m).float()
        else:
            m_hat = out[..., out.size(-1)//2:]
            m_prob = softmax(m_hat.detach() / self.hparams['temp'], dim=1)
            loss += 0.1 * self.gce(m_hat, y)

        m_y_prior = self.get_m_y_prior(y, m_prob)
        m_pred = m_prob.argmax(1)
        p_y_given_m = m_y_prior[m_pred]
        shift = torch.log(p_y_given_m + 1e-4)

        loss += cross_entropy(y_hat + shift, y)

        return loss


class XRM(ERM):
    def __init__(self, hparams, twin_nets, optim, tr_loader, assigns):
        self.device = hparams["device"]
        self.hparams = hparams
        self.net = twin_nets.to(self.device)
        self.optim = optim
        self.tr_loader = tr_loader
        self.y_tr_dynamic = tr_loader.y.clone()
        self.assigns = assigns.to(self.device)

    def get_loss(self, y_hat, y, m=None):
        losses = cross_entropy(y_hat, y.view(-1).long(), reduction='none')
        return sum([losses[y == yi].mean() for yi in y.unique()])

    def flip_y(self, i, pred_ho):
        p_ho, y_ho = pred_ho.softmax(dim=1).detach().max(1)
        p_ho, y_ho = p_ho.cpu(), y_ho.cpu()
        num_y = self.hparams['num_y']

        flip = torch.bernoulli((p_ho - 1 / num_y) * num_y / (num_y - 1)).long()
        self.y_tr_dynamic[i] = flip * y_ho + (1 - flip) * self.y_tr_dynamic[i]

    def update(self, batch):
        i, x, _, _ = batch
        x = x.to(self.device)
        y = self.y_tr_dynamic[i].to(self.device)

        pred_ab = self.net(x)
        pred_a = pred_ab[..., 0]
        pred_b = pred_ab[..., 1]
        pred_hi = pred_a * self.assigns[i] + pred_b * (1 - self.assigns[i])
        pred_ho = pred_a * (1 - self.assigns[i]) + pred_b * self.assigns[i]

        self.optim.zero_grad()
        loss = self.get_loss(pred_hi, y)
        loss.backward()
        self.optim.step()
        if self.optim.lr_scheduler is not None:
            self.optim.lr_scheduler.step()
        self.flip_y(i, pred_ho)

    def cross_mistakes(self, loader):
        ys, pred_ab, _ = self.evaluate(loader)
        pred_a = pred_ab[..., 0]
        pred_b = pred_ab[..., 1]
        return torch.logical_or(
            pred_a.argmax(1).ne(ys),
            pred_b.argmax(1).ne(ys)).long().cpu(), ys


class IRM(ERM):
    def __init__(self, hparams, net, optim):
        super(IRM, self).__init__(hparams, net, optim)
        self.update_count = torch.tensor([0])

    @staticmethod
    def _irm_penalty(logits, y):
        device = "cuda" if logits[0][0].is_cuda else "cpu"
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss_1 = cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def get_loss(self, y_hat, y, m=None):
        penalty_weight = (self.hparams['lambda'] if self.update_count
                          >= self.hparams['penalty_anneal_iters'] else 1.0)
        nll = 0.
        penalty = 0.

        for i in m.unique().int():
            nll += cross_entropy(y_hat[m == i], y[m == i])
            penalty += self._irm_penalty(y_hat[m == i], y[m == i])
        nll /= len(m.unique())
        penalty /= len(m.unique())
        loss = nll + (penalty_weight * penalty)

        self.update_count += 1
        return loss
