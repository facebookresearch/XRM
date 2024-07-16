# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from tqdm import tqdm
import torch
torch.manual_seed(1234)

 
def groups(x, y, m):
    g = len(m.unique()) * y + m
    return [torch.where(g == gi)[0] for gi in g.unique()]
 
def get_acc(net, x, y):
    return net(x).argmax(1).eq(y.view(-1)).float().mean().item()
 
def worst_acc(net, x, y, m):
    return min(get_acc(net, x[g], y[g]) for g in groups(x, y, m))
 
def balanced_cel(p, y):
    losses = torch.nn.functional.cross_entropy(p, y, reduction="none")
    return sum([losses[y == yi].mean() for yi in y.unique()])
 
def build_net(n_in, n_out):
    net = torch.nn.Linear(n_in, n_out, bias=False)
    net.weight.data *= 0.0
    opt = torch.optim.SGD(net.parameters(), lr=1e-2)
    return net, opt
 
def run_dro(x_tr, y_tr, c_tr, x_va, y_va, c_va, x_te, y_te, m_te, n_iters=3000):
    net, opt = build_net(x_tr.size(1), len(y_tr.unique()))
    net = net.cuda()
    loss = torch.nn.CrossEntropyLoss(reduction="none")
 
    g = (len(c_tr.unique()) * y_tr + c_tr).view(-1)
    q = torch.ones(len(c_tr.unique()) * len(y_tr.unique())).cuda()
 
    best_wga = 0
    for k in tqdm(range(n_iters)):
        losses = loss(net(x_tr), y_tr)
        grp_losses = torch.zeros(len(q)).cuda()
        for i in g.unique().int():
            grp_losses[i] = losses[g == i].mean()
            q[i] *= (0.1 * grp_losses[i]).exp().item()
 
        q /= q.sum()
        opt.zero_grad()
        (q * grp_losses).sum().backward()
        opt.step()
 
        wga_va = worst_acc(net, x_va, y_va, c_va)
        if wga_va >= best_wga:
            best_wga = wga_va
            test_wga = worst_acc(net, x_te, y_te, m_te)
 
    return test_wga
 
def run_xrm(x_tr, y_tr, x_va, y_va, n_iters=3000):
    y_tr_clone, nc = y_tr.clone(), len(y_tr.unique())
    net_a, opt_a = build_net(x_tr.size(1), nc)
    net_b, opt_b = build_net(x_tr.size(1), nc)
    ind_a = torch.zeros(len(x_tr), 1).bernoulli_(0.5).long().cuda()
    net_a = net_a.cuda()
    net_b = net_b.cuda()
 
    for iteration in tqdm(range(n_iters)):
        pred_a = net_a(x_tr)
        pred_b = net_b(x_tr)
        pred_hi = pred_a * ind_a + pred_b * (1 - ind_a)
        pred_ho = pred_a * (1 - ind_a) + pred_b * ind_a
 
        opt_a.zero_grad()
        opt_b.zero_grad()
        balanced_cel(pred_hi, y_tr).backward()
        opt_a.step()
        opt_b.step()
 
        p_ho, y_ho = pred_ho.softmax(dim=1).detach().max(1)
        is_flip = torch.bernoulli((p_ho - 1 / nc) * nc / (nc - 1)).long()
        y_tr = is_flip * y_ho + (1 - is_flip) * y_tr

    def cm(x, y):
        return torch.logical_or(
           net_a(x).argmax(1).ne(y),
           net_b(x).argmax(1).ne(y)).long().detach()

    return cm(x_tr, y_tr_clone), cm(x_va, y_va)
 
def load_wb(pt_name="./data/waterbirds/features.pt"):
    pt = torch.load(pt_name)
    return [
        pt["tr"]["x"].cuda(), pt["tr"]["y"].squeeze().long().cuda(), pt["tr"]["m"].squeeze().long().cuda(), # float-2d, long-1d, long-1d
        pt["va"]["x"].cuda(), pt["va"]["y"].squeeze().long().cuda(), pt["va"]["m"].squeeze().long().cuda(),
        pt["te"]["x"].cuda(), pt["te"]["y"].squeeze().long().cuda(), pt["te"]["m"].squeeze().long().cuda()]
 
if __name__ == "__main__":
    torch.manual_seed(0)
    x_tr, y_tr, m_tr, x_va, y_va, m_va, x_te, y_te, m_te = load_wb()
    m_hat_tr, m_hat_va = run_xrm(x_tr, y_tr, x_va, y_va)

    for title, (m_tr_, m_va_) in [
            ['no group label (class label is used instead)', (y_tr, y_va)],
            ['human-annotated group labels', (m_tr, m_va)],
            ['XRM-inferred group labels', (m_hat_tr, m_hat_va)]]:

        test_wga = run_dro(x_tr, y_tr, m_tr_, x_va, y_va, m_va_, x_te, y_te, m_te)
        print(f'{title} -> worst-group-acc: {test_wga:.2f}')
