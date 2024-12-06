# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import glob
import re
import random
import numpy as np
import json
import torch


def set_all_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def prepare_out_dir(hparams):
    os.makedirs(hparams['out_dir'], exist_ok=True)
    if not hparams['resume'] or not os.path.exists(hparams['results_file']):
        with open(hparams['results_file'], 'w') as f:
            json.dump(hparams, f)
    print(hparams)


def read_results_file(file):
    results = {}
    with open(file, "r") as file:
        for line in file:
            res = json.loads(line)
            if 'step' in res.keys() and res['step'] > 0:
                for key, val in res.items():
                    results.setdefault(key, []).append(val)
    return results


def get_star_value_from_file_name(pattern, file_name):
    return int(re.match(pattern.replace('*', r'(\d+)'), file_name).group(1))


def load_checkpoint(hparams, algorithm):
    if os.path.exists(hparams['ckpt_file']):
        ckpt = torch.load(hparams['ckpt_file'])
        algorithm.net.load_state_dict(ckpt['net_state_dict'])
        algorithm.optim.load_state_dict(ckpt['optim_state_dict'])
        if 'lr_scheduler' in ckpt.keys():
            algorithm.optim.lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        if 'XRM' in str(algorithm):
            algorithm.y_tr_dynamic = ckpt['y_tr_dynamic']
            # assigns should be the same if seed is set correctly
            assert (algorithm.assigns == ckpt['assigns']).all()

        return ckpt['last_step'] + 1
    return 0


def save_checkpoint(hparams, algorithm, step):
    to_save = {'net_state_dict': algorithm.net.state_dict(),
               'optim_state_dict': algorithm.optim.state_dict(),
               'last_step': step}
    if algorithm.optim.lr_scheduler is not None:
        to_save['lr_scheduler'] = algorithm.optim.lr_scheduler.state_dict()
    if 'XRM' in str(algorithm):
        to_save['y_tr_dynamic'] = algorithm.y_tr_dynamic
        to_save['assigns'] = algorithm.assigns
    torch.save(to_save, hparams['ckpt_file'])


def report_stats(hparams, algorithm, loaders, step):
    stats = {'step': step}
    if hparams['phase'] == 1:
        if hasattr(algorithm, 'degenerate'):
            # the degenerate case is when a whole class is flipped
            stats['flip_rate'] = 0
        else:
            true_y = algorithm.tr_loader.y
            curr_y = algorithm.y_tr_dynamic
            stats['flip_rate'] = curr_y.ne(true_y).float().mean().item()
    else:
        for split in ['va', 'te']:
            ys, y_hats, ms = algorithm.evaluate(loaders[split])
            metrics = get_metrics(hparams, ys, y_hats, ms)
            for metric, value in metrics.items():
                stats[f'{split}_{metric}'] = value
            if split == 'va' and hasattr(algorithm, 'va_m_hat'):
                stats['va_gi_wga'] = get_metrics(
                    hparams, ys, y_hats, algorithm.va_m_hat.cuda())['wga']
    with open(hparams['results_file'], 'a') as f:
        f.write('\n')
        json.dump(stats, f)

    # printing
    stats_strs = []
    for key, val in stats.items():
        if isinstance(val, float):
            stats_strs.append(f"{key}: {val:.3f}")
        else:
            stats_strs.append(f"{key}: {val}")
    print(', '.join(stats_strs))


def get_metrics(hparams, ys, y_hats, ms):
    gs = (hparams['num_m'] * ys + ms).view(-1)
    avg_acc = y_hats.argmax(1).eq(ys).float().mean().item()
    wga = min([y_hats.argmax(1).eq(ys)[gs == g_i].float().mean().item()
               for g_i in torch.unique(gs)])
    return {'avg_acc': avg_acc, 'wga': wga}


class Iter:
    def __init__(self, loader):
        self.loader = loader
        self.iterator = iter(self.loader)

    def next(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            batch = next(self.iterator)
        return batch


def process_json_files(pattern, json_files, is_phase_1, selection_criterion=None):
    all_values = {
        'hp_comb': [],
        'flip_rate': [],
        'va_wga': [],
        'va_gi_wga': [],  # gi: group-inferred
        'te_wga': [],
        'va_avg_acc': [],
        'te_avg_acc': []
    }

    for file in json_files:
        results = read_results_file(file)
        if len(results) == 0:
            continue
        all_values['hp_comb'].append(
            get_star_value_from_file_name(pattern, file))

        # when phase==1: finds best hp_comb at 'end' step (no early stopping needed)
        # when phase==2: finds best hp_comb at 'best' step (early stopping)
        if is_phase_1:
            # remove_degenerate(results)
            all_values['flip_rate'].append(results['flip_rate'][-1])
        else:
            # index of the last occurrence of max value
            ind = (len(results[selection_criterion]) -
                   np.argmax(results[selection_criterion][::-1]) - 1)
            all_values['va_wga'].append(results['va_wga'][ind])
            if 'va_gi_wga' in results.keys():
                all_values['va_gi_wga'].append(results['va_gi_wga'][ind])
            all_values['te_wga'].append(results['te_wga'][ind])
            all_values['va_avg_acc'].append(results['va_avg_acc'][ind])
            all_values['te_avg_acc'].append(results['te_avg_acc'][ind])

    if len(all_values['va_gi_wga']) == 0:
        all_values.pop('va_gi_wga')
    return all_values


def is_degenerate(algorithm, inferred_groups, ys):
    for split in ['tr', 'va']:
        y = ys[split].cpu()
        g = 2 * y + inferred_groups[split]
        # each class y should be grouped into 2 envs
        # if not, the flip_rate is reported 0
        if len(g.unique()) != 2 * len(y.unique()):
            algorithm.degenerate = True
            return True
    return False


def sort_and_remove_empty(all_values):
    inds = np.argsort(all_values['hp_comb'])
    to_remove = []

    for key in all_values.keys():
        if len(all_values[key]) == len(inds):
            all_values[key] = np.array(all_values[key])[inds]
        else:
            assert len(all_values[key]) == 0
            to_remove.append(key)  # remove empty

    for key in to_remove:
        all_values.pop(key)

    return all_values


def find_best_hparams_comb(dir_, selection_metric, seed):
    # when phase==1: finds best hp_comb at 'end' step (no early stopping needed)
    # when phase==2: finds best hp_comb at 'best' step (early stopping)
    is_phase_1 = 'phase_1' in dir_
    pattern = os.path.join(dir_, f'results_hpcomb_*_seed{seed}.json')
    json_files = glob.glob(pattern)
    assert len(json_files) > 0
    all_values = process_json_files(
        pattern, json_files, is_phase_1, selection_metric)
    all_values = sort_and_remove_empty(all_values)

    if is_phase_1:
        criterion_values = all_values[selection_metric]
        best_hparams_comb = int(all_values['hp_comb'][
            np.argmax(criterion_values)])
    else:
        criterion_values = all_values[selection_metric]
        best_indices = np.argwhere(
            criterion_values == np.max(criterion_values)).flatten()
        # If there are multiple indices with the same
        # maximum value, use 'va_avg_acc' to break the tie
        if len(best_indices) > 1:
            best_index = best_indices[
                np.argmax(all_values['va_avg_acc'][best_indices])]
        else:
            best_index = best_indices[0]
        best_hparams_comb = int(all_values['hp_comb'][best_index])
    return best_hparams_comb
