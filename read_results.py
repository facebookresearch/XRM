# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import glob
import argparse
import numpy as np
from utils import process_json_files, sort_and_remove_empty, find_best_hparams_comb


def read_phase_1(args, pattern, json_files):

    all_values = process_json_files(
        pattern, json_files, True, args.selection_criterion)
    all_values = sort_and_remove_empty(all_values)

    best_hp_comb = find_best_hparams_comb(
        dir_path, args.selection_criterion, seed=0)

    for hp_comb, flip_rate in zip(all_values['hp_comb'],
                                  all_values['flip_rate']):
        suffix = " (best)" if hp_comb == best_hp_comb else ""
        print(f"hp_comb: {hp_comb}, flip_rate: {flip_rate:.3f}"
              f"{suffix}")

    pattern = os.path.join(
        dir_path, f'results_hpcomb_{best_hp_comb}_seed*.json')
    json_files = glob.glob(pattern)
    assert len(json_files) > 0

    all_values = process_json_files(
        pattern, json_files, True, args.selection_criterion)

    print(f'\nAveraged over {len(all_values["flip_rate"])} seeds:')
    print(f"hp_comb: {best_hp_comb}, "
          f"flip_rate: {np.mean(all_values['flip_rate']):.2f} "
          f"(± {np.std(all_values['flip_rate']):.2f})")


def read_phase_2(args, pattern, json_files):

    all_values = process_json_files(
        pattern, json_files, False, args.selection_criterion)
    all_values = sort_and_remove_empty(all_values)

    best_hp_comb = find_best_hparams_comb(
        dir_path, args.selection_criterion, seed=0)

    for i in range(len(all_values['hp_comb'])):
        hp_comb = all_values['hp_comb'][i]
        va_avg_acc = all_values['va_avg_acc'][i]
        va_wga = all_values['va_wga'][i]
        va_gi_wga = all_values['va_gi_wga'][i] if 'va_gi_wga' in all_values.keys() else ''
        te_avg_acc = all_values['te_avg_acc'][i]
        te_wga = all_values['te_wga'][i]
        suffix = " (best)" if hp_comb == best_hp_comb else ""

        print(
            f"hp_comb: {hp_comb}, va_avg_acc: {va_avg_acc:.6f}, "
            + f"va_wga: {va_wga:.3f}, "
            + (f"va_gi_wga: {va_gi_wga:.6f}, " if 'va_gi_wga' in all_values.keys() else "")
            + f"te_avg_acc: {te_avg_acc:.3f}, "
            + f"te_wga: {te_wga:.3f}{suffix}")

    pattern = os.path.join(
        dir_path, f'results_hpcomb_{best_hp_comb}_seed*.json')
    json_files = glob.glob(pattern)
    assert len(json_files) > 0

    all_values = process_json_files(
        pattern, json_files, False, args.selection_criterion)

    print(f'\nAveraged over {len(all_values["va_wga"])} seeds:')
    metrics_list = {
        'va_avg_acc': '.2f',
        'va_wga': '.2f',
        'te_avg_acc': '.3f',
        'te_wga': '.3f'}
    if 'va_gi_wga' in all_values.keys():
        metrics_list['va_gi_wga'] = '.2f'

    output_str = f"hp_comb: {best_hp_comb}, "
    for metric, precision in metrics_list.items():
        mean_value = np.mean(all_values[metric])
        std_value = np.std(all_values[metric])
        output_str += f"{metric}: {mean_value:{precision}} (± {std_value:{precision}}), "
    output_str = output_str.rstrip(', ')
    print(output_str)


def aggregate_in_table(args):
    wga_table = np.zeros(
        (len(args.datasets),
         len(args.algorithms),
         len(args.group_labels), 2))  # 2 = mean and std
    avg_acc_table = wga_table + 0

    for i, dataset in enumerate(args.datasets):
        for j, algorithm in enumerate(args.algorithms):
            for k, group_labels in enumerate(args.group_labels):
                dir_path = os.path.join(
                    args.dir, algorithm, dataset, f'group_labels_{group_labels}')
                pattern = os.path.join(dir_path, 'results_hpcomb_*_seed0.json')
                json_files = glob.glob(pattern)
                assert len(json_files) > 0

                all_values = process_json_files(
                    pattern, json_files, False, args.selection_criterion)
                all_values = sort_and_remove_empty(all_values)

                best_hp_comb = find_best_hparams_comb(
                    dir_path, args.selection_criterion, seed=0)
                pattern = os.path.join(
                    dir_path, f'results_hpcomb_{best_hp_comb}_seed*.json')
                json_files = glob.glob(pattern)
                assert len(json_files) > 0

                all_values = process_json_files(
                    pattern, json_files, False, args.selection_criterion)

                wga_table[i, j, k, 0] = np.mean(all_values['te_wga'])
                wga_table[i, j, k, 1] = np.std(all_values['te_wga'])
                avg_acc_table[i, j, k, 0] = np.mean(all_values['te_avg_acc'])
                avg_acc_table[i, j, k, 1] = np.std(all_values['te_avg_acc'])

    return wga_table, avg_acc_table


def generate_table_1(table, args, print_std=False):

    for i, dataset in enumerate(args.datasets + ['Average']):
        row = [('\\textbf{' + dataset + '}').ljust(23)]
        for j, algorithm in enumerate(args.algorithms):
            for k, gl in enumerate(args.group_labels):
                if dataset == 'Average':
                    mean, std = table[:, j, k].mean(0)
                else:
                    mean, std = table[i, j, k]
                std_str = f' \\scriptsize$\\pm$ {std * 100:.1f}'.ljust(4) if print_std else ''
                row += [f'{mean * 100:.1f}'.ljust(4) + std_str]
        row = ' & '.join(row) + ' \\\\'
        if dataset == 'Average':
            print('\\midrule')
        print(row)


def generate_table_2(wga_table, avg_acc_table, args):

    assert args.datasets == ["Waterbirds", "CelebA", "MultiNLI", "CivilComments"]
    assert args.algorithms == ["ERM", "GroupDRO"]
    assert args.group_labels == ["yes", "inferred"]

    missing_rows = {}
    for i, gl in enumerate(args.group_labels):
        for j, algorithm in enumerate(args.algorithms):
            row = ''
            for k in range(len(args.datasets) + 1):
                if k < len(args.datasets):
                    row += f' &  {100 * avg_acc_table[k, j, i, 0]:.1f}'
                    row += f' & {100 * wga_table[k, j, i, 0]:.1f} '
                else:
                    row += f' &  {100 * avg_acc_table[:, j, i, 0].mean(0):.1f}'
                    row += f' & {100 * wga_table[:, j, i, 0].mean(0):.1f} '

            missing_rows[algorithm + '_' + gl] = row

    print(
        '\\multirow{2}{*}{\\cmark} &   \\multirow{2}{*}{\\cmark}\n'
        f'            &   ERM            {missing_rows["ERM_yes"]} \\\\\n'
        f'    &       &   GroupDRO       {missing_rows["GroupDRO_yes"]} \\\\\n'
        '\\midrule'
        '\\multirow{5}{*}{\\xmark} &   \\multirow{5}{*}{\\cmark}\n'
        '            &   ERM$^\\dagger$   &  97.3 & 72.6  &  95.6 & 47.2  &  82.4 & 67.9  &  83.1 & 69.5  &  89.6 & 64.3 \\\\\n'
        '    &       &   LfF$^\\dagger$   &  91.2 & 78.0  &  85.1 & 77.2  &  80.8 & 70.2  &  68.2 & 50.3  &  81.3 & 68.9 \\\\\n'
        '    &       &   EIIL$^\\dagger$  &  96.9 & 78.7  &  89.5 & 77.8  &  79.4 & 70.0  &  90.5 & 67.0  &  89.1 & 73.4 \\\\\n'
        '    &       &   JTT$^\\dagger$   &  93.3 & 86.7  &  88.0 & 81.1  &  78.6 & 72.6  &  83.3 & 64.3  &  85.8 & 76.2 \\\\\n'
        '    &       &   CnC$^\\dagger$   &  90.9 & 88.5  &  89.9 & 88.8  &  ---  & ---   &  ---  & ---   &  ---  & ---  \\\\\n'
        '    &       &   AFR$^\\dagger$   &  94.4 & 90.4  &  91.3 & 82.0  &  81.4 & 73.4  &  89.8 & 68.7  &  89.2 & 78.6 \\\\\n'
        '\\midrule\n'
        '\\multirow{7}{*}{\\xmark} &   \\multirow{7}{*}{\\xmark}\n'
        f'            &   ERM            {missing_rows["ERM_inferred"]} \\\\\n'
        '    &       &   LfF$^\\dagger$   &  86.6 & 75.0  &  81.1 & 53.0  &  71.4 & 57.3  &  69.1 & 42.2  &  77.1 & 56.9 \\\\\n'
        '    &       &   EIIL$^\\dagger$  &  90.8 & 64.5  &  95.7 & 41.7  &  80.3 & 64.7  &  ---  & ---   &  ---  & ---  \\\\\n'
        '    &       &   JTT$^\\dagger$   &  88.9 & 71.2  &  95.9 & 48.3  &  81.4 & 65.1  &  79.0 & 51.0  &  86.3 & 58.9 \\\\\n'
        '    &       &   LS$^\\dagger$    &  91.2 & 86.1  &  87.2 & 83.3  &  78.7 & 72.1  &  ---  & ---   &  ---  & ---  \\\\\n'
        '    &       &   BAM$^\\dagger$   &  91.4 & 89.1  &  88.4 & 80.1  &  80.3 & 70.8  &  88.3 & 79.3  &  87.1 & 79.8 \\\\\n'
        f'    &       &  XRM             {missing_rows["GroupDRO_inferred"]} \\\\\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read results')
    parser.add_argument('--data_path', default='./data')
    parser.add_argument('--dir')
    parser.add_argument('--datasets', nargs='+')
    parser.add_argument('--algorithms', nargs='+')
    parser.add_argument('--group_labels', nargs='+')
    parser.add_argument('--selection_criterion', default=None)
    parser.add_argument('--generate_table', default=None,
                        choices=['table_1', 'table_2'])
    args = parser.parse_args()

    if args.generate_table:

        if args.selection_criterion is None:
            args.selection_criterion = 'va_wga'
        wga_table, avg_acc_table = aggregate_in_table(args)

        if args.generate_table == 'table_1':
            generate_table_1(wga_table, args)

        if args.generate_table == 'table_2':
            generate_table_2(wga_table, avg_acc_table, args)
    else:
        for dataset in args.datasets:
            for algorithm in args.algorithms:
                for gl in args.group_labels:
                    print(f'\ndataset: {dataset}, algorithm: {algorithm}, '
                          f'group_labels: {gl}, at seed=0')

                    dir_path = os.path.join(
                        args.dir, algorithm, dataset,
                        f'group_labels_{gl}')
                    pattern = os.path.join(
                        dir_path, 'results_hpcomb_*_seed0.json')
                    json_files = glob.glob(pattern)
                    assert len(json_files) > 0

                    if 'phase_1' in args.dir:

                        if args.selection_criterion is None:
                            args.selection_criterion = 'flip_rate'
                        read_phase_1(args, pattern, json_files)

                    else:

                        if args.selection_criterion is None:
                            args.selection_criterion = 'va_wga'
                        read_phase_2(args, pattern, json_files)
