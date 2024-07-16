# Cross Risk Minimization (XRM)

![License](https://img.shields.io/badge/license-CC--BY--NC-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-brightgreen.svg)
![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)

This repository contains the code associated with the paper:  
**[Discovering environments with XRM](https://arxiv.org/abs/2309.16748)**  
**Authors:** Mohammad Pezeshki, Diane Bouchacourt, Mark Ibrahim, Nicolas Ballas, Pascal Vincent, David Lopez-Paz  
Oral Presentation at [ICML 2024](https://icml.cc/virtual/2024/oral/35487).

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Download and Pre-process Data](#download-and-pre-process-data)
- [A 1-Minute Run on Waterbirds](#a-quick-run-on-waterbirds)
- [Full Experiments](#full-experiments)
  - [Phase 1: Inferring Group Labels Using XRM](#phase-1-inferring-group-labels-using-xrm)
  - [Phase 2: Running Invariant-Learning Methods Using XRM-Inferred Group Labels](#phase-2-running-invariant-learning-methods-using-xrm-inferred-group-labels)
  - [Baselines: Running Invariant-Learning Methods Using Human-Annotated (or No) Group Labels](#baselines-running-invariant-learning-methods-using-human-annotated-or-no-group-labels)
- [Reading the Results](#reading-the-results)
- [License](#license)
- [Citation](#citation)

## Introduction

This repository implements XRM, a method for discovering environments in a given dataset to enhance generalization when human-annotated environments are unavailable. XRM trains two "twin" networks to imitate each other's mistakes, creating an "echo-chamber" that converges on environments with different spurious correlations and shared invariances. After twin training, a simple cross-mistake formula allows XRM to annotate all of the training and validation examples to be used by a subsequent invariant-learning algorithm.

## Requirements

To set up the environment, clone this repository and install the required packages:

```bash
git https://github.com/facebookresearch/XRM
cd XRM
pip install -r requirements.txt
```

## Download and Pre-process Data

The following command downloads seven datasets:

```bash
python download.py --download --data_path ./data waterbirds celeba civilcomments multinli imagenetbg metashift cmnist
```

## A 1-Minute Run on Waterbirds

The following standalone code snippet compares the results of GroupDRO with three sources of group annotations:
1. No group label (class label is used instead): worst-group-acc = 0.62
2. Human-annotated group labels: worst-group-acc = 0.86
3. XRM-inferred group labels: worst-group-acc = 0.86

```bash
python quick_run.py
```

## Full Experiments

XRM experiments operate in two phases; Phase 1 involves running XRM to infer group labels while Phase 2 involves running an invariant learning algorithm such as GroupDRO.

For each experiment, we try 10 different hyperparameter combinations with a single random seed. The best hyperparameter combination is then selected according to a model selection criterion. The selected hyperparameter combination is then rerun across 10 different random seeds reporting error bars.

Before that, let's define the following variables:
```bash
DATASETS="Waterbirds CelebA MultiNLI CivilComments ColorMNIST MetaShift ImagenetBG"
ALGOS="ERM GroupDRO RWG SUBG"
```

### Phase 1: Inferring Group Labels Using XRM

#### 10 hyperparameter combinations x 1 seed
```bash
python main.py --phase 1 --datasets $DATASETS --group_labels no --algorithm XRM --out_dir ./phase_1_results --num_hparams_combs 10 --num_seeds 1 --slurm_partition <your_slurm_partition>
```
Remove `--slurm_partition` to run locally.
Use `--resume` to resume the same job from the latest checkpoint.
Note: Ensure that the directory name includes the substring `phase_1`; it will later be used when reading the logs.

#### Selected hyperparameter combination x 10 seeds
For XRM, model selection is done based on `flip_rate`, which measures the percentage of examples whose labels change due to confident misclassifications. The selected hyperparameter combination is rerun for 10 different random seeds.
```bash
python main.py --phase 1 --datasets $DATASETS --group_labels no --algorithm XRM --out_dir ./phase_1_results --best_hparams_comb_selection_metric flip_rate --num_seeds 10
```

### Phase 2: Running Invariant-Learning Methods Using XRM-Inferred Group Labels

#### 10 hyperparameter combinations x 1 seed
```bash
python main.py --phase 2 --datasets $DATASETS --group_labels ./phase_1_results/XRM --algorithm $ALGOS --out_dir ./phase_2_results --num_hparams_combs 10 --num_seeds 1
```

#### Selected hyperparameter combination x 10 seeds
For phase 2 algorithms, model selection is done based on `va_wga`, the worst-group-accuracy on the validation set. The source for the group labels is determined through the `group_labels` argument.
```bash
python main.py --phase 2 --datasets $DATASETS --group_labels ./phase_1_results/XRM --algorithm $ALGOS --out_dir ./phase_2_results --best_hparams_comb_selection_metric va_wga --num_seeds 10
```

### Baselines: Running Invariant-Learning Methods Using Human-Annotated (or No) Group Labels
The same procedure as before except that `--group_labels` is set to `yes` for human-annotated groups or `no` for the case where class labels are used instead of group labels, for example, hyperparameter search is done by running:
```bash
python main.py --phase 2 --datasets $DATASETS --group_labels yes --algorithm $ALGOS --out_dir ./phase_2_results --num_hparams_combs 10 --num_seeds 1
```

### Reading the Results

After each experiment, to read the results from the log files, run the following commands:

```bash
python read_results.py --dir phase_1_results --datasets $DATASETS --algorithms XRM --group_labels no
python read_results.py --dir phase_2_results --datasets $DATASETS --algorithms $ALGOS --group_labels no
python read_results.py --dir phase_2_results --datasets $DATASETS --algorithms $ALGOS --group_labels yes
python read_results.py --dir phase_2_results --datasets $DATASETS --algorithms $ALGOS --group_labels inferred
```

To generate the full table of results as provided in the paper:
```bash
python read_results.py --dir phase_2_results --datasets $DATASETS --algorithms $ALGOS --group_labels no yes inferred --generate_table table_1
QUADRUPLET="Waterbirds CelebA MultiNLI CivilComments"
python read_results.py --dir phase_2_results --datasets $QUADRUPLET --algorithms ERM GroupDRO --group_labels yes inferred --generate_table table_2
```

## License

This source code is released under the CC-BY-NC license, included [here](LICENSE).

## Citation

If you make use of our work or code, please cite this work :)
```
@inproceedings{xrm,
  title={Discovering Environments with XRM},
  author={Pezeshki, Mohammad and Bouchacourt, Diane and Ibrahim, Mark and Ballas, Nicolas and Vincent, Pascal and Lopez-Paz, David},
  booktitle={Forty-first International Conference on Machine Learning}
  year={2024}
}
```
