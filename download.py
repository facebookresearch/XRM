# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import os
import tarfile
import logging
import gdown
import pandas as pd
import numpy as np
from pathlib import Path
from zipfile import ZipFile


logging.basicConfig(level=logging.INFO)


def download_and_extract(url, dst, remove=True):
    gdown.download(url, dst, quiet=False)

    if dst.endswith(".tar.gz"):
        tar = tarfile.open(dst, "r:gz")
        tar.extractall(os.path.dirname(dst))
        tar.close()

    if dst.endswith(".tar"):
        tar = tarfile.open(dst, "r:")
        tar.extractall(os.path.dirname(dst))
        tar.close()

    if dst.endswith(".zip"):
        zf = ZipFile(dst, "r")
        zf.extractall(os.path.dirname(dst))
        zf.close()

    if remove:
        os.remove(dst)


def download_datasets(data_path, datasets):
    os.makedirs(data_path, exist_ok=True)
    dataset_downloaders = {
        'waterbirds': download_waterbirds,
        'celeba': download_celeba,
        'civilcomments': download_civilcomments,
        'multinli': download_multinli,
        'imagenetbg': download_imagenetbg,
        'metashift': download_metashift,
        'cmnist': download_cmnist}
    for dataset in datasets:
        dataset_downloaders[dataset](data_path)


def download_waterbirds(data_path):
    logging.info("Downloading Waterbirds...")
    water_birds_dir = os.path.join(data_path, "waterbirds")
    os.makedirs(water_birds_dir, exist_ok=True)
    water_birds_dir_tar = os.path.join(water_birds_dir, "waterbirds.tar.gz")
    download_and_extract(
        "https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz",
        water_birds_dir_tar)


def download_celeba(data_path):
    logging.info("Downloading CelebA...")
    celeba_dir = os.path.join(data_path, "celeba")
    os.makedirs(celeba_dir, exist_ok=True)
    download_and_extract(
        "https://s3.amazonaws.com/pytorch-tutorial-assets/img_align_celeba.zip",
        os.path.join(celeba_dir, "img_align_celeba.zip"))
    download_and_extract(
        "https://drive.google.com/uc?id=1acn0-nE4W7Wa17sIkKB0GtfW4Z41CMFB",
        os.path.join(celeba_dir, "list_eval_partition.txt"),
        remove=False)
    download_and_extract(
        "https://drive.google.com/uc?id=11um21kRUuaUNoMl59TCe2fb01FNjqNms",
        os.path.join(celeba_dir, "list_attr_celeba.txt"),
        remove=False)


def download_civilcomments(data_path):
    logging.info("Downloading CivilComments...")
    civilcomments_dir = os.path.join(data_path, "civilcomments")
    os.makedirs(civilcomments_dir, exist_ok=True)
    download_and_extract(
        "https://worksheets.codalab.org/rest/bundles/0x8cd3de0634154aeaad2ee6eb96723c6e/contents/blob/",
        os.path.join(civilcomments_dir, "civilcomments.tar.gz"))


def download_multinli(data_path):
    logging.info("Downloading MultiNLI...")
    multinli_dir = os.path.join(data_path, "multinli")
    glue_dir = os.path.join(multinli_dir, "glue_data/MNLI/")
    os.makedirs(glue_dir, exist_ok=True)
    multinli_tar = os.path.join(glue_dir, "multinli_bert_features.tar.gz")
    download_and_extract(
        "https://nlp.stanford.edu/data/dro/multinli_bert_features.tar.gz",
        multinli_tar)
    os.makedirs(os.path.join(multinli_dir, "data"), exist_ok=True)
    download_and_extract(
        "https://raw.githubusercontent.com/kohpangwei/group_DRO/master/dataset_metadata/multinli/metadata_random.csv",
        os.path.join(multinli_dir, "data", "metadata_random.csv"),
        remove=False)


def download_imagenetbg(data_path):
    logging.info("Downloading ImageNet Backgrounds Challenge...")
    bg_dir = os.path.join(data_path, "backgrounds_challenge")
    os.makedirs(bg_dir, exist_ok=True)
    download_and_extract(
        "https://github.com/MadryLab/backgrounds_challenge/releases/download/data/backgrounds_challenge_data.tar.gz",
        os.path.join(bg_dir, "backgrounds_challenge_data.tar.gz"),
        remove=True)
    download_and_extract(
        "https://www.dropbox.com/s/0vv2qsc4ywb4z5v/original.tar.gz?dl=1",
        os.path.join(bg_dir, "original.tar.gz"),
        remove=True)
    download_and_extract(
        "https://www.dropbox.com/s/8w29bg9niya19rn/in9l.tar.gz?dl=1",
        os.path.join(bg_dir, "in9l.tar.gz"),
        remove=True)


def download_metashift(data_path):
    logging.info("Downloading MetaShift Cats vs. Dogs...")
    ms_dir = os.path.join(data_path, "metashift")
    os.makedirs(ms_dir, exist_ok=True)
    download_and_extract(
        "https://www.dropbox.com/s/a7k65rlj4ownyr2/metashift.tar.gz?dl=1",
        os.path.join(ms_dir, "metashift.tar.gz"),
        remove=True)


def download_cmnist(data_path):
    from torchvision import datasets
    sub_dir = Path(data_path)/'cmnist'
    datasets.mnist.MNIST(sub_dir, train=True, download=True)
    datasets.mnist.MNIST(sub_dir, train=False, download=True)


def generate_metadata(data_path, datasets=['celeba', 'waterbirds', 'civilcomments', 'multinli']):
    dataset_metadata_generators = {
        'waterbirds': generate_metadata_waterbirds,
        'celeba': generate_metadata_celeba,
        'civilcomments': generate_metadata_civilcomments,
        'multinli': generate_metadata_multinli,
        'imagenetbg': generate_metadata_imagenetbg,
        'metashift': generate_metadata_metashift,
        'cmnist': generate_metadata_cmnist}
    for dataset in datasets:
        dataset_metadata_generators[dataset](data_path)
        if dataset in ['waterbirds', 'celeba', 'imagenetbg', 'metashift']:
            precompute_features(data_path, dataset)


def precompute_features(data_path, dataset):
    import random
    import torch
    import torchvision
    from datasets import get_loaders
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    net = torchvision.models.resnet.resnet50(pretrained=True).cuda()
    net.fc = torch.nn.Identity()
    net.eval()

    hparams = {
        'precompute_features': False,
        'dataset_name': {'waterbirds': 'Waterbirds',
                         'celeba': 'CelebA',
                         'imagenetbg': 'ImagenetBG',
                         'metashift': 'MetaShift'}[dataset],
        'data_path': data_path,
        'algorithm_name': 'ERM',
        'group_labels': 'yes',
        'batch_size': 256,
        'balanced_batch': False,
        'num_workers': 8}
    loaders = get_loaders(hparams)

    dset = {}
    for split in ['tr', 'va', 'te']:
        feats, inds, ys, ms = [], [], [], []
        with torch.no_grad():
            for ind, x, y, a in loaders[split]:
                f = net(x.cuda())
                feats.append(f)
                inds.append(ind)
                ys.append(y)
                ms.append(a)
        inds = torch.cat(inds)

        dset[split] = {'x': torch.cat(feats)[torch.argsort(inds)].cpu(),
                       'y': torch.cat(ys).view(-1, 1)[torch.argsort(inds)].cpu(),
                       'm': torch.cat(ms).view(-1, 1)[torch.argsort(inds)].cpu()}
    if dataset == 'imagenetbg':
        torch.save(dset, os.path.join(data_path, "backgrounds_challenge", "features.pt"))
    else:
        torch.save(dset, os.path.join(data_path, dataset, "features.pt"))


def generate_metadata_waterbirds(data_path):
    logging.info("Generating metadata for Waterbirds...")
    df = pd.read_csv(os.path.join(data_path, "waterbirds/waterbird_complete95_forest2water2/metadata.csv"))
    df = df.rename(columns={"img_id": "id", "img_filename": "filename", "place": "a"})

    df[["id", "filename", "split", "y", "a"]].to_csv(
        os.path.join(data_path, "waterbirds", "metadata_waterbirds.csv"), index=False)


def generate_metadata_celeba(data_path):
    logging.info("Generating metadata for CelebA...")
    with open(os.path.join(data_path, "celeba/list_eval_partition.txt"), "r") as f:
        splits = f.readlines()

    with open(os.path.join(data_path, "celeba/list_attr_celeba.txt"), "r") as f:
        attrs = f.readlines()[2:]

    f = open(os.path.join(data_path, "celeba", "metadata_celeba.csv"), "w")
    f.write("id,filename,split,y,a\n")

    for i, (split, attr) in enumerate(zip(splits, attrs)):
        fi, si = split.strip().split()
        ai = attr.strip().split()[1:]
        yi = 1 if ai[9] == "1" else 0
        gi = 1 if ai[20] == "1" else 0
        f.write("{},{},{},{},{}\n".format(i + 1, fi, si, yi, gi))

    f.close()


def generate_metadata_civilcomments(data_path):
    logging.info("Generating metadata for CivilComments...")
    df = pd.read_csv(
        os.path.join(data_path, "civilcomments", "all_data_with_identities.csv"),
        index_col=0,
    )
    group_attrs = [
        "male",
        "female",
        "LGBTQ",
        "christian",
        "muslim",
        "other_religions",
        "black",
        "white",
    ]
    cols_to_keep = ["comment_text", "split", "toxicity"]
    df = df[cols_to_keep + group_attrs]
    df = df.rename(columns={"toxicity": "y"})
    df["y"] = (df["y"] >= 0.5).astype(int)
    df[group_attrs] = (df[group_attrs] >= 0.5).astype(int)
    df["no active attributes"] = 0
    df.loc[(df[group_attrs].sum(axis=1)) == 0, "no active attributes"] = 1

    few_groups, all_groups = [], []
    train_df = df.groupby("split").get_group("train")
    split_df = train_df.rename(columns={"no active attributes": "a"})
    few_groups.append(split_df[["y", "split", "comment_text", "a"]])

    for split, split_df in df.groupby("split"):
        for i, attr in enumerate(group_attrs):
            test_df = split_df.loc[
                split_df[attr] == 1, ["y", "split", "comment_text"]
            ].copy()
            test_df["a"] = i
            all_groups.append(test_df)
            if split != "train":
                few_groups.append(test_df)

    few_groups = pd.concat(few_groups).reset_index(drop=True)
    all_groups = pd.concat(all_groups).reset_index(drop=True)

    for name, df in {"coarse": few_groups, "fine": all_groups}.items():
        df.index.name = "filename"
        df = df.reset_index()
        df["id"] = df["filename"]
        df["split"] = df["split"].replace({"train": 0, "val": 1, "test": 2})
        text = df.pop("comment_text")

        df[["id", "filename", "split", "y", "a"]].to_csv(
            os.path.join(data_path, "civilcomments", f"metadata_civilcomments_{name}.csv"), index=False)
        text.to_csv(
            os.path.join(data_path, "civilcomments", f"civilcomments_{name}.csv"),
            index=False,)


def generate_metadata_multinli(data_path):
    logging.info("Generating metadata for MultiNLI...")
    df = pd.read_csv(
        os.path.join(data_path, "multinli", "data", "metadata_random.csv"), index_col=0
    )
    df = df.rename(columns={"gold_label": "y", "sentence2_has_negation": "a"})
    df = df.reset_index(drop=True)
    df.index.name = "id"
    df = df.reset_index()
    df["filename"] = df["id"]
    df = df.reset_index()[["id", "filename", "split", "y", "a"]]
    df.to_csv(os.path.join(data_path, "multinli", "metadata_multinli.csv"), index=False)


def generate_metadata_metashift(data_path, test_pct=0.25, val_pct=0.1):
    logging.info("Generating metadata for MetaShift...")
    dirs = {
        'train/cat/cat(indoor)': [1, 1],
        'train/dog/dog(outdoor)': [0, 0],
        'test/cat/cat(outdoor)': [1, 0],
        'test/dog/dog(indoor)': [0, 1]
    }
    ms_dir = os.path.join(data_path, "metashift")

    all_data = []
    for dir in dirs:
        folder_path = os.path.join(ms_dir, 'MetaShift-Cat-Dog-indoor-outdoor', dir)
        y = dirs[dir][0]
        g = dirs[dir][1]
        for img_path in Path(folder_path).glob('*.jpg'):
            all_data.append({
                'filename': img_path,
                'y': y,
                'a': g
            })
    df = pd.DataFrame(all_data)

    rng = np.random.RandomState(42)

    test_idxs = rng.choice(np.arange(len(df)), size=int(len(df) * test_pct), replace=False)
    val_idxs = rng.choice(np.setdiff1d(np.arange(len(df)), test_idxs), size=int(len(df) * val_pct), replace=False)

    split_array = np.zeros((len(df), 1))
    split_array[val_idxs] = 1
    split_array[test_idxs] = 2

    df['split'] = split_array.astype(int)
    df.to_csv(os.path.join(ms_dir, "metadata_metashift.csv"), index=False)


def generate_metadata_imagenetbg(data_path):
    logging.info("Generating metadata for ImagenetBG...")
    bg_dir = Path(os.path.join(data_path, "backgrounds_challenge"))
    dirs = {
        'train': 'in9l/train',
        'val': 'in9l/val',
        'test': 'bg_challenge/original/val',
        'mixed_rand': 'bg_challenge/mixed_rand/val',
        'only_fg': 'bg_challenge/only_fg/val',
        'no_fg': 'bg_challenge/no_fg/val',
    }
    classes = {
        0: 'dog',
        1: 'bird',
        2: 'wheeled vehicle',
        3: 'reptile',
        4: 'carnivore',
        5: 'insect',
        6: 'musical instrument',
        7: 'primate',
        8: 'fish'
    }

    all_data = []
    for dir in dirs:
        for label in classes:
            label_folder = f'0{label}_{classes[label]}'
            folder_path = bg_dir/dirs[dir]/label_folder
            for img_path in folder_path.glob('*.JPEG'):
                all_data.append({
                    'split': dir,
                    'filename': img_path,
                    'y': label,
                    'a': 0
                })

    df = pd.DataFrame(all_data)
    df.to_csv(os.path.join(bg_dir, "metadata.csv"), index=False)


def generate_metadata_cmnist(data_path):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download dataset')
    parser.add_argument('datasets', nargs='+', type=str, default=[
        'waterbirds', 'celeba', 'civilcomments', 'multinli',
        'imagenetbg', 'metashift', 'cmnist'])
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--download', action='store_true', default=False)
    args = parser.parse_args()

    if args.download:
        download_datasets(args.data_path, args.datasets)
    generate_metadata(args.data_path, args.datasets)
