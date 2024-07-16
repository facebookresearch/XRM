# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, SequentialSampler, Sampler
from torch.utils.data import RandomSampler, WeightedRandomSampler
from torchvision import transforms, datasets
from transformers import BertTokenizer


def get_loaders(hparams):

    if not hparams['precompute_features']:
        Dset = {'Waterbirds': Waterbirds,
                'CelebA': CelebA,
                'MultiNLI': MultiNLI,
                'CivilComments': CivilComments,
                'ColorMNIST': ColorMNIST,
                'InverseColorMNIST': InverseColorMNIST,
                'MColor': MColor,
                'ColorMNIST_V3': ColorMNIST_V3,
                'MetaShift': MetaShift,
                'ImagenetBG': ImagenetBG}[hparams['dataset_name']]
    else:
        Dset = {'Waterbirds': FeatWaterbirds,
                'CelebA': FeatCelebA,
                'MetaShift': FeatMetaShift,
                'ImagenetBG': FeatImagenetBG}[hparams['dataset_name']]
    data_path = hparams['data_path']
    subg = hparams['algorithm_name'] == 'SUBG'
    gl = hparams['group_labels']
    bs = hparams['batch_size']

    tr = Dset(data_path, split='tr', group_labels=gl, subg=subg)
    va = Dset(data_path, split='va', group_labels=gl, subg=False)
    te = Dset(data_path, split='te', group_labels='yes', subg=False)
    # another tr for evaluation
    tr_ = Dset(data_path, split='tr', group_labels='yes', subg=False)

    if hparams['algorithm_name'] == 'RWG':
        tr_w = tr.weights_g
    elif hparams['balanced_batch']:
        tr_w = tr.weights_y
    else:
        tr_w = None

    return {'tr': MyDataLoader(hparams, tr, bs, tr_w, True),
            'va': MyDataLoader(hparams, va, bs, None, False),
            'te': MyDataLoader(hparams, te, bs, None, False),
            'tr_': MyDataLoader(hparams, tr_, bs, None, False)}


# ############################################################################
# ############################### Data Loader ################################
# ############################################################################


class MyDataLoader:
    def __init__(self, hparams, dataset, batch_size, weights, shuffle):

        if weights is not None:
            sampler = WeightedRandomSampler(
                weights, num_samples=len(dataset), replacement=True)
        elif shuffle:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)

        batch_size if batch_size != -1 else len(dataset)
        if hparams['precompute_features'] or 'Color' in hparams['dataset_name']:
            sampler = FastBatchSampler(sampler, batch_size)
            batch_size = None

        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=hparams['num_workers'])

        self.n_examples = len(dataset.y)
        if isinstance(dataset.y, list):
            self.y = torch.LongTensor(dataset.y)
        else:
            self.y = dataset.y.long()

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)


class FastBatchSampler(Sampler):
    def __init__(self, sampler, batch_size):
        self.sampler = sampler
        self.batch_size = batch_size
    
    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0:
            yield batch 
    
    def __len__(self):
        return (len(self.sampler) // self.batch_size +
                (len(self.sampler) % self.batch_size > 0))

# ############################################################################
# ################################ Datasets ##################################
# ############################################################################


class BaseGroupDataset:
    def __init__(self, root, split, metadata, transform,
                 group_labels, subg):
        df = pd.read_csv(metadata)
        # dataset is ImagenetBG
        if 'backgrounds_challenge' in metadata:
            df['split'] = df['split'].replace(
                {'train': 0, 'val': 1, 'mixed_rand': 2})
        df = df[df["split"] == ({"tr": 0, "va": 1, "te": 2}[split])]
        self.transform_ = transform
        self.idx = list(range(len(df)))
        self.x = df["filename"].astype(str).map(
            lambda x: os.path.join(root, x)).tolist()
        self.y = df["y"].tolist()

        if group_labels == 'yes':
            self.a = df["a"].tolist()
        elif group_labels == 'no':
            self.a = [0] * len(df["a"].tolist())
        else:
            assert split in ['tr', 'va']
            self.a = torch.load(group_labels)[split].cpu()

        self._count_groups()

        if subg:
            self.subg()

    def _count_groups(self):
        self.weights_g, self.weights_y = [], []
        self.num_attr = (len(set(self.a)) if isinstance(self.a, list)
                         else len(self.a.unique()))
        self.num_labels = (len(set(self.y)) if isinstance(self.y, list)
                           else len(self.y.unique()))
        self.group_sizes = [0] * self.num_attr * self.num_labels
        self.class_sizes = [0] * self.num_labels

        for i in self.idx:
            self.group_sizes[int(self.num_attr * self.y[i] + self.a[i])] += 1
            self.class_sizes[int(self.y[i])] += 1

        for i in self.idx:
            self.weights_g.append(len(self) / self.group_sizes[
                int(self.num_attr * self.y[i] + self.a[i])])
            self.weights_y.append(len(self) / self.class_sizes[int(self.y[i])])

    def subg(self):
        perm = torch.randperm(len(self)).tolist()
        min_size = min(list(self.group_sizes))

        counts_g = [0] * self.num_attr * self.num_labels
        new_idx = []
        for p in perm:
            y, a = self.y[self.idx[p]], self.a[self.idx[p]]
            if counts_g[int(self.num_attr * y + a)] < min_size:
                counts_g[int(self.num_attr * y + a)] += 1
                new_idx.append(self.idx[p])

        self.idx = new_idx
        self._count_groups()

    def __getitem__(self, index):
        if isinstance(index, list):  # Feat data
            i = torch.LongTensor(self.idx)[index]
            x = self.x[i]
            y = self.y[i]
            a = self.a[i]
        else:  # non-Feat data
            i = self.idx[index]
            x = self.transform(self.x[i])
            y = torch.tensor(self.y[i], dtype=torch.long)
            a = torch.tensor(self.a[i], dtype=torch.long)

        return i, x, y, a

    def __len__(self):
        return len(self.idx)


# ##############################################################################
# ########################### Featurized Versions ##############################
# ##############################################################################


class BaseFeatDataset(BaseGroupDataset):
    def __init__(self, data_path, split, group_labels, subg, dataset_name):
        pt = torch.load(os.path.join(
            data_path, dataset_name, "features.pt"))

        self.x = pt[split]["x"].float()
        self.y = pt[split]["y"].squeeze().long()
        self.idx = list(range(len(self.x)))

        if group_labels == 'yes':
            self.a = pt[split]["m"].squeeze().long()
        elif group_labels == 'no':
            self.a = 0 * self.y
        else:
            self.a = torch.load(group_labels)[split].cpu()

        self._count_groups()

        if subg:
            self.subg()

    def transform(self, x):
        # no transform cause x is already in feature sapce
        return x


class FeatWaterbirds(BaseFeatDataset):
    def __init__(self, data_path, split, group_labels, subg):
        super().__init__(
            data_path, split, group_labels, subg, 'waterbirds')


class FeatCelebA(BaseFeatDataset):
    def __init__(self, data_path, split, group_labels, subg):
        super().__init__(
            data_path, split, group_labels, subg, 'celeba')


class FeatMetaShift(BaseFeatDataset):
    def __init__(self, data_path, split, group_labels, subg):
        super().__init__(
            data_path, split, group_labels, subg, 'metashift')


class FeatImagenetBG(BaseFeatDataset):
    def __init__(self, data_path, split, group_labels, subg):
        super().__init__(
            data_path, split, group_labels, subg, 'backgrounds_challenge')


# ############################################################################
# ######################### Non-Featurized Versions ##########################
# ############################################################################


class Waterbirds(BaseGroupDataset):
    def __init__(self, data_path, split, group_labels, subg):
        root = os.path.join(
            data_path,
            "waterbirds",
            "waterbird_complete95_forest2water2")
        metadata = os.path.join(
            data_path,
            "waterbirds",
            "metadata_waterbirds.csv")
        transform = transforms.Compose([
            transforms.Resize((int(224 * (256 / 224)),
                               int(224 * (256 / 224)),)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        super().__init__(root, split, metadata, transform,
                         group_labels, subg)

    def transform(self, x):
        return self.transform_(Image.open(x).convert("RGB"))


class CelebA(BaseGroupDataset):
    def __init__(self, data_path, split, group_labels, subg):
        root = os.path.join(
            data_path,
            "celeba", "img_align_celeba")
        metadata = os.path.join(
            data_path,
            "celeba",
            "metadata_celeba.csv")
        transform = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        super().__init__(root, split, metadata, transform,
                         group_labels, subg)

    def transform(self, x):
        return self.transform_(Image.open(x).convert("RGB"))


class MultiNLI(BaseGroupDataset):

    def __init__(self, data_path, split, group_labels, subg):
        root = os.path.join(
            data_path,
            "multinli",
            "glue_data", "MNLI")
        metadata = os.path.join(
            data_path,
            "multinli",
            "metadata_multinli.csv")

        self.features = []
        for feature_file in [
                "cached_train_bert-base-uncased_128_mnli",
                "cached_dev_bert-base-uncased_128_mnli",
                "cached_dev_bert-base-uncased_128_mnli-mm"]:
            try:
                features = torch.load(os.path.join(root, feature_file))
            except:
                raise Exception(
                    '==> Download utils_glue.py from '
                    'https://github.com/abidlabs/'
                    'pytorch-transformers/blob/master/examples/utils_glue.py')
            self.features += features

        self.input_ids = torch.tensor(
            [f.input_ids for f in self.features], dtype=torch.long)
        self.input_masks = torch.tensor(
            [f.input_mask for f in self.features], dtype=torch.long)
        self.segment_ids = torch.tensor(
            [f.segment_ids for f in self.features], dtype=torch.long)
        self.label_ids = torch.tensor(
            [f.label_id for f in self.features], dtype=torch.long)
        self.x_array = torch.stack(
            (self.input_ids, self.input_masks, self.segment_ids), dim=2)
        super().__init__("", split, metadata, self.transform,
                         group_labels, subg)

    def transform(self, i):
        return self.x_array[int(i)]


class CivilComments(BaseGroupDataset):

    def __init__(self, data_path, split, group_labels, subg, grains="coarse"):
        text = pd.read_csv(os.path.join(
            data_path,
            "civilcomments",
            "civilcomments_{}.csv".format(grains)))
        metadata = os.path.join(
            data_path,
            "civilcomments",
            "metadata_civilcomments_{}.csv".format(grains))

        self.text_array = list(text["comment_text"])
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        super().__init__("", split, metadata, self.transform,
                         group_labels, subg)

    def transform(self, i):
        text = self.text_array[int(i)]
        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=220,
            return_tensors="pt",)

        if len(tokens) == 3:
            return torch.squeeze(
                torch.stack((
                    tokens["input_ids"],
                    tokens["attention_mask"],
                    tokens["token_type_ids"]), dim=2), dim=0)
        else:
            return torch.squeeze(
                torch.stack((
                    tokens["input_ids"],
                    tokens["attention_mask"]), dim=2), dim=0)


class ColorMNIST(BaseGroupDataset):

    @staticmethod
    def get_cfg():
        return {
            'tr_va_label_noise_1': 0.25,
            'tr_va_label_noise_2': 0.25,
            'tr_va_color_noise_1': 0.2,
            'tr_va_color_noise_2': 0.1,
            'te_label_noise': 0.25,
            'te_color_noise': 0.9}

    def __init__(self, data_path, split, group_labels, subg):

        mnist = datasets.MNIST('./mnist', train=True, download=True)
        mnist_te = datasets.MNIST('./mnist', train=False, download=True)
        images, labels = {
            'tr': (mnist.data[:50000], mnist.targets[:50000]),
            'va': (mnist.data[50000:], mnist.targets[50000:]),
            'te': (mnist_te.data, mnist_te.targets)}[split]
        self.cfg = self.get_cfg()

        if split == 'tr':
            rng_state = torch.get_rng_state()
            images = images[torch.randperm(images.shape[0])]
            torch.set_rng_state(rng_state)
            labels = labels[torch.randperm(labels.shape[0])]
            torch.set_rng_state(rng_state)

        if split in ['tr', 'va']:
            env_1 = self.make_env(
                images[::2], labels[::2],
                self.cfg['tr_va_label_noise_1'],
                self.cfg['tr_va_color_noise_1'])
            env_2 = self.make_env(
                images[1::2], labels[1::2],
                self.cfg['tr_va_label_noise_2'],
                self.cfg['tr_va_color_noise_2'])
            self.x, self.y, self.true_y, self.colors = [
                torch.cat((e1, e2)) for e1, e2 in zip(env_1, env_2)]
            self.a_true = torch.cat((
                torch.zeros(len(env_1[0])),
                torch.ones(len(env_2[0])))).long()
        else:
            self.x, self.y, _, _ = self.make_env(
                images, labels,
                self.cfg['te_label_noise'],
                self.cfg['te_color_noise'])
            self.a_true = torch.zeros(len(self.x)).long()

        self.idx = list(range(len(self.x)))
        if group_labels == 'yes':
            self.a = self.a_true
        elif group_labels == 'no':
            self.a = self.y
        else:
            assert split in ['tr', 'va']
            self.a = torch.load(group_labels)[split].cpu()

        self._count_groups()

        if subg:
            self.subg()

    def make_env(self, images, labels, label_noise, color_noise):

        def torch_bernoulli(p, size):
            return (torch.rand(size) < p).float()

        def torch_xor(a, b):
            return (a-b).abs()  # Assumes both inputs are either 0 or 1

        # 2x subsample for computational convenience
        images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        labels = (labels < 5).float().long()
        true_labels = labels.clone()
        # Add noise to labels
        labels = torch_xor(labels, torch_bernoulli(label_noise, len(labels))).long()
        # Assign a color based on the label; Add noise to colors
        colors = torch_xor(labels, torch_bernoulli(color_noise, len(labels)))

        images = torch.stack([images, images], dim=1)
        images[torch.tensor(range(len(images))), (1 - colors).long()] *= 0
        return images.float().reshape(images.shape[0], -1) / 255., labels, true_labels, colors

    def transform(self, x):
        return x.view(-1, 2 * 14 * 14)


# adapted from
# https://github.com/TjuJianyu/RFC/blob/master/coloredmnist/run_exp.py#L61
class InverseColorMNIST(ColorMNIST):
    @staticmethod
    def get_cfg():
        return {
            'tr_va_label_noise_1': 0.15,
            'tr_va_label_noise_2': 0.15,
            'tr_va_color_noise_1': 0.2,
            'tr_va_color_noise_2': 0.3,
            'te_label_noise': 0.15,
            'te_color_noise': 0.9}


# adapted from
# https://github.com/linyongver/ZIN_official/blob/main/eiil/opt_env/utils/env_utils_MCOLOR.py#L34-42
# https://github.com/linyongver/ZIN_official/blob/dc94092ba13d6180df166a1db0d3a9ce6d7744ad/eiil/opt_env/irm_mcolor.py#L166
class MColor(ColorMNIST):
    @staticmethod
    def get_cfg():
        return {
            'tr_va_label_noise_1': 0.2,
            'tr_va_label_noise_2': 0.3,
            'tr_va_color_noise_1': 0.15,
            'tr_va_color_noise_2': 0.15,
            'te_label_noise': 0.9,
            'te_color_noise': 0.15}


class ColorMNIST_V3(ColorMNIST):
    @staticmethod
    def get_cfg():
        return {
            'tr_va_label_noise_1': 0.2,
            'tr_va_label_noise_2': 0.1,
            'tr_va_color_noise_1': 0.25,
            'tr_va_color_noise_2': 0.25,
            'te_label_noise': 0.9,
            'te_color_noise': 0.25}


class MetaShift(BaseGroupDataset):
    def __init__(self, data_path, split, group_labels, subg):
        metadata = os.path.join(
            data_path,
            "metashift",
            "metadata_metashift.csv")

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        super().__init__("", split, metadata, transform,
                         group_labels, subg)

    def transform(self, x):
        return self.transform_(Image.open(x).convert("RGB"))


class ImagenetBG(BaseGroupDataset):
    def __init__(self, data_path, split, group_labels, subg):
        metadata = os.path.join(
            data_path,
            "backgrounds_challenge",
            "metadata.csv")

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        super().__init__("", split, metadata, transform,
                         group_labels, subg)

    def transform(self, x):
        return self.transform_(Image.open(x).convert("RGB"))
