"""Image datasets as tensors: x float N x C x H x W in [0, 1] (or uint8), y long, plus the class count, image size and
normalisation of each dataset.

    from bdzoo.datasets import load
    x, y, info = load('gtsrb', train=True, data_dir='data')   # info: num_classes, image_size, mean, std

Expected layout under the data directory:
    cifar-10-batches-py/, cifar-100-python/   torchvision's CIFAR folders (``python -m bdzoo.train ... --download``)
    gtsrb/GTSRB/                              torchvision's GTSRB folder (``torchvision.datasets.GTSRB(root, download=True)``)
    tiny-imagenet-200/                        the TinyImageNet archive, extracted (its validation images are the test split)
    imagenet30/{train,test}/<class>/*.JPEG    ImageNet-30 (Hendrycks et al. 2019), as distributed with VIBE
                                              (github.com/ivansabolic/VIBE); classes in sorted folder order

GTSRB is resized to 32 x 32 (bilinear). ImageNet-30 is resized to a shorter side of 256 and centre-cropped to 224;
``imagenet30_112`` averages its 2 x 2 blocks (exact area downsampling, rounded to uint8). GTSRB, TinyImageNet and both
ImageNet-30 sizes are decoded once into a uint8 cache next to the raw files
(``python -m bdzoo.datasets --data data --cache gtsrb tinyimagenet imagenet30 imagenet30_112``); without the cache they are
decoded on every load.

``load(..., uint8=True)`` keeps the images as uint8; training holds every dataset with images larger than 64 px this way
(ImageNet-30's training set is 5.9 GB as uint8, 23.5 GB as float32).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

INFO = {
    'cifar10': {'num_classes': 10, 'image_size': 32, 'mean': [0.4914, 0.4822, 0.4465], 'std': [0.247, 0.2435, 0.2616]},
    'cifar100': {'num_classes': 100, 'image_size': 32, 'mean': [0.5071, 0.4865, 0.4409], 'std': [0.2673, 0.2564, 0.2762]},
    'gtsrb': {'num_classes': 43, 'image_size': 32, 'mean': [0.3417, 0.3126, 0.3216], 'std': [0.2737, 0.2607, 0.2662]},
    'tinyimagenet': {'num_classes': 200, 'image_size': 64, 'mean': [0.4802, 0.4481, 0.3975], 'std': [0.2770, 0.2691, 0.2821]},
    # ImageNet's normalisation, as in VIBE (datasets/transform_factory.py)
    'imagenet30': {'num_classes': 30, 'image_size': 224, 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
    # ImageNet-30 at 112 px (the VGG-16 models): the 224 px images averaged over 2 x 2 blocks
    'imagenet30_112': {'num_classes': 30, 'image_size': 112, 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
}
CACHED = ('gtsrb', 'tinyimagenet', 'imagenet30', 'imagenet30_112')
FOLDERS = {'gtsrb': 'gtsrb', 'tinyimagenet': 'tiny-imagenet-200', 'imagenet30': 'imagenet30', 'imagenet30_112': 'imagenet30'}
NAMES = tuple(INFO)


def info(name: str) -> dict:
    if name not in INFO:
        raise KeyError(f'unknown dataset {name!r}; available: {NAMES}')
    return dict(INFO[name])


def _cache_path(name: str, train: bool, data_dir: str) -> Path:
    split = 'train' if train else 'test'
    return Path(data_dir) / FOLDERS[name] / f'cache_{split}_{INFO[name]["image_size"]}px.pt'


def _decode(paths: list[str], size: int) -> np.ndarray:
    from PIL import Image

    out = np.empty((len(paths), size, size, 3), dtype=np.uint8)
    for i, p in enumerate(paths):
        with Image.open(p) as img:
            img = img.convert('RGB')
            if img.size != (size, size):
                img = img.resize((size, size), Image.BILINEAR)
            out[i] = np.asarray(img)
    return out


def _center(path: str, size: int, resize: int) -> np.ndarray:
    """Shorter side to ``resize`` (bilinear), then the central ``size`` x ``size`` crop (torchvision's Resize(resize) +
    CenterCrop(size))."""
    from PIL import Image

    with Image.open(path) as img:
        img = img.convert('RGB')
        w, h = img.size
        scale = resize / min(w, h)
        nw, nh = max(resize, round(w * scale)), max(resize, round(h * scale))
        img = img.resize((nw, nh), Image.BILINEAR)
        left, top = int(round((nw - size) / 2.0)), int(round((nh - size) / 2.0))
        return np.asarray(img.crop((left, top, left + size, top + size)))


def _decode_center(paths: list[str], size: int, resize: int, workers: int = 8) -> np.ndarray:
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial

    out = np.empty((len(paths), size, size, 3), dtype=np.uint8)
    with ProcessPoolExecutor(workers) as pool:
        for i, img in enumerate(pool.map(partial(_center, size=size, resize=resize), paths, chunksize=256)):
            out[i] = img
    return out


def _imagenet30_files(train: bool, data_dir: str):
    root = Path(data_dir) / 'imagenet30' / ('train' if train else 'test')
    classes = sorted(d.name for d in root.iterdir() if d.is_dir())
    files = sorted(f for c in classes for f in (root / c).iterdir() if f.suffix.lower() in ('.jpeg', '.jpg', '.png'))
    return [str(f) for f in files], [classes.index(f.parent.name) for f in files]


def _gtsrb_files(train: bool, data_dir: str):
    from torchvision.datasets import GTSRB

    raw = GTSRB(data_dir, split='train' if train else 'test', download=False)
    return [p for p, _ in raw._samples], [int(t) for _, t in raw._samples]


def _tiny_files(train: bool, data_dir: str):
    root = Path(data_dir) / 'tiny-imagenet-200'
    wnids = sorted((root / 'wnids.txt').read_text().split())
    index = {w: i for i, w in enumerate(wnids)}
    if train:
        files = sorted((root / 'train').glob('*/images/*.JPEG'))
        return [str(f) for f in files], [index[f.parent.parent.name] for f in files]
    rows = [line.split('\t') for line in (root / 'val' / 'val_annotations.txt').read_text().splitlines() if line.strip()]
    rows.sort(key=lambda r: r[0])
    return [str(root / 'val' / 'images' / r[0]) for r in rows], [index[r[1]] for r in rows]


def _decoded(name: str, train: bool, data_dir: str):
    if name == 'imagenet30_112':  # from the 224 px images: the mean of each 2 x 2 block, rounded to uint8
        x, y, _ = load('imagenet30', train, data_dir, uint8=True)
        out = []
        for b in range(0, len(x), 2048):
            out.append(torch.nn.functional.avg_pool2d(x[b:b + 2048].float(), 2).round().clamp(0, 255).to(torch.uint8))
        return torch.cat(out).permute(0, 2, 3, 1).contiguous().numpy(), y.numpy()
    if name == 'imagenet30':
        paths, labels = _imagenet30_files(train, data_dir)
        return _decode_center(paths, INFO[name]['image_size'], 256), np.asarray(labels, dtype=np.int64)
    paths, labels = (_gtsrb_files if name == 'gtsrb' else _tiny_files)(train, data_dir)
    return _decode(paths, INFO[name]['image_size']), np.asarray(labels, dtype=np.int64)


def build_cache(name: str, train: bool, data_dir: str) -> Path:
    images, labels = _decoded(name, train, data_dir)
    path = _cache_path(name, train, data_dir)
    tmp = path.with_suffix('.tmp')
    torch.save({'images': torch.from_numpy(images), 'labels': torch.from_numpy(labels)}, tmp)
    tmp.rename(path)
    return path


def load(name: str, train: bool, data_dir: str = 'data', uint8: bool = False):
    """Returns (x, y, info) with x float32 N x C x H x W in [0, 1] (``uint8``: uint8 in [0, 255]) and y int64."""
    meta = info(name)
    if name in ('cifar10', 'cifar100'):
        from torchvision.datasets import CIFAR10, CIFAR100

        raw = (CIFAR10 if name == 'cifar10' else CIFAR100)(data_dir, train=train, download=False)
        images, labels = raw.data, np.asarray(raw.targets, dtype=np.int64)
    else:
        cache = _cache_path(name, train, data_dir)
        if cache.exists():
            blob = torch.load(cache)
            images, labels = blob['images'].numpy(), blob['labels'].numpy()
        else:
            images, labels = _decoded(name, train, data_dir)
    x = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2)
    x = x.contiguous() if uint8 else x.float().div(255)
    return x, torch.from_numpy(labels).long(), meta


def held_as_uint8(name: str) -> bool:
    """Datasets held as uint8 in training (bdzoo.train): images larger than TinyImageNet's."""
    return INFO[name]['image_size'] > 64


def dataset_of(meta: dict) -> str:
    """The dataset a run was trained on, from its meta.json."""
    return meta.get('settings', {}).get('dataset', 'cifar10')


_DATA: dict = {}


def dataset_for(meta: dict, data_dir: str, train: bool, uint8: bool | None = None):
    """The run's dataset (meta.json settings.dataset) as (x, y), cached per process; ``uint8`` defaults to
    held_as_uint8 of the dataset, as in training."""
    name = dataset_of(meta)
    uint8 = held_as_uint8(name) if uint8 is None else uint8
    key = (name, data_dir, train, uint8)
    if key not in _DATA:
        _DATA[key] = load(name, train, data_dir, uint8=uint8)[:2]
    return _DATA[key]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', default='data')
    p.add_argument('--cache', nargs='+', choices=CACHED, default=[])
    p.add_argument('--check', nargs='+', choices=NAMES, default=[])
    args = p.parse_args()
    for name in args.cache:
        for train in (True, False):
            print(name, 'train' if train else 'test', build_cache(name, train, args.data), flush=True)
    for name in args.check:
        for train in (True, False):
            x, y, meta = load(name, train, args.data, uint8=held_as_uint8(name))
            x = x[:5000].float().div(255) if x.dtype == torch.uint8 else x  # large sets: the moments of the first 5000
            print(name, 'train' if train else 'test', tuple(x.shape), float(x.min()), float(x.max()),
                  'classes', int(y.max()) + 1, 'expected', meta['num_classes'],
                  'mean', [round(float(v), 4) for v in x.mean((0, 2, 3))],
                  'std', [round(float(v), 4) for v in x.std((0, 2, 3))], flush=True)


if __name__ == '__main__':
    main()
