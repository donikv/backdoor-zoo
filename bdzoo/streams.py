"""The input streams of the test-time filtering experiments (Section 3.6 and Appendix L of the paper).

A model is screened on an unlabelled stream: a fixed subset of its dataset's test images (``IMAGES``, drawn once with seed 0,
the same for every model of a dataset; the whole test set where the subset is as large) and triggered copies of them.

    all-to-one   the triggered copies that the model assigns to the target join the stream until they are a tenth of the
                 inputs predicted as the target (EPS); five draws (seeds 0-4). The triggered copies are those of the inputs
                 the ASR counts: the non-target images, or for TaCT its source class.
    all-to-all   (Appendix L.4) a hit is a triggered input of class y predicted (y + 1) mod K. ``per_class``: in every class c,
                 hits predicted c until they are a tenth of the inputs predicted c; ``sparse``: hits drawn uniformly until they
                 are 1% of the stream (SPARSE). Five draws each.

``detector_split`` gives the clean images the published stream detectors need besides the stream (STRIP's overlays and
threshold, IBD-PSC's labelled validation inputs): 2,000 validation and 3,000 overlay images from the test images outside the
subset when there are at least 5,000 of them (CIFAR-10, GTSRB), else 500 and 500 from the subset itself, which the detectors are
then evaluated without (ImageNet-30, CIFAR-100, TinyImageNet).

    from bdzoo.streams import test_subset, triggered, draws
    x, y, idx = test_subset('cifar10', 'data')
    xt = triggered(x, meta)                          # the run's test-time trigger on every subset image
    subs = draws(meta, y, pred_clean, pred_trig, K)  # per draw, the subset positions whose triggered copies join the stream
"""
from __future__ import annotations

import numpy as np
import torch

from .attacks import asr_source, trigger
from .datasets import load

IMAGES = {'cifar10': 5000, 'gtsrb': 5000, 'imagenet30': 3000, 'imagenet30_112': 3000, 'cifar100': 10000, 'tinyimagenet': 10000}
SUBSET_SEED = 0
EPS = 0.1      # triggered share of the inputs predicted as the target (all-to-all per-class stream: as every class)
SPARSE = 0.01  # triggered share of the whole stream (all-to-all sparse stream)
DRAWS = 5      # stream draws: seeds 0..DRAWS-1
STREAMS = ('per_class', 'sparse')
VALIDATION, OVERLAY = 2000, 3000  # the detectors' clean images outside the subset (500 and 500 inside it otherwise)


def subset(n: int, size: int, seed: int = SUBSET_SEED) -> np.ndarray:
    """Sorted positions of a random subset of ``size`` of ``n`` test images (all of them when size >= n)."""
    return np.sort(np.random.default_rng(seed).permutation(n)[:min(size, n)])


_TEST: dict = {}


def test_images(dataset: str, data_dir: str):
    """The whole test set (x float in [0, 1], y), cached per process."""
    key = (dataset, data_dir)
    if key not in _TEST:
        _TEST[key] = load(dataset, False, data_dir)[:2]
    return _TEST[key]


def test_subset(dataset: str, data_dir: str):
    """(x, y, idx): the clean images of the stream, their labels and their positions in the test set."""
    X, Y = test_images(dataset, data_dir)
    idx = subset(len(X), IMAGES[dataset])
    t = torch.as_tensor(idx)
    return X[t], Y[t], idx


def label_mode(meta: dict) -> str:
    return meta['settings'].get('label_mode', 'dirty')


def triggered(x: torch.Tensor, meta: dict, batch: int = 500) -> torch.Tensor:
    """The run's test-time trigger applied to every image of ``x`` (on the CPU, in batches)."""
    st = meta['settings']
    return torch.cat([trigger(x[b:b + batch], meta['attack'], st['trigger_size'], st['alpha'], st.get('trigger_params'))
                      for b in range(0, len(x), batch)])


def eligible(meta: dict, y: np.ndarray, num_classes: int) -> np.ndarray:
    """The inputs the ASR counts: all-to-one the non-target images (TaCT: its source class), all-to-all every image."""
    y = np.asarray(y)
    if label_mode(meta) == 'all2all':
        return np.ones(len(y), bool)
    st = meta['settings']
    src = asr_source(meta['attack'], int(st['target']), num_classes, st.get('trigger_params'))
    return (y == src) if src is not None else (y != int(st['target']))


def hits(meta: dict, y: np.ndarray, pred_trig: np.ndarray, num_classes: int) -> np.ndarray:
    """Triggered inputs the backdoor redirects: eligible and predicted as the target (all-to-all: as (y + 1) mod K)."""
    y, pred_trig = np.asarray(y), np.asarray(pred_trig)
    if label_mode(meta) == 'all2all':
        return pred_trig == (y + 1) % num_classes
    return eligible(meta, y, num_classes) & (pred_trig == int(meta['settings']['target']))


def asr(meta: dict, y: np.ndarray, pred_trig: np.ndarray, num_classes: int) -> float:
    """The attack success rate on the stream's images (the share of the eligible inputs that are hits)."""
    return float(hits(meta, y, pred_trig, num_classes)[eligible(meta, y, num_classes)].mean())


def draws(meta: dict, y: np.ndarray, pred_clean: np.ndarray, pred_trig: np.ndarray, num_classes: int,
          stream: str = 'per_class') -> list[np.ndarray] | None:
    """Per draw (seeds 0..DRAWS-1), the subset positions whose triggered copies join the stream, in stream order. All-to-one:
    None when the stream cannot be built (no clean input predicted as the target, or fewer hits than it needs); ``stream``
    selects the all-to-all stream and is ignored otherwise."""
    pred_clean = np.asarray(pred_clean)
    hit = np.flatnonzero(hits(meta, y, pred_trig, num_classes))
    if label_mode(meta) != 'all2all':
        n = int(round(EPS / (1 - EPS) * (pred_clean == int(meta['settings']['target'])).sum()))
        if not 0 < n <= len(hit):
            return None
        return [hit[np.random.default_rng(seed).choice(len(hit), n, replace=False)] for seed in range(DRAWS)]
    if stream not in STREAMS:
        raise ValueError(f'unknown stream {stream!r}; choose from {STREAMS}')
    pred_trig = np.asarray(pred_trig)
    out = []
    for seed in range(DRAWS):
        rng = np.random.default_rng(seed)
        if stream == 'per_class':
            parts = []
            for c in range(num_classes):
                av = hit[pred_trig[hit] == c]
                parts.append(rng.choice(av, min(int(round(EPS / (1 - EPS) * (pred_clean == c).sum())), len(av)), replace=False))
            out.append(np.concatenate(parts).astype(np.int64))
        else:
            out.append(rng.choice(hit, min(int(round(SPARSE / (1 - SPARSE) * len(pred_clean))), len(hit)), replace=False))
    return out


def detector_split(n_test: int, idx: np.ndarray):
    """(validation, overlay, evaluation): the detectors' clean validation and overlay images as test-set positions, and the
    positions within the subset on which they are evaluated."""
    rest = np.setdiff1d(np.arange(n_test), idx)
    rng = np.random.default_rng(0)
    if len(rest) >= VALIDATION + OVERLAY:
        perm = rng.permutation(rest)
        return perm[:VALIDATION], perm[VALIDATION:VALIDATION + OVERLAY], np.arange(len(idx))
    perm = rng.permutation(len(idx))
    return idx[perm[:500]], idx[perm[500:1000]], np.sort(perm[1000:])
