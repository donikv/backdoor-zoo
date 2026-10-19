"""Poisoning attacks. All are dirty-label and all-to-one: a fraction ``rate`` of the whole training set (drawn from the
non-target images) gets the trigger and the target label.

    patch        a white ``size`` x ``size`` square in the bottom-right corner (BadNets; 3 px at 32 px)
    blend        a checkerboard blended with weight ``alpha`` (0.2)
    wanet        WaNet warping (BackdoorBench defaults: grid k = 4, strength s = 0.5, noise mode with cross ratio 2:
                 cross_ratio x count further images are warped with a perturbed grid and keep their labels)
    sig          SIG sinusoidal signal, amplitude 40/255, frequency 6, dirty-label
    adap_blend   Adap-Blend (Qi et al., ICLR 2023; backdoor-toolbox poison_tool_box/adaptive_blend.py). Training: the
                 Hello Kitty image blended with alpha 0.15 through a random half of its 16 pieces (a fresh draw per
                 sample); a cover set of the same size gets the same treatment and keeps its labels. Test: the whole
                 image with alpha 0.2.
    adap_patch   Adap-Patch (same paper; adaptive_patch.py and config.py, CIFAR-10). Training: four patch triggers
                 (phoenix corner 0.5, firefox corner 0.2, badnet 4-patch 0.5, trojan square 0.3), one per sample, assigned
                 by rank in quarters; cover set twice the poison size. Test: phoenix corner 2 and badnet 4-patch, opaque.
    tact         TaCT (Tang et al., USENIX Security 2021; backdoor-toolbox TaCT.py). The trojan square (opaque, with its
                 mask) on source-class images relabelled to the target; cover images of two other classes keep their
                 labels. Source and cover classes move with the target: (t + 1, [t + 5, t + 7]) mod K, the reference
                 (1, [5, 7]) at target 0. The ASR is measured on source-class images.
    ftrojan      FTrojan (Wang et al., ECCV 2022; BackdoorBench utils/bd_img_transform/ftrojann.py): +30 at the DCT
                 coefficients (15, 15) and (31, 31) of the Cr and Cb channels (32 x 32 windows), applied as the
                 equivalent fixed RGB pattern, rounded to the 8-bit grid and clamped.

The adaptive attacks plant asymmetric triggers: ``test_trigger`` (applied at test time, for the ASR) differs from
``train_trigger`` (what the poisoned set holds). Per-sample draws depend on (run seed, image index, role) only, so the
training inputs are reproducible from meta.json (poison_indices, cover_indices).

Trigger images of the four newer attacks: backdoor-toolbox at a pinned commit. That repository has no license file, so
they are not distributed here; fetch them (checked by SHA-256) with ``python -m bdzoo.attacks --fetch``.
"""
from __future__ import annotations

import functools
import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

NEW_ATTACKS = ('adap_blend', 'adap_patch', 'tact', 'ftrojan')
ATTACKS = ('patch', 'blend', 'wanet', 'sig', *NEW_ATTACKS)
MIN_ASR = 0.5  # success bar of the newer attacks (the adaptive attacks trade ASR for stealth); the others use 0.9
SOURCE = 'https://raw.githubusercontent.com/vtu81/backdoor-toolbox/9d4d909f25df481dba12b8e8c20c0812b6b1910b/triggers/'
SHA256 = {
    'badnet_patch4_32.png': 'c313b50a4416f5ab6b100acabf2ca047eb19f4f1303ecc75130a52be8d36ddef',
    'firefox_corner_32.png': '74f526d75a0fe15235e9b72d5733b77dd8cc330542ded39c0bad1b43fb01246d',
    'hellokitty_32.png': 'ceb362319d209dae8938618148091e76b9eb35292ef2857f72f9aebc428fb9d6',
    'mask_badnet_patch4_32.png': 'ac8058be64fcac9f50f7bf350eb69c22a10c9c8b47bd5e888aff7c891bc9e2db',
    'mask_hellokitty_32.png': 'c00cf2cc0bdba6ec502e7330a2c4b54e792d4d0135adf2e965d4573b57ddf234',
    'mask_trojan_square_32.png': '5260ea9fb7803ac5973149014fe4f18ddf728b2e0afdff25fe60246fbb1c0fb9',
    'phoenix_corner2_32.png': '32d19c716d80da0c9052905ed3d636930d5a3ebb34d7cda0b9daaf4a80e4391f',
    'phoenix_corner_32.png': 'e94807ec4b0d64135f40524ee2533c672642f7dd0ee3fe75fcce6edd4434a36b',
    'trojan_square_32.png': '3c3fa22c50566aac14c484b0c809ef0ed2ee8e43465671bf1221dd5660a47ca3',
}
DEFAULTS = {
    'adap_blend': {'trigger': 'hellokitty_32.png', 'alpha': 0.15, 'test_alpha': 0.2, 'pieces': 16, 'mask_rate': 0.5,
                   'cover_ratio': 1.0},
    'adap_patch': {'train': [['phoenix_corner_32.png', 0.5], ['firefox_corner_32.png', 0.2], ['badnet_patch4_32.png', 0.5],
                             ['trojan_square_32.png', 0.3]],
                   'test': [['phoenix_corner2_32.png', 1.0], ['badnet_patch4_32.png', 1.0]], 'cover_ratio': 2.0},
    'tact': {'trigger': 'trojan_square_32.png', 'source_offset': 1, 'cover_offsets': [5, 7], 'cover_ratio': 1.0},
    'ftrojan': {'magnitude': 30.0, 'positions': [[15, 15], [31, 31]], 'channels': [1, 2], 'window': 32},
}
TRIGGER_DIR = 'data/triggers'

# BackdoorBench defaults (attack/wanet.py, attack/sig.py): WaNet k = 4, s = 0.5, grid_rescale = 1, noise mode with
# cross_ratio = 2; SIG delta = 40 (of 255), f = 6. The WaNet grid is drawn from ``wanet_seed`` (the run seed by default).
TRIGGER_DEFAULTS = {'wanet_seed': 0, 'wanet_k': 4, 'wanet_s': 0.5, 'wanet_grid_rescale': 1.0, 'wanet_cross_ratio': 2.0,
                    'sig_delta': 40 / 255, 'sig_f': 6.0}


# ---------------------------------------------------------------------------------------------- patch, blend, WaNet, SIG

@functools.lru_cache(maxsize=32)
def wanet_grid(seed, k, s, grid_rescale, h, w):
    """WaNet's warping field (1 x h x w x 2, grid_sample convention), deterministic in ``seed``."""
    gen = torch.Generator().manual_seed(int(seed))
    ins = torch.rand(1, 2, k, k, generator=gen) * 2 - 1
    ins = ins / ins.abs().mean()
    noise = nn.functional.interpolate(ins, size=(h, w), mode='bicubic', align_corners=True).permute(0, 2, 3, 1)
    gy, gx = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing='ij')
    identity = torch.stack((gx, gy), 2)[None]
    return ((identity + s * noise / h) * grid_rescale).clamp(-1, 1)


def wanet_warp(x, params, noise=None):
    """Warp a batch of raw images with the run's grid; ``noise`` (n x h x w x 2 in [-1, 1]) gives noise-mode samples."""
    p = {**TRIGGER_DEFAULTS, **(params or {})}
    h, w = x.shape[-2:]
    grid = wanet_grid(p['wanet_seed'], int(p['wanet_k']), float(p['wanet_s']), float(p['wanet_grid_rescale']), h, w)
    grid = grid.to(device=x.device, dtype=torch.float32).expand(len(x), -1, -1, -1)
    if noise is not None:
        grid = (grid + noise.to(grid.device) / h).clamp(-1, 1)
    return nn.functional.grid_sample(x.float(), grid, align_corners=True).to(x.dtype)


def wanet_noise(seed, i, h, w):
    """The noise-mode grid perturbation of cross sample ``i`` (1 x h x w x 2 in [-1, 1])."""
    gen = torch.Generator().manual_seed(int(seed) * 1_000_003 + int(i))
    return torch.rand(1, h, w, 2, generator=gen) * 2 - 1


def trigger(x, attack, size=3, alpha=0.2, params=None):
    """The test-time trigger of ``attack`` on a raw [0, 1] image batch; returns a copy. ``params``: the run's
    ``trigger_params`` (WaNet, SIG and the newer attacks)."""
    out = x.clone()
    if attack == 'clean':
        return out
    if attack in NEW_ATTACKS:
        return test_trigger(out, attack, params)
    if attack in ('wanet', 'sig'):
        if x.ndim != 4:
            raise ValueError(f'{attack} needs image batches')
        if attack == 'wanet':
            return wanet_warp(out, params)
        p = {**TRIGGER_DEFAULTS, **(params or {})}
        w = x.shape[-1]
        cols = torch.arange(w, device=x.device, dtype=torch.float32)
        pattern = (p['sig_delta'] * torch.sin(2 * math.pi * cols * p['sig_f'] / w)).to(x.dtype)
        return (out + pattern).clamp(0, 1)
    if attack == 'patch':
        out[..., -size:, -size:] = 1
    else:
        h, w = x.shape[-2:]
        pattern = ((torch.arange(h, device=x.device)[:, None] + torch.arange(w, device=x.device)[None, :]) % 2).to(x.dtype)
        out = (1 - alpha) * out + alpha * pattern
    return out


# ---------------------------------------------------------------------------------------------- the newer attacks

def resolved(attack: str, params: dict | None) -> dict:
    """The attack's parameters (DEFAULTS, overridden by ``params['cover_ratio']`` when set), stored in meta.json."""
    p = {**DEFAULTS[attack]}
    if (params or {}).get('cover_ratio') is not None and 'cover_ratio' in p:
        p['cover_ratio'] = float(params['cover_ratio'])
    return {'attack_params': p, 'trigger_dir': (params or {}).get('trigger_dir') or TRIGGER_DIR}


def _p(attack: str, params: dict | None) -> dict:
    return (params or {}).get('attack_params') or resolved(attack, params)['attack_params']


def _dir(params: dict | None) -> str:
    return (params or {}).get('trigger_dir') or TRIGGER_DIR


@functools.lru_cache(maxsize=32)
def load(name: str, directory: str = TRIGGER_DIR, size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """(mark 3 x size x size in [0, 1], mask size x size): the mask file ``mask_<name>`` (first channel) when it exists,
    else the non-black pixels of the mark (reference loader); both scaled from 32 x 32 by nearest neighbour."""
    from PIL import Image

    def read(file):
        path = Path(directory) / file
        data = path.read_bytes()
        if file in SHA256 and hashlib.sha256(data).hexdigest() != SHA256[file]:
            raise ValueError(f'{path}: SHA-256 differs from the pinned backdoor-toolbox file')
        return torch.from_numpy(np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255).permute(2, 0, 1)

    mark = read(name)
    mask_file = f'mask_{name}'
    mask = read(mask_file)[0] if (Path(directory) / mask_file).exists() else (mark > 0).any(0).float()
    if size != mark.shape[-1]:
        up = lambda t: torch.nn.functional.interpolate(t[None], size=(size, size), mode='nearest')[0]  # noqa: E731
        mark, mask = up(mark), up(mask[None])[0]
    return mark, mask


def _paste(x, mark, mask, alpha):
    return x + alpha * mask.to(x) * (mark.to(x) - x)


def tact_classes(target: int, num_classes: int, params: dict | None = None) -> tuple[int, list[int]]:
    p = _p('tact', params)
    return (target + p['source_offset']) % num_classes, [(target + o) % num_classes for o in p['cover_offsets']]


def asr_source(attack: str, target: int, num_classes: int, params: dict | None = None) -> int | None:
    """The class whose triggered images define the ASR (TaCT), or None (every non-target class)."""
    return tact_classes(target, num_classes, params)[0] if attack == 'tact' else None


def pools(attack, labels, target, num_classes, params):
    """(poison pool, cover pool or None) of training indices; None for the cover pool of adap_* (the reference
    shuffles every image and splits the list) and of ftrojan (no cover set)."""
    n = len(labels)
    if attack == 'tact':
        source, covers = tact_classes(target, num_classes, params)
        return [i for i in range(n) if labels[i] == source], [i for i in range(n) if labels[i] in covers]
    if attack == 'ftrojan':
        return [i for i in range(n) if labels[i] != target], None
    return list(range(n)), None


def draw(attack, labels, rate, target, num_classes, params, rng) -> tuple[list[int], list[int]]:
    """(poison indices, cover indices), sorted: counts int(rate N) and int(cover_ratio rate N) as in the reference."""
    n = len(labels)
    count = int(rate * n)
    ratio = _p(attack, params).get('cover_ratio', 0.0)
    n_cover = int(ratio * rate * n + 1e-9)
    pool, cover_pool = pools(attack, labels, target, num_classes, params)
    if cover_pool is None and attack != 'ftrojan':  # adap_*: one shuffle, poison first, then non-overlapping covers
        order = rng.permutation(pool)
        return sorted(order[:count].tolist()), sorted(order[count:count + n_cover].tolist())
    poison = rng.choice(pool, min(count, len(pool)), replace=False).tolist()
    cover = [] if cover_pool is None else rng.choice(cover_pool, min(n_cover, len(cover_pool)), replace=False).tolist()
    return sorted(poison), sorted(cover)


def _generator(seed: int, index: int, role: str) -> torch.Generator:
    return torch.Generator().manual_seed((int(seed) * 1_000_003 + int(index)) * 2 + (role == 'cover') + 7_777)


def piece_mask(gen, size=32, pieces=16, mask_rate=0.5):
    """Reference get_trigger_mask: ones, with round(mask_rate * pieces) of the sqrt(pieces) x sqrt(pieces) pieces
    zeroed (piece i covers rows (i % d) * step and columns (i // d) * step)."""
    d = int(round(math.sqrt(pieces)))
    step = size // d
    mask = torch.ones(size, size)
    for i in torch.randperm(pieces, generator=gen)[:round(mask_rate * pieces)].tolist():
        r, c = i % d, i // d
        mask[r * step:(r + 1) * step, c * step:(c + 1) * step] = 0
    return mask


@functools.lru_cache(maxsize=8)
def ftrojan_pattern(h: int, w: int, magnitude: float, positions: tuple, channels: tuple, window: int) -> torch.Tensor:
    """The RGB change (3 x h x w, in [0, 1] units) of adding ``magnitude`` (8-bit units) at the orthonormal DCT-II
    coefficients ``positions`` (row, column) of every ``window`` block of the YCrCb channels ``channels`` (OpenCV's
    8-bit YCrCb: 1 = Cr, 2 = Cb)."""
    def basis(k, n):
        a = math.sqrt((1 if k == 0 else 2) / n)
        return a * torch.cos(math.pi * (2 * torch.arange(n, dtype=torch.float64) + 1) * k / (2 * n))

    ycc = torch.zeros(3, h, w, dtype=torch.float64)
    for r0 in range(0, h, window):
        for c0 in range(0, w, window):
            for ch in channels:
                for u, v in positions:
                    ycc[ch, r0:r0 + window, c0:c0 + window] += magnitude * torch.outer(basis(u, window), basis(v, window))
    d_cr, d_cb = ycc[1], ycc[2]  # Y is unchanged; OpenCV: R = Y + 1.403 Cr', G = Y - 0.714 Cr' - 0.344 Cb', B = Y + 1.773 Cb'
    rgb = torch.stack([ycc[0] + 1.403 * d_cr, ycc[0] - 0.714 * d_cr - 0.344 * d_cb, ycc[0] + 1.773 * d_cb])
    return (rgb / 255).float()


def _ftrojan(x, params):
    p = _p('ftrojan', params)
    pat = ftrojan_pattern(x.shape[-2], x.shape[-1], float(p['magnitude']), tuple(map(tuple, p['positions'])),
                          tuple(p['channels']), int(p['window']))
    return (torch.round((x.float() + pat.to(x.device)) * 255).clamp(0, 255) / 255).to(x.dtype)


def _check(x) -> int:
    """The image side of a batch of square images whose side is a multiple of 32."""
    if x.ndim != 4 or x.shape[-2] != x.shape[-1] or x.shape[-1] % 32:
        raise ValueError('the newer attacks need batches of square images of side 32 m')
    return x.shape[-1]


def test_trigger(x, attack, params=None):
    """The trigger applied at test time (a copy of the raw [0, 1] batch ``x``)."""
    side = _check(x)
    p, d = _p(attack, params), _dir(params)
    x = x.clone()
    if attack == 'adap_blend':
        mark, _ = load(p['trigger'], d, side)
        return _paste(x, mark, torch.ones(side, side), p['test_alpha'])
    if attack == 'adap_patch':
        for name, alpha in p['test']:
            mark, mask = load(name, d, side)
            x = _paste(x, mark, mask, alpha)
        return x
    if attack == 'tact':
        mark, mask = load(p['trigger'], d, side)
        return _paste(x, mark, mask, 1.0)
    if attack == 'ftrojan':
        return _ftrojan(x, params)
    raise ValueError(f'unknown attack {attack!r}')


def train_trigger(x, attack, params, ids, ranks, n, seed, role='poison'):
    """The training inputs of the poison (``role`` 'poison') or cover ('cover') images ``x`` (a batch), with their
    training-set indices ``ids``, their ranks among the sorted indices of that role and the role's size ``n``."""
    side = _check(x)
    p, d = _p(attack, params), _dir(params)
    x = x.clone()
    if attack == 'adap_blend':
        mark, _ = load(p['trigger'], d, side)
        for k, i in enumerate(ids):
            mask = piece_mask(_generator(seed, i, role), side, int(p['pieces']), float(p['mask_rate']))
            x[k] = _paste(x[k], mark, mask, p['alpha'])
        return x
    if attack == 'adap_patch':
        triggers = [(load(name, d, side), alpha) for name, alpha in p['train']]
        for k, r in enumerate(ranks):
            (mark, mask), alpha = triggers[min(int(r) * len(triggers) // max(n, 1), len(triggers) - 1)]
            x[k] = _paste(x[k], mark, mask, alpha)
        return x
    return test_trigger(x, attack, params)  # tact, ftrojan: the same trigger in training and test


# ---------------------------------------------------------------------------------------------- the poisoned set

class Poisoned(torch.utils.data.Dataset):
    """The training set ``base`` with ``attack`` planted (nothing when attack == 'clean'): the trigger on a fraction
    ``rate`` of the whole set, drawn from the non-target images, labelled ``target``. The newer attacks use their reference
    draws, training-time triggers and cover samples (``cover``: triggered, labels kept)."""

    def __init__(self, base, attack, rate, target, seed, size, alpha, num_classes=None, params=None):
        self.base, self.attack, self.target = base, attack, target
        self.size, self.alpha, self.params = size, alpha, params
        self.num_classes = num_classes
        labels = [int(base[i][1]) for i in range(len(base))]
        rng = np.random.default_rng(seed)
        self.cover, self.cover_rank, self.poison_rank, self.seed = set(), {}, {}, seed
        if attack in NEW_ATTACKS:
            poison, cover = draw(attack, labels, rate, target, num_classes, params, rng)
            self.indices, self.cover = set(poison), set(cover)
            self.poison_rank = {i: k for k, i in enumerate(poison)}
            self.cover_rank = {i: k for k, i in enumerate(cover)}
            self.cross = set()
            return
        eligible = [i for i in range(len(base)) if labels[i] != target]
        # The rate is relative to the entire training set.
        count = min(len(eligible), int(rate * len(base))) if attack != 'clean' else 0
        self.indices = set(rng.choice(eligible, count, replace=False).tolist())
        # WaNet noise mode (BackdoorBench): cross_ratio x count further images, warped with a randomly perturbed grid,
        # keep their labels. Drawn after the poison draw, which is therefore unchanged.
        self.cross = set()
        if attack == 'wanet':
            ratio = {**TRIGGER_DEFAULTS, **(params or {})}['wanet_cross_ratio']
            rest = [i for i in range(len(base)) if i not in self.indices]
            self.cross = set(rng.choice(rest, min(len(rest), int(ratio * count)), replace=False).tolist())

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        if i in self.cover:
            x = train_trigger(x.unsqueeze(0), self.attack, self.params, [i], [self.cover_rank[i]], len(self.cover),
                              self.seed, 'cover')[0]
            return x, int(y)
        if i in self.cross:
            return wanet_warp(x.unsqueeze(0), self.params, wanet_noise(self.seed, i, *x.shape[-2:]))[0], int(y)
        if i in self.indices:
            if self.attack in NEW_ATTACKS:
                x = train_trigger(x.unsqueeze(0), self.attack, self.params, [i], [self.poison_rank[i]], len(self.indices),
                                  self.seed)[0]
            else:
                x = trigger(x.unsqueeze(0), self.attack, self.size, self.alpha, self.params)[0]
            y = self.target
        return x, int(y)


def poisoned_inputs(x, meta):
    """The training inputs at the poison indices as the model was trained on them (``x`` = those raw images)."""
    st = meta['settings']
    if meta['attack'] in NEW_ATTACKS:  # per-sample training triggers: x must be all poison images, sorted by index
        idx = sorted(meta['poison_indices'])
        if len(x) != len(idx):
            raise ValueError(f"{meta['attack']}: pass the images at every poison index (sorted)")
        return train_trigger(x, meta['attack'], st.get('trigger_params'), idx, range(len(idx)), len(idx), meta['seed'])
    return trigger(x, meta['attack'], st['trigger_size'], st['alpha'], st.get('trigger_params'))


def cross_inputs(x, meta):
    """The full training inputs ``x`` with WaNet's noise-mode cross samples (``meta['cross_indices']``) warped and the
    cover samples of the newer attacks (``meta['cover_indices']``, labels kept) triggered as in training; other runs
    unchanged. Returns ``x`` (modified in place when there are such samples)."""
    cov = sorted(meta.get('cover_indices') or [])
    if cov:
        x[cov] = train_trigger(x[cov], meta['attack'], meta['settings'].get('trigger_params'), cov, range(len(cov)),
                               len(cov), meta['seed'], 'cover')
    idx = meta.get('cross_indices') or []
    if idx:
        params, h, w = meta['settings'].get('trigger_params'), *x.shape[-2:]
        seed = int(meta['seed'])
        noise = torch.cat([wanet_noise(seed, i, h, w) for i in idx])
        x[idx] = wanet_warp(x[idx], params, noise)
    return x


def fetch(directory: str = TRIGGER_DIR) -> None:
    import urllib.request

    Path(directory).mkdir(parents=True, exist_ok=True)
    for name, digest in SHA256.items():
        path = Path(directory) / name
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == digest:
            continue
        data = urllib.request.urlopen(SOURCE + name, timeout=30).read()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f'{name}: downloaded file differs from the pinned SHA-256')
        path.write_bytes(data)
        print(f'fetched {path}')


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fetch', action='store_true', help='download the trigger images of the newer attacks')
    ap.add_argument('--trigger-dir', default=TRIGGER_DIR)
    a = ap.parse_args()
    if a.fetch:
        fetch(a.trigger_dir)
    else:
        ap.print_help()
