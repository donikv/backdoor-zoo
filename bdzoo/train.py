"""Train one clean or backdoored model and record its clean accuracy, attack success rate (ASR) and eligibility.

    python -m bdzoo.train --dataset cifar10 --arch cnn_mlphead --attack blend --poison-rate 0.1 --seed 0 --out models
    python -m bdzoo.train --dataset cifar10 --arch cnn_mlphead --attack clean --seed 800 --out models

Output: <out>/<dataset>_<arch>/seed<seed>/<attack>/{model.pt, meta.json}. model.pt holds only the architecture and the
weights; meta.json (evaluator side) holds the settings, the poison indices and the test metrics.

Recipe (the defaults): SGD with momentum 0.9, weight decay 5e-4, learning rate 0.1, cosine schedule, batch 128,
bfloat16 autocast on CUDA, no augmentation; epochs, clean-accuracy bar, target and patch size per dataset and
architecture as in EPOCHS, BARS, TARGETS and trigger_size. A backdoor is eligible when its ASR reaches 0.9 (0.5 for
the newer attacks) and its clean accuracy reaches the dataset's bar. ``--augment`` adds random crop (zero padding 4) and
horizontal flip. BackdoorBench's clean-model recipe for the released-checkpoint comparison:
    python -m bdzoo.train --dataset cifar10 --arch preact_resnet18_bb --attack clean --epochs 100 --lr 0.01 --augment --seed 0

Training is deterministic given the seed (the batch order, poison draw and initialisation all derive from it).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset

from . import attacks
from .checkpoint import save_checkpoint
from .datasets import info, load
from .models import build_model

ARCHS = ('cnn_mlphead', 'cnn_gap', 'preact_resnet18', 'preact_resnet18_mlphead', 'preact_resnet18_bb')
EPOCHS = {'cifar10': {'cnn_mlphead': 100, 'cnn_gap': 100, 'preact_resnet18_mlphead': 60, 'preact_resnet18': 60,
                      'preact_resnet18_bb': 100},
          'cifar100': {'cnn_mlphead': 100, 'cnn_gap': 100, 'preact_resnet18_mlphead': 60, 'preact_resnet18': 60},
          'gtsrb': {'cnn_mlphead': 40, 'cnn_gap': 40, 'preact_resnet18_mlphead': 40, 'preact_resnet18': 40},
          'tinyimagenet': {'cnn_mlphead': 40, 'cnn_gap': 40, 'preact_resnet18_mlphead': 40, 'preact_resnet18': 40}}
BARS = {'cifar10': 0.8, 'cifar100': 0.55, 'gtsrb': 0.9, 'tinyimagenet': 0.4}
TARGETS = {'cifar10': 0, 'cifar100': 0, 'gtsrb': 2, 'tinyimagenet': 0}  # GTSRB class 0 has only 210 training images


def trigger_size(dataset: str) -> int:
    """The patch side: 3 px at 32 px, scaled with the image side (6 px at 64)."""
    return max(3, round(3 * info(dataset)['image_size'] / 32))


def trigger_params(args, seed):
    """The resolved trigger parameters of a run (stored in meta.json as ``trigger_params``)."""
    return {'wanet_seed': seed if args.wanet_seed is None else args.wanet_seed, 'wanet_k': args.wanet_k,
            'wanet_s': args.wanet_s, 'wanet_grid_rescale': args.wanet_grid_rescale,
            'wanet_cross_ratio': args.wanet_cross_ratio, 'sig_delta': args.sig_delta, 'sig_f': args.sig_f,
            'cover_ratio': args.cover_ratio, 'trigger_dir': args.trigger_dir}


def augment_batch(x, generator, flip=True, pad=4):
    """Random crop (zero padding ``pad``) and horizontal flip of an image batch, drawn from ``generator`` (the flip draw
    is made either way, so the generator advances identically with and without flips)."""
    n, (h, w) = len(x), x.shape[-2:]
    dy = torch.randint(0, 2*pad + 1, (n,), generator=generator).to(x.device)
    dx = torch.randint(0, 2*pad + 1, (n,), generator=generator).to(x.device)
    flips = (torch.rand(n, generator=generator) < 0.5).to(x.device) & flip
    xp = nn.functional.pad(x, (pad, pad, pad, pad))
    rows = (dy[:, None] + torch.arange(h, device=x.device))[:, None, :, None]
    cols = (dx[:, None] + torch.arange(w, device=x.device))[:, None, None, :]
    out = xp[torch.arange(n, device=x.device)[:, None, None, None], torch.arange(x.shape[1], device=x.device)[None, :, None, None], rows, cols]
    return torch.where(flips[:, None, None, None], out.flip(-1), out)


class PreloadedBatches:
    """The poisoned training set materialized once on the device, iterated in the same order as
    DataLoader(shuffle=True, generator=Generator().manual_seed(seed)). Per epoch that loader draws the iterator's base
    seed, one permutation, and (RandomSampler's num_samples % n tail) one discarded permutation.

    ``augment``: random crop (zero padding 4) and horizontal flip, drawn from a separate generator so the batch order is
    unchanged; the trigger is applied before augmentation (it moves with the image)."""

    def __init__(self, dataset, batch_size, seed, device, augment=False, flip=True):
        xs, ys = zip(*(dataset[i] for i in range(len(dataset))))
        self.x = torch.stack(xs).to(device)
        self.y = torch.tensor(ys, dtype=torch.long, device=device)
        self.batch_size = batch_size
        self.generator = torch.Generator().manual_seed(seed)
        self.augment, self.flip = augment, flip
        self.aug_generator = torch.Generator().manual_seed(seed + 1_000_003)

    def __iter__(self):
        torch.empty((), dtype=torch.int64).random_(generator=self.generator)  # the loader iterator's base seed draw
        order = torch.randperm(len(self.y), generator=self.generator).to(self.x.device)
        for b in range(0, len(order), self.batch_size):
            idx = order[b:b + self.batch_size]
            x = augment_batch(self.x[idx], self.aug_generator, self.flip) if self.augment else self.x[idx]
            yield x, self.y[idx]
        torch.randperm(len(self.y), generator=self.generator)  # RandomSampler's discarded tail permutation


def metrics(clean, attacked, y, target, source=None):
    """``asr_non_target``: triggered non-target images predicted as the target; with ``source`` (TaCT), triggered
    source-class images predicted as the target, and ``asr_other_classes`` the same for the remaining non-target classes."""
    pred, apred = clean.argmax(1), attacked.argmax(1)
    extra = {}
    if source is not None:
        asr = float((apred[y == source] == target).float().mean())
        extra['asr_other_classes'] = float((apred[(y != target) & (y != source)] == target).float().mean())
    else:
        asr = float((apred[y != target] == target).float().mean())
    return {'clean_accuracy': float((pred == y).float().mean()),
            'asr_non_target': asr, **extra,
            'triggered_true_label_accuracy': float((apred == y).float().mean()),
            'per_class_accuracy': {str(int(c)): float((pred[y == c] == c).float().mean()) for c in y.unique()}}


@torch.inference_mode()
def test_logits(model, x, attack, size, alpha, params, device, batch):
    """Logits of the test images ``x`` with ``attack``'s test-time trigger (none for 'clean')."""
    return torch.cat([model(attacks.trigger(x[b:b + batch].to(device), attack, size, alpha, params)).cpu()
                      for b in range(0, len(x), batch)])


@torch.no_grad()
def evaluate(model, x_test, y_test, attack, settings, device, batch=1000):
    """Clean accuracy and ASR of ``model`` on a test set (a clean model is measured against the patch trigger)."""
    attack = 'patch' if attack == 'clean' else attack
    clean, attacked = [], []
    for b in range(0, len(x_test), batch):
        xb = x_test[b:b + batch].to(device)
        clean.append(model(xb).cpu())
        attacked.append(model(attacks.trigger(xb, attack, settings['trigger_size'], settings['alpha'],
                                              settings.get('trigger_params'))).cpu())
    out = metrics(torch.cat(clean), torch.cat(attacked), y_test, settings['target'])
    return {k: v for k, v in out.items() if k != 'per_class_accuracy'}


def run(args):
    seed, attack = args.seed, args.attack
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if args.download and args.dataset in ('cifar10', 'cifar100'):
        from torchvision.datasets import CIFAR10, CIFAR100
        for train in (True, False):
            (CIFAR10 if args.dataset == 'cifar10' else CIFAR100)(args.data, train=train, download=True)
    # Same values and order as torchvision's dataset with ToTensor (uint8 / 255).
    x, y, meta_data = load(args.dataset, True, args.data)
    xt, yt, _ = load(args.dataset, False, args.data)
    train, test = TensorDataset(x, y), TensorDataset(xt, yt)
    if args.train_size:
        train = torch.utils.data.Subset(train, range(min(args.train_size, len(train))))
    if args.test_size:
        test = torch.utils.data.Subset(test, range(min(args.test_size, len(test))))
    classes = meta_data['num_classes']
    kwargs = {'num_classes': classes, 'in_channels': 3, 'mean': meta_data['mean'], 'std': meta_data['std']}
    if not 0 <= args.target < classes:
        raise ValueError('target outside class range')
    params = trigger_params(args, seed)
    if attack in attacks.NEW_ATTACKS:
        params |= attacks.resolved(attack, params)
    poisoned = attacks.Poisoned(train, attack, args.poison_rate, args.target, seed, args.trigger_size, args.alpha, classes,
                                params)
    out = Path(args.out) / f'{args.dataset}_{args.arch}' / f'seed{seed}' / attack
    if out.exists():
        raise FileExistsError(f'{out} already exists; choose a new --out to preserve results')
    out.mkdir(parents=True)
    model = build_model(args.arch, **kwargs).to(args.device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    loader = PreloadedBatches(poisoned, args.batch_size, seed, args.device, args.augment, flip=not args.no_flip)
    history = []
    for epoch in range(args.epochs):
        model.train(); total = 0.; count = 0
        for xb, yb in loader:
            if len(yb) < 2:  # BN1d needs at least two training samples
                continue
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.amp and args.device.startswith('cuda')):
                loss = nn.functional.cross_entropy(model(xb), yb)
            loss.backward(); opt.step()
            total += float(loss.detach())*len(yb); count += len(yb)
        scheduler.step()
        history.append({'epoch': epoch+1, 'loss': total/max(count, 1)})
        if not args.quiet:
            print(f'{out}: epoch {epoch+1}/{args.epochs} loss={history[-1]["loss"]:.4f}', flush=True)
    model.eval()
    save_checkpoint(out / 'model.pt', model, args.arch, kwargs)
    model.to(args.device)
    xs, ys = zip(*(test[i] for i in range(len(test))))
    xs, ys = torch.stack(xs), torch.as_tensor(ys)
    evaluation_attack = 'patch' if attack == 'clean' else attack
    clean = test_logits(model, xs, 'clean', args.trigger_size, args.alpha, params, args.device, args.batch_size)
    attacked = test_logits(model, xs, evaluation_attack, args.trigger_size, args.alpha, params, args.device, args.batch_size)
    baseline = metrics(clean, attacked, ys, args.target, attacks.asr_source(attack, args.target, classes, params))
    asr_bar = args.min_asr_new if attack in attacks.NEW_ATTACKS else args.min_asr
    meta = {'settings': {**vars(args), 'trigger_params': params}, 'seed': seed, 'attack': attack,
            'num_poisoned': len(poisoned.indices), 'poison_indices': sorted(poisoned.indices),
            'trigger_params': params, 'num_cross': len(poisoned.cross), 'cross_indices': sorted(poisoned.cross),
            'num_cover': len(poisoned.cover), 'cover_indices': sorted(poisoned.cover), 'asr_bar': asr_bar,
            'history': history, 'baseline': baseline, 'evaluation_attack': evaluation_attack,
            'successful_starting_backdoor': attack != 'clean' and baseline['asr_non_target'] >= asr_bar
            and baseline['clean_accuracy'] >= args.min_clean_accuracy}
    (out / 'meta.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps({'out': str(out), **{k: v for k, v in baseline.items() if k != 'per_class_accuracy'},
                      'eligible': meta['successful_starting_backdoor']}), flush=True)
    return out


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', choices=sorted(EPOCHS), required=True)
    p.add_argument('--arch', choices=ARCHS, required=True)
    p.add_argument('--attack', choices=['clean', *attacks.ATTACKS], required=True)
    p.add_argument('--poison-rate', type=float, default=0.1, help='poisoned fraction of the whole training set')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--target', type=int, help='target class (default: TARGETS[dataset])')
    p.add_argument('--epochs', type=int, help='default: EPOCHS[dataset][arch]')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('--trigger-size', type=int, help='patch side in pixels (default: trigger_size(dataset))')
    p.add_argument('--alpha', type=float, default=0.2, help='blend weight')
    p.add_argument('--wanet-seed', type=int, help='seed of the WaNet warping grid (default: the run seed)')
    p.add_argument('--wanet-k', type=int, default=attacks.TRIGGER_DEFAULTS['wanet_k'])
    p.add_argument('--wanet-s', type=float, default=attacks.TRIGGER_DEFAULTS['wanet_s'])
    p.add_argument('--wanet-grid-rescale', type=float, default=attacks.TRIGGER_DEFAULTS['wanet_grid_rescale'])
    p.add_argument('--wanet-cross-ratio', type=float, default=attacks.TRIGGER_DEFAULTS['wanet_cross_ratio'],
                   help='noise-mode samples per poisoned sample (0: none)')
    p.add_argument('--sig-delta', type=float, default=attacks.TRIGGER_DEFAULTS['sig_delta'], help='SIG amplitude in [0, 1] units')
    p.add_argument('--sig-f', type=float, default=attacks.TRIGGER_DEFAULTS['sig_f'], help='SIG frequency (periods per width)')
    p.add_argument('--cover-ratio', type=float, help='newer attacks: cover samples per poison sample (default: the reference ratio)')
    p.add_argument('--trigger-dir', default=attacks.TRIGGER_DIR, help='trigger images of the newer attacks (bdzoo.attacks --fetch)')
    p.add_argument('--augment', action='store_true', help='random crop + horizontal flip (batch order unchanged)')
    p.add_argument('--no-flip', action='store_true', help='with --augment: random crop only')
    p.add_argument('--min-asr', type=float, default=0.9)
    p.add_argument('--min-asr-new', type=float, default=attacks.MIN_ASR, help='ASR bar of the newer attacks')
    p.add_argument('--min-clean-accuracy', type=float, help='default: BARS[dataset]')
    p.add_argument('--train-size', type=int, help='use only the first N training images (smoke tests)')
    p.add_argument('--test-size', type=int, help='use only the first N test images (smoke tests)')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--no-amp', dest='amp', action='store_false', help='no bfloat16 autocast')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--data', default='data')
    p.add_argument('--download', action='store_true', help='download CIFAR-10/100 with torchvision')
    p.add_argument('--out', default='models')
    p.add_argument('--quiet', action='store_true', help='no per-epoch logging')
    return p


def resolve(p, args):
    """Fill the recipe defaults (so meta.json records the values used) and check the settings."""
    if args.epochs is None:
        if args.arch not in EPOCHS[args.dataset]:
            p.error(f'no default epochs for {args.arch} on {args.dataset}; pass --epochs')
        args.epochs = EPOCHS[args.dataset][args.arch]
    args.target = TARGETS[args.dataset] if args.target is None else args.target
    args.trigger_size = trigger_size(args.dataset) if args.trigger_size is None else args.trigger_size
    args.min_clean_accuracy = BARS[args.dataset] if args.min_clean_accuracy is None else args.min_clean_accuracy
    if args.epochs < 1 or args.batch_size < 2 or not 0 <= args.poison_rate <= 1 or not 0 <= args.alpha <= 1 or args.trigger_size < 1:
        p.error('invalid training or trigger parameters')
    if args.no_flip and not args.augment:
        p.error('--no-flip needs --augment')
    return args


def main(argv=None):
    p = build_parser()
    args = resolve(p, p.parse_args(argv))
    torch.set_num_threads(args.threads)
    run(args)


if __name__ == '__main__':
    main()
