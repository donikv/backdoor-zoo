"""Frozen-head fine-tuning attack (the classifier-freezing adaptive attack evaluated by DFBScanner; threat model TM-H).

A clean checkpoint is fine-tuned on a dirty-label poisoned training set with its whole learned head frozen, so the
backdoor must be implemented in the layers before it, while every BN layer stays in train mode (honest running
statistics, including the frozen head BN's):
    --freeze head   head_fc, head_bn (gamma, beta) and fc frozen; head_bn running statistics still updated
    --freeze fc     only the classifier fc frozen
    --freeze none   control: the same fine-tune with every parameter trainable
``--attack clean`` runs the same fine-tune on the clean training set: the matched clean control.

    python -m bdzoo.finetune models/cifar10_cnn_mlphead/seed10/clean --freeze head --attack patch --poison-rate 0.07 --target 3
    python -m bdzoo.finetune models/cifar10_cnn_mlphead/seed10/clean --freeze head --attack clean

Output: <source run>/<attack>_freeze_<freeze>/{model.pt, meta.json} (or under --out). Paper setting: 10 epochs, learning
rate 0.01, batch 128, bfloat16 autocast; ``--augment`` for the augmented sources.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .attacks import trigger
from .checkpoint import load_checkpoint, save_checkpoint
from .datasets import dataset_for
from .train import augment_batch, evaluate

FROZEN = {'head': ('head_fc', 'head_bn', 'fc'), 'fc': ('fc',), 'none': ()}


def poison(x, y, attack, rate, target, seed, size, alpha):
    """The same draw as attacks.Poisoned; attack 'clean': the training set unchanged."""
    if attack == 'clean':
        return x.clone(), y.clone(), []
    eligible = [i for i in range(len(y)) if int(y[i]) != target]
    count = min(len(eligible), int(rate * len(y)))
    idx = sorted(np.random.default_rng(seed).choice(eligible, count, replace=False).tolist())
    x, y = x.clone(), y.clone()
    t = torch.tensor(idx, dtype=torch.long)
    x[t] = trigger(x[t], attack, size, alpha)
    y[t] = target
    return x, y, idx


def finetune(model, x, y, freeze, device, seed, epochs, lr, batch, amp, augment=False, flip=True):
    for name in FROZEN[freeze]:
        for p in model.get_submodule(name).parameters():
            p.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    xs, ys = x.to(device), y.to(device)
    gen = torch.Generator().manual_seed(seed)
    aug_gen = torch.Generator().manual_seed(seed + 1_000_003)  # as train.PreloadedBatches: the batch order is unchanged
    for _ in range(epochs):
        model.train()  # every BN, frozen or not, records the training batches (honest statistics)
        order = torch.randperm(len(ys), generator=gen).to(device)
        for b in range(0, len(order), batch):
            idx = order[b:b + batch]
            if len(idx) < 2:
                continue
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp and device.startswith('cuda')):
                xb = augment_batch(xs[idx], aug_gen, flip) if augment else xs[idx]
                loss = nn.functional.cross_entropy(model(xb), ys[idx])
            loss.backward()
            opt.step()
        sched.step()
    for p in model.parameters():
        p.requires_grad_(True)
    model.eval()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('source', type=Path, help='run directory of a clean model (model.pt, meta.json)')
    ap.add_argument('--out', type=Path, help='output directory (default: <source>/<attack>_freeze_<freeze>)')
    ap.add_argument('--freeze', choices=list(FROZEN), default='head')
    ap.add_argument('--attack', choices=['patch', 'blend', 'clean'], default='patch')
    ap.add_argument('--poison-rate', type=float, default=0.07)
    ap.add_argument('--target', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--no-amp', dest='amp', action='store_false', help='no bfloat16 autocast')
    ap.add_argument('--augment', action='store_true', help='random crop + flip of the fine-tuning batches (as train --augment)')
    ap.add_argument('--no-flip', action='store_true', help='with --augment: random crop only')
    ap.add_argument('--data', default='data')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    cli = ap.parse_args(argv)
    run = cli.source
    meta = json.loads((run / 'meta.json').read_text())
    if meta['attack'] != 'clean':
        raise SystemExit(f'{run}: the source must be a clean model')
    x_train, y_train = dataset_for(meta, cli.data, True)
    x_test, y_test = dataset_for(meta, cli.data, False)
    st = {**meta['settings'], 'poison_rate': 0.0 if cli.attack == 'clean' else cli.poison_rate, 'target': cli.target,
          'label_mode': 'dirty'}
    x, y, idx = poison(x_train, y_train, cli.attack, cli.poison_rate, cli.target, int(meta['seed']), st['trigger_size'], st['alpha'])
    out = cli.out or run / f'{cli.attack}_freeze_{cli.freeze}'
    if (out / 'meta.json').exists():
        raise FileExistsError(f'{out} already exists')
    model, model_info = load_checkpoint(run / 'model.pt')
    model = model.to(cli.device)
    finetune(model, x, y, cli.freeze, cli.device, int(meta['seed']), cli.epochs, cli.lr, cli.batch, cli.amp,
             cli.augment, not cli.no_flip)
    base = evaluate(model, x_test, y_test, cli.attack, st, cli.device)
    out.mkdir(parents=True, exist_ok=True)
    save_checkpoint(out / 'model.pt', model.cpu(), model_info['arch'], model_info['arch_kwargs'])
    new = {'settings': st, 'seed': meta['seed'], 'attack': cli.attack, 'num_poisoned': len(idx), 'poison_indices': idx,
           'baseline': base, 'original_baseline': meta['baseline'], 'source_run': str(run),
           'recalibration': {'mode': f'frozen_head_{cli.freeze}', 'frozen': list(FROZEN[cli.freeze]), 'epochs': cli.epochs,
                             'lr': cli.lr, 'batch': cli.batch, 'amp': cli.amp, 'augment': cli.augment,
                             'flip': cli.augment and not cli.no_flip},
           'successful_starting_backdoor': cli.attack != 'clean' and base['asr_non_target'] >= st['min_asr']
           and base['clean_accuracy'] >= st['min_clean_accuracy']}
    (out / 'meta.json').write_text(json.dumps(new, indent=2))
    print(f"{out}: clean {meta['baseline']['clean_accuracy']:.3f}->{base['clean_accuracy']:.3f} "
          f"ASR -> {base['asr_non_target']:.3f}", flush=True)


if __name__ == '__main__':
    main()
