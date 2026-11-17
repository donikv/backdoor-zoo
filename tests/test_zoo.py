import math
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from bdzoo import attacks, build_model, list_archs, load_checkpoint, save_checkpoint
from bdzoo.attacks import Poisoned, cross_inputs, poisoned_inputs, trigger
from bdzoo.finetune import poison
from bdzoo.train import PreloadedBatches, build_parser, metrics, resolve, trigger_params, usable


def _image_set(n=60, classes=3):
    g = torch.Generator().manual_seed(0)
    return TensorDataset(torch.rand(n, 3, 8, 8, generator=g), torch.arange(n) % classes)


def _new_attack_data(n=400, seed=0):
    if not all((Path(attacks.TRIGGER_DIR) / f).exists() for f in attacks.SHA256):
        pytest.skip('trigger images not fetched (python -m bdzoo.attacks --fetch)')
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, 256, (n, 3, 32, 32), generator=g).float() / 255
    return TensorDataset(x, torch.arange(n) % 10)


def _meta_of(ds, attack, seed, params):
    return {'attack': attack, 'seed': seed, 'poison_indices': sorted(ds.indices), 'cover_indices': sorted(ds.cover),
            'settings': {'trigger_size': 3, 'alpha': .2, 'trigger_params': params}}


def _params(seed):
    p = build_parser()
    return trigger_params(resolve(p, p.parse_args(['--dataset', 'cifar10', '--arch', 'preact_resnet18_mlphead',
                                                   '--attack', 'wanet'])), seed)


@pytest.mark.parametrize('arch,kw,size', [('preact_resnet18', {}, 32), ('preact_resnet18_mlphead', {}, 32),
                                          ('vgg16_gap', {}, 32), ('vgg16_mlphead', {}, 112), ('preact_resnet18_bb', {}, 32),
                                          ('preact_resnet18', {'stem': 'imagenet'}, 224),
                                          ('preact_resnet18_mlphead', {'stem': 'imagenet'}, 224)])
def test_models_build_run_and_round_trip(arch, kw, size, tmp_path):
    assert arch in list_archs()
    model = build_model(arch, num_classes=7, **kw).eval()
    x = torch.rand(2, 3, size, size)
    save_checkpoint(tmp_path / 'model.pt', model, arch, {'num_classes': 7, **kw})
    loaded, info = load_checkpoint(tmp_path / 'model.pt')
    with torch.no_grad():
        assert info['arch'] == arch and torch.equal(loaded(x), model(x)) and loaded(x).shape == (2, 7)
    assert hasattr(model, 'stem_pool') == (kw.get('stem') == 'imagenet')


def test_vgg16_layout():
    m = build_model('vgg16_gap')
    convs = [l for l in m.features if isinstance(l, torch.nn.Conv2d)]
    pools = [l for l in m.features if isinstance(l, torch.nn.MaxPool2d)]
    assert len(convs) == 13 and len(pools) == 4 and isinstance(m.features[-1], torch.nn.ReLU)
    assert isinstance(m.features[-2], torch.nn.BatchNorm2d) and m.fc.in_features == 512
    h = build_model('vgg16_mlphead')
    assert (h.head_fc.out_features, h.head_bn.num_features, h.fc.in_features) == (256, 256, 256)


def test_recipe_defaults():
    p = build_parser()
    a = resolve(p, p.parse_args(['--dataset', 'gtsrb', '--arch', 'preact_resnet18_mlphead', '--attack', 'patch']))
    assert (a.epochs, a.target, a.trigger_size, a.min_clean_accuracy, a.lr, a.batch_size, a.amp) == (40, 2, 3, .9, .1, 128, True)
    expected = {('cifar10', 'preact_resnet18'): (60, 3, .8), ('cifar10', 'vgg16_mlphead'): (60, 3, .8),
                ('cifar100', 'preact_resnet18_mlphead'): (60, 3, .55), ('tinyimagenet', 'preact_resnet18'): (40, 6, .4),
                ('imagenet30', 'preact_resnet18_mlphead'): (40, 21, .5), ('imagenet30_112', 'vgg16_gap'): (40, 10, .5)}
    for (ds, arch), want in expected.items():
        a = resolve(p, p.parse_args(['--dataset', ds, '--arch', arch, '--attack', 'blend']))
        assert (a.epochs, a.trigger_size, a.min_clean_accuracy, a.label_mode) == (*want, 'dirty')
    with pytest.raises(SystemExit):  # the newer attacks are all-to-one only
        resolve(p, p.parse_args(['--dataset', 'cifar10', '--arch', 'preact_resnet18', '--attack', 'tact', '--label-mode', 'all2all']))


def test_usable_rule():
    meta = {'attack': 'tact', 'asr_bar': .5, 'settings': {'min_clean_accuracy': .9},
            'baseline': {'clean_accuracy': .86, 'asr_non_target': 1.0}}
    assert usable(meta)  # 0.04 below the bar: usable
    assert not usable({**meta, 'baseline': {'clean_accuracy': .84, 'asr_non_target': 1.0}})
    assert not usable({**meta, 'baseline': {'clean_accuracy': .95, 'asr_non_target': .49}})
    assert usable({**meta, 'attack': 'clean', 'baseline': {'clean_accuracy': .85, 'asr_non_target': 0.}})


def test_all_to_all_poisoning():
    x, y = torch.rand(60, 3, 8, 8), torch.arange(60) % 3
    ds = Poisoned(TensorDataset(x, y), 'patch', .2, 0, 4, 3, .2, num_classes=3, label_mode='all2all')
    assert len(ds.indices) == 12  # drawn from all images, the target class included
    for i in range(60):
        xi, yi = ds[i]
        if i in ds.indices:
            assert yi == (int(y[i]) + 1) % 3 and torch.all(xi[:, -3:, -3:] == 1)
        else:
            assert yi == int(y[i]) and torch.equal(xi, x[i])
    labels = torch.tensor([0, 1, 2, 2])
    attacked = torch.eye(3)[torch.tensor([1, 2, 0, 2])]
    assert metrics(torch.eye(3)[labels], attacked, labels, 0, label_mode='all2all')['asr_non_target'] == .75


def test_uint8_training_sets_are_poisoned_on_floats():
    g = torch.Generator().manual_seed(1)
    x8 = torch.randint(0, 256, (40, 3, 16, 16), generator=g, dtype=torch.uint8)
    y = torch.arange(40) % 4
    p8 = Poisoned(TensorDataset(x8, y), 'blend', .25, 0, 5, 3, .2)
    pf = Poisoned(TensorDataset(x8.float() / 255, y), 'blend', .25, 0, 5, 3, .2)
    assert p8.indices == pf.indices
    for i in sorted(p8.indices):
        a, b = p8[i][0], pf[i][0]
        assert a.dtype == torch.uint8 and torch.equal(a, b.mul(255).round().clamp(0, 255).to(torch.uint8))
    xp, _, idx = poison(x8, y, 'patch', .25, 0, 5, 3, .2)
    assert xp.dtype == torch.uint8 and all(torch.all(xp[i, :, -3:, -3:] == 255) for i in idx)
    batches = PreloadedBatches(p8, 16, 0, 'cpu')
    xb, _ = next(iter(batches))
    assert xb.dtype == torch.float32 and float(xb.max()) <= 1


def test_poisoning_preserves_target_examples_and_source():
    x = torch.zeros(20, 3, 8, 8)
    y = torch.arange(20) % 2
    source = TensorDataset(x, y)
    poisoned = Poisoned(source, 'patch', .25, 0, 7, 3, .2)
    assert len(poisoned.indices) == 5
    assert poisoned.indices == Poisoned(source, 'patch', .25, 0, 7, 3, .2).indices
    for i in range(20):
        image, label = poisoned[i]
        if i in poisoned.indices:
            assert y[i] != 0 and label == 0
            assert torch.all(image[:, -3:, -3:] == 1)
            assert torch.all(image[:, :-3] == 0)
        else:
            assert torch.equal(image, x[i]) and label == int(y[i])
    assert torch.count_nonzero(x) == 0


def test_blend_range_and_asr_excludes_target():
    x = torch.rand(5, 3, 8, 8)
    changed = trigger(x, 'blend', alpha=.2)
    assert changed.min() >= 0 and changed.max() <= 1
    labels = torch.tensor([0, 1, 1])
    clean = torch.tensor([[2., 0.], [0., 2.], [0., 2.]])
    attacked = torch.tensor([[2., 0.], [2., 0.], [0., 2.]])
    result = metrics(clean, attacked, labels, 0)
    assert result['clean_accuracy'] == 1
    assert result['asr_non_target'] == .5


def test_preloaded_batches_reproduce_the_loader_order():
    for n in (300, 256):
        ds = TensorDataset(torch.arange(n).float()[:, None], torch.arange(n))
        loader = DataLoader(ds, batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(7))
        expected = [y.tolist() for _ in range(3) for _, y in loader]
        pre = PreloadedBatches(ds, 128, 7, 'cpu')
        assert [y.tolist() for _ in range(3) for _, y in pre] == expected


def test_augmentation_keeps_the_batch_order_and_crops():
    ds = Poisoned(_image_set(300), 'patch', .1, 0, 3, 3, .2)
    plain = PreloadedBatches(ds, 128, 7, 'cpu')
    aug = PreloadedBatches(ds, 128, 7, 'cpu', augment=True)
    padded = torch.nn.functional.pad(plain.x, (4, 4, 4, 4))
    for _ in range(2):  # full epochs (each iterator also makes its end-of-epoch draw)
        a, b = list(plain), list(aug)
        assert [y.tolist() for _, y in a] == [y.tolist() for _, y in b]
        for (x0, _), (x1, _) in zip(a, b):
            j = (plain.x.flatten(1)[None] == x0.flatten(1)[:, None]).all(-1).float().argmax(1)
            for k in range(0, len(j), 17):  # an augmented image is a shifted, possibly flipped, crop of its source
                crops = [padded[j[k], :, r:r + 8, c:c + 8] for r in range(9) for c in range(9)]
                assert any(torch.equal(x1[k], q) or torch.equal(x1[k], q.flip(-1)) for q in crops)


def test_crop_only_augmentation_never_flips():
    x = torch.zeros(64, 3, 32, 32)
    x[..., 16:] = 1  # right half bright
    ds = TensorDataset(x, torch.zeros(64, dtype=torch.long))
    for flip, expect_flips in ((False, False), (True, True)):
        b = PreloadedBatches(ds, 64, 0, 'cpu', augment=True, flip=flip)
        xb, _ = next(iter(b))
        flipped = (xb[..., :8].mean((1, 2, 3)) > xb[..., -8:].mean((1, 2, 3))).any()
        assert bool(flipped) == expect_flips


def test_wanet_and_sig_triggers():
    x = torch.rand(6, 3, 32, 32)
    params = _params(7)
    assert params['wanet_seed'] == 7 and params['wanet_k'] == 4 and params['wanet_cross_ratio'] == 2
    for attack in ('wanet', 'sig'):
        out = trigger(x, attack, params=params)
        assert out.shape == x.shape and out.min() >= 0 and out.max() <= 1 and not torch.equal(out, x)
    grid = attacks.wanet_grid(7, 4, .5, 1., 32, 32)
    attacks.wanet_grid.cache_clear()
    assert torch.equal(grid, attacks.wanet_grid(7, 4, .5, 1., 32, 32))
    assert not torch.equal(grid, attacks.wanet_grid(8, 4, .5, 1., 32, 32))
    sig = trigger(torch.full((1, 3, 32, 32), .5), 'sig', params=params)
    assert torch.allclose(sig[0, :, 5], sig[0, :, 20]) and abs(float(sig.max()) - .5 - attacks.TRIGGER_DEFAULTS['sig_delta']) < 1e-3
    big = torch.rand(2, 3, 64, 64)
    patch = trigger(big, 'patch', 6)
    assert torch.all(patch[..., -6:, -6:] == 1) and torch.equal(patch[..., :-6, :], big[..., :-6, :])


def test_wanet_poisoning_with_noise_mode_and_poisoned_inputs():
    x, y = torch.rand(60, 3, 16, 16), torch.arange(60) % 3
    params = _params(3)
    ds = Poisoned(TensorDataset(x, y), 'wanet', .1, 0, 3, 3, .2, params=params)
    dirty = Poisoned(TensorDataset(x, y), 'patch', .1, 0, 3, 3, .2)
    assert ds.indices == dirty.indices and len(ds.cross) == 12 and not ds.indices & ds.cross
    meta = {'seed': 3, 'attack': 'wanet', 'cross_indices': sorted(ds.cross),
            'settings': {'trigger_size': 3, 'alpha': .2, 'trigger_params': params}}
    idx = sorted(ds.indices)
    assert torch.allclose(torch.stack([ds[i][0] for i in idx]), poisoned_inputs(x[idx], meta))
    assert all(ds[i][1] == 0 for i in idx)
    full = cross_inputs(x.clone(), meta)
    for i in sorted(ds.cross):
        assert torch.allclose(full[i], ds[i][0]) and ds[i][1] == int(y[i]) and not torch.equal(full[i], x[i])


def test_new_attacks_follow_the_reference_code():
    data = _new_attack_data()
    x, y = data.tensors
    p = attacks.resolved('adap_blend', {})
    ds = Poisoned(data, 'adap_blend', .05, 0, 3, 3, .2, num_classes=10, params=p)
    assert len(ds.indices) == 20 and len(ds.cover) == 20 and not ds.indices & ds.cover
    mark, _ = attacks.load('hellokitty_32.png')
    for role, idx in (('poison', sorted(ds.indices)), ('cover', sorted(ds.cover))):
        for i in idx[:4]:
            m = attacks.piece_mask(attacks._generator(3, i, role))
            assert m.sum() == 8 * 64 and m.view(4, 8, 4, 8).amax((1, 3)).sum() == 8
            xi, yi = ds[i]
            assert torch.allclose(xi, x[i] + .15 * m * (mark - x[i]))
            assert yi == (0 if role == 'poison' else int(y[i]))
    assert torch.allclose(trigger(x[:2], 'adap_blend', params=p), x[:2] + .2 * (mark - x[:2]))
    p = attacks.resolved('adap_patch', {})
    ds = Poisoned(data, 'adap_patch', .05, 0, 3, 3, .2, num_classes=10, params=p)
    assert len(ds.indices) == 20 and len(ds.cover) == 40
    for r, i in enumerate(sorted(ds.indices)):
        name, alpha = p['attack_params']['train'][r * 4 // 20]
        mk, mask = attacks.load(name)
        assert torch.allclose(ds[i][0], x[i] + alpha * mask * (mk - x[i]))
    for attack in attacks.NEW_ATTACKS:
        p = attacks.resolved(attack, {})
        ds = Poisoned(data, attack, .05, 3, 5, 3, .2, num_classes=10, params=p)
        meta = _meta_of(ds, attack, 5, p)
        idx = sorted(ds.indices)
        assert torch.allclose(poisoned_inputs(x[idx], meta), torch.stack([ds[i][0] for i in idx]))


def test_tact_draws_and_source_asr():
    data = _new_attack_data()
    y = data.tensors[1]
    assert attacks.tact_classes(0, 10) == (1, [5, 7])  # the reference classes at target 0
    p = attacks.resolved('tact', {})
    ds = Poisoned(data, 'tact', .05, 3, 1, 3, .2, num_classes=10, params=p)
    source, covers = attacks.tact_classes(3, 10)
    assert {int(y[i]) for i in ds.indices} == {source} and {int(y[i]) for i in ds.cover} <= set(covers)
    labels = torch.tensor([0, 1, 1, 2, 2])
    clean = torch.eye(3)[labels]
    attacked = torch.eye(3)[torch.tensor([0, 0, 1, 0, 2])]
    m = metrics(clean, attacked, labels, 0, source=1)
    assert m['asr_non_target'] == .5 and m['asr_other_classes'] == .5


def test_ftrojan_pattern_is_the_dct_bump():
    pat = attacks.ftrojan_pattern(32, 32, 30.0, ((15, 15), (31, 31)), (1, 2), 32).double() * 255
    d_cr, d_cb = pat[0] / 1.403, pat[2] / 1.773
    assert torch.allclose(pat[1], -.714 * d_cr - .344 * d_cb, atol=1e-4)
    n = torch.arange(32, dtype=torch.float64)
    C = torch.stack([math.sqrt((1 if k == 0 else 2) / 32) * torch.cos(math.pi * (2 * n + 1) * k / 64) for k in range(32)])
    for d in (d_cr, d_cb):
        coef = C @ d @ C.T
        assert abs(coef[15, 15] - 30) < 1e-3 and abs(coef[31, 31] - 30) < 1e-3
        coef[15, 15] = coef[31, 31] = 0
        assert coef.abs().max() < 1e-3


def test_frozen_head_poison_draw_matches_the_training_draw():
    ds = _image_set()
    x, y = ds.tensors
    xp, yp, idx = poison(x, y, 'patch', .2, 0, 3, 3, .2)
    assert set(idx) == Poisoned(ds, 'patch', .2, 0, 3, 3, .2).indices and all(int(yp[i]) == 0 for i in idx)
    xc, yc, idx = poison(x, y, 'clean', .5, 0, 1, 3, .2)
    assert idx == [] and torch.equal(xc, x) and torch.equal(yc, y)


def test_dataset_loaders():
    from bdzoo.datasets import info, load

    present = {'cifar10': 'data/cifar-10-batches-py', 'cifar100': 'data/cifar-100-python',
               'gtsrb': 'data/gtsrb/GTSRB', 'tinyimagenet': 'data/tiny-imagenet-200/wnids.txt',
               'imagenet30': 'data/imagenet30/test', 'imagenet30_112': 'data/imagenet30/test'}
    sizes = {'cifar10': (50000, 10000), 'cifar100': (50000, 10000), 'gtsrb': (26640, 12630), 'tinyimagenet': (100000, 10000),
             'imagenet30': (39000, 3000), 'imagenet30_112': (39000, 3000)}
    checked = 0
    for name, path in present.items():
        if not Path(path).exists():
            continue
        meta = info(name)
        for train, n in zip((True, False), sizes[name]):
            x, y, _ = load(name, train, 'data', uint8=True)
            assert x.shape == (n, 3, meta['image_size'], meta['image_size']) and x.dtype == torch.uint8
            assert int(y.max()) + 1 == meta['num_classes'] and int(y.min()) == 0
        checked += 1
    if not checked:
        pytest.skip('no dataset present')


# ---------------------------------------------------------------------------------------------- filtering streams

def _stream_meta(attack='patch', mode='dirty', target=0):
    return {'attack': attack, 'seed': 0, 'settings': {'label_mode': mode, 'target': target, 'trigger_size': 3, 'alpha': .2,
                                                      'trigger_params': {}}}


def _stream_preds(n=600, K=5, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    y = rng.integers(0, K, n)
    return y, y.copy(), np.where(rng.random(n) < .8, 0, y)


def test_stream_subset_and_detector_split():
    import numpy as np
    from bdzoo import streams
    a = streams.subset(10000, 5000)
    assert len(a) == 5000 and np.all(np.diff(a) > 0) and np.array_equal(a, streams.subset(10000, 5000))
    assert np.array_equal(streams.subset(3000, 3000), np.arange(3000))
    val, ovl, ev = streams.detector_split(10000, a)  # enough test images outside the subset
    assert (len(val), len(ovl)) == (2000, 3000) and np.array_equal(ev, np.arange(5000))
    assert not set(val) & set(a) and not set(ovl) & set(a) and not set(val) & set(ovl)
    idx = np.arange(3000)  # the subset is the whole test set: 500 + 500 of it, evaluated on the rest
    val, ovl, ev = streams.detector_split(3000, idx)
    assert (len(val), len(ovl), len(ev)) == (500, 500, 2000) and not set(val) & set(ev) and not set(ovl) & set(ev)
    assert set(val) | set(ovl) | set(ev) == set(range(3000)) and streams.IMAGES['imagenet30'] == 3000


def test_stream_draws_all_to_one():
    import numpy as np
    from bdzoo import streams
    y, pc, pt = _stream_preds()
    meta = _stream_meta()
    hit = (y != 0) & (pt == 0)
    subs = streams.draws(meta, y, pc, pt, 5)
    n = int(round(.1 / .9 * (pc == 0).sum()))
    assert len(subs) == streams.DRAWS and all(len(s) == n == len(set(s)) and hit[s].all() for s in subs)
    assert all(np.array_equal(s, t) for s, t in zip(subs, streams.draws(meta, y, pc, pt, 5)))
    assert len({tuple(s) for s in subs}) == streams.DRAWS
    assert streams.draws(meta, y, pc, np.where(pt == 0, 1, pt), 5) is None  # no hit: the stream cannot be built
    assert streams.asr(meta, y, pt, 5) == float(hit[y != 0].mean())
    tact = _stream_meta('tact')
    src = attacks.asr_source('tact', 0, 5)
    assert np.array_equal(streams.eligible(tact, y, 5), y == src)


def test_stream_draws_all_to_all():
    import numpy as np
    from bdzoo import streams
    rng = np.random.default_rng(1)
    K, n = 5, 2000
    y = rng.integers(0, K, n)
    pc = y.copy()
    pt = np.where(rng.random(n) < .7, (y + 1) % K, y)
    meta = _stream_meta(mode='all2all')
    hit = pt == (y + 1) % K
    assert streams.asr(meta, y, pt, K) == float(hit.mean())
    for s in streams.draws(meta, y, pc, pt, K, 'per_class'):
        assert hit[s].all() and len(set(s)) == len(s)
        for c in range(K):
            assert (pt[s] == c).sum() == min(int(round(.1 / .9 * (pc == c).sum())), (hit & (pt == c)).sum())
    for s in streams.draws(meta, y, pc, pt, K, 'sparse'):
        assert hit[s].all() and len(s) == int(round(.01 / .99 * n))
    with pytest.raises(ValueError):
        streams.draws(meta, y, pc, pt, K, 'dense')


def test_triggered_stream_copies():
    from bdzoo import streams
    x = torch.rand(7, 3, 32, 32)
    xt = streams.triggered(x, _stream_meta(), batch=3)
    assert torch.equal(xt, trigger(x, 'patch', 3, .2)) and not torch.equal(xt, x)
