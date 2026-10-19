"""BackdoorBench's released PreAct-ResNet18 checkpoints in this repository's checkpoint format.

BackdoorBench (SCLBD) shares its trained models as one zip per attack and poisoning ratio in a SharePoint folder
(README, 'Model and Data Downloading'); each holds attack_result.pt and the triggered test images. Its PreActResNet18 has
no final BN -> ReLU (architecture preact_resnet18_bb) and normalises in its data transform, which the converted model does
internally, so the converted models take inputs in [0, 1] like the others.

    python -m bdzoo.backdoorbench download --dataset cifar10 --min-ratio 0.01 --root data/backdoorbench
    python -m bdzoo.backdoorbench import data/backdoorbench/cifar10_preactresnet18_*/attack_result.pt --out models/backdoorbench

Each import writes <out>/<attack>_<ratio>/{model.pt, meta.json} with the clean accuracy on the dataset's test set and the
ASR on BackdoorBench's own triggered test images. The released attacks all target class 0.
"""
from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .checkpoint import save_checkpoint
from .datasets import load
from .models import build_model

ARCH = 'preact_resnet18_bb'
NORMALIZATION = {'cifar10': ([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),  # BackdoorBench get_dataset_normalization
                 'cifar100': ([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762])}
TARGET = 0  # BackdoorBench's default all-to-one target (checked against bd_test other_info on import)

SHARE = 'https://cuhko365.sharepoint.com/:f:/s/SDSbackdoorbench/EmYD8BoPY8hAqNCV_Rb_zwsBFdqf88Yx01xi0V8tc4whvw?e=d7oJNc'
SITE = 'https://cuhko365.sharepoint.com/sites/SDSbackdoorbench'
FOLDER = '/sites/SDSbackdoorbench/Shared Documents/backdoorbench'
CA_FILES = ('/etc/pki/tls/certs/ca-bundle.crt', '/etc/ssl/certs/ca-certificates.crt')


def download(dataset: str, min_ratio: float, root: Path, arch='preactresnet18', ca_file: str | None = None):
    """The zips <dataset>_<arch>_<attack>_<ratio>.zip at >= min_ratio that are not there yet, through the guest cookie of
    the share link (REST listing and file download); extracts attack_result.pt next to them."""
    import http.cookiejar
    import ssl
    import time
    import urllib.error
    import urllib.parse
    import urllib.request
    ctx = ssl.create_default_context()
    for ca in ([ca_file] if ca_file else CA_FILES):  # some Python builds ship without a CA store
        if Path(ca).exists():
            ctx.load_verify_locations(ca)
    cj = http.cookiejar.CookieJar()
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj), urllib.request.HTTPSHandler(context=ctx))
    op.addheaders = [('User-Agent', 'Mozilla/5.0')]
    op.open(SHARE, timeout=60).read()
    q = urllib.parse.quote(FOLDER)
    req = urllib.request.Request(f"{SITE}/_api/web/GetFolderByServerRelativeUrl('{q}')/Files?$top=5000&$select=Name,Length",
                                 headers={'Accept': 'application/json;odata=nometadata'})
    files = json.loads(op.open(req, timeout=120).read())['value']
    pat = re.compile(rf'{dataset}_{arch}_([a-z]+)_(0_\d+)\.zip$')
    want = [f for f in files if (m := pat.match(f['Name'])) and float(m.group(2).replace('_', '.')) >= min_ratio - 1e-12]
    (root / 'zips').mkdir(parents=True, exist_ok=True)
    for f in sorted(want, key=lambda f: f['Name']):
        z = root / 'zips' / f['Name']
        if not (z.exists() and z.stat().st_size == int(f['Length'])):
            url = f"{SITE}/_api/web/GetFileByServerRelativeUrl('{urllib.parse.quote(FOLDER + '/' + f['Name'])}')/$value"
            tmp = z.with_suffix('.part')
            for attempt in range(6):  # SharePoint throttles (503) after a few hundred MB: back off and retry
                try:
                    with op.open(url, timeout=600) as r, open(tmp, 'wb') as out:
                        while chunk := r.read(1 << 22):
                            out.write(chunk)
                    break
                except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
                    if attempt == 5:
                        raise
                    print(f"{f['Name']}: {e}; retry in {60 * 2 ** attempt} s", flush=True)
                    time.sleep(60 * 2 ** attempt)
            if tmp.stat().st_size != int(f['Length']):
                raise IOError(f"{f['Name']}: {tmp.stat().st_size} of {f['Length']} bytes")
            tmp.rename(z)
        d = root / z.stem
        if not (d / 'attack_result.pt').exists():
            d.mkdir(exist_ok=True)
            with zipfile.ZipFile(z) as zz, zz.open('attack_result.pt') as src, open(d / 'attack_result.pt', 'wb') as out:
                while chunk := src.read(1 << 22):
                    out.write(chunk)
        print(f"{f['Name']}: {int(f['Length']) / 1e6:.0f} MB, ok", flush=True)
    return len(want)


def safe_load(path) -> dict:
    """attack_result.pt with torch's restricted unpickler; the allowlist is the numpy reconstructors the files reference
    (OrderedDict, torch storages, numpy scalar/ndarray/dtype, _codecs.encode). Needs numpy >= 2."""
    import _codecs

    from numpy._core.multiarray import _reconstruct, scalar
    safe = [(scalar, 'numpy.core.multiarray.scalar'), (_reconstruct, 'numpy.core.multiarray._reconstruct'), np.ndarray,
            np.dtype, _codecs.encode] + [type(np.dtype(t)) for t in ('float32', 'float64', 'int64', 'int32', 'uint8', 'bool')]
    with torch.serialization.safe_globals(safe):
        return torch.load(path, map_location='cpu', weights_only=True)


def to_zoo(result: dict, dataset: str) -> nn.Module:
    if result['model_name'] != 'preactresnet18':
        raise ValueError(f"unsupported BackdoorBench model {result['model_name']!r}")
    mean, std = NORMALIZATION[dataset]
    model = build_model(ARCH, num_classes=int(result['num_classes']), width=int(result['model']['conv1.weight'].shape[0]),
                        mean=mean, std=std)
    missing, unexpected = model.load_state_dict(result['model'], strict=False)
    if set(missing) != {'norm.mean', 'norm.std'} or unexpected:
        raise ValueError(f'state dict mismatch: missing {missing}, unexpected {unexpected}')
    return model.eval()


@torch.no_grad()
def accuracy(model, x, y, device, batch=1000):
    model = model.to(device)
    pred = torch.cat([model(x[i:i + batch].to(device)).argmax(1).cpu() for i in range(0, len(x), batch)])
    return pred, float((pred == y).float().mean())


def triggered_test(zip_path: Path):
    """BackdoorBench's own triggered test images (bd_test_dataset/<original label>/<index>.png) as (x in [0, 1], y)."""
    from io import BytesIO

    from PIL import Image
    xs, ys = [], []
    with zipfile.ZipFile(zip_path) as z:
        for n in z.namelist():
            m = re.match(r'bd_test_dataset/(\d+)/\d+\.png$', n)
            if m:
                xs.append(np.asarray(Image.open(BytesIO(z.read(n))).convert('RGB')))
                ys.append(int(m.group(1)))
    if not xs:  # no stored triggered test set in this zip
        return None, None
    x = torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2).float().div(255)
    return x, torch.tensor(ys)


def import_one(p: Path, out_root: Path, data_dir: str, device: str):
    name = p.parent.name  # <dataset>_preactresnet18_<attack>_<ratio with _ for .>
    m = re.match(r'(cifar10|cifar100)_preactresnet18_([a-z]+)_(0_\d+)$', name)
    if not m:
        raise ValueError(f'cannot parse {name}')
    dataset, attack, ratio = m.group(1), m.group(2), float(m.group(3).replace('_', '.'))
    out = out_root / f'{attack}_{m.group(3)}'
    if (out / 'meta.json').exists():
        print(f'{out}: exists, skipped', flush=True)
        return
    result = safe_load(p)
    bd_test = result.get('bd_test') or {}
    targets = {int(v['other_info'][0]) for v in ((bd_test.get('bd_data_container') or {}).get('data_dict') or {}).values()}
    model = to_zoo(result, dataset)
    x, y, _ = load(dataset, False, data_dir)
    _, acc = accuracy(model, x, y, device)
    zp = p.parent.parent / 'zips' / f'{name}.zip'
    xt, yt = triggered_test(zp)
    if xt is None:
        asr, keep = None, torch.zeros(0, dtype=torch.bool)
    else:
        pred, _ = accuracy(model, xt, yt, device)
        keep = yt != TARGET
        asr = float((pred[keep] == TARGET).float().mean())
    out.mkdir(parents=True, exist_ok=True)
    save_checkpoint(out / 'model.pt', model.cpu(), ARCH, {'num_classes': int(result['num_classes']),
                                                         'mean': NORMALIZATION[dataset][0], 'std': NORMALIZATION[dataset][1]})
    meta = {'source': 'BackdoorBench (SCLBD), released attack_result.pt', 'source_file': p.name, 'attack': attack,
            'label': 'backdoored', 'seed': None,
            'settings': {'dataset': dataset, 'arch': ARCH, 'poison_rate': ratio, 'target': TARGET},
            # None for the training-procedure attacks (blind, lira): no poisoned training set is stored
            'train_poison_fraction': None if result.get('bd_train') is None
            else float(np.asarray(result['bd_train']['poison_indicator']).mean()),
            'bd_test_targets': sorted(targets), 'n_triggered_test': int(keep.sum()),
            'baseline': {'clean_accuracy': acc, 'asr_non_target': asr}}
    (out / 'meta.json').write_text(json.dumps(meta, indent=2))
    print(f"{out}: clean {acc:.4f} ASR {'n/a' if asr is None else f'{asr:.4f}'} targets {sorted(targets)}", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    d = sub.add_parser('download')
    d.add_argument('--dataset', default='cifar10', choices=sorted(NORMALIZATION))
    d.add_argument('--min-ratio', type=float, default=0.01)
    d.add_argument('--root', type=Path, default=Path('data/backdoorbench'))
    d.add_argument('--ca-file', help='CA bundle for HTTPS (default: the usual system bundles)')
    i = sub.add_parser('import')
    i.add_argument('paths', nargs='+', type=Path, help='attack_result.pt files (inside the extracted download)')
    i.add_argument('--out', type=Path, required=True)
    i.add_argument('--data', default='data')
    i.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args(argv)
    if a.cmd == 'download':
        print(f'{download(a.dataset, a.min_ratio, a.root, ca_file=a.ca_file)} checkpoints', flush=True)
        return
    failed = []
    for path in a.paths:
        try:
            import_one(path, a.out, a.data, a.device)
        except Exception as e:  # noqa: BLE001 - one checkpoint must not stop the others
            failed.append(path)
            print(f'{path}: FAILED {type(e).__name__}: {e}', flush=True)
    if failed:
        raise SystemExit(f'{len(failed)} checkpoints failed to import')


if __name__ == '__main__':
    main()
