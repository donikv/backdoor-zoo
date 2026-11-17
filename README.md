# backdoor-zoo

Clean and backdoored image classifiers with BatchNorm, for evaluating backdoor detectors. One command trains one model with
a chosen dataset, architecture, attack, poison rate, target and seed. It saves `model.pt` (architecture and weights only)
and `meta.json` (settings, poison indices, clean accuracy, attack success rate and whether the model is usable).

This is the model zoo of the paper *BAND: Backdoor Detection through Batch Normalization Statistics*. The detector lives in
[BAND](https://github.com/donikv/BAND) and the baselines in [backdoor-baselines](https://github.com/donikv/backdoor-baselines).
The version of the CSF submission is tagged `csf-2027`.

## Install

```bash
pip install -e .
```

Tested with Python 3.12, PyTorch 2.6.0 (CUDA 12.4), torchvision 0.21 and NumPy 2.

## Use

```bash
# a backdoored and a clean CIFAR-10 model with the paper's recipe (CIFAR-10 is downloaded to data/)
python -m bdzoo.train --dataset cifar10 --arch preact_resnet18_mlphead --attack blend --poison-rate 0.01 --seed 7000 --target 5 --download
python -m bdzoo.train --dataset cifar10 --arch preact_resnet18_mlphead --attack clean --seed 8000

# VGG-16 on ImageNet-30 at 112 px, and a fully convolutional ResNet on ImageNet-30 at 224 px
python -m bdzoo.train --dataset imagenet30_112 --arch vgg16_gap --attack sig --poison-rate 0.05 --seed 7001 --target 12
python -m bdzoo.train --dataset imagenet30 --arch preact_resnet18 --attack tact --poison-rate 0.03 --seed 7002 --target 7

# an all-to-all backdoor: a triggered image of class y is labelled (y + 1) mod K
python -m bdzoo.train --dataset gtsrb --arch preact_resnet18 --attack patch --poison-rate 0.05 --seed 7000 --target 4 --label-mode all2all

# frozen-head fine-tuning of a clean model: a backdoor behind the frozen head, and the matched clean control
python -m bdzoo.finetune models/cifar10_preact_resnet18_mlphead/seed7500/clean --freeze head --attack patch --target 5
python -m bdzoo.finetune models/cifar10_preact_resnet18_mlphead/seed7500/clean --freeze head --attack clean

# BackdoorBench's released CIFAR-10 PreAct-ResNet18 checkpoints, and clean models with its training recipe
python -m bdzoo.backdoorbench download --dataset cifar10 --min-ratio 0.01
python -m bdzoo.backdoorbench import data/backdoorbench/cifar10_preactresnet18_*/attack_result.pt --out models/backdoorbench
python -m bdzoo.train --dataset cifar10 --arch preact_resnet18_bb --attack clean --epochs 100 --lr 0.01 --augment --seed 0
```

- **Datasets:** `cifar10`, `gtsrb`, `cifar100`, `tinyimagenet`, `imagenet30` (224 px) and `imagenet30_112` (its 2 × 2 block
  averages). The expected folder layout is in `bdzoo/datasets.py`; decode the larger sets once with
  `python -m bdzoo.datasets --data data --cache gtsrb tinyimagenet imagenet30 imagenet30_112`.
- **Architectures:** `preact_resnet18_mlphead` and `vgg16_mlphead` (MLP head GAP → Linear → BN1d → ReLU → Linear),
  `preact_resnet18` and `vgg16_gap` (fully convolutional), and `preact_resnet18_bb` (BackdoorBench's PreActResNet18).
  The ResNets use the ImageNet stem at 224 px.
- **Attacks:** `patch`, `blend`, `wanet`, `sig`, `adap_blend`, `adap_patch`, `tact`, `ftrojan`, all-to-one by default;
  `--label-mode all2all` for the first four. The four newer attacks scale their 32 px triggers by whole factors, so they
  need an image side of 32 m (not 112 px).
- **Trigger images:** the four newer attacks use images from backdoor-toolbox, which has no license. Fetch them with
  `python -m bdzoo.attacks --fetch`.
- **Recipe (the defaults):** SGD 0.1 with a cosine schedule, batch 128, no augmentation; 60 epochs on CIFAR-10 and CIFAR-100,
  40 on the others; patch side 3 px at 32 px (6, 10 and 21 px at 64, 112 and 224 px). Clean-accuracy bars 0.80 (CIFAR-10),
  0.90 (GTSRB), 0.55 (CIFAR-100), 0.40 (TinyImageNet) and 0.50 (ImageNet-30).
- **Usable models:** clean accuracy at most 0.05 below the dataset's bar and, for a backdoor, attack success at least 0.9
  (patch, blend, WaNet, SIG) or 0.5 (Adap-Blend, Adap-Patch, TaCT, FTrojan); `meta.json` records it as `usable`.

`bdzoo.streams` builds the input streams of the paper's test-time filtering experiments: a fixed subset of the test images
with the run's triggered copies (a tenth of the inputs predicted as the target, five draws; for all-to-all models the
per-class and sparse streams), and the clean images the published input-level detectors need.

Training is deterministic given the seed on the CPU. On GPUs the models are not bit-reproducible because cuDNN kernels
are nondeterministic.

## The paper's model sets

Per dataset and architecture: 160 backdoored test slots (8 attacks × poison rates 1, 2, 3, 5 and 10% × seeds 7000–7003,
seed 700i with test target i), 20 clean test models (seeds 7500–7519), 15 clean references (seeds 8000–8014) and ten
labelled poisoned configuration models (patch and blend at 10%, seeds 8050–8054, seed 805i with configuration target i).
A slot that fails the usable criterion is retrained with seed + 100 j (j = 1, 2, 3), except when all four first attempts at
10% fail for that attack and setting. VGG-16 uses CIFAR-10 and ImageNet-30 at 112 px; at 112 px only patch, blend, WaNet
and SIG.

| Dataset | Test targets (seeds 7000–7003) | Configuration targets (seeds 8050–8054) |
|---|---|---|
| CIFAR-10 | 5, 3, 4, 7 | 6, 2, 8, 0, 4 |
| GTSRB | 4, 2, 9, 5 | 35, 25, 4, 7, 11 |
| CIFAR-100 | 93, 52, 89, 37 | 52, 71, 1, 38, 37 |
| TinyImageNet | 10, 122, 67, 40 | 100, 40, 7, 17, 153 |
| ImageNet-30 (both sizes) | 26, 12, 7, 14 | 12, 3, 26, 21, 7 |

- **Frozen head:** the 20 clean MLP-head ResNet test models of CIFAR-10 and ImageNet-30 (seeds 7500–7519), fine-tuned with
  patch and blend at 10% with `--freeze head`, `fc` and `none` (source i takes test target i mod 4), and the clean
  controls with `--freeze head` and `none`.
- **All-to-all:** patch and blend at 3, 5 and 10%, both ResNet heads, CIFAR-10 and GTSRB, seeds 7000–7003.
- **BackdoorBench:** the 32 released CIFAR-10 PreAct-ResNet18 checkpoints at poison ratio 1% or more, and 20 clean models
  with its recipe (seeds 0–9 as references, 10–19 as clean test models).

## License

MIT. BackdoorBench's checkpoints and the backdoor-toolbox trigger images are downloaded from their sources, not distributed here.
