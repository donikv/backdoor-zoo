# backdoor-zoo

Clean and backdoored image classifiers with BatchNorm, for evaluating backdoor detectors. One command trains one model with
a chosen dataset, architecture, attack, poison rate, target and seed. It saves `model.pt` (architecture and weights only)
and `meta.json` (settings, poison indices, clean accuracy, attack success rate and whether the backdoor is eligible).

This is the model zoo of the paper *BAND: Backdoor Detection through Batch Normalization Statistics*. The detector lives in
[BAND](https://github.com/donikv/BAND) and the baselines in [backdoor-baselines](https://github.com/donikv/backdoor-baselines).

## Install

```bash
pip install -e .
```

Tested with Python 3.12, PyTorch 2.6.0 (CUDA 12.4), torchvision 0.21 and NumPy 2.

## Use

```bash
# a backdoored and a clean CIFAR-10 model with the paper's recipe (CIFAR-10 is downloaded to data/)
python -m bdzoo.train --dataset cifar10 --arch cnn_mlphead --attack blend --poison-rate 0.1 --seed 0 --download
python -m bdzoo.train --dataset cifar10 --arch cnn_mlphead --attack clean --seed 800

# frozen-head fine-tuning of a clean model: a backdoor, and the matched clean control
python -m bdzoo.finetune models/cifar10_cnn_mlphead/seed800/clean --freeze head --attack patch --poison-rate 0.07 --target 3
python -m bdzoo.finetune models/cifar10_cnn_mlphead/seed800/clean --freeze head --attack clean

# BackdoorBench's released CIFAR-10 PreAct-ResNet18 checkpoints, and clean models with its training recipe
python -m bdzoo.backdoorbench download --dataset cifar10 --min-ratio 0.01
python -m bdzoo.backdoorbench import data/backdoorbench/cifar10_preactresnet18_*/attack_result.pt --out models/backdoorbench
python -m bdzoo.train --dataset cifar10 --arch preact_resnet18_bb --attack clean --epochs 100 --lr 0.01 --augment --seed 0
```

- **Datasets:** `cifar10`, `cifar100`, `gtsrb`, `tinyimagenet`. The expected folder layout is in `bdzoo/datasets.py`.
- **Architectures:** `cnn_mlphead`, `preact_resnet18_mlphead` (MLP head Linear → BN1d → ReLU → Linear), and `cnn_gap`, `preact_resnet18` (fully convolutional).
- **Attacks:** `patch`, `blend`, `wanet`, `sig`, `adap_blend`, `adap_patch`, `tact`, `ftrojan`.
- **Trigger images:** the four newer attacks use images from backdoor-toolbox, which has no license. Fetch them with `python -m bdzoo.attacks --fetch`.
- **Paper's settings:** poison rates 1–15%, targets 0, 3 and 7, and `--augment` for the augmented models. The defaults reproduce the recipe: epochs, accuracy bars, target and patch size per dataset (see `bdzoo/train.py`).

Training is deterministic given the seed for the CNNs. On GPUs the PreAct-ResNets are not bit-reproducible because cuDNN
kernels are nondeterministic.

## License

MIT. BackdoorBench's checkpoints and the backdoor-toolbox trigger images are downloaded from their sources, not distributed here.
