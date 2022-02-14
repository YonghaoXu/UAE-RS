<h1 align="center">Universal Adversarial Examples in Remote Sensing:<br>Methodology and Benchmark</h1>

This is an official PyTorch implementation of the black-box adversarial attack methods for remote sensing data in our paper [Universal adversarial examples in remote sensing: Methodology and benchmark](https://arxiv.org).


### Dataset
We collect the generated universal adversarial examples in the dataset named UAE-RS, which is the first dataset that provides black-box adversarial samples in the remote sensing field.

To build UAE-RS, we use the *Mixcut-Attack* method to attack `ResNet18` with 1050 test samples from the **UCM dataset** and 5000 test samples from the **AID dataset** for scene classification, and use the *Mixup-Attack* method to attack `FCN-8s` with 5 test images from the **Vaihingen dataset** (image IDs: 11, 15, 28, 30, 34) and 5 test images from the **Zurich Summer** dataset (image IDs: 16, 17, 18, 19, 20) for semantic segmentation.

### Supported methods and models
This repo contains implementations of black-box adversarial attacks for remote sensing data on both scene classification and semantic segmentation tasks.
- Supported adversarial attack methods:
  - [FGSM](https://arxiv.org/abs/1412.6572)
  - [I-FGSM](https://arxiv.org/abs/1611.01236)
  - [C&W](https://arxiv.org/abs/1608.04644)
  - [TPGD](https://arxiv.org/abs/1901.08573)
  - [Jitter](https://arxiv.org/abs/2105.10304)
  - [Mixup-Attack](https://arxiv.org)
  - [Mixcut-Attack](https://arxiv.org)
- Supported scene classification models:
  - [AlexNet](https://arxiv.org/abs/1412.6572)
  - [VGG11, VGG16, VGG19](https://arxiv.org/abs/1611.01236)
  - [Inception-v3](https://arxiv.org/abs/1608.04644)
  - [ResNet18, ResNet50, ResNet101](https://arxiv.org/abs/1901.08573)
  - [ResNeXt50, ResNeXt101](https://arxiv.org/abs/2105.10304)
  - [DenseNet121, DenseNet169, DenseNet201](https://arxiv.org)
  - [RegNetX-400MF, RegNetX-8GF, RegNetX-16GF](https://arxiv.org)
- Supported semantic segmentation models:
  - [FCN-32s, FCN-16s, FCN-8s](https://arxiv.org/abs/1412.6572)
  - [DeepLab-v2, DeepLab-v3+](https://arxiv.org/abs/1611.01236)
  - [SegNet](https://arxiv.org/abs/1608.04644)
  - [ICNet](https://arxiv.org/abs/1901.08573)
  - [ContextNet](https://arxiv.org/abs/2105.10304)
  - [SQNet](https://arxiv.org)
  - [PSPNet](https://arxiv.org)
  - [U-Net](https://arxiv.org)
  - [LinkNet](https://arxiv.org)
  - [FRRNet-A](https://arxiv.org)
  - [FRRNet-B](https://arxiv.org)
### Preparation
- Package requirements: The scripts in this repo are tested with `torch==1.10` and `torchvision==0.11` using two NVIDIA Tesla V100 GPUs.
- Remote sensing datasets used in this repo:
  - [UCM dataset](http://weegee.vision.ucmerced.edu/datasets/landuse.html)
  - [AID dataset](https://captain-whu.github.io/AID/)
  - [Vaihingen dataset](http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-vaihingen.html)
  - [Zurich Summer dataset](https://sites.google.com/site/michelevolpiresearch/data/zurich-dataset)
- Pretraining the models for scene classification
```
CUDA_VISIBLE_DEVICES=0,1 python pretrain_cls.py --network 'alexnet' --dataID 1 --root_dir <THE-ROOT-PATH-OF-DATA>
CUDA_VISIBLE_DEVICES=0,1 python pretrain_cls.py --network 'resnet18' --dataID 1 --root_dir <THE-ROOT-PATH-OF-DATA>
CUDA_VISIBLE_DEVICES=0,1 python pretrain_cls.py --network 'inception' --dataID 1 --root_dir <THE-ROOT-PATH-OF-DATA>
...
```
- Pretraining the models for semantic segmentation
```
cd ./segmentation
CUDA_VISIBLE_DEVICES=0 python pretrain_seg.py --model 'fcn8s' --dataID 1 --root_dir <THE-ROOT-PATH-OF-DATA>
CUDA_VISIBLE_DEVICES=0 python pretrain_seg.py --model 'deeplabv2' --dataID 1 --root_dir <THE-ROOT-PATH-OF-DATA>
CUDA_VISIBLE_DEVICES=0 python pretrain_seg.py --model 'segnet' --dataID 1 --root_dir <THE-ROOT-PATH-OF-DATA>
...
```
Please replace `<THE-ROOT-PATH-OF-DATA>` with the local path where you store the remote sensing datasets.
### Adversarial attacks on scene classification
```
CUDA_VISIBLE_DEVICES=0,1 python pretrain_cls.py --network 'alexnet' --dataID 1 --root_dir <THE-ROOT-PATH-OF-DATA>
CUDA_VISIBLE_DEVICES=0,1 python pretrain_cls.py --network 'resnet18' --dataID 1 --root_dir <THE-ROOT-PATH-OF-DATA>
CUDA_VISIBLE_DEVICES=0,1 python pretrain_cls.py --network 'inception' --dataID 1 --root_dir <THE-ROOT-PATH-OF-DATA>
...
```
