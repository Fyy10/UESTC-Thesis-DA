# UESTC-Thesis-DA

Undergraduate thesis at UESTC, a DA model.

NOTE: The project is in progress.

## Usage

Use the following command to see help information:

```bash
python main.py -h
```

Output:

```plain
usage: main.py [-h] [--model str] [--record_folder str] [--batch_size int]
               [--num_epoch int] [--checkpoint str] [--load str] [--seed int]
               [--dataset str] [--source str] [--target str] [--lr float]
               [--optim str] [--use_cuda] [--gpu_num int] [--save] [--vis]
               [--eval] [--grad_step int] [--multi_card]

A model for Domain Adaptation

optional arguments:
  -h, --help           show this help message and exit
  --model str          select model (lenet, alexnet, alexnet_org, resnet50,
                       vit_b16, default: lenet)
  --record_folder str  record folder (default: record)
  --batch_size int     batch size for training (default: 128)
  --num_epoch int      number of epochs for training (default: 10)
  --checkpoint str     checkpoint folder (default: checkpoint)
  --load str           load pretrained model, provide the filename (default:
                       None)
  --seed int           random seed (default: 1)
  --dataset str        select dataset (d5, o31, dn, default: d5)
  --source str         select source domain (default: svhn)
  --target str         select target domain (default: mnistm)
  --lr float           learning rate (default: 1e-3)
  --optim str          optimizer (default: adam)
  --use_cuda           use gpu (default: False)
  --gpu_num int        select a gpu (default: 0)
  --save               save model or not (default: False)
  --vis                save features for visualize (default: False)
  --eval               evaluate without training (default: False)
  --grad_step int      number of gradient accumulation steps (default: 1)
  --multi_card         distributed training and evaluating (default: False)
```

## Datasets

The `--dataset` option.

Both `--source` and `--target` should be domains specified in the following sections.

### Digit-Five (d5)

[Download](https://github.com/VisionLearningGroup/VisionLearningGroup.github.io/tree/master/M3SDA/code_MSDA_digit#digit-five-download)

Suggested backbone (model): `lenet`

Image Preprocessing:

- Resize to `32x32`
- ToTensor
- Normalize with `mean=0.5, std=0.5`

Domains:

- mnist
- mnistm
- svhn
- syn
- usps

### Office-31 (o31)

[Download](https://faculty.cc.gatech.edu/~judy/domainadapt/#datasets_code)

Suggested backbone (model): `alexnet`, `alexnet_org`

Image Preprocess:

- Resize to `224x224`
- ...

Domains:

- Amazon (A)
- DSLR (D)
- Webcam (W)

### DomainNet (dn)

[Download](https://ai.bu.edu/M3SDA/#dataset)

Suggested backbone: `resnet50`, `vit_b16`

Image Preprocess:

- Resize to `224x224`
- ...

Domains:

- Clipart (clp)
- Infograph (info)
- Painting (pnt)
- Quickdraw (qdr)
- Real (real)
- Sketch (skt)

## Models

The `--model` option.

### Feature Extractor

Feature extractors take an image batch of size `[N, C, H, W]` and outputs features of size `[N, num_features]`.

#### LeNet-5 (lenet)

Input size: `32x32`

Feature dimension: `84`

#### AlexNet (alexnet)

Input size: `224x224`

Feature dimension: `4096`

#### AlexNetOrg (alexnet_org)

Input size: `224x224` or `256x256`

Feature dimension: `4096`

#### ResNet50 (resnet50)

Input size: `224x224`

Feature dimension: `2048`

#### ViT-B_16 (vit_b16)

Input size: `224x224`

Feature dimension: `xxxx`

### Classifier

A single fully-connected layer that maps `[N, num_features]` to `[N, num_classes]`.

## Optimizer

`--optim` option.

- adam

## Load model

`--load` option.

Load a trained model, specify the filename (no filetype suffix).

## Metric

Assume that $f_1, f_2 \in \mathbb{R}^{N \times F}$, where $N$ and $F$ represent the batch size and the number of features, respectively.

### MMDLoss

### KMomenLoss
