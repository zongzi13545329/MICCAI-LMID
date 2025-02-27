Pytorch implementation for MICCAI 908

## Installation
Install [anaconda/miniconda](https://docs.conda.io/en/latest/miniconda.html)  
Required packages
```
  $ conda env create --name miccai --file env.yml
  $ conda activate miccai
```
Install [PyTorch](https://pytorch.org/get-started/locally/)  
Install [OpenSlide and openslide-python](https://pypi.org/project/openslide-python/).  
[Tutorial 1](https://openslide.org/) and [Tutorial 2 (Windows)](https://www.youtube.com/watch?v=0i75hfLlPsw).  

## Download GRAOE.zip
unzip GRAPE.zip

## Training on default datasets.
>Train model on GRAPE dataset:
```
  $ python train_grape.py
```

### Useful arguments:
``
[--num_classes]       # Number of non-negative classes, for a binary classification (postive/negative), this is set to 2
[--feats_size]        # Size of feature vector (depends on the backbone)
[--lr]                # Initial learning rate [0.0001]
[--num_epochs]        # Number of training epochs [50]
[--stop_epochs]       # Skip remaining epochs if training has not improved after N epochs [10]
[--weight_decay]      # Weight decay [1e-3]
[--dataset]           # Dataset folder name, this is set to GRAPE
[--split]             # Training/validation split [0.2]
