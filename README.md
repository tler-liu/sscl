# sscl

Semi-Supervised Contrastive Learning (SSCL)

This repository contains an implementation of a semi-supervised contrastive learning recipe (encoder with two heads: classification + projection) suitable for experiments on CIFAR-10 and CIFAR-100. The codebase includes:

-   `model/model.py` — PyTorch module `SSCLModel` (ResNet-50 backbone, classifier head, projection head) and loss helpers (supervised CE, pseudolabel semi-loss, InfoNCE contrastive loss).
-   `train.py` — training script that pairs labeled and unlabeled data, applies weak/strong augmentations, computes losses, and saves checkpoints.
-   `validate.py` - validation script that computes validation accuracy on a model and generates loss plots.
-   `eval.py` — load a checkpoint and compute test top-1 accuracy.

## Requirements

The project expects Python 3.8+ and the packages listed in `requirements.txt`. At minimum you need:

-   torch
-   torchvision
-   tqdm

Install with pip:

```bash
python -m pip install -r requirements.txt
```

## Training

The main training entrypoint is `train.py`. The script detects a GPU (CUDA) or Apple MPS when available and falls back to CPU.

Example script to initiate training on a given dataset, percentage of labeled data, and hyperparameters:

```bash
python train.py \
	--dataset cifar10 \
	--labeled-fraction 0.1 \
	--lambda-sup 1.0 \
	--lambda-cont 1.0 \
	--tau 1.0 \
	--epochs 100 \
	--image-size 32 \
	--save-dir ./checkpoints \
    --final-model path_of_model.pth \
    --val-metrics path_of_val_metrics.json \
    --train-plot path_of_training_losses_plot.png
```

Notes:

-   `--dataset` : `cifar10` or `cifar100`.
-   `--labeled_fraction` : fraction of the training set used as labeled examples (rest treated as unlabeled).
-   `--lambda_sup` and `--lambda_cont` : weights for supervised and contrastive terms. The semi-supervised weight (lambda_semi) is computed internally as lambda_semi = min(1, exp(-tau \* L_sup)) and is not a CLI argument.
-   `--tau` : scalar used when computing the bounded semi weight (see above).
-   `--save_dir` : directory where checkpoints are written (per-epoch and final checkpoint).
-   `--final-model` : filepath to save the final model weights (.pth file)
-   `--val-metrics` : filepath to save validation metrics (JSON file)
-   `--train-plot` : filepath to save training losses plot (.png file)
-   `--image-size` : for cifar, this should be 32

Checkpoints are saved as `checkpoint_epoch_{epoch}.pth` and `checkpoint_last.pth` if `final-model` is not specified. Each checkpoint contains the model state dict, optimizer/scheduler states, epoch, and `num_classes` so the model can be reconstructed for evaluation.


## Validation

`validate.py` can be used to run validation testing on a given model file. Note that the settings should match exactly that of
the settings used for training. 

Example script to initiate validation on a given dataset, percentage of labeled data, and hyperparameters:

```bash
python validate.py \
    --ckpt ./checkpoints/model.pth \
    --train-metrics ./checkpoints/train_metrics.json \
	--dataset cifar10 \
	--labeled-fraction 0.1 \
    --image-size 32 \
	--plot-out path_for_loss_plot.png \
    --val-metrics-out path_for_val_metrics.json
```

## Evaluation

After training you can evaluate a checkpoint with `eval.py`. Example:

```bash
python eval.py --ckpt ./checkpoints/checkpoint_last.pth --dataset cifar10 --batch-size 256 --image-size 224
```

Notes:

-   `--ckpt` : the filepath of the model's .pth file

The script will reconstruct the `SSCLModel` using the `num_classes` stored in the checkpoint and print test top-1 accuracy.
