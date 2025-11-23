import argparse
import json
import os
import torch
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from train import prepare_cifar_datasets, evaluate_top1
from model.model import SSCLModel


def select_device(device_arg: str = 'auto') -> torch.device:
    if device_arg != 'auto':
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device('cuda')
    try:
        if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
            return torch.device('mps')
    except Exception:
        pass
    return torch.device('cpu')


def load_checkpoint(ckpt_path: str, device: torch.device) -> dict:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    return ckpt


def plot_train_metrics(metrics_list, out_path: str):
    epochs_list = [r.get('epoch', i + 1) for i, r in enumerate(metrics_list)]
    sup_vals = [r.get('avg_L_sup', 0.0) for r in metrics_list]
    semi_vals = [r.get('avg_L_semi', 0.0) for r in metrics_list]
    contr_vals = [r.get('avg_L_contr', 0.0) for r in metrics_list]

    plt.figure()
    plt.plot(epochs_list, sup_vals, label='L_sup')
    plt.plot(epochs_list, contr_vals, label='L_cont')
    plt.plot(epochs_list, semi_vals, label='L_semi')
    plt.xlabel('Epoch')
    plt.ylabel('Average loss')
    plt.title('Training losses per epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Validate checkpoint and recreate training plot')
    parser.add_argument('--ckpt', type=str, required=True, help='path to checkpoint .pth file')
    parser.add_argument('--train-metrics', type=str, required=True, help='path to train_metrics.json produced during training')
    parser.add_argument('--dataset', choices=['cifar10', 'cifar100'], default='cifar10')
    parser.add_argument('--root', type=str, default='./data')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--val-size', type=int, default=5000, help='number of validation examples (must match training split used)')
    parser.add_argument('--labeled-fraction', type=float, default=0.1, help='labeled fraction used during training split')
    parser.add_argument('--seed', type=int, default=42, help='random seed used to create splits during training')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
    parser.add_argument('--plot-out', type=str, default=None, help='path to save recreated training plot PNG')
    parser.add_argument('--val-metrics-out', type=str, default=None, help='optional path to write validation metrics JSON')
    args = parser.parse_args()

    device = select_device(args.device)
    print(f"Using device: {device}")

    # load checkpoint
    ckpt = load_checkpoint(args.ckpt, device)
    num_classes = ckpt.get('num_classes', None)
    if num_classes is None:
        raise RuntimeError('Checkpoint missing num_classes; cannot instantiate model')

    model = SSCLModel(num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()

    # reconstruct datasets (uses same splitting logic as train.py)
    print('Reconstructing validation split...')
    _, _, val_subset, _, num_classes2 = prepare_cifar_datasets(dataset_name=args.dataset, root=args.root, image_size=args.image_size, val_size=args.val_size, labeled_fraction=args.labeled_fraction, seed=args.seed)
    if num_classes2 != num_classes:
        print(f"Warning: checkpoint num_classes={num_classes} differs from dataset num_classes={num_classes2}")

    # evaluate on validation split using train.evaluate_top1
    print('Evaluating validation set...')
    val_metrics = evaluate_top1(model, val_subset, device=device, image_size=args.image_size, batch_size=args.batch_size)
    print(f"Validation top-1: {val_metrics['val_top1']*100:.2f}% loss={val_metrics['val_loss']:.4f} over {val_metrics.get('val_samples', 'N/A')} samples")

    # optionally save val metrics
    if args.val_metrics_out:
        out_dir = os.path.dirname(args.val_metrics_out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        try:
            with open(args.val_metrics_out, 'w') as vf:
                json.dump(val_metrics, vf, indent=2)
            print(f"Saved validation metrics to: {args.val_metrics_out}")
        except Exception as e:
            print(f"Failed to save validation metrics to {args.val_metrics_out}: {e}")

    # load train_metrics.json and recreate plot
    if not os.path.exists(args.train_metrics):
        print(f"train_metrics file not found: {args.train_metrics}; skipping plot recreation")
        return

    try:
        with open(args.train_metrics, 'r') as f:
            train_metrics = json.load(f)
    except Exception as e:
        print(f"Failed to read train metrics JSON {args.train_metrics}: {e}")
        return

    # determine plot output path
    if args.plot_out:
        plot_path = args.plot_out
    else:
        # place alongside train_metrics (same directory) with a descriptive name
        metrics_dir = os.path.dirname(args.train_metrics)
        base = os.path.splitext(os.path.basename(args.ckpt))[0]
        plot_name = f"train_losses_{base}.png"
        plot_path = os.path.join(metrics_dir if metrics_dir else '.', plot_name)

    try:
        os.makedirs(os.path.dirname(plot_path) or '.', exist_ok=True)
    except Exception:
        pass

    try:
        plot_train_metrics(train_metrics, plot_path)
        print(f"Saved recreated training loss plot to: {plot_path}")
    except Exception as e:
        print(f"Failed to create/save training plot: {e}")


if __name__ == '__main__':
    main()
