import itertools
import math
from typing import Callable, Sequence, Optional

import os
import torch
import torch.nn as nn
import json
from typing import List
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from tqdm import tqdm

from model.model import SSCLModel


def get_weak_transform(image_size: int = 224) -> Callable:
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_strong_transform(image_size: int = 224) -> Callable:
    # Use stronger color jitter and RandAugment if available
    transform_list = [
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    ]
    try:
        # torchvision >=0.9 provides RandAugment
        transform_list.append(transforms.RandAugment())
    except Exception:
        # fallback: add an extra random erase later
        pass
    transform_list += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25),
    ]
    return transforms.Compose(transform_list)


class LabeledAugmentedDataset(Dataset):
    """Wrap a base labeled dataset to return weak view + an aug-stack for pairwise supervised term.

    Each __getitem__ returns: (x_weak, y, aug_stack)
      - x_weak: Tensor shaped (C,H,W)
      - y: int
      - aug_stack: Tensor shaped (n_views, C, H, W)
    """

    def __init__(self, base_dataset: Dataset, weak_transform: Callable, n_views: int = 2):
        self.base = base_dataset
        self.weak = weak_transform
        self.n_views = n_views

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        # produce one weak view and n_views weak augmentations (for pair term)
        x_weak = self.weak(img)
        views = [self.weak(img) for _ in range(self.n_views)]
        aug_stack = torch.stack(views, dim=0)
        return x_weak, label, aug_stack


class UnlabeledAugmentedDataset(Dataset):
    """Wrap an unlabeled dataset to return (weak, strong_stack).

    Each __getitem__ returns: (x_weak, strong_stack)
      - x_weak: Tensor (C,H,W)
      - strong_stack: Tensor (n_strong, C, H, W)
    """

    def __init__(self, base_dataset: Dataset, weak_transform: Callable, strong_transform: Callable, n_strong: int = 2):
        self.base = base_dataset
        self.weak = weak_transform
        self.strong = strong_transform
        self.n_strong = n_strong

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img = self.base[idx]
        # some unlabeled datasets return PIL images directly; allow both
        if isinstance(img, tuple) or isinstance(img, list):
            img = img[0]
        x_weak = self.weak(img)
        strong_views = [self.strong(img) for _ in range(self.n_strong)]
        strong_stack = torch.stack(strong_views, dim=0)
        return x_weak, strong_stack


def infinite_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def prepare_cifar_datasets(
    dataset_name: str = 'cifar10',
    root: str = './data',
    image_size: int = 224,
    val_size: int = 5000,
    labeled_fraction: float = 0.1,
    seed: int = 42,
):
    """Prepare CIFAR10 or CIFAR100 datasets and split into labeled/unlabeled/val/test subsets.

    Returns: labeled_dataset, unlabeled_dataset, val_dataset, test_dataset
    """
    assert dataset_name in ('cifar10', 'cifar100')
    if dataset_name == 'cifar10':
        DatasetCls = datasets.CIFAR10
        num_classes = 10
    else:
        DatasetCls = datasets.CIFAR100
        num_classes = 100

    # load full training set without transforms (we'll apply transforms in wrappers)
    full_train = DatasetCls(root=root, train=True, download=True, transform=None)
    test = DatasetCls(root=root, train=False, download=True, transform=None)

    num_train = len(full_train)  # typically 50000
    assert num_train > val_size

    gen = torch.Generator()
    gen.manual_seed(seed)
    perm = torch.randperm(num_train, generator=gen).tolist()

    val_indices = perm[:val_size]
    train_indices = perm[val_size:]

    # split labeled vs unlabeled within training remainder
    num_labeled = int(len(train_indices) * float(labeled_fraction))
    labeled_indices = train_indices[:num_labeled]
    unlabeled_indices = train_indices[num_labeled:]

    labeled_subset = torch.utils.data.Subset(full_train, labeled_indices)
    unlabeled_subset = torch.utils.data.Subset(full_train, unlabeled_indices)
    val_subset = torch.utils.data.Subset(full_train, val_indices)

    # test dataset - attach evaluation transform later when evaluating
    return labeled_subset, unlabeled_subset, val_subset, test, num_classes


class EvalDataset(Dataset):
    """Wrap a Subset or Dataset and apply a transform for evaluation."""

    def __init__(self, base_dataset: Dataset, transform: Optional[Callable] = None):
        self.base = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        # base dataset item usually (img, label)
        if isinstance(item, tuple) or isinstance(item, list):
            img, label = item
        else:
            # some datasets may return image only (unlikely for val set)
            img = item
            label = -1
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def evaluate_top1(model: nn.Module, val_dataset: Dataset, device: torch.device, image_size: int = 224, batch_size: int = 256) -> dict:
    """Evaluate top-1 accuracy and average cross-entropy loss on a labeled validation dataset.

    Returns a dict with keys: 'val_loss' and 'val_top1'
    """
    # evaluation transform (resize/center-crop + normalize)
    eval_t = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    wrapped = EvalDataset(val_dataset, transform=eval_t)
    loader = DataLoader(wrapped, batch_size=batch_size, shuffle=False, num_workers=4)

    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_samples = 0
    correct = 0

    loss_fn = nn.CrossEntropyLoss(reduction='sum')

    with torch.no_grad():
        for xb, yb in tqdm(loader, desc='Eval', unit='batch'):
            xb = xb.to(device)
            yb = yb.to(device)
            logits, _ = model.forward(xb)
            # accumulate loss (sum reduction) so we can average later
            batch_loss = loss_fn(logits, yb)
            total_loss += float(batch_loss.detach().cpu().item())
            preds = logits.argmax(dim=1)
            correct += int((preds == yb).sum().detach().cpu().item())
            total_samples += xb.size(0)

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    top1 = correct / total_samples if total_samples > 0 else 0.0

    return {'val_loss': avg_loss, 'val_top1': top1, 'val_samples': total_samples}

def train_sscl(
    model: nn.Module,
    labeled_dataset: Dataset,
    unlabeled_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    device: torch.device = torch.device('cpu'),
    epochs: int = 10,
    batch_size_l: int = 32,
    batch_size_u: int = 128,
    lr: float = 0.03,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    lambda_sup: float = 1.0,
    lambda_cont: float = 1.0,
    tau: float = 1.0,
    image_size: int = 224,
    log_every: int = 50,
    save_dir: Optional[str] = None,
    final_model_filename: Optional[str] = None,
    val_metrics_filename: Optional[str] = None,
    save_every_epochs: int = 1,
    save_last: bool = True,
    train_plot_filename: Optional[str] = None,
):
    weak_t = get_weak_transform(image_size)
    strong_t = get_strong_transform(image_size)

    labeled = LabeledAugmentedDataset(labeled_dataset, weak_t, n_views=2)
    unlabeled = UnlabeledAugmentedDataset(unlabeled_dataset, weak_t, strong_t, n_strong=2)

    loader_l = DataLoader(labeled, batch_size=batch_size_l, shuffle=True, num_workers=4, drop_last=True)
    loader_u = DataLoader(unlabeled, batch_size=batch_size_u, shuffle=True, num_workers=4, drop_last=True)

    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(loader_l))

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    u_iter = infinite_loader(loader_u)

    model.train()
    global_step = 0
    # track per-epoch averaged metrics
    epoch_history: List[dict] = []
    for epoch in range(epochs):
        pbar = tqdm(total=len(loader_l), desc=f"Epoch {epoch+1}/{epochs}", unit='it')
        # accumulators for this epoch
        sum_L_sup = 0.0
        sum_L_semi = 0.0
        sum_L_contr = 0.0
        batch_count = 0
        for i, batch in enumerate(loader_l):
            x_l, y_l, aug_stack = batch  # x_l: (B, C, H, W), aug_stack: (B, n_views, C,H,W)
            # get corresponding unlabeled batch
            x_ul, strong_stack = next(u_iter)

            # prepare tensors and move to device
            x_l = x_l.to(device)
            y_l = y_l.to(device)
            # build list of labeled augmentations for compute_losses
            # aug_stack shape: (B, n_views, C, H, W)
            x_l_aug_list = [aug_stack[:, v].to(device) for v in range(aug_stack.shape[1])]

            x_ul = x_ul.to(device)
            x_ul_aug_list = [strong_stack[:, v].to(device) for v in range(strong_stack.shape[1])]

            optimizer.zero_grad()
            total, metrics = model.compute_losses(
                x_l=x_l,
                y_l=y_l,
                x_l_aug_list=x_l_aug_list,
                x_ul=x_ul,
                x_ul_aug_list=x_ul_aug_list,
                lambda_sup=lambda_sup,
                lambda_cont=lambda_cont,
                tau=tau,
            )
            total.backward()
            optimizer.step()
            scheduler.step()

            if global_step % log_every == 0:
                # update tqdm postfix and also write a log line above the progress bar
                # try to coerce total to a float for nice display; fall back gracefully
                try:
                    total_val = float(total.item())
                except Exception:
                    try:
                        total_val = float(total)
                    except Exception:
                        total_val = None
                postfix = {'total': total_val}
                postfix.update({k: float(v) for k, v in metrics.items()})
                pbar.set_postfix(postfix)
                if total_val is not None:
                    pbar.write(f"Epoch {epoch+1} Step {i} GlobalStep {global_step} total={total_val:.4f}")
                else:
                    pbar.write(f"Epoch {epoch+1} Step {i} GlobalStep {global_step} total={total}")
                pbar.write(str({k: float(v) for k, v in metrics.items()}))

            # accumulate metrics (convert to float); metrics keys from model: 'L_sup', 'L_semi', 'L_contr'
            try:
                sum_L_sup += float(metrics.get('L_sup', 0.0))
            except Exception:
                sum_L_sup += 0.0
            try:
                sum_L_semi += float(metrics.get('L_semi', 0.0))
            except Exception:
                sum_L_semi += 0.0
            try:
                sum_L_contr += float(metrics.get('L_contr', 0.0))
            except Exception:
                sum_L_contr += 0.0
            batch_count += 1

            pbar.update(1)
            global_step += 1
        pbar.close()
        # compute epoch averages (avoid div by zero)
        if batch_count > 0:
            avg_L_sup = sum_L_sup / batch_count
            avg_L_semi = sum_L_semi / batch_count
            avg_L_contr = sum_L_contr / batch_count
        else:
            avg_L_sup = avg_L_semi = avg_L_contr = 0.0

        epoch_record = {
            'epoch': epoch + 1,
            'avg_L_sup': avg_L_sup,
            'avg_L_semi': avg_L_semi,
            'avg_L_contr': avg_L_contr,
            'batches': batch_count,
        }
        epoch_history.append(epoch_record)

        # log epoch summary
        print(f"Epoch {epoch+1} summary: avg_L_sup={avg_L_sup:.6f}, avg_L_semi={avg_L_semi:.6f}, avg_L_contr={avg_L_contr:.6f} over {batch_count} batches")

        # save epoch history to JSON inside save_dir when available
        if save_dir is not None:
            try:
                metrics_path = os.path.join(save_dir, 'train_metrics.json')
                with open(metrics_path, 'w') as mf:
                    json.dump(epoch_history, mf, indent=2)
            except Exception as e:
                print(f"Failed to save epoch metrics to {metrics_path}: {e}")
        # end of epoch: optionally save a checkpoint
        if save_dir is not None and ((epoch + 1) % save_every_epochs == 0):
            ckpt = {
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'num_classes': getattr(model.classifier, 'out_features', None),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            ckpt_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(ckpt, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    # Save final checkpoint if requested
    if save_dir is not None and save_last:
        ckpt = {
            'epoch': epochs - 1,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'num_classes': getattr(model.classifier, 'out_features', None),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        # determine final checkpoint path: if a filename/path was provided use it; otherwise default
        if final_model_filename:
            # if user passed an absolute path, use it directly; otherwise place inside save_dir
            if os.path.isabs(final_model_filename):
                ckpt_path = final_model_filename
            else:
                ckpt_path = os.path.join(save_dir, final_model_filename)
        else:
            ckpt_path = os.path.join(save_dir, f'checkpoint_last.pth')
        torch.save(ckpt, ckpt_path)
        print(f"Saved final checkpoint: {ckpt_path}")

    # After training, optionally evaluate on validation set and save metrics
    if val_dataset is not None:
        print("Evaluating validation set...")
        metrics = evaluate_top1(model, val_dataset, device=device, image_size=image_size)
        print(f"Validation top-1: {metrics['val_top1']*100:.2f}% loss={metrics['val_loss']:.4f} over {metrics['val_samples']} samples")
        if save_dir is not None or val_metrics_filename is not None:
            # choose metrics path: prefer explicit filename, allow absolute or relative
            if val_metrics_filename:
                if os.path.isabs(val_metrics_filename):
                    metrics_path = val_metrics_filename
                else:
                    # if save_dir provided, place inside it; otherwise use relative path as-is
                    metrics_path = os.path.join(save_dir, val_metrics_filename) if save_dir else val_metrics_filename
            else:
                # fallback to default name inside save_dir
                metrics_path = os.path.join(save_dir, 'val_metrics.json') if save_dir else 'val_metrics.json'
            try:
                # ensure directory exists for metrics_path when it's inside save_dir or absolute with directories
                metrics_dir = os.path.dirname(metrics_path)
                if metrics_dir:
                    os.makedirs(metrics_dir, exist_ok=True)
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=2)
                print(f"Saved validation metrics: {metrics_path}")
            except Exception as e:
                print(f"Failed to save validation metrics to {metrics_path}: {e}")

    # After optionally saving validation metrics, produce a plot of training losses per epoch
    # epoch_history is collected during training; if present, attempt to plot
    try:
        if 'epoch_history' in locals() and epoch_history:
            try:
                import matplotlib
                import matplotlib.pyplot as plt
            except Exception as e:
                print(f"matplotlib not available, skipping training loss plot: {e}")
            else:
                epochs_list = [r['epoch'] for r in epoch_history]
                sup_vals = [r['avg_L_sup'] for r in epoch_history]
                semi_vals = [r['avg_L_semi'] for r in epoch_history]
                contr_vals = [r['avg_L_contr'] for r in epoch_history]

                plt.figure()
                plt.plot(epochs_list, sup_vals, label='L_sup')
                plt.plot(epochs_list, contr_vals, label='L_cont')
                plt.plot(epochs_list, semi_vals, label='L_semi')
                plt.xlabel('Epoch')
                plt.ylabel('Average loss')
                plt.title('Training losses per epoch')
                plt.legend()
                plt.grid(True)

                # choose save path: prefer explicit train_plot_filename when provided
                if train_plot_filename:
                    # if user passed an absolute path, use it; else place inside save_dir if provided
                    if os.path.isabs(train_plot_filename):
                        plot_path = train_plot_filename
                    else:
                        plot_path = os.path.join(save_dir, train_plot_filename) if save_dir else train_plot_filename
                else:
                    plot_name = 'train_losses.png'
                    if save_dir:
                        try:
                            os.makedirs(save_dir, exist_ok=True)
                        except Exception:
                            pass
                        plot_path = os.path.join(save_dir, plot_name)
                    else:
                        plot_path = plot_name

                try:
                    plt.savefig(plot_path)
                    plt.close()
                    print(f"Saved training loss plot to: {plot_path}")
                except Exception as e:
                    print(f"Failed to save training loss plot to {plot_path}: {e}")
    except Exception:
        # be resilient: plotting is optional
        pass


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train SSCL on CIFAR10/100 (minimal CLI)')
    parser.add_argument('--dataset', choices=['cifar10', 'cifar100'], default='cifar10', help='dataset to use')
    parser.add_argument('--labeled-fraction', type=float, default=0.1, help='fraction of training remainder to treat as labeled (0-1)')
    parser.add_argument('--lambda-sup', type=float, default=1.0, help='weight for supervised loss')
    parser.add_argument('--lambda-cont', type=float, default=1.0, help='weight for contrastive loss')
    parser.add_argument('--tau', type=float, default=1.0, help='tau used to compute bounded lambda_semi = min(1, exp(-tau * L_sup))')
    parser.add_argument('--epochs', type=int, default=1, help='number of training epochs')
    parser.add_argument('--image-size', type=int, default=64, help='image size for training (keep small for smoke runs)')
    parser.add_argument('--save-dir', type=str, default='./checkpoints', help='directory to save checkpoints')
    parser.add_argument('--final-model', type=str, default=None, help='filename or path for final model output (overrides default checkpoint_last.pth)')
    parser.add_argument('--val-metrics', type=str, default=None, help='filename or path to save validation metrics JSON (overrides default val_metrics.json)')
    parser.add_argument('--train-plot', type=str, default=None, help='filename or path to save training plot PNG (overrides default train_losses.png inside --save-dir)')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'], help='device override')
    args = parser.parse_args()

    # select device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            try:
                if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
                    device = torch.device('mps')
                else:
                    device = torch.device('cpu')
            except Exception:
                device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    labeled_subset, unlabeled_subset, val_subset, test_set, num_classes = prepare_cifar_datasets(
        dataset_name=args.dataset,
        root='./data',
        image_size=args.image_size,
        val_size=5000,
        labeled_fraction=args.labeled_fraction,
        seed=42,
    )

    model = SSCLModel(num_classes=num_classes, pretrained=False)

    train_sscl(
        model=model,
        labeled_dataset=labeled_subset,
        unlabeled_dataset=unlabeled_subset,
        val_dataset=val_subset,
        device=device,
        epochs=args.epochs,
        batch_size_l=32,
        batch_size_u=128,
        lr=0.04,
        lambda_sup=args.lambda_sup,
        lambda_cont=args.lambda_cont,
        tau=args.tau,
        image_size=args.image_size,
        log_every=10,
        save_dir=args.save_dir,
        final_model_filename=args.final_model,
        val_metrics_filename=args.val_metrics,
        train_plot_filename=args.train_plot,
        save_every_epochs=50,
        save_last=True,
    )
    if args.final_model:
        if os.path.isabs(args.final_model):
            final_path = args.final_model
        else:
            final_path = os.path.join(args.save_dir, args.final_model) if args.save_dir else args.final_model
        print(f"Training finished. Final model saved to: {final_path}")
    else:
        print(f"Training finished. Checkpoints saved to: {args.save_dir}")
