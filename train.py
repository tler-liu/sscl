import itertools
import math
from typing import Callable, Sequence, Optional

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets

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


def train_sscl(
    model: nn.Module,
    labeled_dataset: Dataset,
    unlabeled_dataset: Dataset,
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
    save_every_epochs: int = 1,
    save_last: bool = True,
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
    for epoch in range(epochs):
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
                print(f"Epoch {epoch} Step {i} GlobalStep {global_step} total={total.item():.4f}")
                print({k: float(v) for k, v in metrics.items()})

            global_step += 1

        # end of epoch: optionally save a checkpoint
        if save_dir is not None and ((epoch + 1) % save_every_epochs == 0):
            ckpt = {
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            ckpt_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(ckpt, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")


if __name__ == '__main__':
    # Example usage: small smoke run with CIFAR10 (downloads if missing).
    # Prefer CUDA if available, then Apple's MPS (on macOS ARM), otherwise CPU.
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        # Check for MPS (Apple Silicon). Use both is_available and is_built when present.
        mps_available = False
        try:
            mps_available = getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available()
        except Exception:
            mps_available = False
        if mps_available:
            device = torch.device('mps')
        else:
            device = torch.device('cpu')

    # Use small image size to speed up the smoke run
    image_size = 64

    # Prepare example datasets using torchvision CIFAR10 for demonstration
    base_transform = transforms.Resize((image_size, image_size))
    cifar_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=base_transform)

    # split a tiny labeled set and an unlabeled set for demonstration
    labeled_subset = torch.utils.data.Subset(cifar_train, list(range(0, 1024)))
    unlabeled_subset = torch.utils.data.Subset(cifar_train, list(range(1024, 8192)))

    model = SSCLModel(num_classes=10, pretrained=False)

    train_sscl(
        model=model,
        labeled_dataset=labeled_subset,
        unlabeled_dataset=unlabeled_subset,
        device=device,
        epochs=1,
        batch_size_l=32,
        batch_size_u=128,
        lr=0.01,
        image_size=image_size,
        log_every=10,
    )
