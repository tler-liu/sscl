import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model.model import SSCLModel


def select_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        try:
            if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
                return torch.device('mps')
        except Exception:
            pass
    return torch.device('cpu')


def get_eval_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def evaluate_checkpoint(ckpt_path: str, dataset_name: str = 'cifar10', root: str = './data', batch_size: int = 256, image_size: int = 224):
    device = select_device()
    print(f"Using device: {device}")

    ckpt = torch.load(ckpt_path, map_location='cpu')
    num_classes = ckpt.get('num_classes', None)
    if num_classes is None:
        raise RuntimeError('Checkpoint does not contain num_classes. Recreate checkpoint with num_classes saved.')

    model = SSCLModel(num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()

    # build test dataset
    if dataset_name == 'cifar10':
        DatasetCls = datasets.CIFAR10
    else:
        DatasetCls = datasets.CIFAR100

    test_ds = DatasetCls(root=root, train=False, download=True, transform=get_eval_transform(image_size))
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits, _ = model.forward(x)
            preds = logits.argmax(dim=1).cpu()
            correct += (preds == y).sum().item()
            total += y.size(0)

    acc = 0.0 if total == 0 else float(correct) / total
    print(f"Test top-1 accuracy: {acc * 100:.2f}%")
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='./checkpoints/checkpoint_last.pth')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--root', type=str, default='./data')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--image-size', type=int, default=224)
    args = parser.parse_args()

    evaluate_checkpoint(args.ckpt, dataset_name=args.dataset, root=args.root, batch_size=args.batch_size, image_size=args.image_size)


if __name__ == '__main__':
    main()
