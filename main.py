import torch

from model.model import SSCLModel


def _dummy_run():
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

	# toy configuration
	num_classes = 10
	batch_l = 8
	batch_u = 12
	C, H, W = 3, 224, 224

	model = SSCLModel(num_classes=num_classes, pretrained=False).to(device)

	# Create dummy tensors simulating augmentations
	x_l = torch.randn(batch_l, C, H, W, device=device)
	y_l = torch.randint(0, num_classes, (batch_l,), device=device)
	x_l_aug_list = [
		torch.randn(batch_l, C, H, W, device=device),
		torch.randn(batch_l, C, H, W, device=device),
	]

	# Unlabeled sets (weak and augmentations). Use same batch size
	x_ul = torch.randn(batch_u, C, H, W, device=device)
	x_ul_aug_list = [
		torch.randn(batch_u, C, H, W, device=device),
		torch.randn(batch_u, C, H, W, device=device),
	]

	total, metrics = model.compute_losses(
		x_l=x_l,
		y_l=y_l,
		x_l_aug_list=x_l_aug_list,
		x_ul=x_ul,
		x_ul_aug_list=x_ul_aug_list,
		lambda_sup=1.0,
		lambda_cont=1.0,
		tau=1.0,
	)

	print('Total loss:', total.item())
	print({k: float(v) for k, v in metrics.items()})


if __name__ == '__main__':
	_dummy_run()