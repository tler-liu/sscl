import torch

from model.model import SSCLModel


def _dummy_run():
	device = torch.device('cpu')

	# toy configuration
	num_classes = 10
	batch_l = 8
	batch_u = 12
	C, H, W = 3, 224, 224

	model = SSCLModel(num_classes=num_classes, pretrained=False).to(device)

	# Create dummy tensors simulating augmentations
	x_l = torch.randn(batch_l, C, H, W, device=device)
	y_l = torch.randint(0, num_classes, (batch_l,), device=device)

	# Unlabeled sets (weak and two strong augmentations). Use same batch size
	x_uw = torch.randn(batch_u, C, H, W, device=device)
	x_us1 = torch.randn(batch_u, C, H, W, device=device)
	x_us2 = torch.randn(batch_u, C, H, W, device=device)

	total, metrics = model.compute_losses(
		x_l=x_l,
		y_l=y_l,
		x_uw=x_uw,
		x_us1=x_us1,
		x_us2=x_us2,
		lambda_1=1.0,
		lambda_2=1.0,
		lambda_3=1.0,
		tau=1.0,
	)

	print('Total loss:', total.item())
	print({k: float(v) for k, v in metrics.items()})


if __name__ == '__main__':
	_dummy_run()