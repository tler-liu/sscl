
import math
from typing import Optional, Tuple, Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SSCLModel(nn.Module):
	"""Semi-supervised model with a shared encoder and two heads:
	- classification head for supervised and semi-supervised losses
	- projection head for contrastive learning (InfoNCE)

	Losses:
	  L = lambda_sup * L_sup + lambda_semi * L_semi + lambda_cont * L_contr

	Semi-sup bound for lambda_semi is computed as:
	  lambda_semi = min(1, exp(-tau * L_supervised))

	Notes:
	  - Encoder: ResNet50 (pretrained optional)
	  - Projection head: MLP
	  - compute_losses expects:
	    • labeled weak batch (x_l) and labels (y_l)
	    • list of labeled augmentations (x_l_aug_list), each a batch tensor matching x_l
	    • unlabeled weak batch (x_ul)
	    • list of unlabeled augmentations (x_ul_aug_list), each a batch tensor matching x_ul
	"""

	def __init__(
		self,
		num_classes: int,
		proj_dim: int = 128,
		proj_hidden: int = 2048,
		pretrained: bool = True,
		temperature: float = 0.1,
	) -> None:
		super().__init__()
		# Encoder: ResNet50 with final fc replaced by identity so we get 2048-d features
		# Use the new `weights` argument when available to avoid torchvision deprecation warnings.
		weights_enum = getattr(models, 'ResNet50_Weights', None)
		if weights_enum is not None:
			weights = weights_enum.DEFAULT if pretrained else None
			resnet = models.resnet50(weights=weights)
		else:
			# Fallback for older torchvision versions that use `pretrained=`
			resnet = models.resnet50(pretrained=pretrained)
		resnet.fc = nn.Identity()  # feature extractor -> (batch, 2048)
		self.encoder = resnet

		feat_dim = 2048
		# Classification head (linear)
		self.classifier = nn.Linear(feat_dim, num_classes)

		# Projection head for contrastive learning (MLP)
		self.projection = nn.Sequential(
			nn.Linear(feat_dim, proj_hidden),
			nn.ReLU(inplace=True),
			nn.Linear(proj_hidden, proj_dim),
		)

		self.temperature = temperature

	def encode(self, x: torch.Tensor) -> torch.Tensor:
		"""Return encoder features (before any head)."""
		return self.encoder(x)

	def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		"""Return classification logits and projection vectors for input x."""
		feats = self.encode(x)  # (B, feat_dim)
		logits = self.classifier(feats)
		proj = self.projection(feats)
		return logits, proj

	@staticmethod
	def _info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
		"""Compute InfoNCE (NT-Xent) loss for a batch of paired vectors z1, z2.

		z1, z2: (N, D)
		Returns scalar loss.
		"""
		assert z1.shape == z2.shape
		device = z1.device
		N = z1.shape[0]

		z1 = F.normalize(z1, dim=1)
		z2 = F.normalize(z2, dim=1)

		z = torch.cat([z1, z2], dim=0)  # (2N, D)

		# similarity matrix
		sim = torch.matmul(z, z.T) / temperature

		# mask to remove similarity of samples to themselves
		diag_mask = torch.eye(2 * N, device=device).bool()
		sim_masked = sim.masked_fill(diag_mask, float('-inf'))

		# For each i in [0..2N), positive index is i+N (mod 2N) if i < N else i-N
		positives = torch.cat([torch.arange(N, 2 * N, device=device), torch.arange(0, N, device=device)])

		# compute log-softmax over rows
		log_probs = F.log_softmax(sim_masked, dim=1)

		# gather positive log-probs
		positive_log_probs = log_probs[torch.arange(2 * N, device=device), positives]

		loss = -positive_log_probs.mean()
		return loss

	def compute_losses(
		self,
		x_l: torch.Tensor,
		y_l: torch.Tensor,
		x_l_aug_list: Sequence[torch.Tensor],
		x_ul: torch.Tensor,
		x_ul_aug_list: Sequence[torch.Tensor],
		lambda_sup: float = 1.0,
		lambda_cont: float = 1.0,
		tau: float = 1.0,
	) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
		"""Compute L_sup, L_semi, L_contr and total loss.

		Loss computed as:
		  total = lambda_sup * L_sup + lambda_semi * L_semi + lambda_cont * L_contr

		where lambda_semi = min(1, exp(-tau * L_supervised)).

		Returns (total_loss, metrics_dict)
		metrics_dict contains 'L_sup', 'L_sup_cls', 'L_sup_pair', 'L_semi', 'L_contr', 'lambda_semi'

		Notes on inputs:
		  - x_l, x_ul, and all tensors in x_l_aug_list and x_ul_aug_list must be 4D float tensors (B, C, H, W).
		  - If your inputs are PIL images or numpy arrays, convert them to tensors before calling this method.
		"""
		# Supervised loss on labeled weakly augmented data
		logits_l, _ = self.forward(x_l)
		L_sup_cls = F.cross_entropy(logits_l, y_l)

		# Semi-supervised loss: get pseudolabels from classifier on unlabeled weak
		logits_ul, _ = self.forward(x_ul)
		with torch.no_grad():
			probs_ul = F.softmax(logits_ul, dim=1)
			pseudo_labels = probs_ul.argmax(dim=1)

		# Use standard CE with pseudo-labels (detached)
		L_semi = F.cross_entropy(logits_ul, pseudo_labels)

		# Supervised pair term over labeled augmentations (Eq. 2 second term)
		# CE(q(i), q(j)) averaged over all ordered pairs i != j
		if x_l_aug_list is None or len(x_l_aug_list) < 2:
			L_sup_pair = torch.tensor(0.0, device=logits_l.device)
		else:
			# Compute classifier outputs for each augmentation
			logits_l_augs = [self.forward(x_aug)[0] for x_aug in x_l_aug_list]
			log_probs_list = [F.log_softmax(lg, dim=1) for lg in logits_l_augs]
			probs_list = [F.softmax(lg, dim=1).detach() for lg in logits_l_augs]

			num_views = len(logits_l_augs)
			pair_losses = []
			for i in range(num_views):
				for j in range(num_views):
					if i == j:
						continue
					pair_loss = -(probs_list[i] * log_probs_list[j]).sum(dim=1).mean()
					pair_losses.append(pair_loss)
			L_sup_pair = torch.stack(pair_losses).mean() if pair_losses else torch.tensor(0.0, device=logits_l.device)

		# Contrastive loss on unlabeled augmentations (use first two views if available)
		if x_ul_aug_list is None or len(x_ul_aug_list) < 2:
			L_contr = torch.tensor(0.0, device=logits_l.device)
		else:
			_, proj_ul_1 = self.forward(x_ul_aug_list[0])
			_, proj_ul_2 = self.forward(x_ul_aug_list[1])
			L_contr = self._info_nce_loss(proj_ul_1, proj_ul_2, temperature=self.temperature)

		# Total supervised term per Eq. (2)
		L_sup = L_sup_cls + L_sup_pair

		# Compute bounded lambda_semi (no gradient through this scalar)
		Lsup_val = float(L_sup.detach().cpu().item())
		lambda_semi = min(1.0, math.exp(-tau * Lsup_val))

		total = lambda_sup * L_sup + lambda_semi * L_semi + lambda_cont * L_contr

		metrics = {
			'L_sup': L_sup.detach(),
			'L_sup_cls': L_sup_cls.detach(),
			'L_sup_pair': L_sup_pair.detach(),
			'L_semi': L_semi.detach(),
			'L_contr': L_contr.detach(),
			'lambda_semi': torch.tensor(lambda_semi, device=L_sup.device),
			'total': total.detach(),
		}
		return total, metrics


__all__ = ["SSCLModel"]

