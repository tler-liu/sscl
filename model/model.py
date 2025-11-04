
import math
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SSCLModel(nn.Module):
	"""Semi-supervised model with a shared encoder and two heads:
	- classification head for supervised and semi-supervised losses
	- projection head for contrastive learning (InfoNCE)

	Losses:
	  L = lambda_1 * L_sup + lambda_2 * L_semi + lambda_3 * L_contr

	Semi-sup bound for lambda_semi is computed as:
	  lambda_semi = min(1, exp(-tau * L_supervised))

	Notes:
	  - Encoder: ResNet50 (pretrained optional)
	  - Projection head: MLP
	  - compute_losses expects tensors for: labeled weak (x_l, y), unlabeled weak (x_uw),
		unlabeled strong 1 (x_us1), unlabeled strong 2 (x_us2).
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
		x_uw: torch.Tensor,
		x_us1: torch.Tensor,
		x_us2: torch.Tensor,
		lambda_1: float = 1.0,
		lambda_2: float = 1.0,
		lambda_3: float = 1.0,
		tau: float = 1.0,
	) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
		"""Compute L_sup, L_semi, L_contr and total loss.

		Returns (total_loss, metrics_dict)
		metrics_dict contains 'L_sup', 'L_semi', 'L_contr', 'lambda_semi'
		"""
		# Supervised loss on labeled weakly augmented data
		logits_l, _ = self.forward(x_l)
		L_sup = F.cross_entropy(logits_l, y_l)

		# Semi-supervised loss: get pseudolabels from classifier on unlabeled weak
		logits_uw, _ = self.forward(x_uw)
		with torch.no_grad():
			probs_uw = F.softmax(logits_uw, dim=1)
			pseudo_labels = probs_uw.argmax(dim=1)

		# Use standard CE with pseudo-labels (detached)
		L_semi = F.cross_entropy(logits_uw, pseudo_labels)

		# Contrastive loss on the two strong augmentations
		# Get projections for both strong views
		_, proj1 = self.forward(x_us1)
		_, proj2 = self.forward(x_us2)
		L_contr = self._info_nce_loss(proj1, proj2, temperature=self.temperature)

		# Compute bounded lambda_semi
		# Use detached numeric L_sup to compute scalar weight (no gradient through lambda)
		Lsup_val = float(L_sup.detach().cpu().item())
		lambda_semi = min(1.0, math.exp(-tau * Lsup_val))

		total = lambda_1 * L_sup + lambda_2 * (lambda_semi * L_semi) + lambda_3 * L_contr

		metrics = {
			'L_sup': L_sup.detach(),
			'L_semi': L_semi.detach(),
			'L_contr': L_contr.detach(),
			'lambda_semi': torch.tensor(lambda_semi, device=L_sup.device),
			'total': total.detach(),
		}
		return total, metrics


__all__ = ["SSCLModel"]

