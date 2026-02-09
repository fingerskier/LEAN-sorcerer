import torch
import torch.nn as nn
import torch.nn.functional as F


class TacticHead(nn.Module):
    """Dual-purpose output head for tactic prediction.

    Produces three outputs from the refined latent:
      1. retrieval_emb  [B, dim]  - L2-normalized embedding for FAISS lookup
      2. coarse_logits  [B, C]    - classification over tactic families
      3. confidence     [B, 1]    - scalar confidence for adaptive TTC

    The retrieval embedding is the primary output: at inference, find the
    nearest tactic in a pre-built FAISS index.  The coarse classifier
    provides auxiliary gradients during training.
    """

    def __init__(self, dim=256, num_coarse_classes=64):
        super().__init__()
        self.retrieval_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.classifier = nn.Linear(dim, num_coarse_classes)
        self.confidence = nn.Linear(dim, 1)

    def forward(self, x):
        """
        x:  [B, dim]  refined latent from TTC loop
        Returns:
            emb:     [B, dim]  L2-normalized retrieval embedding
            logits:  [B, C]    coarse tactic family logits
            conf:    [B, 1]    sigmoid confidence score
        """
        emb = F.normalize(self.retrieval_proj(x), dim=-1)
        logits = self.classifier(x)
        conf = torch.sigmoid(self.confidence(x))
        return emb, logits, conf
