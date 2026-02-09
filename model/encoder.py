import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class LeanStateEncoder(nn.Module):
    """Encode Lean proof state text into a fixed-dim latent vector.

    Uses a frozen pretrained code model (e.g. CodeBERT) as backbone,
    with a learned linear projection to the model's latent dimension.

    Input:  tokenized text  (input_ids, attention_mask)
    Output: [B, dim]
    """

    def __init__(self, dim=256, backbone="microsoft/codebert-base"):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone)
        for p in self.backbone.parameters():
            p.requires_grad = False  # frozen backbone

        hidden = self.backbone.config.hidden_size  # typically 768
        self.proj = nn.Linear(hidden, dim)

    def forward(self, input_ids, attention_mask=None):
        """
        input_ids:      [B, seq_len]
        attention_mask:  [B, seq_len]  (optional)
        Returns:         [B, dim]
        """
        with torch.no_grad():
            out = self.backbone(input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]  # CLS token: [B, hidden]
        return self.proj(cls)              # [B, dim]

    @staticmethod
    def get_tokenizer(backbone="microsoft/codebert-base"):
        return AutoTokenizer.from_pretrained(backbone)
