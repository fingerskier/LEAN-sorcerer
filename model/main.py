import torch
import torch.nn as nn
import math

from .encoder import LeanStateEncoder
from .heads import TacticHead


# ──────────────────────────────────────────────────────────────
# Rotary Embeddings Helper
# ──────────────────────────────────────────────────────────────

def apply_rotary_emb(x, freqs_cos, freqs_sin):
    """Apply rotary positional embeddings.
    x: [B, nh, seq, hd]
    freqs_cos / freqs_sin: [1, 1, seq, hd]
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    return (x * freqs_cos) + (rotated * freqs_sin)


class RotaryEmbedding(nn.Module):
    """Precomputed RoPE frequencies."""
    def __init__(self, dim, base=10000.0, max_seq_len=32):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)  # [max_seq, dim]
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, x):
        seq_len = x.shape[2]
        return (
            self.cos_cached[:, :, :seq_len, ...].to(x.device, x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(x.device, x.dtype),
        )


# ──────────────────────────────────────────────────────────────
# Multi-head Latent Attention with RoPE
# ──────────────────────────────────────────────────────────────

class MultiHeadLatentAttention(nn.Module):
    """MLA with KV low-rank compression + RoPE (decoupled flavor)."""
    def __init__(self, dim, num_heads=8, latent_dim=None, rope_base=10000.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.latent_dim = latent_dim or (self.head_dim // 4)  # compression ratio ~4x typical

        # RoPE module
        self.rope = RotaryEmbedding(self.head_dim, base=rope_base, max_seq_len=16)

        # Projections
        self.q_proj   = nn.Linear(dim, dim)
        self.kv_down  = nn.Linear(dim, self.latent_dim, bias=False)      # A
        self.k_proj_up = nn.Linear(self.latent_dim, dim, bias=False)     # B_k
        self.v_proj_up = nn.Linear(self.latent_dim, dim, bias=False)     # B_v

        self.out_proj = nn.Linear(dim, dim)

        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        """
        x: [B, seq_len=3, dim]   # three expert views as short sequence
        """
        B, L, D = x.shape
        assert L <= 16, "Increase RoPE max_seq_len if needed"

        # Q: full projection + RoPE
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        freqs_cos, freqs_sin = self.rope(q)
        q = apply_rotary_emb(q, freqs_cos, freqs_sin)

        # KV: compress → decompress → RoPE on K
        latent_kv = self.kv_down(x)                     # [B, L, latent_dim]
        k_full = self.k_proj_up(latent_kv)              # [B, L, D]
        v_full = self.v_proj_up(latent_kv)

        k = k_full.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v_full.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        k = apply_rotary_emb(k, freqs_cos, freqs_sin)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale   # [B, nh, L, L]
        attn = attn.softmax(dim=-1)

        out = attn @ v                                  # [B, nh, L, hd]
        out = out.transpose(1, 2).contiguous().view(B, L, D)

        return self.out_proj(out)


# ──────────────────────────────────────────────────────────────
# Expert Definitions (simplified placeholders)
# ──────────────────────────────────────────────────────────────

class TransformerExpert(nn.Module):
    """Lightweight transformer-style expert."""
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: [B, dim] → fake seq=1
        return self.proj(x)


class DiffusionExpert(nn.Module):
    """Simplified diffusion-style latent refinement."""
    def __init__(self, dim, noise_steps=5):
        super().__init__()
        self.noise_steps = noise_steps
        self.denoise_net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        for _ in range(self.noise_steps):
            noise = torch.randn_like(x)
            noisy = x + noise
            pred = self.denoise_net(torch.cat([noisy, noise], dim=-1))
            x = noisy - pred
        return x


class SSMExpert(nn.Module):
    """Simple state-space / Mamba-like expert."""
    def __init__(self, dim):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # add dummy seq dim
        x = self.layer(x) + x
        return x.mean(dim=1)


# ──────────────────────────────────────────────────────────────
# Full Heterogeneous TTC Model
# ──────────────────────────────────────────────────────────────

class HeterogeneousTTCModel(nn.Module):
    """
    Heterogeneous experts + TTC loops + MLA with RoPE.
    All experts run every loop → fused via MLA → residual refinement.
    """
    def __init__(
        self,
        dim=256,
        num_heads=8,
        latent_dim=None,
        rope_base=10000.0,
        backbone="microsoft/codebert-base",
        num_coarse_classes=64,
    ):
        super().__init__()
        self.dim = dim

        # Lean proof-state encoder (frozen backbone + learned projection)
        self.encoder = LeanStateEncoder(dim=dim, backbone=backbone)

        # Experts
        self.expert_transformer = TransformerExpert(dim)
        self.expert_diffusion   = DiffusionExpert(dim)
        self.expert_ssm         = SSMExpert(dim)

        # MLA fusion with RoPE
        self.mla = MultiHeadLatentAttention(
            dim=dim,
            num_heads=num_heads,
            latent_dim=latent_dim,
            rope_base=rope_base,
        )

        # Final projection after mean-pooling MLA output
        self.final_agg = nn.Linear(dim, dim)

        # Output head: retrieval embedding + coarse classifier + confidence
        self.head = TacticHead(dim=dim, num_coarse_classes=num_coarse_classes)

    def forward(self, input_ids, attention_mask=None, num_loops=8):
        """
        input_ids:       [B, seq_len]   tokenized Lean proof state
        attention_mask:  [B, seq_len]   (optional)
        num_loops:       test-time compute iterations
        Returns:
            emb:    [B, dim]  L2-normalized retrieval embedding
            logits: [B, C]    coarse tactic family logits
            conf:   [B, 1]    confidence score
        """
        x = self.encoder(input_ids, attention_mask)  # → [B, dim]

        for _ in range(num_loops):
            # Run all experts
            e1 = self.expert_transformer(x)
            e2 = self.expert_diffusion(x)
            e3 = self.expert_ssm(x)

            # Stack as short sequence of expert views
            expert_views = torch.stack([e1, e2, e3], dim=1)  # [B, 3, dim]

            # MLA + RoPE fusion
            attended = self.mla(expert_views)                # [B, 3, dim]

            # Aggregate + residual
            fused = attended.mean(dim=1)                     # [B, dim]
            x = self.final_agg(fused) + x

        return self.head(x)


# ──────────────────────────────────────────────────────────────
# Example usage
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)
    model = HeterogeneousTTCModel(dim=256, num_heads=8, latent_dim=64)

    # Simulate tokenized Lean proof state (batch=4, seq_len=32)
    input_ids = torch.randint(0, 1000, (4, 32))
    attention_mask = torch.ones(4, 32, dtype=torch.long)

    emb, logits, conf = model(input_ids, attention_mask, num_loops=6)
    print(f"emb:    {emb.shape}")     # [4, 256]
    print(f"logits: {logits.shape}")  # [4, 64]
    print(f"conf:   {conf.shape}")    # [4, 1]
    print("Model ready for training / inference.")