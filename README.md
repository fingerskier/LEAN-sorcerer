# LEAN-sorcerer
LEAN predictive prosthetic

**Heterogeneous TTC Model with MLA (No Routing – All Experts Active Every Loop)**

**Architecture**
* AI model purpose built for LEAN
* Vector DB pre-populated with LEAN proofs & tactics
  * using the model's encoder

# Heterogeneous TTC Model with MLA + RoPE (No Routing – All Experts Active Every Loop)

## Overall Purpose
- Combine three architecturally diverse experts (Transformer, Diffusion, SSM)
- Use **test-time compute (TTC)** via iterative refinement loops
- Fuse expert outputs using **Multi-head Latent Attention (MLA)** with **RoPE** for efficient, position-aware cross-expert communication
- No routing: every expert processes the current latent state in **every** loop iteration
- Predict next tactic via **contrastive retrieval** against a tactic embedding index

## Model Components

### 1. Lean State Encoder
- `self.encoder`: `LeanStateEncoder` — frozen CodeBERT backbone + learned linear projection
- Backbone: `microsoft/codebert-base` (125M params, frozen)
- Projection: Linear(768 → dim) — only learnable part (~197K params)
- Input: tokenized Lean proof state `(input_ids, attention_mask)`
- Output: `[batch, dim]`

### 2. Experts (run in parallel every loop)
| Expert              | Architecture Type      | Key Operation                              | Output Shape    | Role / Strength                          |
|---------------------|------------------------|--------------------------------------------|-----------------|------------------------------------------|
| `expert_transformer` | Transformer-style     | Simple projection + mean pooling           | `[batch, dim]`  | Sequence/context reasoning               |
| `expert_diffusion`   | Diffusion-based       | Iterative latent denoising (simplified DDPM) | `[batch, dim]`  | Iterative refinement / noise handling    |
| `expert_ssm`         | State-space (Mamba-like) | Residual linear state transitions         | `[batch, dim]`  | Efficient linear-time processing         |

- All three experts receive the **same current latent state** `x` each loop
- Outputs treated as three parallel "views" of the refined representation

### 3. Fusion via Multi-head Latent Attention (MLA) with RoPE
- `self.mla`: MultiHeadLatentAttention(dim, num_heads, latent_dim)
- **RoPE Integration**:
  - **Decoupled RoPE** variant: RoPE is applied **after** KV decompression, only to the positional-sensitive portions of Q and K heads.
  - Precompute frequency bases (`cos`, `sin`) for max sequence length (here small, e.g., seq_len ≤ 3 + buffer).
  - Rotate Q and K vectors in 2D subspaces (pairs of dimensions) using rotary matrices based on position indices (0, 1, 2 for the three expert views).
  - This injects **relative positional information** into attention scores without absolute positional encodings.
- Input to MLA: stacked expert outputs → `[batch, 3, dim]` (seq_len = 3)
  - Token 0 = Transformer expert output (position 0)
  - Token 1 = Diffusion expert output (position 1)
  - Token 2 = SSM expert output (position 2)
- MLA + RoPE mechanism:
  - Q: full linear projection → apply RoPE rotation
  - KV: low-rank compression → latent_dim → decompression → apply RoPE rotation to decompressed K and V
  - Attention computed over the 3 "expert tokens" with position-rotated Q/K
  - Output: `[batch, 3, dim]`
- Aggregation after MLA:
  - Mean pool across the 3 positions → `[batch, dim]`
  - (Alternative options: take first token, weighted sum via learned scalar, max, etc.)

### 4. Final Aggregation & Residual Update
- `self.final_agg`: Linear(dim → dim)
- Update rule (inside loop):
```
fused = final_agg(mean(mla_output, dim=1))
x_new = fused + x          # residual connection
```
- Enables progressive refinement of the latent state across loops

### 5. Test-Time Compute (TTC) Loop
- Controlled by `num_loops` (hyperparameter, e.g. 4–16)
- Pseudocode flow per forward pass:
```
x ← encoder(proof_state)
for i in 1 to num_loops:
    e1 ← transformer_expert(x)
    e2 ← diffusion_expert(x)
    e3 ← ssm_expert(x)
    expert_views ← stack([e1, e2, e3], dim=1)          # [B, 3, dim]
    attended     ← mla(expert_views)                    # [B, 3, dim] with RoPE applied internally
    fused        ← mean(attended, dim=1)                # [B, dim]
    x            ← final_agg(fused) + x                 # residual
emb, logits, conf ← head(x)
return emb, logits, conf
```

### 6. Output Head (`TacticHead`)
- **Retrieval projection**: MLP → L2-normalize → `[batch, dim]` embedding for FAISS nearest-neighbor lookup against pre-embedded tactics
- **Coarse classifier**: Linear(dim → C) over ~64 tactic families (simp, ring, apply, intro, etc.) — provides auxiliary gradients during training
- **Confidence scorer**: Linear(dim → 1) + sigmoid — predicts whether the model's top retrieval is correct, enables adaptive TTC (run more loops when confidence is low)

## Training

### Objective
Combined loss: `L = L_contrastive + 0.3 * L_classifier + 0.1 * L_confidence`

| Loss | Type | Purpose |
|------|------|---------|
| `L_contrastive` | InfoNCE (temperature=0.07) | Pull retrieval embedding toward correct tactic, push away from in-batch negatives |
| `L_classifier` | Cross-entropy | Coarse tactic family prediction — clean gradients early in training |
| `L_confidence` | Binary cross-entropy | Train confidence to predict retrieval correctness |

### Why contrastive + TTC?
The iterative refinement loop progressively pushes the latent toward the correct tactic embedding. Contrastive loss directly rewards this — the latent should be closer to the correct tactic than to any other tactic in the batch. More loops = better alignment.

### Data
- Source: LeanDojo (proof_state, tactic) pairs
- Tactic embeddings: produced by the same frozen encoder, stored in a FAISS index
- Coarse labels: ~64 tactic families derived from tactic name prefixes

## Key Design Properties
- **Heterogeneity**: Three different inductive biases active simultaneously
- **No routing overhead**: All experts always compute (trade compute for diversity)
- **Efficient attention with position awareness**: MLA compresses KV (low memory); RoPE adds relative positioning without extra parameters or KV cache bloat
- **Iterative refinement**: TTC loop allows error correction & progressive improvement
- **Retrieval-based prediction**: Open-ended tactic space handled via embedding similarity, not fixed classification
- **Modular & extensible**: Easy to swap/replace experts or extend RoPE to longer sequences
