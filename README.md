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

## Model Components

### 1. Input Embedding
- `self.embed`: Linear(dim_in → dim)
- Projects raw input → fixed latent dimension
- Shape transformation: `[batch, input_features]` → `[batch, dim]`

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
text- Enables progressive refinement of the latent state across loops
```

### 5. Test-Time Compute (TTC) Loop
- Controlled by `num_loops` (hyperparameter, e.g. 4–16)
- Pseudocode flow per forward pass:
```
x ← embed(input)
for i in 1 to num_loops:
e1 ← transformer_expert(x)
e2 ← diffusion_expert(x)
e3 ← ssm_expert(x)
expert_views ← stack([e1, e2, e3], dim=1)          # [B, 3, dim]
attended     ← mla(expert_views)                    # [B, 3, dim] with RoPE applied internally
fused        ← mean(attended, dim=1)                # [B, dim]
x            ← final_agg(fused) + x                 # residual
return x
```

## Key Design Properties
- **Heterogeneity**: Three different inductive biases active simultaneously
- **No routing overhead**: All experts always compute (trade compute for diversity)
- **Efficient attention with position awareness**: MLA compresses KV (low memory); RoPE adds relative positioning without extra parameters or KV cache bloat
- **Iterative refinement**: TTC loop allows error correction & progressive improvement
- **Modular & extensible**: Easy to swap/replace experts or extend RoPE to longer sequences
