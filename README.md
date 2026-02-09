# LEAN-sorcerer
LEAN predictive prosthetic

**Heterogeneous TTC Model with MLA (No Routing – All Experts Active Every Loop)**

**Architecture**
* AI model purpose built for LEAN
* Vector DB pre-populated with LEAN proofs & tactics

## Overall Purpose
- Combine three architecturally diverse experts (Transformer, Diffusion, SSM)
- Use **test-time compute (TTC)** via iterative refinement loops
- Fuse expert outputs using **Multi-head Latent Attention (MLA)** for efficient cross-expert communication
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

### 3. Fusion via Multi-head Latent Attention (MLA)
- `self.mla`: MultiHeadLatentAttention(dim, num_heads, latent_dim)
- Input to MLA: stacked expert outputs → `[batch, 3, dim]` (seq_len = 3)
  - Token 0 = Transformer expert output
  - Token 1 = Diffusion expert output
  - Token 2 = SSM expert output
- MLA mechanism:
  - Q: full linear projection
  - KV: low-rank compression → latent_dim → decompression to K and V
  - Attention computed over the 3 "expert tokens"
  - Output: `[batch, 3, dim]`
- Aggregation after MLA:
  - Mean pool across the 3 positions → `[batch, dim]`
  - (Alternative options: take first token, weighted sum, max, etc.)

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
x ← embed(input)
for i in 1 to num_loops:
e1 ← transformer_expert(x)
e2 ← diffusion_expert(x)
e3 ← ssm_expert(x)
expert_views ← stack([e1, e2, e3], dim=1)          # [B, 3, dim]
attended     ← mla(expert_views)                    # [B, 3, dim]
fused        ← mean(attended, dim=1)                # [B, dim]
x            ← final_agg(fused) + x                 # residual
return x
```

## Key Design Properties
- **Heterogeneity**: Three different inductive biases active simultaneously
- **No routing overhead**: All experts always compute (trade compute for diversity)
- **Efficient attention**: MLA compresses KV (low memory in loops)
- **Iterative refinement**: TTC loop allows error correction & progressive improvement
- **Modular & extensible**: Easy to swap/replace individual experts

## Typical Use Cases
- Latent refinement for downstream tasks (classification, regression, embedding)
- Iterative reasoning / puzzle solving
- Domain-specific fine-tuning (e.g. Lean tactic embedding refinement)
- Prototyping hybrid architectures without full sparse MoE training

This structure balances architectural diversity, test-time scaling, and memory-efficient fusion in a clean, trainable package.
