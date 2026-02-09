"""Training loop for LEAN-sorcerer.

Combined objective:
  L = L_contrastive + 0.3 * L_classifier + 0.1 * L_confidence

Data: (proof_state_text, tactic_text, coarse_label) triples.
Tactic embeddings are produced by the same frozen encoder used for proof states.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model import HeterogeneousTTCModel
from model.encoder import LeanStateEncoder


# ── Losses ───────────────────────────────────────────────────

def info_nce_loss(query, positive, temperature=0.07):
    """InfoNCE contrastive loss (in-batch negatives).

    query:    [B, dim]  model output embeddings (L2-normalized)
    positive: [B, dim]  target tactic embeddings (L2-normalized)
    Returns:  scalar loss
    """
    # Similarity matrix: [B, B]
    logits = (query @ positive.T) / temperature
    labels = torch.arange(query.size(0), device=query.device)
    return F.cross_entropy(logits, labels)


def combined_loss(emb, logits, conf, tactic_emb, coarse_labels,
                  temperature=0.07, w_cls=0.3, w_conf=0.1):
    """Combined training objective.

    emb:           [B, dim]  model retrieval embedding
    logits:        [B, C]    coarse tactic family logits
    conf:          [B, 1]    confidence score
    tactic_emb:    [B, dim]  ground-truth tactic embedding
    coarse_labels: [B]       coarse tactic family index
    """
    l_contrastive = info_nce_loss(emb, tactic_emb, temperature)
    l_classifier = F.cross_entropy(logits, coarse_labels)

    # Confidence target: 1 if the model's top retrieval was correct
    with torch.no_grad():
        sims = emb @ tactic_emb.T  # [B, B]
        top1 = sims.argmax(dim=1)
        correct = (top1 == torch.arange(emb.size(0), device=emb.device)).float()
    l_confidence = F.binary_cross_entropy(conf.squeeze(-1), correct)

    return l_contrastive + w_cls * l_classifier + w_conf * l_confidence


# ── Dataset ──────────────────────────────────────────────────

class TacticDataset(Dataset):
    """Dataset of (proof_state, tactic, coarse_label) triples.

    Expects pre-tokenized data or raw text with a tokenizer.
    Override `load_data()` to plug in LeanDojo or custom sources.
    """

    def __init__(self, data, tokenizer, max_len=256):
        """
        data: list of dicts with keys:
              'proof_state' (str), 'tactic' (str), 'coarse_label' (int)
        tokenizer: HuggingFace tokenizer (e.g. CodeBERT)
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        state_enc = self.tokenizer(
            item["proof_state"],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tactic_enc = self.tokenizer(
            item["tactic"],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "state_ids": state_enc["input_ids"].squeeze(0),
            "state_mask": state_enc["attention_mask"].squeeze(0),
            "tactic_ids": tactic_enc["input_ids"].squeeze(0),
            "tactic_mask": tactic_enc["attention_mask"].squeeze(0),
            "coarse_label": torch.tensor(item["coarse_label"], dtype=torch.long),
        }


# ── Training ─────────────────────────────────────────────────

def train_epoch(model, tactic_encoder, dataloader, optimizer, device,
                num_loops=8, temperature=0.07):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        state_ids = batch["state_ids"].to(device)
        state_mask = batch["state_mask"].to(device)
        tactic_ids = batch["tactic_ids"].to(device)
        tactic_mask = batch["tactic_mask"].to(device)
        coarse_labels = batch["coarse_label"].to(device)

        # Forward: proof state → model → (emb, logits, conf)
        emb, logits, conf = model(state_ids, state_mask, num_loops=num_loops)

        # Encode target tactics with the same frozen encoder
        with torch.no_grad():
            tactic_emb = tactic_encoder(tactic_ids, tactic_mask)
            tactic_emb = F.normalize(tactic_emb, dim=-1)

        loss = combined_loss(emb, logits, conf, tactic_emb, coarse_labels,
                             temperature=temperature)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, tactic_encoder, dataloader, device, num_loops=8, k=10):
    """Top-k retrieval accuracy on a held-out set."""
    model.eval()
    correct_at_k = 0
    total = 0

    for batch in dataloader:
        state_ids = batch["state_ids"].to(device)
        state_mask = batch["state_mask"].to(device)
        tactic_ids = batch["tactic_ids"].to(device)
        tactic_mask = batch["tactic_mask"].to(device)

        emb, _, _ = model(state_ids, state_mask, num_loops=num_loops)

        tactic_emb = tactic_encoder(tactic_ids, tactic_mask)
        tactic_emb = F.normalize(tactic_emb, dim=-1)

        sims = emb @ tactic_emb.T  # [B, B]
        topk = sims.topk(min(k, sims.size(1)), dim=1).indices
        targets = torch.arange(emb.size(0), device=device).unsqueeze(1)
        correct_at_k += (topk == targets).any(dim=1).sum().item()
        total += emb.size(0)

    return correct_at_k / total if total > 0 else 0.0


# ── Entry point ──────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train LEAN-sorcerer")
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-loops", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--backbone", type=str, default="microsoft/codebert-base")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Model
    model = HeterogeneousTTCModel(
        dim=args.dim,
        num_heads=args.num_heads,
        backbone=args.backbone,
    ).to(device)

    # Separate tactic encoder (shares backbone weights via same pretrained model)
    tactic_encoder = LeanStateEncoder(dim=args.dim, backbone=args.backbone).to(device)
    tactic_encoder.eval()

    tokenizer = LeanStateEncoder.get_tokenizer(args.backbone)

    # ── Placeholder: replace with real data (e.g. LeanDojo) ──
    dummy_data = [
        {"proof_state": "⊢ ∀ n : ℕ, 0 + n = n", "tactic": "intro n", "coarse_label": 0},
        {"proof_state": "n : ℕ ⊢ 0 + n = n", "tactic": "simp", "coarse_label": 1},
    ] * 64  # repeat for minimal batch training
    dataset = TacticDataset(dummy_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Only optimize learnable params (encoder projection + experts + MLA + head)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, tactic_encoder, dataloader, optimizer, device,
                           num_loops=args.num_loops, temperature=args.temperature)
        acc = evaluate(model, tactic_encoder, dataloader, device,
                       num_loops=args.num_loops)
        scheduler.step()

        print(f"Epoch {epoch:3d} | loss={loss:.4f} | top-10 acc={acc:.3f}")

    print("Training complete.")


if __name__ == "__main__":
    main()
