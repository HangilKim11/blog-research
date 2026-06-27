"""Encoder + head shape and gradient smoke tests on a tiny tokenizer."""

from __future__ import annotations

import torch

from amex.models.transformer import (
    ClassificationHead,
    ContrastiveHead,
    EncoderConfig,
    FeatureEmbedder,
    ReconstructionHead,
    SequenceEncoder,
    count_parameters,
)
from amex.ssl.tokenizer import TokenizerArtifact


def _tiny_tokenizer() -> TokenizerArtifact:
    return TokenizerArtifact(
        numeric_cols=[f"num_{i}" for i in range(5)],
        categorical_cols=["B_30", "D_63"],
        numeric_mean={f"num_{i}": 0.0 for i in range(5)},
        numeric_std={f"num_{i}": 1.0 for i in range(5)},
        categorical_vocab={"B_30": {"0": 1, "1": 2, "2": 3}, "D_63": {"CR": 1, "CO": 2}},
    )


def _fake_batch(B: int = 3, T: int = 13, F: int = 5, C: int = 2) -> dict[str, torch.Tensor]:
    return {
        "numeric_values": torch.randn(B, T, F),
        "numeric_mask": torch.zeros(B, T, F, dtype=torch.bool),
        "categorical_ids": torch.randint(0, 3, (B, T, C)),
        "attention_mask": torch.tril(torch.ones(B, T, dtype=torch.bool), diagonal=0)[
            :B
        ],  # variable lengths
    }


def test_feature_embedder_shape() -> None:
    tok = _tiny_tokenizer()
    cfg = EncoderConfig(d_model=64)
    emb = FeatureEmbedder(tok, cfg)
    batch = _fake_batch()
    out = emb(batch["numeric_values"], batch["numeric_mask"], batch["categorical_ids"])
    assert out.shape == (3, 13, 64)


def test_encoder_forward_shape_with_cls() -> None:
    tok = _tiny_tokenizer()
    cfg = EncoderConfig(d_model=64, n_layers=2, use_cls_token=True)
    enc = SequenceEncoder(tok, cfg)
    batch = _fake_batch()
    out = enc(
        batch["numeric_values"],
        batch["numeric_mask"],
        batch["categorical_ids"],
        batch["attention_mask"],
    )
    assert out.shape == (3, 14, 64)  # T+1 with CLS


def test_encoder_forward_shape_no_cls() -> None:
    tok = _tiny_tokenizer()
    cfg = EncoderConfig(d_model=64, n_layers=2, use_cls_token=False)
    enc = SequenceEncoder(tok, cfg)
    batch = _fake_batch()
    out = enc(
        batch["numeric_values"],
        batch["numeric_mask"],
        batch["categorical_ids"],
        batch["attention_mask"],
    )
    assert out.shape == (3, 13, 64)


def test_encoder_causal_mask_blocks_future() -> None:
    """With causal=True, position t's output must not depend on inputs at t+1."""
    tok = _tiny_tokenizer()
    cfg = EncoderConfig(d_model=64, n_layers=2, use_cls_token=False, dropout=0.0)
    enc = SequenceEncoder(tok, cfg).eval()
    batch = _fake_batch(B=1, T=8)
    with torch.no_grad():
        out_a = enc(
            batch["numeric_values"],
            batch["numeric_mask"],
            batch["categorical_ids"],
            torch.ones(1, 8, dtype=torch.bool),
            causal=True,
        )
        # Perturb only the LAST timestep's input
        perturbed = batch["numeric_values"].clone()
        perturbed[:, -1, :] += 10.0
        out_b = enc(
            perturbed,
            batch["numeric_mask"],
            batch["categorical_ids"],
            torch.ones(1, 8, dtype=torch.bool),
            causal=True,
        )
    # All positions except the last must be unchanged.
    torch.testing.assert_close(out_a[:, :-1], out_b[:, :-1], atol=1e-6, rtol=1e-5)
    # And the last position must change (perturbation visible).
    assert (out_a[:, -1] - out_b[:, -1]).abs().sum() > 1e-3


def test_reconstruction_head_shapes() -> None:
    tok = _tiny_tokenizer()
    head = ReconstructionHead(tok, d_model=64)
    hidden = torch.randn(3, 13, 64)
    num_pred, cat_logits = head(hidden)
    assert num_pred.shape == (3, 13, tok.n_numeric)
    assert len(cat_logits) == tok.n_categorical
    for i, col in enumerate(tok.categorical_cols):
        assert cat_logits[i].shape == (3, 13, tok.cat_vocab_size(col))


def test_contrastive_head_unit_norm() -> None:
    head = ContrastiveHead(d_model=64, proj_dim=32)
    cls = torch.randn(3, 64)
    z = head(cls)
    assert z.shape == (3, 32)
    norms = z.norm(dim=-1)
    torch.testing.assert_close(norms, torch.ones(3), atol=1e-5, rtol=1e-5)


def test_classification_head_logits_shape() -> None:
    head = ClassificationHead(d_model=64)
    x = torch.randn(3, 64)
    out = head(x)
    assert out.shape == (3,)


def test_full_forward_backward_smoke() -> None:
    """End-to-end gradient flow through encoder + reconstruction head."""
    tok = _tiny_tokenizer()
    cfg = EncoderConfig(d_model=64, n_layers=2)
    enc = SequenceEncoder(tok, cfg)
    head = ReconstructionHead(tok, d_model=64)
    batch = _fake_batch()

    out = enc(
        batch["numeric_values"],
        batch["numeric_mask"],
        batch["categorical_ids"],
        batch["attention_mask"],
    )
    # drop the CLS, keep per-timestep slots
    per_t = out[:, 1:, :]  # (B, T, D)
    num_pred, _ = head(per_t)
    loss = (num_pred - batch["numeric_values"]).pow(2).mean()
    loss.backward()
    grads = [p.grad for p in enc.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


def test_param_count_in_expected_range() -> None:
    """Small config should be 1-3M params; sanity-check parameter budget."""
    tok = _tiny_tokenizer()  # F=5, C=2 -- tiny; real has F=177 C=11
    cfg = EncoderConfig(d_model=128, n_layers=4, n_heads=4, dim_feedforward=512)
    enc = SequenceEncoder(tok, cfg)
    n = count_parameters(enc)
    # encoder alone (excluding heads) on tiny tokenizer is well under 1M
    assert 50_000 < n < 1_500_000, f"unexpected param count: {n:,}"
