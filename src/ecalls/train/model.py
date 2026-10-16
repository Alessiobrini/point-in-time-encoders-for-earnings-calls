"""The vintage encoder's architecture, held fixed across every vintage.

Two constraints decide the shape, and they pull in opposite directions.

**Section 5 item 2 requires the architecture, dimension, hyperparameters, seed and training budget to be
identical across vintages, so that only the data varies.** A later vintage trained longer because it had
more text would confound data volume with training length.

**The comparison that justifies Phase 4 is against `bge-small-en-v1.5`, measured at 0.5378 on the
retrieval benchmark.** That model is 12 layers, 384 hidden, 33.4 M parameters. Matching its capacity means
a difference in the result is attributable to the training data and the domain rather than to size, which
is the comparison worth running. Beating it with a larger model would show only that larger models help.

The backbone is ModernBERT's, which supplies the recipe the methodology notes specify: no next-sentence
prediction, rotary positions, gated linear units in the feedforward block, alternating local and global
attention, and unpadding. Using the published implementation rather than writing one keeps the
architecture side of the comparison uncontroversial.

`max_position_embeddings` is 768 because rotary positions need a bound and the packed corpus is packed to
exactly 768, which was itself measured to cover 99.95% of chunks whole.
"""
import json

CONTEXT = 768
HIDDEN = 384
LAYERS = 12
HEADS = 12
# GLU doubles the first feedforward projection, so matching bge-small’s 33.4 M total needs a narrower
# intermediate size than its 1536. The value here is chosen by counting parameters, see `describe`.
INTERMEDIATE = 1024
TARGET_PARAMS = 33.4e6


def config(vocab_size: int, hidden: int = HIDDEN, layers: int = LAYERS, heads: int = HEADS,
           intermediate: int = INTERMEDIATE, context: int = CONTEXT):
    from transformers import ModernBertConfig

    return ModernBertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        intermediate_size=intermediate,
        max_position_embeddings=context,
        # BERT's special tokens, at the ids the tokenizer assigns them.
        pad_token_id=0, bos_token_id=2, eos_token_id=3, cls_token_id=2, sep_token_id=3,
        # Masked language modeling only. Next-sentence prediction is gone from every modern recipe.
        tie_word_embeddings=True,
    )


def build_mlm(vocab_size: int, seed: int = 0, **kw):
    """A freshly initialized encoder with a masked language modeling head.

    The seed is part of the fixed budget rather than a convenience: two vintages initialized differently
    would differ for a reason that has nothing to do with their data.
    """
    import torch
    from transformers import ModernBertForMaskedLM

    torch.manual_seed(seed)
    return ModernBertForMaskedLM(config(vocab_size, **kw))


def build_encoder(vocab_size: int, seed: int = 0, **kw):
    """The same backbone without the masked language modeling head, for the contrastive stage."""
    import torch
    from transformers import ModernBertModel

    torch.manual_seed(seed)
    return ModernBertModel(config(vocab_size, **kw))


def count_params(model) -> dict:
    emb = model.get_input_embeddings().weight.numel()
    tot = sum(p.numel() for p in model.parameters())
    # The head is tied to the input embedding, so it costs nothing extra and is not double counted.
    return {"total": tot, "embedding": emb, "encoder": tot - emb,
            "embedding_share": emb / tot}


def describe(vocab_size: int, **kw) -> dict:
    """Parameter counts for a candidate shape, so the size choice is measured rather than assumed."""
    m = build_encoder(vocab_size, **kw)
    c = m.config
    return {"hidden": c.hidden_size, "layers": c.num_hidden_layers, "heads": c.num_attention_heads,
            "intermediate": c.intermediate_size, "context": c.max_position_embeddings,
            "vocab": c.vocab_size, **count_params(m)}


def save_config(path, vocab_size: int, window: dict, seed: int = 0, **kw) -> None:
    """Write the architecture and the training window together.

    Section 5 item 4: a vintage whose training window cannot be stated cannot support a claim. The window
    travels with the architecture so a model directory is self-describing.
    """
    from pathlib import Path

    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    m = build_encoder(vocab_size, seed=seed, **kw)
    m.config.save_pretrained(p)
    (p / "_window.json").write_text(json.dumps({**window, "seed": seed,
                                                **count_params(m)}, indent=2) + "\n")
