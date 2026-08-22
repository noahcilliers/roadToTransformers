"""
Do a multilingual model's middle layers represent a sentence and its translation
with the same internal vectors?

Run it, no arguments, no inputs:

    python3 crosslingual_probe.py

The first run downloads ~1.1GB of BLOOM weights; later runs use the cache.

WHAT IT MEASURES, AND WHY IT'S BUILT THIS WAY
---------------------------------------------
Raw cosine between hidden states is close to useless on its own. Transformer
activations are anisotropic: they occupy a narrow cone, so two totally unrelated
sentences still sit around 0.98. Four choices below exist to get around that:

  fp32           transformers v5 loads BLOOM in its checkpoint dtype (fp16),
                 whose resolution near 1.0 is 0.000488 -- coarser than the
                 effect we're measuring. Forced to fp32.
  many pairs     One sentence pair is an anecdote. Averaged over 8.
  centering      Subtract the per-layer mean vector to remove the common
                 direction. This is what separates signal from the cone.
  retrieval      "Does each English sentence's nearest French neighbour happen
                 to be its own translation?" Invariant to any shared offset,
                 so the cone can't fake it.

Pooling matters too. Sentences differ in token count, so each layer's
(seq_len, hidden) activation is collapsed to one (hidden,) vector:
  mean  -- average over tokens; swamped by a few huge outlier dimensions
  last  -- final token, which in a causal LM has attended to everything
Both are reported. For decoder-only models `last` is the sharper probe.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "bigscience/bloom-560m"

# Literal translation pairs, matched in content.
PAIRS = [
    ("The cat sat on the mat.",                 "Le chat était assis sur le tapis."),
    ("She drank a glass of cold water.",        "Elle a bu un verre d'eau froide."),
    ("The train arrives at nine o'clock.",      "Le train arrive à neuf heures."),
    ("He wrote a long letter to his mother.",   "Il a écrit une longue lettre à sa mère."),
    ("The children played in the garden.",      "Les enfants jouaient dans le jardin."),
    ("This book is very difficult to read.",    "Ce livre est très difficile à lire."),
    ("The stock market fell after the meeting.", "La bourse a chuté après la réunion."),
    ("We ate bread and cheese for lunch.",      "Nous avons mangé du pain et du fromage au déjeuner."),
]

POOLINGS = ("mean", "last")


def banner(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def load():
    banner("SETUP")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # dtype=float32 is deliberate; see module docstring.
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    model.eval()
    print(f"model        : {MODEL_NAME}")
    print(f"param dtype  : {next(model.parameters()).dtype}  (forced; checkpoint ships fp16)")
    print(f"hidden size  : {model.config.hidden_size}")
    print(f"layers       : {model.config.n_layer}")
    return tokenizer, model


def layer_vectors(text, pool, tokenizer, model):
    """One pooled (hidden,) vector per layer -> (n_layers + 1, hidden)."""
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    # out.hidden_states: tuple of (batch, seq_len, hidden), embeddings at index 0
    if pool == "mean":
        return torch.stack([h[0].mean(dim=0) for h in out.hidden_states])
    return torch.stack([h[0, -1] for h in out.hidden_states])


def show_shapes(tokenizer, model):
    """The original sanity check: confirm what comes out of the model."""
    banner("SHAPES")
    en, fr = PAIRS[0]
    for label, text in (("English", en), ("French", fr)):
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        hs = out.hidden_states
        n_tok = inputs["input_ids"].shape[1]
        print(f"{label:>8}: {text}")
        print(f"{'':>8}  {n_tok} tokens, {len(hs)} hidden states "
              f"(embeddings + {len(hs) - 1} layers), each {tuple(hs[0].shape)}")
    print("\nToken counts differ, so the sequence axis is pooled away before comparing.")


def stats(E, Fr):
    """Mean matched sim, mean mismatched sim, and top-1 retrieval accuracy."""
    sim = F.normalize(E, dim=1) @ F.normalize(Fr, dim=1).T          # (n, n)
    matched = sim.diag()
    mismatched = sim[~torch.eye(len(sim), dtype=torch.bool)]
    top1 = (sim.argmax(dim=1) == torch.arange(len(sim))).float().mean()
    return matched.mean().item(), mismatched.mean().item(), top1.item()


def probe(pool, tokenizer, model):
    en = torch.stack([layer_vectors(e, pool, tokenizer, model) for e, _ in PAIRS])
    fr = torch.stack([layer_vectors(f, pool, tokenizer, model) for _, f in PAIRS])

    rows = []
    for L in range(en.shape[1]):
        E, Fr = en[:, L], fr[:, L]
        raw_t, raw_c, _ = stats(E, Fr)
        mu = torch.cat([E, Fr]).mean(dim=0, keepdim=True)           # per-layer common direction
        cen_t, cen_c, acc = stats(E - mu, Fr - mu)
        rows.append({
            "layer": L, "raw_t": raw_t, "raw_c": raw_c, "raw_gap": raw_t - raw_c,
            "cen_t": cen_t, "cen_c": cen_c, "gap": cen_t - cen_c, "top1": acc,
        })

    banner(f"POOLING: {pool}")
    print(f"{'layer':>5} | {'raw tr':>7} {'raw ctl':>7} {'gap':>7} | "
          f"{'cen tr':>7} {'cen ctl':>7} {'gap':>7} | {'top1':>5} | centered gap")
    print("-" * 78)
    hi = max(r["gap"] for r in rows)
    for r in rows:
        label = "emb" if r["layer"] == 0 else str(r["layer"])
        bar = "#" * int(round(18 * max(r["gap"], 0) / hi)) if hi > 0 else ""
        print(f"{label:>5} | {r['raw_t']:7.4f} {r['raw_c']:7.4f} {r['raw_gap']:7.4f} | "
              f"{r['cen_t']:7.4f} {r['cen_c']:7.4f} {r['gap']:7.4f} | "
              f"{r['top1']:5.2f} | {bar}")

    if pool == "last":
        print("\nnote: the emb row reads 1.0000 with top1 at chance because every sentence\n"
              "      ends in '.' -- identical last tokens. That row is a pipeline check.")
    return rows


def verdict(results):
    banner("VERDICT")
    chance = 1.0 / len(PAIRS)
    for pool, rows in results.items():
        best = max(rows, key=lambda r: r["gap"])
        depth = 100 * best["layer"] / (len(rows) - 1)
        final = rows[-1]
        print(f"[{pool}] peak centered gap {best['gap']:.4f} at layer {best['layer']}"
              f" of {len(rows) - 1} ({depth:.0f}% depth), top1 {best['top1']:.2f}")
        print(f"{'':7}final layer gap {final['gap']:.4f}, top1 {final['top1']:.2f}")

    print(f"\nchance top-1 with {len(PAIRS)} candidates: {chance:.2f}")
    print(
        "\nRead the CENTERED gap, not the raw similarity. A gap that rises through\n"
        "the early layers, peaks past the middle, then falls toward the output is\n"
        "the signature of the model building a language-independent representation\n"
        "and then converting it back into language-specific next-token predictions.\n"
        "\nCaveats: 8 pairs is small and top-1 saturates at 1.00, so the peak layer is\n"
        "approximate. English/French share Latin roots and subword tokens, so some\n"
        "similarity is shared surface form rather than shared meaning -- BLOOM also\n"
        "covers Chinese, Arabic, Hindi and Swahili if you want a harder test; swap\n"
        "the French half of PAIRS above and rerun."
    )


def main():
    tokenizer, model = load()
    show_shapes(tokenizer, model)
    verdict({pool: probe(pool, tokenizer, model) for pool in POOLINGS})


if __name__ == "__main__":
    main()
