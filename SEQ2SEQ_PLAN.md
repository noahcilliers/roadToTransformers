# Seq2Seq Architecture Plan

Planned: 2026-08-18. **Not yet implemented.**

English → Afrikaans translation. A faithful build of Sutskever, Vinyals & Le (2014),
*Sequence to Sequence Learning with Neural Networks*. No attention, no bidirectional
layers — both are deliberate omissions, added later as single-variable changes.

## Configuration

| | |
|---|---|
| vocabulary | 16,000 joint BPE, shared English + Afrikaans |
| embedding dim | 1024, shared encoder/decoder, tied to output |
| hidden dim | 1024 |
| layers | 4 encoder + 4 decoder, unidirectional |
| encoder/decoder weights | separate (per the paper) |
| total parameters | **83,574,400** |

## Encoder

Reads the **reversed** source sentence. Only the final state matters — no per-token
outputs are kept, because without attention nothing downstream reads them.

| layer | input | output | state carried to t+1 | params |
|---|---|---|---|---:|
| embedding | `[B, S]` ids | `[B, S, 1024]` | — | 16,384,000 |
| enc 1 | `[B, S, 1024]` | `[B, S, 1024]` | `h₁, c₁` | 8,396,800 |
| enc 2 | `[B, S, 1024]` | `[B, S, 1024]` | `h₂, c₂` | 8,396,800 |
| enc 3 | `[B, S, 1024]` | `[B, S, 1024]` | `h₃, c₃` | 8,396,800 |
| enc 4 | `[B, S, 1024]` | `[B, S, 1024]` | `h₄, c₄` | 8,396,800 |

Per layer: `4H(input + H + 2) = 4 × 1024 × (1024 + 1024 + 2) = 8,396,800`.
Layers 2–4 cost the same as layer 1 because embedding dim equals hidden dim.

Within a layer, `h` and `c` both advance through time. Between layers, **only `h`
moves up** — each layer keeps a private cell state. Four layers means four
independent memories of the sentence.

## The handoff

The entire source sentence becomes two tensors:

```
h_enc  [4, B, 1024]      c_enc  [4, B, 1024]
```

8,192 numbers per sentence, fixed size regardless of source length. This is the
whole representation the decoder receives, and it is the bottleneck the model is
built to demonstrate. Expect clean output on short sentences and degradation past
roughly 25 tokens.

Source reversal is the only mitigation available without attention. Reverse the
source only, never the target. The paper reports perplexity 5.8 → 4.7 from this
alone.

## Decoder

Initialized directly from the encoder's final state: `h_dec ← h_enc`, `c_dec ← c_enc`,
layer for layer. Trained with teacher forcing on the gold target shifted right by
one and prefixed with `<sos>`.

| layer | input | output | params |
|---|---|---|---:|
| embedding | `[B, T]` ids | `[B, T, 1024]` | shared, 0 |
| dec 1 | `[B, T, 1024]` | `[B, T, 1024]` | 8,396,800 |
| dec 2 | `[B, T, 1024]` | `[B, T, 1024]` | 8,396,800 |
| dec 3 | `[B, T, 1024]` | `[B, T, 1024]` | 8,396,800 |
| dec 4 | `[B, T, 1024]` | `[B, T, 1024]` | 8,396,800 |
| output | `[B, T, 1024]` | `[B, T, 16000]` | 16,000 (bias only) |

The output projection reuses the embedding matrix transposed, so it adds only a bias.

## Parameter budget

| component | params | share |
|---|---:|---:|
| shared embedding (tied to output) | 16,384,000 | 19.6% |
| encoder, 4 layers | 33,587,200 | 40.2% |
| decoder, 4 layers | 33,587,200 | 40.2% |
| output bias | 16,000 | 0.0% |
| **total** | **83,574,400** | |

Recurrence is 80.4% of the model. The PTB baseline had this inverted — 74% sat in the
output projection — which is why that model was wider than it was deep in effect.

Size alternates, if the data supports a change after the first full run:

| V | H | params |
|---:|---:|---:|
| 16k | 768 | 50.1M |
| **16k** | **1024** | **83.6M** |
| 32k | 1024 | 100.0M |
| 16k | 1536 | 175.7M |

## Memory and throughput

| | |
|---|---|
| weights, fp32 | 334 MB |
| weights + gradients | 669 MB |
| + Adam moments | 1.34 GB |
| activations @ B=128, T=50 | ~2.5 GB |
| logits + softmax backward | ~1.2 GB |

Roughly 4 GB at batch 128 — comfortable on 24 GB.

Speed is the constraint, not memory. Measured on the M4: 7,378 tok/s for one 4-layer
H=1024 stack, ~3,700 tok/s for encoder plus decoder, giving **~9.4 h/epoch** and 6–10
days for a full run. Not viable. Estimated ~20x on the 4090 → ~26 min/epoch. Build and
debug on the M4, train on the 4090.

## Training

| | |
|---|---|
| init | uniform [-0.08, 0.08] |
| optimizer | SGD, lr 0.7, halve each half-epoch after epoch 5 |
| gradient clipping | norm 5 |
| batch size | 128 |
| dropout | 0.3 on non-recurrent connections only |
| decoding | beam search, beam 2 |
| eval | FLORES-200 devtest, chrF and sacreBLEU |

Adam at lr 1e-3 is a reasonable modern substitute for the SGD schedule and worth
trying as a variant, but the paper's schedule is the documented-working baseline.

## Deliberately excluded

- **Attention** (Bahdanau/Luong) — ~3M params. The single largest quality gain
  available, added once the bottleneck is observed firsthand.
- **Bidirectional encoder layer 1** — without attention only the final state is read,
  and a backward pass's final state corresponds to the start of the sentence.
- **Residual connections** — not needed at 4 layers; required past ~6.
