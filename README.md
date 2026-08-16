# Road to Transformers

A from-scratch learning repo. The goal is to understand the Transformer by building every architecture that leads up to it, in order, by hand — no `nn.RNN`, no `nn.LSTM`, no copy-pasted reference implementations.

```
feed-forward nets  ->  n-gram LM  ->  RNN (Elman)  ->  LSTM  ->  seq2seq  ->  attention  ->  Transformer
      done              done            done          done        next
```

Everything from the n-gram model onward trains on **PennTreebank** next-token prediction, so each architecture is directly comparable to the last one on the same task.

## Where I am now

The manual LSTM works and beats the Elman RNN on held-out next-token accuracy. The cell math is done — concatenate `[embedding, h]`, compute all four gates in parallel, update `c` and `h` — and training now tracks validation loss and perplexity instead of just training loss.

**Next up: seq2seq, then attention, then the Transformer.** Coming out of the LSTM, the motivating question is the one attention answers — a single fixed-size hidden state has to carry everything the model knows about the past, and that becomes the bottleneck.

## Results so far

All on PennTreebank next-token prediction, `vocab_size = 10,001`.

| Model | Config | Held-out accuracy | Valid loss / perplexity |
|---|---|---:|---|
| Feed-forward n-gram | ctx 5, emb 64, hidden 512 | 21.20% (1000 examples) | — |
| Elman RNN | emb 56, hidden 512, epoch 20 | 17.87% (1024 examples) | — |
| Elman RNN | emb 56, hidden 512, epoch 40 | 16.80% (1024 examples) | — |
| Manual LSTM | emb 64, hidden 512, 5 epochs | 19.91% (82,304 examples) | 5.2389 / 188.47 |
| Manual LSTM | emb 64, hidden 512, +10 epochs | **22.37%** (82,304 examples) | 5.0633 / 158.11 |

Caveats worth keeping in mind: the RNN tests sampled ~1k examples while the LSTM evaluated the full test stream, so those numbers aren't perfectly apples-to-apples. And the n-gram's 21.20% looks competitive only because exact next-token accuracy is a weak metric — validation loss and perplexity are the honest comparison, which is why they were added at the LSTM stage.

Full LSTM run tables live in `LSTM_RESULTS.md` (untracked, local). The dated narrative of what was learned each day is in [LEARNING_LOG.md](LEARNING_LOG.md).

## Repo layout

| File | What it is |
|---|---|
| [learning/tensors.py](learning/tensors.py) | Tensor basics — creation, shapes, devices, matmul, in-place ops |
| [learning/datasets.py](learning/datasets.py) | FashionMNIST datasets and `DataLoader` batching |
| [learning/transforms.py](learning/transforms.py) | Image transforms and one-hot label transforms |
| [learning/nn.py](learning/nn.py) | First feed-forward net with `nn.Module` / `nn.Sequential` |
| [learning/autodiff.py](learning/autodiff.py) | Autograd — `requires_grad`, `grad_fn`, `backward()`, reading gradients |
| [manualFashionMNIST.py](manualFashionMNIST.py) | Full CNN training script — conv, pooling, cross entropy, Adam, checkpointing |
| [ngram.py](ngram.py) | Feed-forward n-gram LM. Fixed context window -> embeddings -> flatten -> MLP -> vocab logits |
| [rnn.py](rnn.py) | Elman RNN. `h = tanh(W_x·x + W_h·h)`, truncated BPTT, gradient clipping |
| [lstm.py](lstm.py) | Manual LSTM cell — forget/input/candidate/output gates, cell state, validation + perplexity |

## Setup

Built against `torch 2.13.0`, `torchvision 0.28.0`, and **`torchtext 0.6.0`** — the old version, which is deliberate. The language-model scripts use the legacy `Field` / `PennTreebank.splits(...)` API, so a newer torchtext will not work.

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install torch torchvision "torchtext==0.6.0" numpy
```

PennTreebank downloads itself into `.data/` on first run. Checkpoints (`*.pth`), `.data/`, and `.venv/` are all gitignored.

## Running things

`ngram.py`, `rnn.py`, and `manualFashionMNIST.py` take a positional flag:

```bash
python3 ngram.py train
```

Valid flags are `train`, `continue`, `test`, and — for `ngram.py` and `rnn.py` — `generate`.

`lstm.py` uses argparse instead:

```bash
python3 lstm.py --fn train
```

Add `--cont` to resume from `lstm_model.pth`. `--fn test` runs next-token evaluation. The LSTM saves a checkpoint only when validation loss improves, so an interrupted run still keeps the best model it reached; it prints an overfitting warning when validation stops improving while training loss keeps falling.

Two rough edges to know about: `rnn.py` has its checkpoint filenames hardcoded to `rnn_model_bigger_epoch_20.pth` (load) and `rnn_model_bigger_epoch_40.pth` (save), so continuing a run means editing those lines. And `lstm.py`'s help text mentions `generate`, but LSTM generation isn't implemented yet.

## Things that took real debugging

Kept here because these were the actual sticking points, not the smooth parts.

**The RNN memory explosion.** PennTreebank arrives as one enormous row of ~929k tokens. Backpropagating through the whole thing builds a single computation graph over the entire corpus. The fix was `batchify(...)` to reshape the corpus into contiguous parallel streams, plus truncated BPTT: process `seq_len` chunks, carry `h` forward for context, and `h.detach()` between chunks to keep the hidden *values* while cutting the gradient history. Gradient clipping via `clip_grad_norm_` went in at the same time to tame unstable updates.

**The first LSTM draft had the gates wrong.** They were built as sequential `nn.Sequential` stages — each gate feeding the next. They should be computed *in parallel* from the same `[embedding, h]` concatenation, then combined by the LSTM equations. Defining `nn.Linear` layers in `__init__` and applying `sigmoid` / `tanh` in `forward` is what made this click.

**Why sigmoid for gates and tanh for candidates.** `sigmoid` maps to `0..1`, which behaves like a knob — how much to keep, how much to write. `tanh` maps to `-1..1`, which is signed content. Gates are decisions; the candidate is information.

**Training loss is not the metric.** The bigger RNN's training loss fell from 3.38 to 2.96 between epochs 20 and 40 while held-out accuracy *dropped* from 17.87% to 16.80%. That result is what motivated adding a validation split, and then perplexity — `exp(validation_loss)`, since PyTorch's `CrossEntropyLoss` computes `-ln(p_correct)` in nats, not bits. Perplexity reads as roughly how many next tokens the model is still torn between.

**`CrossEntropyLoss` wants raw logits.** It applies the softmax internally. Feeding it softmax output means applying softmax twice.

## Reading behind the code

- Elman, *Finding Structure in Time* — the original recurrent architecture, and why temporal XOR needs memory rather than a wider input window
- Olah, *Understanding LSTM Networks* — the friendly gate diagrams; worth reading against the original LSTM paper, whose memory-cell diagram has the self-loop and the input/output gates but no forget gate
- 3Blue1Brown on compression as intelligence — the intuition that cross entropy measures how surprised a model is by the correct answer
