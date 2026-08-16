# LSTM Results

Recorded: 2026-08-14

## Model Configuration

This run is for the manual LSTM language model in `lstm.py`.

- Dataset: PennTreebank
- Task: next-token prediction
- Vocabulary size: 10,001
- Embedding dimension: 64
- Hidden dimension: 512
- Sequence length: 64
- Batch size: 64
- Optimizer: Adam, learning rate 0.001
- Checkpoint: `lstm_model.pth`

The checkpoint is saved whenever validation loss improves, so an interrupted run can still preserve the best validation model reached so far.

## Initial Training Run

Command:

```bash
python3 lstm.py --fn train
```

| Epoch | Train Loss | Valid Loss | Perplexity |
|---:|---:|---:|---:|
| 1 | 6.4194 | 5.9750 | 393.45 |
| 2 | 5.7511 | 5.6195 | 275.75 |
| 3 | 5.4354 | 5.4386 | 230.12 |
| 4 | 5.2231 | 5.3199 | 204.36 |
| 5 | 5.0613 | 5.2389 | 188.47 |

Initial test result after this run:

```text
Testing results: 16390 correct and 65914 wrong...  19.91 %
```

## Continued Training Run

Command:

```bash
python3 lstm.py --fn train --cont
```

| Continued Epoch | Train Loss | Valid Loss | Perplexity | Note |
|---:|---:|---:|---:|---|
| 1 | 4.9785 | 5.1864 | 178.83 | validation improved |
| 2 | 4.8069 | 5.1359 | 170.02 | validation improved |
| 3 | 4.6831 | 5.1040 | 164.68 | validation improved |
| 4 | 4.5729 | 5.0836 | 161.35 | validation improved |
| 5 | 4.4749 | 5.0651 | 158.40 | validation improved |
| 6 | 4.3853 | 5.0633 | 158.11 | best validation in this run |
| 7 | 4.3045 | 5.0646 | 158.32 | overfitting warning |
| 8 | 4.2302 | 5.0647 | 158.34 | overfitting warning |
| 9 | 4.1576 | 5.0688 | 158.99 | overfitting warning |
| 10 | 4.0875 | 5.0808 | 160.90 | overfitting warning |

The run was interrupted after epoch 10. Because the code now saves on validation improvement, the best checkpoint from this continuation run should correspond to continued epoch 6, where validation loss reached `5.063283065212591`.

Test result after the continuation run:

```text
Testing results: 18411 correct and 63893 wrong...  22.37 %
```

That is `18411 / 82304` correct next-token predictions.

## Interpretation

The continuation run shows a clear pattern:

- Training loss kept improving from `4.9785` to `4.0875`.
- Validation loss improved through continued epoch 6, then flattened and drifted upward.
- The best validation point was around `5.0633` perplexity `158.11`.
- Test accuracy improved from `19.91%` to `22.37%`.

This suggests the LSTM learned useful sequence structure beyond the first checkpoint, but at learning rate `0.001` it started to plateau around validation loss `5.06`.

## Comparison Notes

The LSTM result is encouraging compared with the previous Elman RNN experiments:

- Bigger RNN epoch-20 test result: `183/1024 (17.87%)`
- Bigger RNN epoch-40 test result: `172/1024 (16.80%)`
- LSTM continued test result: `18411/82304 (22.37%)`

These are not perfectly apples-to-apples because the RNN test sampled fewer examples, while the LSTM test evaluated a much larger test stream. Still, the LSTM's result is a strong signal that the gated cell state is helping next-token prediction.

## Next Experiments

- Continue from the best checkpoint with a lower learning rate, such as `0.0003`.
- Save richer checkpoints with model state, optimizer state, best validation loss, and epoch count.
- Add LSTM generation to compare qualitative text samples against the RNN.
- Keep tracking validation loss and test accuracy together, since training loss alone kept improving even after validation plateaued.

---

# Generation and Decoding

Recorded: 2026-08-16

Generation from the 158-perplexity checkpoint produced long runs of `<unk>`:

```text
context:  industry science and technology told it that he was n't convinced that the purchase is likely to be of net
generated: income <eos> the <unk> of the <unk> of the <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk>
```

## The `<unk>` tokens are not a vocabulary bug

Counting `.data/penn-treebank/ptb.train.txt` directly:

| token | count | share |
|---|---:|---:|
| `the` | 50,770 | 5.72% |
| `<unk>` | 45,020 | 5.07% |
| `N` | 32,481 | 3.66% |
| `of` | 24,400 | 2.75% |

Total: 887,521 tokens, 9,999 distinct types.

`<unk>` is the second most frequent token in the corpus and appears as a literal string in the raw text file. The Mikolov-preprocessed PennTreebank is distributed already unked — rare words were replaced before release — so no vocabulary change can recover the originals. A model that predicts `<unk>` often is behaving correctly on this data.

Escaping `<unk>` entirely would require a different corpus. WikiText-2 is the natural next step: comparable size, ~33k vocab, preserved casing and punctuation, and only ~2.7% `<unk>`.

## The actual cause was greedy decoding

`__generate__` used `logits.argmax(dim=1)`, which always takes the single most probable token and therefore falls into the highest-frequency attractor. The same checkpoint produced fluent text before collapsing, which indicated the model was not the problem:

```text
generated: said it will sell $ n million of its common shares outstanding <eos>
```

## Fix: sampling with temperature and top-k

Added `sample()` and replaced both `argmax` calls in `__generate__`. `__test__` still uses `argmax`, since accuracy should measure the single best guess.

```bash
python3 lstm.py --fn generate --temperature 1.1 --top-k 40
```

New flags: `--temperature` (default `0.8`) and `--top-k` (default `40`).

Output from the same checkpoint at `--temperature 1.1 --top-k 40`:

```text
context:  times which this week also unveiled a <unk> <eos> new york times co. is expected to report lower earnings for
generated: the quarter <eos> the latest quarter were $ n a share a year earlier <eos> the treasury results included that

context:  <unk> 's investment banker because it is also a creditor <eos> it said it chose lazard in part because of
generated: the new york stock exchange <eos> futures fell about $ n million or $ n a share in the quarter
```

The repetition loops are gone with no change to the model weights. `<unk>` still appears at roughly 18% of generated tokens against 5% in the corpus; that gap reflects an undertrained model hedging toward frequent tokens and should close as perplexity improves.

Note that temperature `0.8` *sharpens* the distribution and so was not what fixed the loop — replacing `argmax` with `torch.multinomial` was. Raising temperature above `1.0` reduces `<unk>` frequency by flattening the distribution away from the most common tokens.

---

# Run 3 Configuration: Scaled and Regularized

Recorded: 2026-08-16. **Training not yet run — results pending.**

## Configuration

| | Baseline (ppl 158) | Run 3 |
|---|---:|---:|
| Embedding dimension | 64 | 1024 |
| Hidden dimension | 512 | 1024 |
| Dropout | none | 0.5 |
| Learning rate | 0.001 | 0.001 |
| Epochs | 15 (+10 continued) | 40 |
| Sequence length | 64 | 64 |
| Batch size | 64 | 32 |
| Parameters | 6,952,273 | 28,884,753 |
| Checkpoint | `lstm_model.pth` | `lstm_model_28M.pth` |

This run changes several things at once and should be treated as one new configuration rather than an isolated test of any single change.

## Parameter allocation

The baseline spent most of its capacity on the output projection rather than on the recurrence:

| component | baseline | share | run 3 | share |
|---|---:|---:|---:|---:|
| embeddings | 640,064 | 9.2% | 10,241,024 | 35.5% |
| four gates | 1,181,696 | 17.0% | 8,392,704 | 29.1% |
| `hidden_to_output` | 5,130,513 | 73.8% | 10,251,025 | 35.5% |
| **total** | **6,952,273** | | **28,884,753** | |

The LSTM cell itself grew from 1.18M to 8.39M parameters.

## Dropout placement

Following Zaremba, Sutskever & Vinyals (2014), dropout is applied only to the non-recurrent connections:

- `embeddings` output, before the concatenation with `h`
- `h`, on its way into `hidden_to_output`

The `h` returned to the next timestep is left untouched. Applying a fresh mask to the recurrent path each step would leave a `(1-p)^T = 0.5^64 ≈ 5.4e-20` chance of any unit retaining information across a 64-step chunk, destroying the memory the cell exists to provide.

Verified by smoke test: the returned `h` has 0.00% exact zeros in train mode.

## Timing

Benchmarked on MPS at batch 32, sequence length 64:

| config | params | min/epoch |
|---|---:|---:|
| e64 / h512 | 6.95M | 1.1 |
| e1024 / h1024 | 18.6M (tied) | 3.6 |
| e1500 / h1500 | 33.0M (tied) | 4.5 |

Roughly 2.5 hours for 40 epochs. Two measured observations: fusing the four gate matmuls into one gives no speedup at these dimensions, because the 10,001-way output projection dominates each step; moving that projection and the cross entropy outside the timestep loop saves 30–35% and is not yet implemented.

## What to watch

- Epoch 1 should start near `9.21` (`ln(10001)`, uniform over the vocab) and fall quickly.
- The train/validation **gap**, not the train loss. The baseline reached train `4.09` against validation `5.06`. Train loss will look worse here because dropout handicaps training only; success is the two staying close.
- When "Overfitting detected" first appears. The baseline hit it at continued epoch 6. Much later or never would suggest undertraining rather than overfitting, in which case the next move is more epochs rather than more regularization.

Reference targets: Zaremba medium (2x650, dropout 0.5) reaches ~82 perplexity; large (2x1500, dropout 0.65) reaches ~78.

## Results

Pending.

## Planned Next Run

Weight tying, as the first single-variable change against this configuration:

```python
self.hidden_to_output.weight = self.embeddings.weight
```

Both matrices are `[10001, 1024]` and encode word identity in opposite directions. Tying cuts 10,241,024 parameters (28.88M to 18.64M) at no compute cost, and doubles the gradient signal per word — which matters because 8,015 of 9,999 vocabulary words appear fewer than 50 times in training, and each currently has a 1,024-parameter output row estimated from those few examples.

References: Press & Wolf (arXiv:1608.05859), Inan et al. (arXiv:1611.01462).
