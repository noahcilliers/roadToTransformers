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

| Epoch | Train Loss | Valid Loss | Perplexity | Note |
|---:|---:|---:|---:|---|
| 1 | 5.8145 | 5.2632 | 193.09 | |
| 2 | 5.1815 | 5.0223 | 151.76 | |
| 3 | 4.9269 | 4.9107 | 135.74 | |
| 4 | 4.7597 | 4.8581 | 128.78 | |
| 5 | 4.6305 | 4.8186 | 123.79 | |
| 6 | 4.5290 | 4.7968 | 121.13 | |
| 7 | 4.4430 | 4.7822 | 119.36 | |
| 8 | 4.3697 | 4.7742 | 118.41 | |
| 9 | 4.3055 | 4.7710 | 118.03 | |
| 10 | 4.2495 | 4.7681 | 117.69 | **best validation** |
| 11 | 4.1967 | 4.7728 | 118.25 | overfitting warning |
| 12 | 4.1512 | 4.7759 | 118.61 | overfitting warning |
| ... | | | | |
| 22 | 3.8454 | 4.8406 | 126.55 | run stopped |

**Best: epoch 10, validation 4.7681, perplexity 117.69.** Down from the 158.11 baseline, a 26% reduction.

Checkpoint preserved as `lstm_model_ppl11769_e1024_h1024_drop05.pth`.

### Interpretation

Dropout worked. Despite the model being 4.2x larger, the train/validation gap at the best epoch *narrowed* against the baseline: 0.5186 here versus 0.6780 before. Without regularization, quadrupling capacity on 887k tokens would have widened it considerably.

It is still not enough dropout. From epoch 11 onward, train loss fell 0.40 while validation rose 0.07, and the gap widened monotonically from 0.52 to 1.00 by epoch 22.

The failure mode is overfitting, not optimization. The per-epoch deltas were smooth and monotonic throughout — `-0.0560, -0.0528, -0.0455, -0.0408` on the train side with no oscillation. A learning rate set too high produces noisy, bouncing validation loss and a stalling train loss; none of that appears here. Lowering the learning rate would only descend more precisely into the same overfit basin.

Running 40 epochs was wasteful for this configuration, since the optimum arrived at epoch 10.

---

# Run 4: Weight Tying

Recorded: 2026-08-17

Single-variable change against Run 3:

```python
self.hidden_to_output.weight = self.embeddings.weight
```

Both matrices are `[10001, 1024]` and encode word identity in opposite directions — the embedding maps token to vector, the output projection maps vector to token scores. Sharing them cuts 10,241,024 parameters (28.88M to 18.64M) at no compute cost, and doubles the gradient signal per word, which matters because 8,015 of 9,999 vocabulary words appear fewer than 50 times in training.

References: Press & Wolf (arXiv:1608.05859), Inan et al. (arXiv:1611.01462).

## Initialization caveat

Tying silently breaks the output layer's initialization. `nn.Embedding` initializes to `N(0, 1)` while `nn.Linear` uses `U(-1/sqrt(in), 1/sqrt(in))`, so after the assignment the embedding's much larger weights become the output projection. Measured initial loss was **24.45 instead of `ln(10001) = 9.21`**, with logit std 6.22 and absmax 29.5 — the model starts wildly overconfident about random predictions.

Fixed with an explicit small init on the shared matrix:

```python
nn.init.uniform_(self.embeddings.weight, -0.1, 0.1)
nn.init.zeros_(self.hidden_to_output.bias)
```

Initial loss after the fix: 9.208. This is why every tied language model specifies its own initialization rather than relying on module defaults. Zaremba uses `U(-0.05, 0.05)`, which lands in the same place.

## Results

| Epoch | Train Loss | Valid Loss | Perplexity | Note |
|---:|---:|---:|---:|---|
| 1 | 5.9262 | 5.3106 | 202.48 | |
| 2 | 5.1862 | 4.9768 | 145.02 | |
| 3 | 4.8549 | 4.8040 | 121.99 | |
| 4 | 4.6258 | 4.6951 | 109.42 | already beats Run 3's best |
| 5 | 4.4431 | 4.6361 | 103.14 | |
| 6 | 4.2941 | 4.6009 | 99.57 | under 100 |
| 7 | 4.1621 | 4.5805 | 97.57 | **best validation** |
| 8 | 4.0502 | 4.5826 | 97.77 | plateau |
| 9 | 3.9465 | 4.5905 | 98.54 | plateau |
| 10 | 3.8561 | 4.5825 | 97.76 | plateau |
| 11 | 3.7719 | 4.5873 | 98.23 | plateau |
| 12 | 3.6957 | 4.5999 | 99.47 | genuine divergence begins |
| 13 | 3.6286 | 4.6091 | 100.39 | |
| 14 | 3.5665 | 4.6304 | 102.56 | run interrupted |

**Best: epoch 7, validation 4.5805, perplexity 97.57.**

Checkpoint preserved as `lstm_model_ppl9757_tied_e1024.pth`.

## Head-to-head

| | Run 3 untied | Run 4 tied |
|---|---:|---:|
| Parameters | 28,884,753 | 18,643,729 |
| Best perplexity | 117.69 | **97.57** |
| Best epoch | 10 | 7 |
| Train loss at best | 4.2495 | 4.1621 |
| Gap at best | 0.5186 | 0.4184 |

## Interpretation

Tying improved perplexity by 17% while removing 36% of the parameters.

Three observations:

**Epoch 1 was worse** (202.48 vs 193.09). The shared matrix serves both roles from a fresh small init, so it starts slower. It crossed over at epoch 2 and the lead widened every epoch after: -6.75, -13.74, -19.37.

**Train loss is lower despite 10.2M fewer parameters** — 4.6258 vs 4.7597 at epoch 4. This is the doubled gradient signal: every word trains its vector from both the input and the output side, so the model fits faster per epoch. Fewer parameters would normally mean slower fitting.

**The gap narrowed at the same time** (0.4184 vs 0.5186). Fitting better *and* generalizing better simultaneously is the signature of a better-specified model rather than a differently-regularized one — regularization alone buys a smaller gap at the cost of a higher train loss.

The earlier turn (epoch 7 vs 10) is not earlier overfitting. The tied model reached train loss 4.1621 in seven epochs where the untied model needed ten to reach 4.2495; it traverses the trajectory faster and hits its optimum sooner.

Epochs 7 through 11 are a plateau, not a climb: 97.56, 97.77, 98.54, 97.76, 98.23. Those swings are at the noise level. The "Overfitting detected" message at epoch 8 was premature — four further epochs produced essentially equivalent models. Genuine divergence starts at epoch 12.

Overfitting pressure remains high. By epoch 14 the gap reached 1.064 against the untied run's 0.715, because train loss dives to 3.5665.

## Test Accuracy

```text
Testing results: 21457 correct and 60911 wrong...  26.05 %
```

Exact next-token accuracy across the runs:

| Model | Test accuracy |
|---|---:|
| Bigger RNN, epoch 40 | 172/1024 (16.80%) |
| LSTM baseline, first run | 16390/82304 (19.91%) |
| LSTM baseline, continued | 18411/82304 (22.37%) |
| LSTM tied, ppl 97.57 | **21457/82304 (26.05%)** |

## Generation

From the 97.57 checkpoint at `--temperature 1.0 --top-k 40`:

```text
context:  the overall collapse in stock prices could permanently erode the base of <unk> support the otc market was struggling to
generated: offset the stock exchange by the end of n <eos> the dow jones industrial average rose n points to n

context:  about $ n million purchase price and cancellation of a software license provided by the morris units to information international
generated: wire services <eos> for the n months ended aug. n its quarterly profit rose n n to n million yen

context:  an economist at the national association of manufacturers <eos> but sung won <unk> chief economist at <unk> corp. in minneapolis
generated: said in recent years the u.s. department also has engaged in <unk> for a number of banks from other companies

context:  put back to the company in n was priced at n basis points above the treasury 's 10-year note <eos>
generated: the amex stock market 's index of trading was the high n 's index <eos> just for the s&p n
```

The output now has real syntactic structure: subject-verb agreement, plausible noun phrases, and correctly formed financial idiom. No repetition loops appear at all.

`<unk>` frequency in generated text, measured across 400 generated tokens:

| Model | `<unk>` rate |
|---|---:|
| ppl 158, greedy decoding | dominated by `the <unk> of the <unk>` loops |
| ppl 117.69, temperature 1.1 | ~18% |
| ppl 97.57, temperature 1.0 | **13.2%** |
| PTB corpus itself | 5.07% |

Four of twenty sampled sequences contained no `<unk>` at all. The gap to the corpus rate of 5.07% is the model still hedging toward frequent tokens under uncertainty, and should keep closing as perplexity improves.

## Next Experiments

- **Dropout 0.5 to 0.65.** One line, and the most direct lever on the remaining overfitting. Zaremba's value for the large model.
- **Depth: 2 layers at 650 units, tied.** 13,275,851 parameters and 71% of the current compute — deeper, smaller, and faster at once. This is the shape of Zaremba's medium model, which reaches ~82 perplexity. Requires making the cell a reusable submodule and threading two `(h, c)` pairs through the training loop.
- **Batched output projection.** Stack the hidden states to `[B, T, H]` and do one matmul plus one cross entropy call outside the timestep loop. Measured at 30-35% faster, no effect on results.
- **`ReduceLROnPlateau`.** Worth single-digit perplexity, not tens. Lower priority than regularization.
- Consider stopping runs at 20 epochs; both runs found their optimum well before 40.
