# Learning Log

This is the intended progression:

Feed-forward neural nets -> RNNs -> LSTM -> seq2seq -> attention -> Transformer

## 2026-07-29

- Started with PyTorch tensor basics in `learning/tensors.py`.
- Practiced creating tensors from Python lists and NumPy arrays.
- Looked at tensor shapes, device placement, moving tensors to the accelerator/CPU, indexing, matrix multiplication, transposes, and in-place operations.
- Noted that CPU tensors and NumPy arrays can share memory, while moving tensors to an accelerator creates a separate copy.

## 2026-07-31

- Worked with FashionMNIST datasets and dataloaders in `learning/datasets.py`.
- Loaded train/test datasets with torchvision and converted images into normalized float tensors.
- Used `DataLoader` batching and inspected batch shapes for images and labels.
- Explored transforms in `learning/transforms.py`, including image transforms and one-hot label transforms.

## 2026-08-01

- Built a first feed-forward neural network in `learning/nn.py`.
- Used `nn.Module`, `nn.Flatten`, `nn.Sequential`, `nn.Linear`, and `nn.ReLU`.
- Ran a forward pass on a fake FashionMNIST-shaped input and converted logits into a predicted class with softmax and argmax.

## 2026-08-02

- Studied autograd in `learning/autodiff.py`.
- Practiced `requires_grad`, `grad_fn`, scalar loss calculation, `loss.backward()`, and reading gradients from weights and biases.
- Built a more complete FashionMNIST training script in `manualFashionMNIST.py`.
- Added a CNN-style model with convolution, ReLU, max pooling, linear layers, cross entropy loss, Adam, model saving/loading, and train/test/continue modes.
- Saved a trained FashionMNIST model checkpoint as `fashion_mnist_model.pth`.

## 2026-08-03

- Started planning an n-gram language model in `ngram.py`.
- Settled on the core idea: use a fixed number of previous tokens to predict the next token.
- Clarified that tokens become integer IDs first, then `nn.Embedding` learns vector representations during training.
- Chose PennTreebank as the simplest dataset for language modeling.

## 2026-08-04

- Worked through PyTorch and torchtext setup issues.
- Found that the installed `torchtext` version used the older `Field` and `PennTreebank.splits(...)` API.
- Built the first `NGramModel` as a feed-forward neural network:
  - token IDs -> embedding lookup
  - flatten context embeddings
  - linear layer + ReLU
  - final projection to vocab-sized logits
- Added n-gram example creation with `xs` as context windows and `ys` as next-token labels.
- Added batching with `TensorDataset` and `DataLoader`.
- Trained the first working version and saw the loss drop to about `6.15`.

## 2026-08-05

- Added model loading with `load_state_dict`.
- Built a test path that:
  - loads the saved model
  - makes test n-grams
  - runs predictions with `argmax`
  - prints context, real next word, and predicted next word
- Updated testing to evaluate 1000 examples while printing only 10 samples.
- Discussed `argmax` vs `softmax`, and why `CrossEntropyLoss` should receive raw logits.
- Added explanatory comments for the training and testing blocks.
- Wrote down the feed-forward equation flow:
  - `E = Embedding(x)`
  - `z = flatten(E)`
  - `h = ReLU(zW1 + b1)`
  - `logits = hW2 + b2`

## 2026-08-06

- Tuned model size and training settings.
- Increased context size, embedding size, and hidden dimension.
- Observed that larger settings initially got worse, then improved after using a larger hidden dimension.
- Discussed overfitting, optimization instability, learning rate, batch size, Adam, AdamW, and weight decay.
- Reached about `21.20%` exact next-token accuracy over 1000 test examples.
- Added a `.gitignore` for generated artifacts like `.venv/`, `.data/`, `__pycache__/`, and `*.pth`.
- Added a second hidden layer to the feed-forward n-gram model.
- Started bridging toward RNNs by adding sequential generation:
  - pick a starting context
  - predict the next token
  - append the prediction
  - drop the oldest token
  - repeat until `<eos>` or a max token limit
- Discussed why greedy `argmax` generation can collapse into repeated `<unk>` tokens.
- Discussed the next conceptual step: RNNs store context in a hidden state instead of a fixed token window.

## 2026-08-07

- Read Elman's paper `Finding Structure in Time`.
- Focused on the difference between a feed-forward network and an Elman recurrent network.
- Studied how context units copy the previous hidden activations forward so the network can use information from earlier time steps.
- Connected the paper's temporal structure examples to the idea that sequences contain regularities that are not visible from a single input alone.

## 2026-08-08

- Finished reading Elman's `Finding Structure in Time`.
- Reviewed why temporal XOR requires memory of previous inputs rather than just the current input.
- Watched 3Blue1Brown's video on compression as intelligence.
- Learned the basic idea of cross entropy as measuring how surprised a model is by the correct answer.
- Connected cross entropy to language modeling: the model outputs scores over possible next tokens, and training rewards higher probability on the true next token.
- Clarified that PyTorch's `CrossEntropyLoss` should receive raw logits because it handles the softmax/log-probability calculation internally.

## 2026-08-09

- Reviewed Elman's RNN architecture and the idea that sequence structure lives in the hidden state, not just in a fixed token window.
- Started building a simple Elman-style RNN in `rnn.py` for PennTreebank next-token prediction.
- Corrected the model design from two separate vocab predictions to the recurrent form:
  - current token embedding -> hidden-sized input projection
  - previous hidden state -> hidden-sized context projection
  - combine both projections, apply `tanh`, then project the new hidden state to vocab-sized logits
- Built the first RNN training loop that carries hidden state forward through a sequence and predicts the next token at each step.
- Diagnosed the major memory spike as a sequence-shape problem: PennTreebank was effectively one very long row, so full backpropagation through the whole row created an enormous computation graph.
- Reworked the training data flow with `batchify(...)` so the single long corpus becomes contiguous batch streams.
- Added truncated backpropagation through time with `seq_len` chunks:
  - carry `h` forward as context
  - use `h.detach()` between chunks to keep the hidden values while cutting old gradient history
  - backpropagate once per chunk
- Added gradient clipping with `clip_grad_norm_` to reduce unstable RNN updates.
- Added per-token epoch loss reporting and checkpoint saving/loading with `rnn_model.pth`.

## 2026-08-12

- Continued the bigger Elman RNN experiment with `embedding_dim = 56` and `hidden_dim = 512`.
- Trained the bigger model from epoch 20 to epoch 40 with `python3 rnn.py continue`.
- Recorded the continued-training loss curve:
  - epoch 20 continuation started around `3.3804`
  - final continued-training loss reached `2.955093368711435`
- Tested the continued bigger model and saw held-out exact next-token accuracy drop to `172/1024 (16.80%)`.
- Compared this against the bigger model's epoch-20 test result of `183/1024 (17.87%)` and the smaller RNN baseline of `187/1024 (18.26%)`.
- Noted the important experiment result: training loss kept improving while held-out accuracy got worse, suggesting overfitting or weaker generalization.
- Generated qualitative samples from the epoch-40 bigger model.
- Observed that some generated text still had plausible business/news phrases, but also more odd phrase collisions such as `french mushrooms hotel`, `proxy navy`, and `journal flush`.
- Saved the bigger-model experiment trail in `outputs/rnn_bigger_model_epoch_20.md`.

## 2026-08-13

- Studied LSTM architecture after reading the friendlier Google/Olah explanation and comparing it with the original research-paper notation.
- Interpreted the original LSTM memory-cell diagram:
  - the self-loop with weight `1.0` represents carrying cell state forward across time
  - the input gate controls whether new candidate memory is written
  - the output gate controls whether memory is exposed to the rest of the network
  - the original diagram does not show the modern forget gate
- Clarified that the circular arrow is symbolic for recurrence across time, not a literal instantaneous loop inside one timestep.
- Discussed why gate matrices learn their roles without direct supervision:
  - architecture gives each gate a specific kind of control
  - the loss punishes bad predictions
  - gradients shape each gate based on whether keeping, writing, forgetting, or exposing information helped prediction
- Clarified the difference between concatenating input embeddings with the previous hidden state and adding vectors:
  - concatenation places vectors side by side
  - a later linear layer learns how to mix the combined vector
  - this is mathematically similar to separate input and hidden projections whose results are added
- Started a manual LSTM implementation in `lstm.py`.
- Reviewed the first LSTM draft and found the main issue: gates should be computed in parallel from the current embedding plus previous hidden state, then combined with the LSTM equations, rather than built as sequential modules.
- Identified concrete fixes for `lstm.py`:
  - add the missing `=` in the `add_to_cell_gate` definition
  - avoid defining `i_t` and `C_t` inside `nn.Sequential`
  - use `nn.Linear(...)` layers in `__init__` and apply `sigmoid`/`tanh` in `forward`
  - pass both hidden state `h` and cell state `c` through `forward`
  - use `hidden_dim` for both `h` and `c` at first to keep the implementation simple

## 2026-08-14

- Clarified the difference between `sigmoid` and `tanh` in LSTMs:
  - `sigmoid` squashes values to `0..1`, so it works naturally as a gate or knob
  - `tanh` squashes values to `-1..1`, so it works naturally as signed candidate content
  - LSTM gates use sigmoid for forget/input/output decisions, while the candidate memory uses tanh
- Reviewed the updated manual LSTM cell in `lstm.py`.
- Confirmed that the core LSTM equations are now in the right shape:
  - concatenate current embedding and previous hidden state
  - compute forget, input, candidate, and output gates in parallel
  - update cell state with `c = f * c + i * candidate`
  - update hidden state with `h = o * tanh(c)`
- Found the remaining LSTM implementation work is mostly in training/validation wiring rather than the cell math.
- Reviewed the validation implementation and identified key fixes:
  - define `stream_len = data.size(1)` inside `__valid__`
  - return average validation loss with `epoch_loss / token_count`
  - detach both `h` and `c` during truncated backpropagation/validation
  - call `__train__` with the required train dataset, validation dataset, optimizer, epoch count, sequence length, and batch size
  - track best validation loss with `float("inf")` or `None` rather than a sentinel like `-1000`
- Discussed why exact next-token accuracy is a limited language-model metric.
- Introduced validation loss and perplexity as better language-model evaluation tools.
- Clarified that PyTorch `CrossEntropyLoss` computes `-ln(p_correct)`, not `-log2(p_correct)`.
- Clarified that `p_correct` is the probability assigned to the true next token, not necessarily the probability of the token the model guessed with `argmax`.
- Defined perplexity as `exp(validation_loss)` when using PyTorch's natural-log cross entropy.
- Interpreted perplexity as a rough measure of how many plausible next-token choices the model is still uncertain among.
- Added and reviewed LSTM test-mode wiring in `lstm.py`:
  - batchify the test dataset instead of accidentally using training data
  - initialize both hidden state `h` and cell state `c`
  - wrap evaluation in `torch.no_grad()`
  - compare `argmax` next-token predictions against the shifted target tokens
- Switched `--cont` to a normal command-line flag with `action="store_true"`.
- Added checkpoint saving when validation loss improves, so interrupted continuation runs can still keep the best validation model reached so far.
- Ran the first manual LSTM training pass:
  - epoch 1 train/valid loss: `6.4194 / 5.9750`
  - epoch 5 train/valid loss: `5.0613 / 5.2389`
  - perplexity improved from `393.45` to `188.47`
  - test accuracy reached `16390/82304 (19.91%)`
- Continued LSTM training from the checkpoint with `python3 lstm.py --fn train --cont`.
- Observed validation improvement through continued epoch 6:
  - train loss `4.3853`
  - validation loss `5.0633`
  - perplexity `158.11`
- Observed validation plateau/early overfitting warnings after continued epoch 6 while training loss kept decreasing.
- Tested the best continued checkpoint and reached `18411/82304 (22.37%)` exact next-token accuracy.
- Saved the experiment record in `LSTM_RESULTS.md`.

## 2026-08-16

### Diagnosing the `<unk>`-heavy generation output

- Investigated why LSTM generation produced long runs of `<unk>` such as `the <unk> of the <unk> <unk> <unk>`.
- Counted the raw corpus and found this is not a vocabulary bug:
  - `ptb.train.txt` contains 887,521 tokens and 9,999 distinct types
  - `<unk>` is the second most frequent token: 45,020 occurrences, 5.07% of the corpus
  - only `the` is more common at 5.72%
- Learned that the Mikolov-preprocessed PennTreebank ships *already* unked — rare words were replaced with the literal string `<unk>` before distribution, so the original words cannot be recovered by building a larger vocab.
- Concluded the model predicting `<unk>` frequently is correct behavior for this corpus.
- Identified the actual bug as greedy `argmax` decoding, which always takes the single most probable token and therefore falls into high-frequency loops.

### Sampling, temperature, and top-k

- Added a `sample()` function to `lstm.py` and replaced both `argmax` calls in `__generate__`.
- Added `--temperature` and `--top-k` command line flags (defaults `0.8` and `40`).
- Kept `argmax` in `__test__`, since accuracy should measure the model's single best guess.
- Learned that temperature is not part of the softmax function; it is a rescaling of the logits applied before it, and comes from the Boltzmann distribution in statistical physics.
- Learned that softmax is shift-invariant but not scale-invariant, so only the *gaps* between logits matter:
  - adding 100 to every logit leaves the probabilities bit-for-bit identical
  - dividing by `T` uniformly stretches or compresses every gap
- Derived why the division goes inside the exponent: `exp(z/T) = exp(z)^(1/T)`, so temperature scaling in logit space equals raising probabilities to the power `1/T` and renormalizing.
- Noted the limits: `T -> 0` is argmax (greedy), `T = 1` is the model's honest distribution, `T -> inf` is uniform.
- Understood that scaling before the softmax rather than after is chosen for numerical stability (log-space avoids underflow of ~1e-8 probabilities) and because `-inf` top-k masking only means "impossible" in logit space.
- Clarified that `T = 0.8` *sharpens* the distribution, so temperature was not what fixed the loop — replacing `argmax` with `torch.multinomial` was.
- Confirmed generation improved substantially at `--temperature 1.1 --top-k 40`.

### How cross entropy consumes logits

- Confirmed `nn.CrossEntropyLoss` applies the softmax internally; it is `log_softmax` + `nll_loss` fused, so raw logits must be passed in.
- Learned the fusion exists for numerical stability via the log-sum-exp trick.
- Verified that the gradient of cross entropy with respect to the logits is exactly `p - y`, the predicted distribution minus the one-hot target.
- Understood that a single token's loss therefore touches all 10,001 output logits: the correct one is pushed up by `1 - p`, and every other is pushed down in proportion to the probability it wrongly claimed.
- Realized this is why top-k masking must never be used during training — `-inf` logits receive zero gradient and `-log(0)` produces `inf` loss.
- Noted that probabilities are a generation-time concept in this code; training never leaves log space.

### Dropout

- Read that dropout randomly zeroes a fraction `p` of activations during training and scales survivors by `1/(1-p)`.
- Learned the two standard explanations: preventing co-adaptation between units, and approximating an ensemble of exponentially many thinned subnetworks.
- Understood why the `1/(1-p)` scaling matters: it matches `E[dropout(x)] = x`, so downstream layers see the same input magnitude at train and eval time.
- Worked out the consequences of getting it wrong in this model specifically:
  - the sigmoid/tanh gates would saturate, jamming the forget and input gates open
  - logits would be uniformly 2x too large, which is *identical* to running permanently at temperature 0.5
- Noted that the scaling deliberately fixes only the first moment — dropout still injects variance, and that variance is the entire regularizing mechanism.
- Learned this is "inverted dropout"; the original formulation scaled weights by `(1-p)` at test time instead, and the inverted version won because inference needs no dropout-aware code.
- Confirmed `nn.Dropout` reads its own `self.training` flag on every forward pass and becomes an identity function in eval mode.
- Learned `model.train()` / `model.eval()` set that boolean recursively on all submodules, and that `eval()` is literally `train(False)`.
- Noted `torch.no_grad()` and `model.eval()` are unrelated: one disables the autograd graph, the other changes layer behavior.
- Noted the trap that `F.dropout` defaults to `training=True`, so the functional form silently stays on during eval unless `training=self.training` is passed.

### Where dropout goes in a recurrent network

- Read Zaremba, Sutskever & Vinyals (2014), "Recurrent Neural Network Regularization" (arXiv:1409.2329).
- Learned the paper's notation: subscripts are timesteps, superscripts are layers.
  - `h^(l-1)_t` is the **input** from the layer below (superscript changed = depth)
  - `h^l_(t-1)` is the **previous hidden state** (subscript changed = time)
- Realized the paper's two separate affine transforms `T_n,n` are equivalent to this code's single `Linear` applied to `torch.cat([embeds, h])`, since `[W_x | W_h] · [x ; h] = W_x·x + W_h·h`.
- Learned the rule: dropout goes on superscript transitions (across layers), never on subscript transitions (across time).
- Simulated why fresh per-timestep masks destroy memory: survival probability is `(1-p)^T`, so at `p=0.5` and `T=64` it is `5.4e-20` — zero of 10,000 units retained information.
- Understood this attacks precisely the near-identity `c = c*f + ...` path that the LSTM exists to provide.
- Contrasted with additive noise in a feedback loop, which is a random walk: std grows as `sqrt(T)`, roughly 8x over 64 timesteps.
- Learned that a *fixed* mask reused across the whole sequence preserves 50% of units perfectly, because perfectly correlated noise does not accumulate — this is Gal & Ghahramani (2015), arXiv:1512.05287, which reports 73.4 test perplexity on PTB.
- Applied dropout to `embeds` and to `h` before `hidden_to_output`, deliberately leaving the returned `h` undropped.
- Verified with a smoke test that the returned `h` has 0.00% exact zeros in train mode (dropout leaking onto the recurrent path would show ~50%).

### Parameter analysis and scaling

- Broke down the old `e64 / h512` model and found the capacity was badly allocated:
  - `hidden_to_output`: 5,130,513 params (73.8%)
  - four gates: 1,181,696 params (17.0%)
  - embeddings: 640,064 params (9.2%)
- Benchmarked training speed on MPS and found compute is not the constraint:
  - `e64 / h512`: ~1.1 min/epoch
  - `e1024 / h1024`: ~3.6 min/epoch
  - `e1500 / h1500`: ~4.5 min/epoch
- Measured that fusing the four gate matmuls into one gives no speedup at these dimensions, because the 10,001-way output projection dominates each step.
- Measured that moving the output projection and cross entropy *outside* the timestep loop (stack `h` to `[B, T, H]`, one matmul, one CE call) saves 30–35%. Not yet implemented.
- Concluded the real constraint is data: 887,521 training tokens against 28.9M parameters is ~33 parameters per token, which only works because of dropout.
- Learned about weight tying but deliberately deferred it to keep this run's changes smaller:
  - `embeddings.weight` and `hidden_to_output.weight` are both `[10001, 1024]` and encode the same word-identity information in opposite directions
  - tying them would cut 10,241,024 parameters (28.88M -> 18.64M) at no compute cost
  - it also doubles the gradient signal per word, which matters because 8,015 of 9,999 vocabulary words appear fewer than 50 times in training
  - references: Press & Wolf (arXiv:1608.05859), Inan et al. (arXiv:1611.01462)

### New training configuration

- Rebuilt the model as `e1024 / h1024` with `nn.Dropout(0.5)` and learning rate `0.001`, for 40 epochs.
- Moved the checkpoint path into a single `CHECKPOINT` constant after finding the save path had been renamed to `lstm_model_28M.pth` while all three load paths still pointed at `lstm_model.pth`.
- Backed up the 158-perplexity checkpoint as `lstm_model_ppl158_e64_h512.pth` before changing dimensions.
- Smoke tested the new model: 28,884,753 parameters, one full train step, initial loss `9.23` against the expected `ln(10001) = 9.21` for an untrained model.
- Reference points for what to aim at: Zaremba medium (2x650, dropout 0.5) reaches ~82 perplexity, large (2x1500, dropout 0.65) reaches ~78.
