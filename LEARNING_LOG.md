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
