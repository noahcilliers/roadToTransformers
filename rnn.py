import torch
import numpy as np
import sys
import random
#datasets
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms import v2
from torchtext.data import Field
from torchtext.datasets import PennTreebank
#dataloaders: make it easy to iterate over the data
from torch.utils.data import DataLoader, TensorDataset

from torch import nn
import torch.nn.functional as F

"""

In this file I will:

1. Demonstrate Elman's RNN architecture for a neural network

2. Show the RNN's capability of producing better results than the ngram model in the 
PennTreebank dataset

3. Show the capability of the RNN architecture in generating longer responses

4. Test the RNN's capability when giving longer context sequences

"""

################################################################################
# 1. First we will create the proper datasets for training and testing the model
################################################################################

# TEXT is a field which has these specifications
# We will use this to intialize our data properly
# lowercase and tokenize to spit strings
TEXT = Field(lower=True, tokenize=str.split)

# Data intialization
train_data, valid_data, test_data = PennTreebank.splits(
    text_field=TEXT,
    root=".data"
)

# vocabullary initialization
# voacb is the index to word ratio which allows us to compare our vec output to actual words
TEXT.build_vocab(train_data)


################################################################################
# 2. Now we need to set the framework for our RNN
################################################################################

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.flatten = nn.Flatten()
        # W_x
        self.input_to_hidden = nn.Linear(embedding_dim, hidden_dim)
        #W_h
        self.context_to_hidden = nn.Linear(hidden_dim, hidden_dim)
        #f
        self.hidden_to_output = nn.Linear(hidden_dim, vocab_size)

        self.hidden_dim = hidden_dim
            

    def forward(self, x, h):
        embeds = self.embeddings(x)
        # h_t = tanh(W_x*x_t + W_h * h_(t-1))
        h = torch.tanh(self.input_to_hidden(embeds) + self.context_to_hidden(h))
        logits = self.hidden_to_output(h)
        return logits, h


################################################################################
# 3. The next step is to write our training functions
################################################################################
loss_fn = nn.CrossEntropyLoss() 

def batchify(dataset, batch_size, device):
    """The single Example -> [batch_size, stream_len] of contiguous streams."""
    tokens = dataset[0].text                    # the one row: ~929k tokens
    ids = torch.tensor([TEXT.vocab.stoi[t] for t in tokens], dtype=torch.long)

    stream_len = len(ids) // batch_size         # drop the ragged tail
    ids = ids[: stream_len * batch_size]
    return ids.view(batch_size, stream_len).to(device)


def _train_(model, optimizer, train_data, device, epochs, batch_size=32, seq_len=35):
    data = batchify(train_data, batch_size, device)      # [B, L]
    stream_len = data.size(1)
    model.train()

    for e in range(epochs):
        epoch_loss = 0.0
        token_count = 0
        h = torch.zeros(batch_size, model.hidden_dim, device=device)

        # one pass = the entire corpus, seq_len tokens at a time
        for start in range(0, stream_len - 1, seq_len):
            length = min(seq_len, stream_len - 1 - start)   # last chunk is short
            x_chunk = data[:, start : start + length]           # [B, length]
            y_chunk = data[:, start + 1 : start + 1 + length]   # [B, length]

            h = h.detach()          # carry the context, cut the gradient

            optimizer.zero_grad()
            chunk_loss = 0.0
            for t in range(length):
                logits, h = model(x_chunk[:, t], h)             # logits [B, vocab]
                chunk_loss = chunk_loss + loss_fn(logits, y_chunk[:, t])

            chunk_loss = chunk_loss / length
            chunk_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += chunk_loss.item() * length * batch_size
            token_count += length * batch_size

        print(f"epoch {e + 1}: {epoch_loss / token_count}")






def train(embedding_dim, hidden_dim, vocab_size, cont, epochs):
    # OUTER FUNCTION
    # assemble proper parts and call the training function
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    
    model = RNN(vocab_size, embedding_dim, hidden_dim).to(device)
 
    if cont:
        model.load_state_dict(torch.load("rnn_model_bigger_epoch_20.pth", map_location=device))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    

    _train_(model, optimizer, train_data, device, epochs)
    torch.save(model.state_dict(), "rnn_model_bigger_epoch_40.pth")

################################################################################
# 4. Now we must write the testing functions
# This will consist of next word prediction given a certain amount of context
# Then we will move onto an generation 
################################################################################

def _next_word_pred_(model, test_data, device, batch_size, seq_len=35, max_examples=1000, print_examples=10):
    data = batchify(test_data, batch_size, device)
    model.eval()
    h = torch.zeros(batch_size, model.hidden_dim, device=device)
    stream_len = data.size(1)
    correct = 0
    total = 0
    shown = 0

    # This loops through the stream in chunks that match training.
    with torch.no_grad():
        for start in range(0, stream_len - 1, seq_len):
            length = min(seq_len, stream_len - 1 - start)

            for t in range(length):
                x = data[:, start + t]
                y = data[:, start + t + 1]
                logits, h = model(x, h)
                preds = logits.argmax(dim=1)

                correct += (preds == y).sum().item()
                total += y.numel()

                if shown < print_examples:
                    for input_id, real_id, pred_id in zip(x.cpu(), y.cpu(), preds.cpu()):
                        if shown >= print_examples:
                            break

                        input_word = TEXT.vocab.itos[input_id.item()]
                        real_word = TEXT.vocab.itos[real_id.item()]
                        pred_word = TEXT.vocab.itos[pred_id.item()]
                        print("input:    ", input_word)
                        print("real:     ", real_word)
                        print("predicted:", pred_word)
                        print()
                        shown += 1

                if total >= max_examples:
                    break

            if total >= max_examples:
                break

    percent_correct = correct / total * 100
    print(f"Correct: {correct}/{total} ({percent_correct:.2f}%)")


def test(embedding_dim, hidden_dim, vocab_size):
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    
    model = RNN(vocab_size, embedding_dim, hidden_dim).to(device)
 
    model.load_state_dict(torch.load("rnn_model_bigger_epoch_40.pth", map_location=device))

    _next_word_pred_(model, test_data, device, 32)


def _generate_(model, test_data, device, batch_size):
    """
    So I think I want to load the context in with a few words before I start
    the generation. This is going to help the model start on track.
    """
    data = batchify(test_data, batch_size, device)
    model.eval()
    stream_len = data.size(1)
    # generate three different sequences every time
    with torch.no_grad():
        for i in range(10):
            print(f"Sequence Starting...")
            h = torch.zeros(1, model.hidden_dim, device=device)
            # first step is to get the random number for the row we are using
            r = random.randrange(batch_size)
            start = random.randrange(stream_len - 13)
            # then we have to load in the context for the first 2 words and pass the 3rd word
            for j in range(3):
                x = data[r, j + start]
                print(f"{TEXT.vocab.itos[x.item()]}" , end=" ")
                logits, h = model(x.view(1), h)
    
            pred = logits.argmax(dim=1)
            pred_word = TEXT.vocab.itos[pred.item()]
            for word in range(10):
                print(f"{pred_word}", end=" ")
                logits, h = model(pred, h)

                unk_id = TEXT.vocab.stoi["<unk>"]
                logits[0, unk_id] = -float("inf")


                pred = logits.argmax(dim=1)
                pred_word = TEXT.vocab.itos[pred.item()]
            print(f"\nSequence Over \n Generating Next Sequence")




def generate(embedding_dim, hidden_dim, vocab_size):
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    
    model = RNN(vocab_size, embedding_dim, hidden_dim).to(device)
 
    model.load_state_dict(torch.load("rnn_model_bigger_epoch_40.pth", map_location=device))

    _generate_(model, test_data, device, 1000)
    


def main():
     #arguments
    flag = sys.argv[1]

    #env vars
    embedding_dim = 56
    hidden_dim = 512
    vocab_size = len(TEXT.vocab)
    epochs = 20
    
    #execution
    if flag == "train":
        print("Training starting now...")
        train(embedding_dim, hidden_dim, vocab_size, False, epochs)
    elif flag == "continue":
        print("Continue training starting now...")
        train(embedding_dim, hidden_dim, vocab_size, True, epochs)
    elif flag == "test":
        print("Testing starting now...")
        test(embedding_dim, hidden_dim, vocab_size)
    elif flag == "generate":
        print("Generating starting now...")
        generate(embedding_dim, hidden_dim, vocab_size)
    else:
        print("Improper request. Requires flag")






if __name__ == "__main__":
    main()
