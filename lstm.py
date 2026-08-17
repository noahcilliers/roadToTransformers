import torch
import numpy as np
import sys
import random
import argparse
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
Here is my manual attempt at writing an lstm 

Ideas:

Inputs every time step: Cell state, hidden state, input
Outputs: Hidden state, cell state

Gates:
1. Forget gate: What does the cell state need to forget

2. Add memory states: What do we need to add to our cell state

3. Output gate: What do we need to remember from our cell state


Input -> vocab -> embeddings -> Gate 1 -> Gate 2 -> Gate 3 -> logits -> Cross Entropy Loss/-log_2(p)-> Backprop



"""
############################################################
### Here is the model                                    ###
############################################################

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.flatten = nn.Flatten()
        self.hidden_dim = hidden_dim
        self.epochs = 0

        combined_dim = hidden_dim+embedding_dim
        ### GATES
        self.forget_gate = nn.Linear(combined_dim, hidden_dim)
        self.input_gate = nn.Linear(combined_dim, hidden_dim)
        self.candidate_gate = nn.Linear(combined_dim, hidden_dim)
        self.output_gate = nn.Linear(combined_dim, hidden_dim)

        ###Get Logits
        self.hidden_to_output = nn.Linear(hidden_dim, vocab_size)
        assert embedding_dim == hidden_dim, "weight tying needs embedding_dim == hidden_dim"
        self.hidden_to_output.weight = self.embeddings.weight

        # nn.Embedding inits to N(0,1), nn.Linear to U(-1/sqrt(in), 1/sqrt(in)).
        # after tying, the embedding's much larger weights become the output
        # projection, which starts the model wildly overconfident (loss ~24
        # instead of ln(vocab) = 9.21). tied models need an explicit small init.
        nn.init.uniform_(self.embeddings.weight, -0.1, 0.1)
        nn.init.zeros_(self.hidden_to_output.bias)

        # applied to the non-recurrent connections only: embedding -> cell,
        # and cell -> logits. never to the h we hand to the next timestep.
        self.dropout = nn.Dropout(0.5)

        
    def forward(self, x, h, c):
        """
        So the cell state doesnt necassarily pass through a gate:
        it just is edited by these matmuls and then 
        used to create the hidden dim. So i guess it goes through some matrices
        but the matrices it goes through are different every time
        """


        embeds = self.dropout(self.embeddings(x))
        combined = torch.cat([embeds, h], dim=1)
        # Gate functions below
        # These operations get all the values from our gates
        f = torch.sigmoid(self.forget_gate(combined))
        i = torch.sigmoid(self.input_gate(combined))
        candidate = torch.tanh(self.candidate_gate(combined))
        o = torch.sigmoid(self.output_gate(combined))
        # These are the operations we provide for the cell state and the hidden 
        # forget(c * f), add info(i * x), new hidden with memory (o * tanh(c))
        c = c * f + i * candidate
        h = o * torch.tanh(c)

        # logits size of vocab of c
        logits = self.hidden_to_output(self.dropout(h))

        return logits, h, c



############################################################
### Now for the dataloaders                              ###
############################################################

# field for making data
TEXT = Field(lower=True, tokenize=str.split)
# make the data vars
train_data, valid_data, test_data = PennTreebank.splits(
    text_field=TEXT,
    root=".data"
)
# make the vocab through the field
TEXT.build_vocab(train_data)

def batchify(dataset, batch_size, device):
    #The single Example -> [batch_size, stream_len] of contiguous streams.
    tokens = dataset[0].text                    # the one row: ~929k tokens
    ids = torch.tensor([TEXT.vocab.stoi[t] for t in tokens], dtype=torch.long)

    stream_len = len(ids) // batch_size         # drop the ragged tail
    ids = ids[: stream_len * batch_size]
    return ids.view(batch_size, stream_len).to(device)


############################################################
### Training functions below                             ###
############################################################
loss_fn = nn.CrossEntropyLoss()

# one place, so train/test/generate can never drift onto different checkpoints
CHECKPOINT = "lstm_model_18.6M_tied.pth"

def __valid__(model, device, valid_dataset, seq_len, batch_size):
    data= batchify(valid_dataset, batch_size, device)
    stream_len = data.size(1)
    model.eval()
    with torch.no_grad():
        epoch_loss = 0.0
        token_count = 0
        h = torch.zeros(batch_size, model.hidden_dim, device=device)
        c = torch.zeros(batch_size, model.hidden_dim, device=device)
        # one pass = the entire corpus, seq_len tokens at a time
        for start in range(0, stream_len - 1, seq_len):
            length = min(seq_len, stream_len - 1 - start)   # last chunk is short
            x_chunk = data[:, start : start + length]           # [B, length]
            y_chunk = data[:, start + 1 : start + 1 + length]   # [B, length]

            h = h.detach()          # carry the context, cut the gradient
            c = c.detach()

            #optimizer.zero_grad()
            chunk_loss = 0.0
            for t in range(length):
                logits, h, c = model(x_chunk[:, t], h, c)             # logits [B, vocab]
                chunk_loss = chunk_loss + loss_fn(logits, y_chunk[:, t])

            chunk_loss = chunk_loss / length
           # chunk_loss.backward()
           # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
           # optimizer.step()

            epoch_loss += chunk_loss.item() * length * batch_size
            token_count += length * batch_size
    avg_loss = epoch_loss / token_count
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return avg_loss, perplexity





def __train__(model, device, train_dataset, valid_dataset, optimizer, epochs, seq_len, batch_size, cont):
    data= batchify(train_dataset, batch_size, device)
    stream_len = data.size(1)

    
    best_valid = float("inf")
    if cont:
        best_valid, ppl = __valid__(model, device, valid_dataset, seq_len, batch_size)
        print(f"resuming: valid {best_valid:.4f}, ppl {ppl:.2f}")
    

    model.train()
    for e in range(epochs):
        epoch_loss = 0.0
        token_count = 0
        h = torch.zeros(batch_size, model.hidden_dim, device=device)
        c = torch.zeros(batch_size, model.hidden_dim, device=device)
        # one pass = the entire corpus, seq_len tokens at a time
        for start in range(0, stream_len - 1, seq_len):
            length = min(seq_len, stream_len - 1 - start)   # last chunk is short
            x_chunk = data[:, start : start + length]           # [B, length]
            y_chunk = data[:, start + 1 : start + 1 + length]   # [B, length]

            h = h.detach()          # carry the context, cut the gradient
            c = c.detach()

            optimizer.zero_grad()
            chunk_loss = 0.0
            for t in range(length):
                logits, h, c = model(x_chunk[:, t], h, c)             # logits [B, vocab]
                chunk_loss = chunk_loss + loss_fn(logits, y_chunk[:, t])

            chunk_loss = chunk_loss / length
            chunk_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += chunk_loss.item() * length * batch_size
            token_count += length * batch_size
        
        valid_loss, valid_ppl = __valid__(model, device, valid_dataset, seq_len, batch_size)
        if valid_loss > best_valid:
            print(f"Overfitting detected... New loss: {valid_loss} Old loss: {best_valid}")
        else:
            best_valid = valid_loss
            torch.save(model.state_dict(), CHECKPOINT)
        model.train()

        model.epochs += 1
        print(f"epoch {e + 1}: train {epoch_loss / token_count:.4f}, valid {valid_loss:.4f}, ppl {valid_ppl:.2f}")


def train(vocab_size, embedding_dim, hidden_dim, cont, epoch, seq_len, batch_size):
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    model = LSTM(vocab_size, embedding_dim, hidden_dim).to(device)

    if cont:
        model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    __train__(model, device, train_data, valid_data, optimizer, epoch, seq_len, batch_size, cont)

############################################################
### Testing functions below                             ###
############################################################

def __test__(model, device, test_dataset, seq_len, batch_size):
    # do testing next token prediction for now
    data = batchify(test_dataset, batch_size, device)
    stream_len = data.size(1)

    model.eval()

    correct = 0
    total = 0

    h = torch.zeros(batch_size, model.hidden_dim, device=device)
    c = torch.zeros(batch_size, model.hidden_dim, device=device)

    with torch.no_grad():
        for start in range(0, stream_len - 1, seq_len):
                length = min(seq_len, stream_len - 1 - start)   # last chunk is short
                x_chunk = data[:, start : start + length]           # [B, length]
                y_chunk = data[:, start + 1 : start + 1 + length]   # [B, length]

                h = h.detach()          # carry the context, cut the gradient
                c = c.detach()

                for t in range(length):
                    logits, h, c = model(x_chunk[:, t], h, c)             # logits [B, vocab]
                # now we need to check our logits against the predicted value
                # select the single argmaxxed value
                    predictions = logits.argmax(dim=1)
                # check all of them
                    correct += (predictions == y_chunk[:, t]).sum().item()
                    total += batch_size
        print(f"Testing results: {correct} correct and {total-correct} wrong...  {(correct / total) * 100:.2f} %")
                

                

    


def test(vocab_size, embedding_dim, hidden_dim, batch_size, seq_len):
    # set up the testing model and such
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    model = LSTM(vocab_size, embedding_dim, hidden_dim).to(device)

    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))

    __test__(model, device, test_data, seq_len, batch_size)


############################################################
### Generation functions below                           ###
############################################################

def sample(logits, temperature, top_k):
    """
    Pick the next token by sampling instead of argmax.

    argmax always takes the single most likely token, which walks the model
    straight into its highest-frequency loop ("the <unk> of the <unk> ...").
    Sampling from the distribution keeps the output varied.

    temperature < 1 sharpens the distribution (more conservative),
    temperature > 1 flattens it (more random).
    top_k discards everything outside the k most likely tokens so we never
    sample from the long tail of near-zero-probability junk.
    """
    logits = logits / temperature

    if top_k:
        # the k-th best score per row, everything below it becomes impossible
        kth = logits.topk(top_k, dim=1).values[:, -1:]      # [B, 1]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    probs = F.softmax(logits, dim=1)
    return torch.multinomial(probs, num_samples=1).squeeze(1)


def __generate__(model, dataset, device, batch_size, seq_len, context, temperature, top_k):
    data = batchify(dataset, batch_size, device)
    stream_len = data.size(1)
    model.eval()

    h = torch.zeros(batch_size, model.hidden_dim, device=device)
    c = torch.zeros(batch_size, model.hidden_dim, device=device)

    context_tokens = [[] for _ in range(batch_size)]
    generated_tokens = [[] for _ in range(batch_size)]
    
    with torch.no_grad():
        start = random.randrange(stream_len - seq_len)
        x_chunk = data[:, start : start + context]

        for t in range(context):
            logits, h, c = model(x_chunk[:, t], h, c)
            for row in range(batch_size):
                context_tokens[row].append(x_chunk[row, t].item())

        predictions = sample(logits, temperature, top_k)

        for _ in range(seq_len - context):
            for row in range(batch_size):
                generated_tokens[row].append(predictions[row].item())
            logits, h, c = model(predictions, h, c)
            predictions = sample(logits, temperature, top_k)

    #printing function
    for row in range(batch_size):
        context_words = [TEXT.vocab.itos[token_id] for token_id in context_tokens[row]]
        generated_words = [TEXT.vocab.itos[token_id] for token_id in generated_tokens[row]]
        print(f"\nSequence {row + 1:02d}")
        print(f"context:  {' '.join(context_words)}")
        print(f"generated: {' '.join(generated_words)}")


        
def generate(vocab_size, embedding_dim, hidden_dim, batch_size, temperature, top_k):
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    model = LSTM(vocab_size, embedding_dim, hidden_dim).to(device)

    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))

    batch_size = 10
    seq_len = 40
    context = 20

    __generate__(model, test_data, device, batch_size, seq_len, context, temperature, top_k)








parser = argparse.ArgumentParser()

def main():
    parser.add_argument("--fn", type=str, default="empty")
    parser.add_argument("--cont", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)

    args = parser.parse_args()

    
    
    embedding_dim = 1024
    hidden_dim = 1024
    vocab_size = len(TEXT.vocab)
    epochs = 40

    seq_len = 64
    batch_size = 32


    if args.fn == "empty":
        print("You must specify argument --fn. train, test, or generate")

    elif args.fn == "train":
        print("LSTM training starting now")
        train(vocab_size, embedding_dim, hidden_dim, args.cont, epochs, seq_len, batch_size)

    elif args.fn == "test":
        print("LSTM testing starting now")
        test(vocab_size, embedding_dim, hidden_dim, batch_size, seq_len)
    elif args.fn == "generate":
        print("LSTM generating starting now")
        generate(vocab_size, embedding_dim, hidden_dim, batch_size, args.temperature, args.top_k)
        



if __name__ == "__main__":
    main()
