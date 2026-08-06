# torch + numpy
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
n-gram model: given n terms we must predict the next word

word/token embeddings

tokenizer just splits the words and assigns the proper tokens to the numbers

Numbers -> vector embeddings

vector embeddings -> input into the NLP

input shape = batch, context size, embedding dim 

NLP -> linear stack with ReLU (delinearization)

logits output which we Cross entropy Loss with the next token

one word output which we will compare to our label in the pretraining

"""

#
# First we will define our dataset and dataloader
# no transform, train, or download here
# This is the most basic dataset for language modeling
#

TEXT = Field(lower=True, tokenize=str.split)

train_data, valid_data, test_data = PennTreebank.splits(
    text_field=TEXT,
    root=".data"
)

TEXT.build_vocab(train_data)

"""
So now we have our vocabullary created so each word corresponds to a number in .vocab

But these numbers must correlate to a vecotor embedding AND we dont want to preassign these

They shall be assigned randomly and adjusted during our training

we will see this in our model below

"""

#
# MODEL
# input shape: [batch_size, context_size]
# after embedding: [batch_size, context_size, embedding_dim]
# after flatten: [batch_size, context_size * embedding_dim]

class NGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_amount, hidden_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            #size of input = 28 * 28
            nn.Linear(context_amount*embedding_dim, hidden_dim),
            #relu mean rectified linear unit
            # it just takes all the negative numbers and makes them 0
            nn.ReLU(),

            ##second hidden layer
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            #output hidden_dim->vocab size: this is because we assign each word a number for output
            nn.Linear(hidden_dim, vocab_size)
        )

    def forward(self, x):
        embeds = self.embeddings(x)
        x = self.flatten(embeds)
        logits = self.linear_relu_stack(x)
        return logits


"""
Training functions below:

Training builds fixed-size n-gram examples from the PennTreebank text.
Each input is a context of word IDs, and each label is the next word ID.

During training, the model predicts vocab-sized logits for each context.
CrossEntropyLoss compares those logits to the true next-word labels, then
backprop updates both the neural network weights and the embedding table.

Equation flow:
E = Embedding(x)
z = flatten(E)
h = ReLU(zW1 + b1)
logits = hW2 + b2

Conceptually, this is:
embedded input -> input projection -> hidden activation -> output projection.

"""

loss_fn = nn.CrossEntropyLoss()

def _train_(model, optimizer, train_loader, device, epochs):
    #indicate training 
    model.train()
    total_loss = 0

    for i in range(epochs):
        total_loss = 0

        for x_batch, y_batch in train_loader:

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            loss = loss_fn(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"epoch {i + 1}: {avg_loss}")



    torch.save(model.state_dict(), "n_gram_model.pth")


def make_ngrams(data, context_size):
    xs = []
    ys = []
    #for every row
    for row in data.examples:
        # set the proper token and id values
        tokens = row.text
        ids = [TEXT.vocab.stoi[token] for token in tokens]

        # we will iterate them and add the rows
        for i in range(len(tokens) - context_size):
            xs.append(ids[i:i+ context_size])
            ys.append(ids[i + context_size])

    return torch.tensor(xs), torch.tensor(ys)



def train(embedding_dim, context_amount, hidden_dim, vocab_size, cont, epochs):
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    model = NGramModel(vocab_size, embedding_dim, context_amount, hidden_dim).to(device)

    if cont:
        model.load_state_dict(torch.load("n_gram_model.pth", map_location=device))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    # data procesor
    xs, ys = make_ngrams(train_data, context_amount)
    train_dataset = TensorDataset(xs, ys)
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )

    _train_(model, optimizer, train_loader, device, epochs)



"""

Testing functions below:

Testing uses the same n-gram batching process, but the model is put in eval mode
and gradients are disabled.

The model predicts the most likely next word with argmax, then prints the context,
the real next-word sentence, and the predicted next-word sentence for inspection.


"""

def _test_(model, test_loader, device, max_examples=1000, print_examples=10):
    model.eval()
    shown = 0
    total_correct = 0
    total_examples = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            preds = logits.argmax(dim=1)

            x_batch = x_batch.cpu()
            y_batch = y_batch.cpu()
            preds = preds.cpu()

            for context_ids, real_id, pred_id in zip(x_batch, y_batch, preds):
                if total_examples >= max_examples:
                    percent_correct = total_correct / total_examples * 100
                    print(f"Correct: {total_correct}/{total_examples} ({percent_correct:.2f}%)")
                    return

                if pred_id.item() == real_id.item():
                    total_correct += 1
                total_examples += 1

                if shown >= print_examples:
                    continue

                context_words = [TEXT.vocab.itos[i] for i in context_ids.tolist()]
                real_word = TEXT.vocab.itos[real_id.item()]
                pred_word = TEXT.vocab.itos[pred_id.item()]

                print("context:  ", " ".join(context_words))
                print("real:     ", " ".join(context_words + [real_word]))
                print("predicted:", " ".join(context_words + [pred_word]))
                print()

                shown += 1

    percent_correct = total_correct / total_examples * 100
    print(f"Correct: {total_correct}/{total_examples} ({percent_correct:.2f}%)")



def test(embedding_dim, context_amount, hidden_dim, vocab_size):
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    model = NGramModel(vocab_size, embedding_dim, context_amount, hidden_dim).to(device)
    model.load_state_dict(torch.load("n_gram_model.pth", map_location=device))

    # data procesor
    xs, ys = make_ngrams(test_data, context_amount)
    test_dataset = TensorDataset(xs, ys)
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=True
    )

    _test_(model, test_loader, device, max_examples=1000, print_examples=10)


# this method will generate a response based off the context amount. 
#so we will only grab one input from the data loader and then build on it till
# our token = <eos>
def _generate_(model, device, context_amount, max_tokens=30):
    model.eval()
    xs, ys = make_ngrams(test_data, context_amount)
    random_num = random.randrange(len(xs))
    context = xs[random_num].tolist()
    generated = context.copy()

    print("starting context:", " ".join(TEXT.vocab.itos[i] for i in context))

    with torch.no_grad():
        for _ in range(max_tokens):
            x = torch.tensor([context]).to(device)
            logits = model(x)

            unk_id = TEXT.vocab.stoi["<unk>"]
            logits[0, unk_id] = -float("inf")
            pred_id = logits.argmax(dim=1).item()


            pred_id = logits.argmax(dim=1).item()
            pred_word = TEXT.vocab.itos[pred_id]

            generated.append(pred_id)
            print(pred_word)

            if pred_word == "<eos>":
                break

            # Remove the oldest token and add the predicted token to the end.
            context = context[1:] + [pred_id]

    print("generated:", " ".join(TEXT.vocab.itos[i] for i in generated))


def generate(embedding_dim, context_amount, hidden_dim, vocab_size):
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    model = NGramModel(vocab_size, embedding_dim, context_amount, hidden_dim).to(device)
    model.load_state_dict(torch.load("n_gram_model.pth", map_location=device))

    _generate_(model, device, context_amount)





def main():
    #arguments
    flag = sys.argv[1]

    #env vars
    embedding_dim = 64
    context_amount = 5
    hidden_dim = 512
    vocab_size = len(TEXT.vocab)
    epochs = 5
    
    #execution

    if flag == "train":
        print("Training starting now...")
        train(embedding_dim, context_amount, hidden_dim, vocab_size, False, epochs)
    elif flag == "continue":
        print("Continue training starting now...")
        train(embedding_dim, context_amount, hidden_dim, vocab_size, True, epochs)
    elif flag == "test":
        print("Testing starting now...")
        test(embedding_dim, context_amount, hidden_dim, vocab_size)
    elif flag == "generate":
        print("Generating starting now...")
        generate(embedding_dim, context_amount, hidden_dim, vocab_size)
    else:
        print("Improper request. Requires flag")


    


if __name__ == "__main__":
    main()
