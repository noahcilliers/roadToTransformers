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
Okay so the goal here is to translate an english sequence to the same sequence in affricans.

A regular lstm would struggle at this because the weights would have to learn not only what the sentence is but also
the translation

So that brings the encoder decoder method. The encoders read all the data and get the proper knowledge for what we are 
trying to say

And then the decoders take those cell and hidden states and make the correct output

We will use the reverse order for inputs to make the translation easier

"""

# vocabullary

PAD, SOS, EOS = 0, 1, 2








####################################################################################################
# So first we want to define what the different architecutres are for the enocder and decoders layers
# The encoders inputs are the sequence, prior hidden and cell units
# the decoders inputs are the prior token given, prior hidden and cell units
##################################################################################################


## Single LSTM cell logic. we can send some data in and get this data out
class LSTMCell(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        combined_dim = hidden_dim+embedding_dim
        ### GATES
        self.forget_gate = nn.Linear(combined_dim, hidden_dim)
        self.input_gate = nn.Linear(combined_dim, hidden_dim)
        self.candidate_gate = nn.Linear(combined_dim, hidden_dim)
        self.output_gate = nn.Linear(combined_dim, hidden_dim)

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


        embeds = self.dropout(x)
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


        return h, c


class StackedLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers=4):
        super().__init__()
        self.cells = nn.ModuleList([
            LSTMCell(embedding_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
    
    # dont we need c for each proir layer
    def forward(self, x, state=None):
        # shape of the input
        B, T, _ = x.shape
        
        if state is None:
            h = [x.new_zeros(B, self.hidden_dim) for _ in range(self.num_layers)]
            c = [x.new_zeros(B, self.hidden_dim) for _ in range(self.num_layers)]
        else:
            h, c = [list(s.unbind(0)) for s in state]

        outs = []
        for t in range(T):
            # now we can actually run the forward pass through the stacked lstm
            inp = x[:, t]
            for l, cell in enumerate(self.cells):
                h[l], c[l] = cell(inp, h[l], c[l])
                inp = h[l]
            outs.append(inp)

        return torch.stack(outs, 1), (torch.stack(h), torch.stack(c))




class Encoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, emb, num_layers=4):
        super().__init__()
        self.lstm = StackedLSTM(embedding_dim, hidden_dim, num_layers)
        self.emb = emb

    def forward(self, x):
        _, state = self.lstm(self.emb(x))
        return state



class Decoder(nn.Module):  
    def __init__(self, embedding_dim, vocab_size, hidden_dim, emb, num_layers=4):
        super().__init__()
        self.emb = emb
        self.lstm = StackedLSTM(embedding_dim, hidden_dim, num_layers)
        self.out = nn.Linear(hidden_dim, vocab_size)
        self.out.weight = emb.weight

    def forward(self, x, state):
        outs, state = self.lstm(self.emb(x), state)
        logits = self.out(outs)
        return logits, state





class Seq2Seq(nn.Module):   # owns the shared embedding, wires enc -> dec
    def __init__(self, vocab_size, embedding_dim, hidden_dim, emb):
        super().__init__()
        emb = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = Encoder(embedding_dim, hidden_dim, emb, num_layers=4)
        self.decoder = Decoder(embedding_dim, vocab_size, hidden_dim, emb, num_layers=4)

    # can we input the entire sequence into the forward pass
    # this forward pass is for training only. For generation we need to make a different 
    # function. Since the previous input must be used instead of an array
    def forward(self, src, tgt_in):
        # run the loop for encoder
        state = self.encoder(src)
        # run the loop for decoder
        logits, _ = self.decoder(tgt_in, state)
        return logits


    # generate one sequence
    @torch.no_grad()
    def generate(self, src, max_len=100):
        # get the state with the encoder
        state = self.encoder(src)
        tok = torch.full((src.size(0), 1), SOS, device=src.device)
        out = []
        # run until decoder says <eos>
        for _ in range(max_len):
            logits, state = self.decoder(tok, state)      # [B, 1, V]
            tok = logits[:, -1].argmax(-1, keepdim=True)  # [B, 1]
            if tok.item() == EOS: break
            out.append(tok)
            #output the tensor of outputs 
        return torch.cat(out, 1)

def main():
    pass




if __name__ == "__main__":
    main()