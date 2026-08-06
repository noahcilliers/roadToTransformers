import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

"""
DEFINE OUR CLASS FOR A NN

FLATTEN PUTS THE INPUT INTO A FLAT TENSOR

EVERY NN SHOULD BE A SUBCLASS OF nn.Module

THIS STACK IS VERY SIMPLE

JUST A FEW LAYERS TURNING INTO LEARNED VALUES

"""

class NueralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            #size of input = 28 * 28
            nn.Linear(28*28, 512),
            #relu mean rectified linear unit
            # it just takes all the negative numbers and makes them 0
            nn.ReLU(),
            #one layer of 1x1
            nn.Linear(512, 512),
            #non linearization
            nn.ReLU(),
            #output 512->10
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits





def main():
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    model = NueralNetwork().to(device)
    
    X = torch.rand(1, 28, 28, device=device)
    logits = model(X)
    pred_probab = nn.Softmax(dim=1)(logits)
    y_pred = pred_probab.argmax(1)
    print(f"Prediced class: {y_pred}")





if __name__ == "__main__":
    main()