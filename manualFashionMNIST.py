# torch + numpy
import torch
import numpy as np
import sys
#datasets
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms import v2
#plotting with this 
import matplotlib.pyplot as plt
#dataloaders: make it easy to iterate over the data
from torch.utils.data import DataLoader

from torch import nn
import torch.nn.functional as F

"""
OKAY SO WHHAT DO WE NEED

LOOP:
- load data
- transform the data
- send it through
- get the response
- back propogate


# load
model = NueralNetwork()
model.load_state_dict(torch.load("fashion_mnist_model.pth"))
model.eval()


So MLP (Multilayer Perception)is the intial implementation for this, but it doesnt work too well becauser it requires us to flatten the image so we can't easily get the full picture

SO a CNN (Convolutional Neural Network) would be much better here (this is what we used for the PPO carracing)

Backward step only calculates the gradient :: THE OPTIMIZER CHANGES THE WEIGHTS


"""
#-----------------------------------------------------------
# HERE IS THE SECTION WHERE WE DEFINE THE DATA VARS AND FUNC
#___________________________________________________________
training_data = datasets.FashionMNIST(
    root="data",
    train = True,
    download = True,
    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

def grab_batch(data_iter):
    #next and iter are py native. 
    imgs, labels = next(data_iter)
    return imgs, labels

#-------------------------------------------
# HERE IS THE SECTION WHERE WE DEFINE THE NN
#___________________________________________

class NueralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.cnn_stack = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.linear_relu_stack = nn.Sequential(
            #size of input = 28 * 28
            nn.Linear(64*7*7, 512),
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
        x = self.cnn_stack(x)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


loss_fn = nn.CrossEntropyLoss()

def train(device, model, optimizer):
    for epochs in range(10):
        for imgs, labels in train_dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            #now we have 64 imgs, and 64 labels
            logits = model(imgs)
            #back propogate
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), "fashion_mnist_model.pth")

def test(device, model):
    total, c = 0, 0
    with torch.no_grad():
        for imgs, labels in test_dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            logits = model(imgs)
            preds = logits.argmax(dim=1)

            c += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"Model got {c}/{total} correct, {100 * c/total:.2f}%")
        



def main():
    flag = sys.argv[1]
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    model = NueralNetwork().to(device)
    #model.load_state_dict(torch.load("fashion_mnist_model.pth"))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    if(flag == "train"):
        model.train()
        train(device, model, optimizer)
    elif(flag == "test"):
        model.load_state_dict(torch.load("fashion_mnist_model.pth"))
        model.eval()
        test(device, model)
    elif(flag == "continue"):
        model.load_state_dict(torch.load("fashion_mnist_model.pth"))
        model.train()
        train(device, model, optimizer)








if __name__ == "__main__":
    main()