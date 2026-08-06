# torch + numpy
import torch
import numpy as np
#datasets
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import v2
#plotting with this 
import matplotlib.pyplot as plt
#dataloaders: make it easy to iterate over the data
from torch.utils.data import DataLoader


# initalize the data 

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
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)

#
# intialize the dataloaders 
# (called this because they help load data into the transformers during training)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

def grab_batch():
    #next and iter are py native. 
    imgs, labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {imgs.size()}")
    print(f"Labels batch shape: {labels.size()}")
    fig = plt.figure()
    for i in range(64):
        print(f"Label: {labels[i]}")
        plt.title(labels[i])
        plt.imshow(imgs[i].squeeze(), cmap="gray")
        plt.show(block=False)
        plt.pause(2)
        plt.close(fig)

def show_data(x):
    figure = plt.figure(figsize=(8,8))
    img, label = training_data[x]
    plt.title(label)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
    plt.show()

def main():
    #grab_batch()
    #print(training_data.__len__())
    #show_data(0)
    pass
   
    

if __name__ == "__main__":
    main()
