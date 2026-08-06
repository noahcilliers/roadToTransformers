import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from datasets import grab_batch


"""
So transforms take the data we recieve from the datasets and make them able to be trained from

This is by using transform= and target_transform=

So transform= is used to make sure we have the features in the correct format which is the normalized tensor

And then for target_transform= the labels are put in one hot encoded tensors

one hot encoded tensors are tensors with len=total labels and 1 in the correct position

so if theres 10 labels and the position is 2 it would look like [0,0,1,0,0,0,0,0,0,0]

"""

#here we make the dataset object and set the transform
ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    #first make the tensor with ToImage (to image tensor)
    #then we make into float32 and normalize from 0-1
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32,scale=True)]),
    # for labels we use the lambda which allows any user function
    # we use one_hot to turn it into a one hot tesnor, 10 long, and a float to match the Dtype
    target_transform=v2.Lambda(
        lambda y: F.one_hot(torch.tensor(y),num_classes=10).float()
    )
)

def show(x):
    feature, label = ds[x]
    print(label)
    fig = plt.figure()
    plt.title(label)
    plt.axis("off")
    plt.imshow(feature.squeeze(), cmap="gray")
    plt.show(block=False)
    plt.pause(2)
    plt.close(fig)


def main():
    grab_batch()


if __name__ == "__main__":
    main()