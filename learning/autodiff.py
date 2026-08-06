import torch

"""
if we want to turn off the tracking for a tensors requries_grad we can just disable it using torch.no_grad

this is when we are applying the forward pass only
it looks like 
with torch.no_grad():
    forward pass here


FULL JACOBIAN VS JACOBIAN PRODUCT

FULL JACOBIAN CALCULARTES EACH GRADIENT FOR EVERY SINGLE OUTPUT PARAMETER

JACOBIAN PRODUCT IS ABLE TO CALCULATE THE GRADIENT IN RESPECT TO ONLY THE SCALAR LOSS


"""



def main():
    x = torch.ones(5)
    #tensor([1., 1., 1., 1., 1.])
    y = torch.zeros(3)
    #tensor([0., 0., 0.])
    w = torch.randn(5, 3, requires_grad=True)
    """
tensor([[-1.5893, -0.5276, -0.3328],
        [-1.4554,  2.1665, -0.3905],
        [ 0.1680,  0.7090, -1.1540],
        [-0.6980,  1.2247,  1.8812],
        [ 0.0693,  0.0319,  0.2901]], requires_grad=True)

    """
    b = torch.randn(3, requires_grad=True)
    #tensor([-0.6778, -1.5185,  0.1642], requires_grad=True)
    z = torch.matmul(x, w)+b
    #tensor([-4.1832,  2.0860,  0.4581], grad_fn=<AddBackward0>)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
    print(f"Gradient function for z = {z.grad_fn}")
    print(f"Gradient function for loss = {loss.grad_fn}")
    loss.backward()
    print(w.grad)
    print(b.grad)
    




if __name__ == "__main__":
    main()