import torch
import numpy as np


def main():

# Tensor from list
    data = [[1,2],[3,4]]
    x_data = torch.tensor(data)
    print(x_data)

# Tensor from Numpy arr
    np_array = np.array(data)
    x_np = torch.from_numpy(np_array)
    print(x_np)

# We can also copy a tensor and retain the shape but with different numbers
    x_ones = torch.ones_like(x_data)
    x_rand = torch.rand_like(x_data, dtype=torch.float)
    print(f"Ones tensor: \n {x_ones}\n\n")
    print(f"Rand tensor: \n {x_rand}\n\n")

# What device is this tensor stored on? 
	
    print(f"This tensor is stored on: {x_data.device}\n")
    
    fastDevice = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
    x_data = x_data.to(fastDevice)
    print(f"This tensor is stored on: {x_data.device}\n")

# These kind of operations are very similar to what we saw with matlab
    size = (4,4)
    tensor = torch.ones(size)
    print(f"\n{tensor}\n")
    tensor[:,1] = 0
    print(tensor)

#matrix multiplication and just integer multiplication with matrices
    
    print(f"\n{tensor * tensor}\n")
    print(tensor @ tensor.T)

    print(tensor.t_())
    print(tensor.add_(1))


# When you create a numpy from a tensor or a tensor from a numpy they actually share their data location on the cpu. so if you go ahead and change the tensor or numpy it will be reflected on both
# THIS IS ONLY ON THE CPU. WHEN WE MOVE TO THE GPU THE TENSOR IS COPIED SO IT IS NOT THE SAME 	








if __name__ == "__main__":
    main()

