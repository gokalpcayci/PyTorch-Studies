
# Taking the random out of random

# Start with random numbers -> tensor operations ->
# update random numbers to try and make them better representations of the data ->
# again -> again -> again...

# To reduce the randomness in neural networks and pytorch comes the concept of a random seed
# Essentially what the random seed does is "flavour" the randomness

import torch

# Create two random tensors
random_tensor_A = torch.rand(3, 4)
random_tensor_B = torch.rand(3, 4)

print(random_tensor_A)
print(random_tensor_B)
print(random_tensor_A == random_tensor_B)
# ----------
# Let's make some random but reproducible tensors
print("-----------------")
# Set the random seed
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
random_tensor_C = torch.rand(3,4)
torch.manual_seed(RANDOM_SEED)
random_tensor_D = torch.rand(3, 4)
print(random_tensor_C)
print(random_tensor_D)
print(random_tensor_C == random_tensor_D)


if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")

# Create a tensor (default on the CPU)

tensor = torch.tensor([1,2,3])
print(tensor, tensor.device)

# Move tensor to GPU (if available)

tensor_on_gpu = tensor.to(mps_device)
print(tensor_on_gpu)


# https://www.learnpytorch.io
