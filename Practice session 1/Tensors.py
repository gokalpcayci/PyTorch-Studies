import torch

import numpy as np


# Directly from data datatype is automatically generated
data = ([[1,2,3], [4,5,6]])
x_data = torch.tensor(data,dtype=torch.float32)
print(x_data, x_data.dtype)


# From Numpy array
np_array = np.array([[23,45,67],[456, 32, 31]])
x_data = torch.from_numpy(np_array)
print(x_data)
print(x_data.type(), x_data.dtype) # torch.LongTensor torch.int64

# From another tensor:

x_ones = torch.ones_like(x_data)
print(f"Ones Tensor: \n {x_ones} \n")
x_rand = torch.rand_like(x_data, dtype=torch.float32)
print(f"Random Tensor: \n {x_rand} \n")


shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")


# Tensor to NumPy array

t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

# -----------------------------------
# ---------------------------------------

# TRY IT YOURSELF

new_tensor = torch.rand(7, 7)
print(new_tensor.shape)

new_tensor_2 = torch.rand(1, 7)

print(new_tensor * new_tensor_2)

random_seed = 0
torch.manual_seed(random_seed)
random_tensor = torch.rand(7, 1)
torch.manual_seed(random_seed)
random_tensor_2 = torch.rand(7, 1)
print(random_tensor * random_tensor_2)



random_seed = 1234
torch.manual_seed(random_seed)
random_tensor_3 = torch.rand(1,4, device="mps")
print(random_tensor_3)

# -----------------------------------
tensor_x = torch.rand(1,4)
tensor_y = torch.rand(4,4)
matrix_mul = torch.matmul(tensor_x, tensor_y)
print(matrix_mul)
print(torch.max(matrix_mul))
print(torch.min(matrix_mul))

# -------------------------------

random_tensor = torch.rand(1,1,1, 10)

squeezed_tensor = torch.squeeze(random_tensor)
print(random_tensor)
print(squeezed_tensor)
print(random_tensor.shape, squeezed_tensor.shape)
torch.manual_seed(7)
new_tensor = torch.rand(10)
print(new_tensor)

print(torch.rand( size=(224, 224, 3)))
