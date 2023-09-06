import torch

# Manipulating tensors (tensor operations)
# ----------------------------------------
# Tensor operations include:
# addition, subtraction, multiplication, division,
# matrix multiplication, element-wise products, element-wise exponentiation and more.

# Create a tensor

tensor = torch.tensor([1, 2, 3])
print(tensor + 10)  # tensor([11, 12, 13])
print(tensor * 10)  # tensor([10, 20, 30])
print(tensor - 10)  # tensor([-9, -8, -7])

# Try out PyTorch in-built functions
print(torch.mul(tensor, 10))  # multiply
print(torch.add(tensor, 10))  # addition

# Matrix multiplication

# Two main ways of performing multiplication in neural networks and deep learning
# - element-wise multiplication
# - Matrix multiplication

# Element wise multiplication
print(tensor * tensor)  # tensor([1, 4, 9])

# Matrix multiplication
print(torch.matmul(tensor, tensor)) # tensor(14) 1 + 4 + 9
# so basically you summ it up before showing it

# don't use for loop for matrix multiplication instead use pytorch build up method up there
# it's nearly 10 times faster than for loop

# There are two main rules that performing matrix multiplication needs to satisfy:
# 1. The inner dimensions must match:
# (3, 2) @ (3, 2) => this won't work. ❌
# (2, 3) @ (3, 2) => this will work ✅
# (3, 2) @ (2, 3) => this will work ✅
# ----------------
# The resulting matrix has the shape of the outer dimensions:
# (2, 3) @ (3, 2) -> (2, 2)
print(tensor @ tensor) # "@" symbol also means matrix multiplication but use matmul instead for readibility.

# One of the most common errors in deep learning: shape errors
#  To fix our tensor shape issues, we can manipulate the shape of one of our tensors using a transpose
# A transpose switches the axes or dimensions of a given tensor.

tensor_B = torch.tensor([[7,10],
                         [8, 11],
                         [9, 12]])

print(tensor_B.T) # tensor([[ 7,  8,  9],
                  #        [10, 11, 12]])

print(tensor_B.T.shape) # torch.Size([2, 3])
print(tensor_B.shape) # torch.Size([3, 2])
print(tensor_B.T.dtype, torch.rand(3, 2).dtype)
print(torch.matmul(tensor_B.T, torch.rand(3, 2)))

