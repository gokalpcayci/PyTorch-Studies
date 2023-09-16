
import torch
# Float 32 tensor
# you can look at all the tensor datatypes here: https://pytorch.org/docs/stable/tensors.html
# Most common ones you will use are: 32-bit floating point and 16-bit floating point
# precision in computer science is how many bits of information you're using to represent a number.
float_32_tensor = torch.tensor([1, 2, 3],
                               dtype=None, # datatype is the tensor (e.g. float32, float16 etc. )
                               device="cpu", # device is the hardware (e.g. cpu, gpu)
                               requires_grad=False)  # if you want pytorch to track the gradients of this tensor when it
# You don't have to specify the data type device or requires_grad. Pytorch will do that for you
print(float_32_tensor.dtype) # torch.int64

# -----------------------
# -----------------------
# Note: Tensor datatypes is one of the 3 big issues you'll run into when working with tensors

# 1. Tensors not right datatypes
# if you try to do operations with different datatypes you'll get an error for example:
# tensor_1 = torch.tensor([1,2,3], dtype=torch.float32)
# tensor_2 = torch.tensor([1,2,3], dtype=torch.int32)
# tensor_1 + tensor_2 # this will give you an error
# --------------------------------
# 2. Tensors not in right shape
# if you try to do operations with different shapes you'll get an error for example:
# tensor_1 = torch.tensor([1,2,3])
# tensor_2 = torch.tensor([1,2,3,4])
# tensor_1 + tensor_2 # this will give you an error
# --------------------------------
# 3. Tensors not on right device (CPU or GPU)
# if you try to do operations with different devices you'll get an error for example:
# tensor_1 = torch.tensor([1,2,3], device="cpu") # tensor on cpu
# tensor_2 = torch.tensor([1,2,3], device="cuda:0") # tensor on gpu
# tensor_1 + tensor_2 # this will give you an error
# -----------------------
# -----------------------

float_16_tensor = torch.tensor([1, 2, 3],dtype=torch.float16)
print(float_16_tensor.dtype) # torch.float16

# Getting information from tensors

# 1. Tensors not right datatype - to do get datatype from a tensor, can use `tensor.dtype`
# 2. Tensors not right shape - to do get shape from a tensor, can use `tensor.shape`
# 3. Tensors not right device - to do get device from a tensor, can use `tensor.device`
