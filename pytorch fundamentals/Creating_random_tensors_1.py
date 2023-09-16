import torch
# Why Random tensors?

# Random tensors are important because the way many neural networks learn is that
# they start with tensors full of random number and then adjust those random numbers
# to better represent the data

# Start with random numbers -> look at data -> update random numbers ->
# look at data -> update random numbers


# Create a random tensor of size (3, 4)
# random_tensor = torch.rand(1, 10, 10)
# print(random_tensor)
# print(random_tensor.ndim)

# Create a random tensor with similar shape to an image tensor
# random_image_size_tensor = torch.rand(size=(224, 224, 3)) # height, width, colour channels (R, G, B)
# print(random_image_size_tensor)

# ---------------------------
# PRACTICE
random_tensor = torch.rand(3, 2, 1)
random_tensor_2 = torch.rand(1,2,3)
random_tensor_3 = torch.rand(6,6)
random_tensor_4 = torch.rand(6,6, 1)
print(random_tensor)
print(random_tensor_2)
print(random_tensor_3)
print(random_tensor_4)

# with size attribute
random_tensor_5 = torch.rand(size=(3,3,3))
random_tensor_6 = torch.rand(3,3,3)
print(random_tensor_5)
print(random_tensor_6)
# ---------------------------

# Zeros and ones
# Create a tensor of all zeros
zeros = torch.zeros(3,4)
random_tensor = torch.rand(3,4)
print(zeros)
print(zeros * random_tensor)
# if you're working with a random tensor and you wanted to mask out all of the numbers in the first column
# you could create a tensor with zeros of that column and multiply with your target tensor
# and you would zero all those numbers

# Create a tensor of all ones
ones = torch.ones(3,4)
print(ones)
# show the data type of tensor
print(ones.dtype) # torch.float32
# whenever you create a tensor with pytorch (unless you explicitly define what the data type is) it starts of with
# tourch.float32

# -----------------------
# Creating a range of tensors and tensors-like

# Use torch.range()
# Quick note: torch.range will be depreceted instead use torch.arange()
print(torch.arange(0, 10)) # index starts with [included, excluded]
print(torch.__version__) # show pytorch version

# Use torch.range()
one_to_ten = torch.arange(start=1, end=10, step=1)
print(one_to_ten)

# Creating tensors like
ten_zeros = torch.zeros_like(input=one_to_ten)
print(ten_zeros)
