import torch
x = torch.rand(5, 3)
# print(x)
#  https://pytorch.org/docs/stable/tensors.html
# A torch.Tensor is a multidimensional matrix containing elements of a single data type.
# PyTorch tensors are created using `torch.Tensor()`
# Pytorch indexes are 0 indexed
scalar = torch.tensor(7)
print(scalar)

print(torch.tensor([[1., -1.], [1., -1.]]))

# Get tensor back as Phyton int
print(scalar.item())
# -------------------------
# VECTOR
vector = torch.tensor([7, 7])
print(vector)
# show how many dimensions vector have
# you can see how many dimension something has by looking at how many nested array there is (number of square brackets)

print(vector.ndim)
#
print(vector.shape)
# -------------------------
# MATRIX
matrix = torch.tensor([[1,2], [3,4]])
print("number of dimensions in matrix: " + str(matrix.ndim))

# any time you encode data into numbers it's a tensor data type
# -------------------------
# TENSOR
tensor = torch.tensor([[[1,2,3],
                        [4,5,6],
                        [7,8,9]]])
print("number of dimensions in tensor: " + str(tensor.ndim))
print("Shape of the tensor: " + str(tensor.shape))
# Most of the time you won't be crafting tensors by hand. PyTorch will do a lot of that behing the scene

# -------------------------
# PRACTICE

scalar_ex = torch.tensor(5)
vector_ex = torch.tensor([9,5,12,56])
matrix_ex = torch.tensor([[12,34,65,45,21], [1,2,5,34,21]])
tensor_ex = torch.tensor([[[4,67,23,21],
                           [56,23,12,78],
                           [44,123,567,3400]]])
dictionary = {"scalar": scalar_ex, "vector": vector_ex, "matrix": matrix_ex,"tensor": tensor_ex}
print("--------------\nShapes and Dimensions of tensor.\n")
for key, val in dictionary.items():
    print(key + " has " + str(val.ndim) + " dimension.")
    print(key + "'s shape: " + str(val.shape))
    print("-----------------")

# -------------------------
