import torch

# Reshaping: reshapes an input tensor to a defined shape
# View: Return a view of an input tensor of certain shape but keep the same memory as the original tensor
# Stacking: combine multiple tensors on top of each other (vstack) or side by side (hstack)
# Squeeze: removes all `1` dimensions from a tensor
# Unsqueze: add a `1` dimension to a target tensor
# Permute: Return a view of the input with dimensions permuted (swapped) in a certain way

x = torch.arange(1., 10.)
print(x)
print(x.shape)

# -------------------------------------------------------
# Add an extra dimension (reshaping)
x_reshaped = x.reshape(3, 3)  # you have to much the nuber of elements in the tensor to your shape
# let's say you have 9 elements, you can't reshape it to 4,4 but if you had 16 element you could have
x_reshaped_2 = x.reshape(9, 1)
print(x_reshaped, x_reshaped.shape)
print(x_reshaped_2, x_reshaped_2.shape)

# -------------------------------------------------------
# Change the view
# changing z changes x (because a view of a tensor shares the same memory as the original input)
z = x.view(1, 9)
print(z, z.shape)
z[:, 0] = 5
print("------------------")
print(z, x)
# -------------------------------------------------------
# Stack tensors on top of each other
print("------------------")
x_stacked = torch.stack([x, x, x, x], dim=0)
# tensor([[5., 2., 3., 4., 5., 6., 7., 8., 9.],
#         [5., 2., 3., 4., 5., 6., 7., 8., 9.],
#         [5., 2., 3., 4., 5., 6., 7., 8., 9.],
#         [5., 2., 3., 4., 5., 6., 7., 8., 9.]])
# ----
y_stacked = torch.stack([x, x, x, x], dim=1)
# tensor([[5., 5., 5., 5.],
#         [2., 2., 2., 2.],
#         [3., 3., 3., 3.],
#         [4., 4., 4., 4.],
#         [5., 5., 5., 5.],
#         [6., 6., 6., 6.],
#         [7., 7., 7., 7.],
#         [8., 8., 8., 8.],
#         [9., 9., 9., 9.]])
# ----
# dim=2 wouldn't work because original shape of x isn't compatible
# z_stacked = torch.stack([x, x, x, x], dim=3)
print(x_stacked)
print(y_stacked)
# print(z_stacked) # IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 3)

# -------------------------------------------------------
# squeeze tensors
# torch.squeeze() - removes all single dimensions from a target tensor
print("-------------------")
x_reshaped_3 = x_reshaped.reshape(1, 9)
print(x_reshaped_3)
print(torch.squeeze(x_reshaped_3))
print(x_reshaped_3.size())
print(torch.squeeze(x_reshaped_3).size())
x_squeezed = x_reshaped_3.squeeze()
print(x_squeezed)
# torch.Size([1, 9]) 0th dimension = 1, first dimension = 9
# tensors can have unlimited dimensions
# -------------------------------------------------------
# unsqueeze tensors
# torch.unsqueeze() - adds a single dimension to a target tensor at a specific dim (dim = dimension)
print(f"Previous target: {x_squeezed}")
print(f"Previous shape: {x_squeezed.shape}")

# Add a extra dimension
# instead of adding the 0th dimension at the start unsqueeze will add the new dimension at the end
x_unsqueezed = x_squeezed.unsqueeze(dim=0)
print(f"New tensor: {x_unsqueezed}")
print(f"New shape: {x_unsqueezed.shape}")

# -------------------------------------------------------
# torch.permute - rearranges the dimensions of a target tensor in a specified order

x = torch.rand(2, 5, 3)
print(x)
x_reordered = torch.permute(x, (2, 0, 1))
print(x.size())
print(x_reordered.size())

x_original = torch.rand(size=(224, 224, 3))  # [height, width, colour_channels]

# Permute the original tensor to rearrange the axis (or dim) order
x_permuted = torch.permute(x_original, (2, 0, 1)) # shifts axis 0->1, 1->2, 2->0
print(x_original.size())  # torch.Size([224, 224, 3])
print(x_permuted.size())  # torch.Size([3, 224, 224])
x_original[0, 0, 0] = 93475
print(x_original)
print(x_permuted)
# changing the value of x_original to highlight the fact that permute returns a different view of the original tensor
# and a view in pytorch shares memory with that tensor
# -------------------------------------------------------
