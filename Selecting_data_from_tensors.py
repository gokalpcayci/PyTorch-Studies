
import torch
# INDEXING

# indexing with pytorch is similar to indexing
# numphy uses arrays as its data type phytorch uses tensors
x = torch.arange(1, 10).reshape(1, 3, 3)
print(x)

# Let's index on our new tensor
print(x[0])
# tensor([[1, 2, 3],
#         [4, 5, 6],
#         [7, 8, 9]])

# Let's index on the middle bracket (dim=1)
print(x[0][0]) # tensor([1, 2, 3])
# Let's index on the most inner bracket (last dimension)
print(x[0][2][2]) #tensor(1)

# You can also use ":" to select "all" of a target dimension
# Get all the values of 0th and 1st dimensions but only index 1 of 2nd dimension
# print(x[1, 1])

# Get all values of the 0 dimension but only the 1 index value of 1st and 2nd dimension
print(x[:,1, 1 ])
# x[:,1, 1 ] and x[0][1][1] almost the same thing. The difference is the semicolon gives us all the dimension so ve
# get square brackets

# Get index 0 of 0th and 1st dimension and all values of 2nd dimension
print(x[0,0, :])

# Practice: Index on x to return 9:
print(x[0,2,2])
print(x[0,:,2]) #dıştan içe doğru gidiyoruz
