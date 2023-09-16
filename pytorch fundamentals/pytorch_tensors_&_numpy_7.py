
import  torch
import numpy as np
# NumPy is a popular scientific Phyton numerical computing library
# and because of this, PyTorch has functionality to interact with it.
# - Data in NumPy, want in PyTorch tensor -> torch.from_numpy(ndarray)
# - PyTorch tensor -> NumPy -> torch.Tensor.numpy()

# Numpy array to tensor
array = np.arange(1.0,8.0)
tensor = torch.from_numpy(array) # warning: when converting from numpy -> pytorch, pytorch reflects numpy's default
# datatype of float64 unless specified otherwise
# print(tensor, array)

# Change the value of array, what will this do to tensor?
array = array + 1
print(array, tensor)
# [2. 3. 4. 5. 6. 7. 8.] tensor([1., 2., 3., 4., 5., 6., 7.], dtype=torch.float64)

# Tensor to NumPy array
tensor = torch.ones(7)
numpy_tensor = tensor.numpy()
print(tensor, numpy_tensor)

# Change the tensor, what happens to numpy_tensor ?
tensor = tensor + 1
print(tensor, numpy_tensor) # tensor([2., 2., 2., 2., 2., 2., 2.]) [1. 1. 1. 1. 1. 1. 1.]

