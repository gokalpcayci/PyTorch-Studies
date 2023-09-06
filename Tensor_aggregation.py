import torch

# Finding the min, max, mean, sum, etc

x = torch.arange(0, 100, 10)
# tensor([ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
print(x)

# Find the min
print(torch.min(x)) # tensor(0) also we can write tha same thing like this
print(x.min())

# Find the max
print(torch.max(x)) # tensor(90) also we can write tha same thing like this
print(x.max())

# Fint the avarage
# print(torch.mean(x)) # Input dtype must be either a floating point or complex dtype. Got: Long
# we can do this instead
print(torch.mean(x.type(torch.float32))) # tensor(45.) also we can write tha same thing like this
print(x.type(torch.float32).mean())


# Find the sum
print(torch.sum(x)) # tensor(450) also we can write tha same thing like this
print(x.sum())

# i think it's better to write like torch.<command>()
# choose one and stick with it through out your code

# ----------------------------
# arg finds the index of whatever you are looking for min, max etc.
print(torch.argmin(x))
print(torch.argmax(x))
