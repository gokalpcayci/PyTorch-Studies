import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn


# Data can be almost anything... in machine learning

# Excel speadsheet
# images of any kind
# videos (youtube has lots of data...)
# audio like songs or podcasts
# DNA
# Text

# Machine learning is a game of two parts:
# 1. Get data into a numerical representation.
# 2. Build a model to learn patterns in that numerical representation.

# To showcase this, let's create some known data using the linear regression formula.

# We'll use linear regression formula to make a straight line with known parameters

# Create known parameters
weight = 0.7
bias = 0.3

# Create
start = 0
end = 1
step = 0.02
# X in machine learning is typically a Matrix or tensor and A Capital represents a matrix or tensor
# and a lower case represents a vector
X = torch.arange(start, end, step).unsqueeze(dim=1)

y = weight * X + bias
print(X[:10], "\n-------\n", y[: 10])
print(len(X), len(y))

# Splitting data into training and test sets (one of the most important concepts in machine learning general)

# Let's create a training and text set with our data

# Create a train/test split
# set - split same thing

train_split = int(0.8 * len(X))
print(train_split)
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

print(len(X_test), len(X_train))
print(len(y_test), len(y_train))


# fig, ax = plt.subplots()
# ax.plot(X_train, y_train)
# plt.show()


def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=None):
    # Plots training data, test data and compares predictions.
    plt.figure(figsize=(10, 7))
    #   Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
    #   Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")
    #   Are there predictions
    if predictions is not None:
        #         Plot the predictions if they exist
        plt.scatter(test_data, predictions, c="r", label="Predictions")
    #   Show the legend
    plt.legend()
    plt.show()


plot_predictions()


# Create linear regression model class
class LinearRegressionModel(nn.Module):  # <- almost everything in PyTorch
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1,
                                                requires_grad=True,
                                                dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1,
                                             requires_grad=True,
                                             dtype=torch.float))

        # Forward method to define the computation in the model

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # <- "x" is the input data
        return self.weights * x + self.bias  # this is the linear formula


# What our model does:
# - Start with random values (weight & bias)
# - Look at training data and adjust the random values to better
# represent (or get closer to) the ideal values (the weight & bias values we used to create the data)

# How does it do so?
# 1. Gradient descent
# 2. Backpropagation
# -----------------------------
# Note -  Pytorch model building essentials:
# - torch.nn contains all of the buildings for computational graphs (a neural network can be considered a computational
# graph)
# - torch.nn Parameter - what parameters should our model try and learn, often a PyTorch layer from torch.nn will set these for us
# - torch.nn Module - The base class for all neural network modules, if you subclass it, you should overwrite forward()
# - torch.optim - this where the optimizers in pytorch live, they will help with gradient descent
# - def forward() - All nn.Module subclasses require you to overwrite forward(), this method defines what happens in the forward computation

# ------------------------

# Checking the contents of our PyTorch model
# now we've created a model, let's see what's inside...
# So we can check our model parameters or what's inside our model using `.parameters()`

# Create a random seed

torch.manual_seed(42)

# Create an instance of the model (this is a subclass of nn.Module)
model_0 = LinearRegressionModel()

print(model_0)  # LinearRegressionModel()

# it didn't give us much so let's check the parameters

print(list(model_0.parameters()))
# [Parameter containing:
# tensor([0.3367], requires_grad=True), Parameter containing:
# tensor([0.1288], requires_grad=True)]

# List named parameters
# print(model_0.state_dict())
# print(weight, bias)

# Making predictions using `torch.inference_mode()` To check our model's predictive power, let's see how well it predicts `y_test` based on `X_test` when we pass data through our model,
# it's going to run it through the `forward()` method. print(X_test) Make predictions with model inference mode turns off the gradient tracking. Because when we are doing inference we are not doing
# training so we don't need to keep track of the gradient to keep track of how we should update our models. So inference mode

# disables all of the useful things that are available during training. What's the benefit of this? Pytorch behind the scenes keeps track of less data so in turn it will with our small dataset,
# it probably won't be too dramatic, but with a larger dataset, it means that your predictions will potentially be a lot faster. You can do something similar with torch.no_grad(), however,
# inference_mode() is preferred
with torch.inference_mode(mode=True):
    y_preds = model_0(X_test)

# print(y_preds)
# if you have notImplementedError control the spacing in our class

# y_preds = model_0(X_test)
# print(y_preds)
# plot_predictions(predictions=y_preds)

# The whole idea of trainning is for a model to move from some unknown parameters (these may be random) to some known
# parameters or in other words from a poor representation of the data to a better representation of the data

# One way to measure how poor or how wrong your models predictions are is to use a loss function.
# Note: Loss function may also be called cost function or criterion in different areas. For our case, we're going to refer to it as a loss function.
# ---------
# Note - Things we need to train:
# Loss function: A function to measure how wrong your model's predictions are to the ideal outputs, lower is better
# Optimizer: Takes into account the loss of a model and adjusts the model's parameters (e.g. weight & biases)

# And specifically for PyTorch, we need:
# * A training loop
# * A testing loop
# --------
# list(model_0.parameters())
# Check out our model's parameters (a parameter is a value that the model sets itself)
print(model_0.state_dict())

# Setup a loss function
loss_fn = nn.L1Loss()

# Setup an optimizer (stochastic gradient descent)
optimizer = torch.optim.SGD(model_0.parameters(), lr=0.01)  # lr = learning rate possibly the most important hyperparameter
# hyperparameter is a value that us as a data scientist or a machine learning engineer set ourselves
# why give lr=0.01 these type of values come with experience
#  a lot of the default settings are pretty good in torch optimizers
# ----------------
# Build a training loop (and a testing loop) in PyTorch

# A couple of things we need in a training loop:
# 0. Loop through the data
# 1. Forward pass (this involves data moving through our model's `forward()` functions) to make predictions on data - also called forward propagation
# 2. Calculate the loss (compare forward pass predictions to ground truth labels)
# 3. Optimizer zero grad
# 4. Loss backward - move backwards through the network to calculate the gradients of each of the parameters of our model with respect to the loss ( backpropagation )
# 5. Optimizer step - use the optimizer to adjust our model's parameters to try and improve the loss ( Gradient descent )

# ---------------------------------

# epoch is one loop through the data... (this is a hyperparameter because we've set it ourselves)
torch.manual_seed(42)

epochs = 200

# TRack different values
epoch_count = []
loss_values = []
test_loss_values = []

#  0. Loop through the data
for epoch in range(epochs):
    # set the model to training mode
    model_0.train()  # train mode in PyTorch sets all parameters that require gradients to require gradients

    # 1. Forward pass
    y_pred = model_0(X_train)

    # 2. Calculate the loss
    loss = loss_fn(y_pred, y_train)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Perform backpropagation on the loss with respect to the parameters of the model
    loss.backward()

    # 5. Step the optimizer (perform gradient descent)
    optimizer.step()  # by default how the optimizer changes will accumulate through the loop so... we have to zero them above in step 3

    # Testing
    model_0.eval()  # turns off different settings in the model not needed for evaluation/testing (dropout/ batch norm layers)
    with torch.inference_mode():
        # turns off gradient tracking & couple more things  behind the scenes
        # with torch.no_grad(): you may also see with torch.no_grad(): in older PyTorch code
        # 1. do the forward pass
        test_pred = model_0(X_test)

        # 2. Calculate the loss
        test_loss = loss_fn(test_pred, y_test)
    # print out what's happening
    if epoch % 10 == 0:
        loss_values.append(loss)
        epoch_count.append(epoch)
        test_loss_values.append(test_loss)
        # print(f"Epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}")
        # Print out model state_dict()
        # print(model_0.state_dict())
#
# print(f"\n------\nepoch counts:\n {epoch_count}, \n--------\nloss values:\n {loss_values}, \n----------\ntest loss values:\n{test_loss_values}")
with torch.inference_mode():
    y_preds_new = model_0(X_test)

# print(model_0.state_dict())

# plot_predictions(predictions=y_preds)
#
# plot_predictions(predictions=y_preds_new)

# Plot the loss curves

# plt.plot(epoch_count, np.array(torch.tensor(loss_values).numpy()), label="Train loss")
# plt.plot(epoch_count, test_loss_values, label="Test loss")
# plt.title("Training and test loss curves")
# plt.ylabel("Loss")
# plt.xlabel("Epochs")
# plt.legend()
# plt.show()


# note - Saving a model in PyTorch
# maybe you want to use this model a weeks time later or want to show it to a friend

# There are 3 main methods you should know about for saving and loading models in pytorch.
# 1.  `torch.save()` - allows you save a PyTorch object in Phyton's pickle format
# 2. `torch.load()` - allows you to load a saved PyTorch object
# 3. `torch.nn.Module.load_state_dict()` - this allows you to load a model's saved state dictionary
from pathlib import  Path
print(model_0.state_dict())

# Create models directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# Create model save path
MODEL_NAME = "01_pytorch_workflow_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
#  Save the model state dict
print(MODEL_SAVE_PATH)
torch.save(obj=model_0.state_dict(), f=MODEL_SAVE_PATH)

model = torch.load(MODEL_SAVE_PATH)
# model.eval()

# Since we saved our model's `state_dict()` rather the entire model, we'll create a new instance of our model class and load the saved `state_dict()` into that.
print(model_0.state_dict())

# To load in a saved state_dict we have to instantiate a new instance of our model class
loaded_model_0 = LinearRegressionModel()

# Load the saved state_dict of model_0 (this will update the new instance with updated parameters)

loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

print(loaded_model_0.state_dict())
# Make some predictions with our loaded model
loaded_model_0.eval()
with torch.inference_mode():
    loaded_model_preds = loaded_model_0(X_test)

print(loaded_model_preds)

# Compare loaded model preds with original model preds
print(y_preds == loaded_model_preds)

model_0.eval()
with torch.inference_mode():
    original_model_preds = model_0(X_test)

print(original_model_preds)

print(original_model_preds == loaded_model_preds)
