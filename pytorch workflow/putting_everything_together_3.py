import torch
import matplotlib.pyplot as plt
from torch import nn

# Create device agnostik code.
# This means if we've got access to a GPU, our code will use it (for potentially faster computing)
# if no GPU is available the code will default to using CPU


device = "cpu"

if torch.backends.mps.is_available():
    device = "mps"
    print(f"using device: {device}")

weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

# Training split
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]


class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)


torch.manual_seed(42)
model_1 = LinearRegression()

with torch.inference_mode(mode=True):
    y_predictions = model_1(X_test)

loss_function = nn.L1Loss()
optimizer = torch.optim.SGD(model_1.parameters(), lr=0.01)




torch.manual_seed(42)
epochs = 200

model_1 = model_1.to(device)

X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)
#  Put data on the target device (device agnostic code for data)


for epoch in range(epochs):
    model_1.train()
    y_predictions = model_1(X_train)
    loss = loss_function(y_predictions, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model_1.eval()
    with torch.inference_mode():
        test_predictions = model_1(X_test)
        test_loss = loss_function(test_predictions, y_test)

    if epoch % 10 == 0:
       print(f"epoch: {epoch} | loss: {loss} | prediction: test loss: {test_loss} ")




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


with torch.inference_mode():
    test_predictions = model_1(X_test)

# plot_predictions(predictions=test_predictions)


# Saving and loading our DL model

from pathlib import  Path

# 1. Create models directory

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# Create model save path
MODEL_NAME = "01_pytorch_workflow_model_1.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_1.state_dict(), f=MODEL_SAVE_PATH)

# Load a PyTorch
# Create a new instance of linear regression
loaded_model_1 = LinearRegression()

# Load the saved model_1 state dict

loaded_model_1.load_state_dict(torch.load(MODEL_SAVE_PATH))

# Put the loaded model to device
print(loaded_model_1.to(device))
print(next(loaded_model_1.parameters()).device)
print(loaded_model_1.state_dict())

loaded_model_1.eval()
with torch.inference_mode():
    loaded_model_1_preds = loaded_model_1(X_test)

print(test_predictions == loaded_model_1_preds)
