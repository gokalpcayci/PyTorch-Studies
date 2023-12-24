import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn

weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02

X = torch.arange(start, end, step).unsqueeze(dim=1)

y = weight * X + bias

train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=None):
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c="b",
                s=4, label="Training data")
    plt.scatter(test_data, test_labels,
                c="g", s=4, label="Test data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c="r")

    plt.legend()
    plt.show()


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, requires_grad=True))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias


torch.manual_seed(42)

model_0 = LinearRegressionModel()

with torch.inference_mode(mode=True):
    y_predictions = model_0(X_test)

plot_predictions(predictions=y_predictions)

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(model_0.parameters(), lr=0.01)

torch.manual_seed(42)
epochs = 100

epoch_count = []
loss_values = []
test_loss_values = []

for epoch in range(epochs):
    model_0.train()
    y_predictions = model_0(X_train)
    loss = loss_fn(y_predictions, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model_0.eval()
    with torch.inference_mode():
        test_predictions = model_0(X_test)
        test_loss = loss_fn(test_predictions, y_test)

    if epoch % 10 == 0:
        loss_values.append(loss)
        epoch_count.append(epoch)
        test_loss_values.append(test_loss)


with torch.inference_mode():
    y_predictions_new = model_0(X_test)


plot_predictions(predictions=y_predictions_new)

# plt.plot(epoch_count, np.array(torch.tensor(loss_values).numpy()), label="Train loss")
# plt.plot(epoch_count, test_loss_values, label="Test loss")
# plt.title("Training and test loss curves")
# plt.ylabel("Loss")
# plt.xlabel("Epochs")
# plt.legend()
# plt.show()
