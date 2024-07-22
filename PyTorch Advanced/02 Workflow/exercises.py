"""
Exercises & Extra-curriculum

See exercises for this notebook here: https://www.learnpytorch.io/01_pytorch_workflow/#exercises
"""
import torch
from torch import nn
import matplotlib.pyplot as plt
from pathlib import Path

# Setup device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

"""
1. Create a straight line dataset using the linear regression formula (weight * X + bias).
* Set weight=0.3 and bias=0.9 there should be at least 100 datapoints total.
* Split the data into 80% training, 20% testing.
* Plot the training and testing data so it becomes visual.
"""
# Set weight=0.3 and bias=0.9 there should be at least 100 datapoints total.
weight = 0.3
bias = 0.9
start = 0
end = 1
step = 0.01
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias
print(y, len(y))

# Split the data into 80% training, 20% testing.
train_split = int(0.8 * len(X))
print(train_split)
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]
print(len(X_train), len(y_train), len(X_test), len(y_test))

# Plot the training and testing data so it becomes visual.
def plot_predictions(train_data=X_train, train_labels=y_train,
                     test_data=X_test, test_labels=y_test,
                     predictions=None):
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c='b', s=4, label='Training data')
    plt.scatter(test_data, test_labels, c='g', s=4, label='Testing data')
    if predictions is not None:
        plt.scatter(test_data, predictions, c='r', s=4, label='Predictions')
    plt.legend(prop={'size': 14})
    plt.show()

# plot_predictions()

"""
2. Build a PyTorch model by subclassing nn.Module.
* Inside should be a randomly initialized nn.Parameter() with requires_grad=True, one for weights and one for bias.
* Implement the forward() method to compute the linear regression function you used to create the dataset in 1.
* Once you've constructed the model, make an instance of it and check its state_dict().
* Note: If you'd like to use nn.Linear() instead of nn.Parameter() you can.
"""
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Inside should be a randomly initialized nn.Parameter() with requires_grad=True, one for weights and one for bias.
        self.weights = nn.Parameter(torch.rand(1, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.rand(1, requires_grad=True, dtype=torch.float))

    # Implement the forward() method to compute the linear regression function you used to create the dataset in 1.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias

# Once you've constructed the model, make an instance of it and check its state_dict().
model = LinearRegressionModel()
print(model.state_dict())

"""
3. Create a loss function and optimizer using nn.L1Loss() and torch.optim.SGD(params, lr) respectively.
* Set the learning rate of the optimizer to be 0.01 and the parameters to optimize should be the model parameters from the model you created in 2.
* Write a training loop to perform the appropriate training steps for 300 epochs.
* The training loop should test the model on the test dataset every 20 epochs.
"""
# Set the learning rate of the optimizer to be 0.01 and the parameters to optimize
# should be the model parameters from the model you created in 2.
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)

# Write a training loop to perform the appropriate training steps for 300 epochs.
epochs = 300
epoch_count = []
loss_values = []
test_loss_values = []
for epoch in range(epochs):
    model.train()
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        test_pred = model(X_test)
        test_loss = loss_fn(test_pred, y_test)

    # The training loop should test the model on the test dataset every 20 epochs.
    if epoch % 20 == 0:
        epoch_count.append(epoch)
        loss_values.append(loss.item())
        test_loss_values.append(test_loss.item())
        print(f'Epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}')
        # Print out model state_dict()
        print(model.state_dict())

"""
4. Make predictions with the trained model on the test data.
* Visualize these predictions against the original training and testing data
(note: you may need to make sure the predictions are not on the GPU if you want to use 
non-CUDA-enabled libraries such as matplotlib to plot).
"""
with torch.inference_mode():
    y_preds_new = model(X_test)
plt.plot(epoch_count, loss_values, label='Train loss')
plt.plot(epoch_count, test_loss_values, label='Test loss')
plt.title('Training and test loss curves')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()
plt.show()
# Visualize these predictions against the original training and testing data
# plot_predictions(predictions=y_preds_new)

"""
5 Save your trained model's state_dict() to file.
* Create a new instance of your model class you made in 2. and load in the state_dict() you just saved to it.
* Perform predictions on your test data with the loaded model and confirm they match the original model predictions from 4.
"""
MODEL_PATH = Path('/home/michel/models')
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_NAME = 'exercise_workflow_model.pt'
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
print(f'Saving model to: {MODEL_SAVE_PATH}')
torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)

# Create a new instance of your model class you made in 2. and load in the state_dict() you just saved to it.
loaded_model = LinearRegressionModel()
loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

# Perform predictions on your test data with the loaded model and confirm they match the original model predictions from 4.
loaded_model.eval()
with torch.inference_mode():
    loaded_model_preds = loaded_model(X_test)

print(y_preds_new == loaded_model_preds)
