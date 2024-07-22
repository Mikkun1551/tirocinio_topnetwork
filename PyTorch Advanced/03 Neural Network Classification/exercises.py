"""
Exercises & Extra-curriculum

See exercises for this notebook here: https://www.learnpytorch.io/02_pytorch_classification/#exercises
"""
import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from torch import nn
from torchmetrics import Accuracy
import matplotlib.pyplot as plt
import requests
from pathlib import Path
import numpy as np

# Setup device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

"""
1. Make a binary classification dataset with Scikit-Learn's make_moons() function.
"""
# For consistency, the dataset should have 1000 samples and a random_state=42.
X_moon, y_moon = make_moons(n_samples=1000,
                            noise=0.07,
                            random_state=42)

# Turn the data into PyTorch tensors.
X_moon = torch.from_numpy(X_moon).type(torch.float)
y_moon = torch.from_numpy(y_moon).type(torch.float)

# Split the data into training and test sets using
# train_test_split with 80% training and 20% testing.
X_moon_train, X_moon_test, y_moon_train, y_moon_test = train_test_split(X_moon, y_moon,
                                                                        test_size=0.2,
                                                                        random_state=42)
# plt.figure(figsize=(10, 7))
# plt.scatter(X_moon[:, 0], X_moon[:, 1], c=y_moon, cmap=plt.cm.RdYlBu)
# plt.show()

"""
2. Build a model by subclassing nn.Module that incorporates non-linear
activation functions and is capable of fitting the data you created in 1.
"""
# Feel free to use any combination of PyTorch layers (linear and non-linear) you want.
class MoonModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units):
        """
        :param input_features: int, number of inputs to the model
        :param output_features: int, number of output classes
        :param hidden_units: int, numbers of neurons for layer, default 8
        """
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features))

    def forward(self, x):
        return self.linear_layer_stack(x)

# print(X_moon.shape) -> 2
# print(torch.unique(y_moon)) -> 1
model_moon = MoonModel(2, 1, hidden_units=10).to(device)

"""
3. Setup a binary classification compatible loss function 
and optimizer to use when training the model.
"""
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model_moon.parameters(), lr=0.1)

"""
4. Create a training and testing loop to fit the model
you created in 2 to the data you created in 1.
"""
# To measure model accuracy, you can create your own
# accuracy function or use the accuracy function in TorchMetrics.
acc_fn = Accuracy(task="multiclass", num_classes=2).to(device)

epochs = 1500
X_moon_train, y_moon_train = X_moon_train.to(device), y_moon_train.to(device)
X_moon_test, y_moon_test = X_moon_test.to(device), y_moon_test.to(device)
for epoch in range(epochs):
    model_moon.train()

    y_logits = model_moon(X_moon_train).squeeze()
    y_pred_probs = torch.sigmoid(y_logits)
    y_preds = torch.round(y_pred_probs)

    loss = loss_fn(y_logits, y_moon_train)
    acc = acc_fn(y_preds, y_moon_train.int())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_moon.eval()
    with torch.inference_mode():
        test_logits = model_moon(X_moon_test).squeeze()
        test_preds = torch.round(torch.sigmoid(test_logits))

        test_loss = loss_fn(test_logits, y_moon_test)
        test_acc = acc_fn(test_preds, y_moon_test)
    if epoch % 100 == 0:
        print(f'Epoch: {epoch} | Loss: {loss:.4f}  '
              f'Acc: {acc:.2f}% | Test loss: {test_loss:.4f}  '
              f'Test acc: {test_acc:.2f}%')


"""
5. Make predictions with your trained model and plot them using the 
plot_decision_boundary() function created in this notebook.
"""
model_moon.eval()
with torch.inference_mode():
    y_logits = model_moon(X_moon_test)

if len(torch.unique(y_moon)) > 2:
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
else:
    y_pred = torch.round(torch.sigmoid(y_logits))

if Path('helper_functions.py').is_file():
    print('helper_functions.py already exists, skipping download')
else:
    print('Download helper_functions.py')
    request = requests.get('https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py')
    with open('helper_functions.py', 'wb') as f:
        f.write(request.content)
from helper_functions import plot_predictions, plot_decision_boundary

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Train')
plot_decision_boundary(model_moon, X_moon_train, y_moon_train)
plt.subplot(1, 2, 2)
plt.title('Test')
plot_decision_boundary(model_moon, X_moon_test, y_moon_test)
plt.show()

"""
6. Replicate the Tanh (hyperbolic tangent) activation function in pure PyTorch.
"""
# Feel free to reference the ML cheatsheet website for the formula.
tensor_A = torch.arange(-100, 100, 1)
plt.plot(tensor_A)
plt.show()
plt.plot(torch.tanh(tensor_A))
plt.show()
def tanh(x):
  return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))

plt.plot(tanh(tensor_A))
plt.show()

"""
7. Create a multi-class dataset using the spirals 
data creation function from CS231n (see below for the code).
"""
np.random.seed(42)
N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j

# Split the data into training and test sets (80% train, 20% test)
# as well as turn it into PyTorch tensors.
X = torch.from_numpy(X).type(torch.float) # features as float32
y = torch.from_numpy(y).type(torch.LongTensor) # labels need to be of type long
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
acc_fn = Accuracy(task="multiclass", num_classes=3).to(device) # send accuracy function to device

# Construct a model capable of fitting the data (you may need a
# combination of linear and non-linear layers).
class SpiralModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.linear1 = nn.Linear(in_features=2, out_features=10)
    self.linear2 = nn.Linear(in_features=10, out_features=10)
    self.linear3 = nn.Linear(in_features=10, out_features=3)
    self.relu = nn.ReLU()

  def forward(self, x):
    return self.linear3(self.relu(self.linear2(self.relu(self.linear1(x)))))

model_1 = SpiralModel().to(device)

# Build a loss function and optimizer capable of handling multi-class data
# (optional extension: use the Adam optimizer instead of SGD, you may have to experiment
# with different values of the learning rate to get it working).
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_1.parameters(),
                             lr=0.02)

# Make a training and testing loop for the multi-class data and train
# a model on it to reach over 95% testing accuracy (you can use any
# accuracy measuring function here that you like).
epochs = 1000
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)
for epoch in range(epochs):
    ## Training
    model_1.train()

    y_logits = model_1(X_train)
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)

    loss = loss_fn(y_logits, y_train)
    acc = acc_fn(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    ## Testing
    model_1.eval()
    with torch.inference_mode():
        # 1. Forward pass
        test_logits = model_1(X_test)
        test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
        # 2. Caculate loss and acc
        test_loss = loss_fn(test_logits, y_test)
        test_acc = acc_fn(test_pred, y_test)

    # Print out what's happening
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | "
              f"Loss: {loss:.2f} Acc: {acc:.2f} | "
              f"Test loss: {test_loss:.2f} Test acc: {test_acc:.2f}")

# Plot the decision boundaries on the spirals dataset from your model
# predictions, the plot_decision_boundary() function should work for this dataset too.
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_1, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_1, X_test, y_test)
plt.show()
