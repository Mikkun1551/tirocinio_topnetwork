"""
03. Neural network classification with pytorch

Classification is a problem of predictind whatever something is one thing or another
(there can be multiple things as the options)

Book of this lecture: https://www.learnpytorch.io/02_pytorch_classification/
All other resources: https://github.com/mrdbourke/pytorch-deep-learning
Ask a question: https://github.com/mrdbourke/pytorch-deep-learning/discussions
"""
import torch
from torch import nn
from sklearn.datasets import make_circles
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import requests
from pathlib import Path
from torchmetrics import Accuracy

"""
1. Make classification data and get it ready
"""
# Make 1000 examples
n_samples = 1000
# Create circles
X, y = make_circles(n_samples, noise=0.03, random_state=42)
print(len(X), len(y)) # 1000 1000
print(f'First 5 examples of X: \n{X[:5]}')
print(f'First 5 examples of y: \n{y[:5]}')
# Make DataFrame of circle data
circles = pd.DataFrame({'X1': X[:, 0],
                        'X2': X[:, 1],
                        'label': y})
print(circles)
# Visualize x 3
"""
plt.scatter(x=X[:, 0],
            y=X[:, 1],
            c=y,
            cmap=plt.cm.RdYlBu)
# plt.show()
"""
"""
Note: the data we're working with is often referred to as a toy dataset, a dataset
that is small enough to experiment but still sizeable enough to
practice the fundamentals



1.1 Check input and output shapes
"""
print(X.shape, y.shape) # (1000, 2) (1000,)
# View the first example of features and labels
X_sample = X[0]
y_sample = y[0]
print(f'Values for one sample of X: {X_sample} and the same for y: {y_sample}') # [0.75424625 0.23148074], 1
print(f'Shapes for one sample of X: {X_sample.shape} and the same for y: {y_sample.shape}') # (2, ), ()

"""
1.2 Turn data into tensors and create train and test splits
"""
# Turn data into tensors
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)
print(X[:5], y[:5]) # tensor
print(type(X), X.dtype, y.dtype) # <class 'torch.Tensor'> torch.float32 torch.float32
# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, # 0.2 = 20% of data will be test & 80% will be train
                                                    random_state=42)
print(len(X_train), len(X_test), len(y_train), len(y_test)) # 800 200 800 200
print(n_samples) # 1000

"""
2. Building a model
Let's build a model to classify our blue and red dots
To do so we want:
1. Setup device agnostic code so our code will run on an accellerato (GPU) if there is one
2. Construct a model (by subclassing nn.Module)
3. Define loss function and optimizer
4. Create a training and test loop
"""
# Make device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

"""
Now we've setup device agnostic code let's creat a model
1. Subbclasses nn.module (almost all models in Pytorch subclass nn.module
2. Create 2 nn.linear() laters that are capable of handling the shapes of our data
3. Define a forward() method that outlines the forward pass (or forward computation)
4. Instatiate an instance of our model class and send it to target device)
"""
# 1. Construct a model that subclass nn.Module
print(X_train.shape) # torch.Size([800, 2])
print(y_train[:5]) # tensor([1., 0., 0., 0., 1.])
class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        # 2. Create 2 nn.Linear layers capable of handling the shape of our data
        self.layer_1 = nn.Linear(in_features=2, out_features=5) # takes 2 features and upscales to 5 features
        # The next in features must match the previous out features
        self.layer_2 = nn.Linear(in_features=5, out_features=1) # takes in 5 features from previous layer and outputs a single feature (same shape as y)

        # Or you can use sequential, if is a non complex network
        # self.two_linear_layers = nn.Sequential(
        #                nn.Linear(in_features=2, out_features=5),
        #                nn.Linear(in_features=5, out_features=1))
    # 3. Define a forward() method that outlines the forward pass
    def forward(self, x):
        return self.layer_2(self.layer_1(x)) # x (input) -> layer 1 -> layer 2 -> output
        # return self.two_linear_layers(x)

# 4. Instantiate an instance of our model class and send it to the target device
model_0 = CircleModelV0().to(device)
print(model_0)

# Let's replicate the model above using nn.Sequential
model_0 = nn.Sequential(nn.Linear(in_features=2, out_features=5),
                         nn.Linear(in_features=5, out_features=1)).to(device)
print(model_0)
print(model_0.state_dict())
# Make predictions
with torch.inference_mode():
    untrained_preds = model_0(X_test.to(device))
print(f'Lenght of predictions: {len(untrained_preds)}, ' # 200
      f'Shape: {untrained_preds.shape}') # torch.Size([200, 1]
print(f'Lenght of test samples: {len(X_test)}, ' # 200
      f'Shape: {X_test.shape}') # torch.Size([200, 2]
print(f'\nFirst 10 predictions:\n{torch.round(untrained_preds[:10])}')
print(f'\nFirst 10 labels:\n{y_test[:10]}')
print(X_test[:10], y_test[:10])

"""
2.1 Setup loss function and optimizer
Which loss function or optimizer should you use?
Again... this is problem specific
For example for regression you might want MAE or MSE
For classification you might want binary cross entropy or categorical cross entropy
As a reminder, the loss function measures how wrong your models predictions are
And for optimizers two of the most common and useful are SGD and adam,
however there are many built-in options with pytorch
* For the loss function we're going to use torch.nn.BECWithLogitsLoss(),
for more check binary cross entropy (BCE) and logits in deep learning
* For different optimizers check torch.optim on pytorch site
"""
# Setup the loss function
# loss_fn = nn.BCELoss # BCELoss = requires inputs to have gone through the sigmoid activation function prior to input to BCELoss
loss_fn = nn.BCEWithLogitsLoss() # = sigmoid activation function built-in, basically sigmoid + BCELoss but more stable
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.1)
# Calculate accuracy - what percentage does our model get right
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred)) * 100
    return acc

"""
3. Train model
To train our model, we're going to need to build a training loop:
1. Forward pass
2. Calculate the loss
3. Optimizer zero grad
4. Loss backward (backpropagation)
5. Optimizer step (gradient descent)



3.1 Going from: raw logits -> prediction probabilities -> prediction labels

Our model outputs are going to be raw logits
We can convert this logits into prediction probabilities by passing them to
some kind of activation function (sigmoid for binary classification and 
softmax for multiclass classification)
Then we convert our model's prediction probabilities to prediction labels
by either rounding them or taking the argmax()
"""
# 1. View the first 5 outputs of the forward pass on the test data
# If you are doing prediction use eval and inference
model_0.eval()
with torch.inference_mode():
    y_logits = model_0(X_test.to(device))[:5]
print(y_logits)
print(y_test[:5])
# Use the sigmoid activation function on our model logits to turn them into prediction probabilities
y_pred_probs = torch.sigmoid(y_logits)
print(y_pred_probs)
"""
For our prediction probability values, we need to perform a range-style rounding on them:
* y_pred_probs >= 0.5 y=1 (class 1)
* y_pred_probs < 0.5 y=0 (class 0)
"""
print(torch.round(y_pred_probs))
# Find the predicted labels
y_preds = torch.round(y_pred_probs)
# In full (logits -> pred probs -> pred labels)
y_pred_labels = torch.round(torch.sigmoid(model_0(X_test.to(device))[:5]))
# Check for equality
print(torch.eq(y_preds.squeeze(), y_pred_labels.squeeze()))
# Get rid of extra dimension
y_preds.squeeze()
print(y_preds)

"""
3.2 Building a training and test loop
"""
torch.manual_seed(42)
torch.cuda.manual_seed(42)
# Set number of epochs
epochs = 100
# Put train data to target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)
# Build training and evaluation loop
for epoch in range(epochs):
    # Training
    model_0.train()

    # 1. Forward pass
    y_logits = model_0(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits)) # turns logits -> pred probs -> pred label

    # 2. Calculate loss/accuracy
    # loss = loss_fn(torch.sigmoid(y_logits), y_train) # nn.BCELoss expects prediction probabilities as input
    loss = loss_fn(y_logits, y_train) # nn.BCEWithLogitsLoss expects raw logits as input
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backward (backpropagation)
    loss.backward()

    # 5. Gradient descent (optimizer step)
    optimizer.step()

    # Testing
    model_0.eval()
    with torch.inference_mode():
        # 1. Forward pass
        test_logits = model_0(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        # 2. Calculate the test loss
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)
    # Print out what's happening
    if epoch % 10 == 0:
        print(f'Epoch: {epoch} | Loss: {loss:.5f}  '
              f'Acc: {acc:.2f}% | Test loss: {test_loss:.5f}  '
              f'Test acc: {test_acc:.2f}%')

"""
4. Make predictions and evaluate the model
From the metrics it looks like the model isn't learning...
So to inspect it let's make some predictions and make them visual
So like always, visualize! 
To do so we're going to import a function called plot_decision_boundary()
https://github.com/mrdbourke/pytorch-deep-learning/blob/main/helper_functions.py
"""
# Download helper functions from Learn PyTorch repo (if it's not already downloaded)
if Path('helper_functions.py').is_file():
    print('helper_functions.py already exists, skipping download')
else:
    print('Download helper_functions.py')
    request = requests.get('https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py')
    with open('helper_functions.py', 'wb') as f:
        f.write(request.content)
from helper_functions import plot_predictions, plot_decision_boundary
# Plot decision boundary of the model
"""
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Train')
plot_decision_boundary(model_0, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title('Test')
plot_decision_boundary(model_0, X_test, y_test)
# plt.show()
"""
# 1. Create models directory
MODEL_PATH = Path('/home/michel/models')
MODEL_PATH.mkdir(parents=True, exist_ok=True)
# 2. Create model save path
MODEL_NAME = 'nn_classification_model_0.pt'
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
# 3. Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_0.state_dict(),
           f=MODEL_SAVE_PATH)
"""
5. Improve a model (from a model perspective)
* Add more layers - give the model more chances to learn about patterns in the data
* Add more hidden units - go from 5 hidden units to 10 hidden units
* Fit for longer
* Changing the activation function
* Change the learning rate
* Change the loss function
These options are all from a model's perspective because they deal directly with
the model rather than the data
And because these options are all values we can change, they are referred
as "hyperparameters"
Let's try and improve our model by, one at the time:
* Adding more hidden units: 5 -> 10
* Increase the number of layers: 2 -> 3
* Increase the number of epochs: 100 -> 1000
"""
class CircleModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2,
                                 out_features=10)
        self.layer_2 = nn.Linear(in_features=10,
                                 out_features=10)
        self.layer_3 = nn.Linear(in_features=10,
                                 out_features=1)

    def forward(self, x):
        # z = self.layer_1(x)
        # z = self.layer_2(z)
        # z = self.layer_3(z)
        # return z
        return self.layer_3(self.layer_2(self.layer_1(x))) # this way of writing operations leverages speed ups where possible behind scenes

model_1 = CircleModelV1().to(device)
print(model_1)
# Create a loss function
loss_fn = nn.BCEWithLogitsLoss()
# Create an optimizer
optimizer = torch.optim.SGD(params=model_1.parameters(),
                            lr=0.1)
# Write a training and evaluation loop for model_1
torch.manual_seed(42)
torch.cuda.manual_seed(42)
# Train for longer
epochs = 1000
# Put data on the target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)
for epoch in range(epochs):
    # Training
    model_1.train()

    # 1. Forward pass
    y_logits = model_1(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))  # turns logits -> pred probs -> pred label

    # 2. Calculate loss/accuracy
    # loss = loss_fn(torch.sigmoid(y_logits), y_train) # nn.BCELoss expects prediction probabilities as input
    loss = loss_fn(y_logits, y_train)  # nn.BCEWithLogitsLoss expects raw logits as input
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backward (backpropagation)
    loss.backward()

    # 5. Gradient descent (optimizer step)
    optimizer.step()

    # Testing
    model_1.eval()
    with torch.inference_mode():
        # 1. Forward pass
        test_logits = model_1(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        # 2. Calculate the test loss
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)
    # Print out what's happening
    if epoch % 100 == 0:
        print(f'Epoch: {epoch} | '
              f'Loss: {loss:.5f}  Acc: {acc:.2f}% | '
              f'Test loss: {test_loss:.5f}  Test acc: {test_acc:.2f}%')

# Plot decision boundary of the model
"""
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Train')
plot_decision_boundary(model_1, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title('Test')
plot_decision_boundary(model_1, X_test, y_test)
# plt.show()
"""
# 1. Create models directory
MODEL_PATH = Path('/home/michel/models')
MODEL_PATH.mkdir(parents=True, exist_ok=True)
# 2. Create model save path
MODEL_NAME = 'nn_classification_model_1.pt'
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
# 3. Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_1.state_dict(),
           f=MODEL_SAVE_PATH)
"""
5.1 Preparing data to see if our model can fit a straight line
One way to troubleshoot to a larger problem is to test out a smaller problem
"""
# Create scome data (same as notebook 01)
weight = 0.7
bias = 0.3
start = 0
end = 1
step = 0.01
# Create data
X_regression = torch.arange(start, end, step).unsqueeze(dim=1)
y_regression = weight * X_regression + bias # linear regression formula (without epsilon)
# Check the data
print(len(X_regression))
print(X_regression[:5], y_regression[:5])
# Create train and test splits
train_split = int(0.8 * len(X_regression))
X_train_regression, y_train_regression = X_regression[:train_split], y_regression[:train_split]
X_test_regression, y_test_regression = X_regression[train_split:], y_regression[train_split:]
# Check the lenght of each
print(len(X_train_regression), len(X_test_regression), len(y_train_regression), len(y_test_regression))
"""
plot_predictions(train_data=X_train_regression,
                 train_labels=y_train_regression,
                 test_data=X_test_regression,
                 test_labels=y_test_regression)
# plt.show()
"""
print(X_train_regression[:10], y_train_regression[:10])
# One feature for one label
"""
5.2 Adjust model_1 to fit a straight line
"""
# Same architecture as model_1 (but using nn.Sequential())
model_2 = nn.Sequential(
    nn.Linear(in_features=1, out_features=10),
    nn.Linear(in_features=10, out_features=10),
    nn.Linear(in_features=10, out_features=1),).to(device)
print(model_2)
# Loss and optimizer
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_2.parameters(), lr=0.01)
# Train the model
torch.manual_seed(42)
torch.cuda.manual_seed(42)
# Set number of epochs
epochs = 1000
# Put the data on the target device
X_train_regression, y_train_regression = X_train_regression.to(device), y_train_regression.to(device)
X_test_regression, y_test_regression = X_test_regression.to(device), y_test_regression.to(device)
# Training
for epoch in range(epochs):
    # Training
    y_pred = model_2(X_train_regression)
    loss = loss_fn(y_pred, y_train_regression)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Testing
    model_2.eval()
    with torch.inference_mode():
        test_pred = model_2(X_test_regression)
        test_loss = loss_fn(test_pred, y_test_regression)
    if epoch % 100 == 0:
        print(f'Epoch: {epoch} | '
              f'Loss: {loss:.5f}  | '
              f'Test loss: {test_loss:.5f}')

# Turn on evaluation mode
model_2.eval()
# Make predictions (inference)
with torch.inference_mode():
    y_preds = model_2(X_test_regression)
# Plot data and predictions, check if they are on cpu
"""
plot_predictions(train_data=X_train_regression,
                 train_labels=y_train_regression,
                 test_data=X_test_regression,
                 test_labels=y_test_regression,
                 predictions=y_preds)
# plt.show()
"""
"""
6. The missing piece: non-linearity
What patterns could you draw if you were given an infinite amount of a straight
and non-straight lines?
Or in machine learning terms, an infinite (but really it is finite) of linear and
non-linear function?



6.1 Recreating non-linear data (red and blue circles)
"""
# Make and plot data
n_samples = 1000
X, y = make_circles(n_samples,
                    noise=0.03,
                    random_state=42)
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
# plt.show()
# Convert data to tensors
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)
# and then to train and test splits
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)
print(X_train[:5], y_train[:5])

"""
6.2 Building a model with non-linearity
* Linear = straight line
* Non-linear = non-straight lines

Artificial Neural Networks are a large combination of linear (straight) and
non-linear (non-straight) functions which are potentially able
to find patterns in data
"""
# Build a model with non-linear activation functions
class CircleModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2,
                                 out_features=10)
        self.layer_2 = nn.Linear(in_features=10,
                                 out_features=10)
        self.layer_3 = nn.Linear(in_features=10,
                                 out_features=1)
        self.relu = nn.ReLU() # relu is a non-linear activation function

    def forward(self, x):
        # Where should we put our non-linear activation functions? Between layers
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))

model_3 = CircleModelV2()
print(model_3)

# Setup loss and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model_3.parameters(), lr=0.1)

"""
6.3 Training a model with non linearity
"""
# Random seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)
# Put all data on target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)
# Loop through data
epochs = 1500
for epoch in range(epochs):
    # Training
    model_3.train()

    # 1. Forward pass
    y_logits = model_3(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))  # turns logits -> pred probs -> pred label

    # 2. Calculate loss/accuracy
    loss = loss_fn(y_logits, y_train)  # nn.BCEWithLogitsLoss expects raw logits as input
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backward (backpropagation)
    loss.backward()

    # 5. Gradient descent (optimizer step)
    optimizer.step()

    # Testing
    model_3.eval()
    with torch.inference_mode():
        # 1. Forward pass
        test_logits = model_3(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        # 2. Calculate the test loss
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)
    # Print out what's happening
    if epoch % 100 == 0:
        print(f'Epoch: {epoch} | '
              f'Loss: {loss:.5f}  Acc: {acc:.2f}% | '
              f'Test loss: {test_loss:.5f}  Test acc: {test_acc:.2f}%')

"""
6.4 Evaluating a model trained with non-linear activation functions
"""
# Makes predictions
model_3.eval()
with torch.inference_mode():
    y_preds = torch.round(torch.sigmoid(model_3(X_test))).squeeze()
print(y_preds[:10], y_test[:10])
# Plot decision boundaries
"""
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Train model 1')
plot_decision_boundary(model_1, X_train, y_train) # model_1 = NOT non-linearity
plt.subplot(1, 2, 2)
plt.title('Test model 3')
plot_decision_boundary(model_3, X_test, y_test) # model_3 = HAS non-linearity
plt.show()
"""
# 1. Create models directory
MODEL_PATH = Path('/home/michel/models')
MODEL_PATH.mkdir(parents=True, exist_ok=True)
# 2. Create model save path
MODEL_NAME = 'nn_classification_model_3.pt'
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
# 3. Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_3.state_dict(),
           f=MODEL_SAVE_PATH)
"""
7. Replicating non-linear activation function
Neural networks, rather than us telling the model what to learn,
we give it the tools to discorver patterns in data and it tries to
figure out the patterns on its own.
And these tools are linear & non-linear functions
"""
# Create tensor
A = torch.arange(-10, 10, 1, dtype=torch.float32)
print(A.dtype)
# Visualize the tensor
# plt.plot(A)
# plt.show()

# Let's see relu
# plt.plot(torch.relu(A))
# plt.show()
# Let's create a function which return the relu
def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.maximum(torch.tensor(0), x) # inputs must be tensors

print(relu(A))
# Plot relu activation function
# plt.plot(relu(A))
# plt.show()

# Now let's do the same to sigmoid
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

# plt.plot(torch.sigmoid(A))
# plt.show()
print(sigmoid(A))
# plt.plot(sigmoid(A))
# plt.show()

"""
8. Putting it all  together with a multi-class classification problem
* Binary classification - one thing or another (cat vs dog, spam
vs not spam, fraud vs not fraud)
* Multi-class classification - more than one thing or another (cat vs
dog vs chicken)



8.1 Creating a toy multi-class dataset
"""
# Set the hyperparameters for data creation
NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42
# 1. Create multi-class data
X_blob, y_blob = make_blobs(n_samples=1000,
                            n_features=NUM_FEATURES,
                            centers=NUM_CLASSES,
                            cluster_std=1.5, # give the clusters a little shake up
                            random_state=RANDOM_SEED)
# 2. Turn data into tensors
X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)
# 3. Split into train and test
X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob,
                                                                        y_blob,
                                                                        test_size=0.2,
                                                                        random_state=RANDOM_SEED)
# 4. Plot data (visualize x 3)
"""
plt.figure(figsize=(10, 7))
plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu)
plt.show()
"""


"""
8.2 Building a multi-class classification model in Pytorch
"""
# Create device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Build a multi-class classification model
class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
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

# Create an instance of BlobModel and send it to target device
model_4 = BlobModel(input_features=2, # must match with the features of x -> x.shape
                    output_features=4 # must match with how many classes are in the data -> torch.unique(y)
                    ).to(device)
print(model_4)

"""
8.3 Create a loss function and an optimizer for a multi-class classification model
"""
# Create a loss function for multi-class classification
# loss function measures how wrong our model's predictions are
loss_fn = nn.CrossEntropyLoss()
# Create an optimizer for multi-class classification
# optimizer updates our model parameters to try and reduce the loss
optimizer = torch.optim.SGD(params=model_4.parameters(),
                            lr=0.1) # learning rate is a hyperparameter you can change

"""
8.4 Getting prediction probabilities for a multi-class Pytorch model
In order to evaluate and train and test our model, we need to convert our model's
outputs (logits) to prediction probabilities and the to prediction labels
- Logits (raw output of the model) 
-> pred probs (use torch.softmax)
-> pred labels (take the argmax ot the prediction probabilities)
"""
# Let's get some raw outputs of our model (logits)
model_4.eval()
with torch.inference_mode():
    y_logits = model_4(X_blob_test.to(device))
print(y_logits[:10])
print(y_blob_test[:10])
# Convert our model's logit outputs to prediction probabilities
y_pred_probs = torch.softmax(y_logits, dim=1)
print(y_logits[:5])
print(y_pred_probs[:5])
# The pred probs sum makes 1, so they are probabilities
print(y_pred_probs[0])
print(torch.sum(y_pred_probs[0]))
# So we want the one with the highest probability
print(torch.max(y_pred_probs[0]))
# Get the index
print(torch.argmax(y_pred_probs[0]))
# Convert our model's prediction probabilities to prediction labels
y_preds = torch.argmax(y_pred_probs, dim=1)
print(y_preds)
print(y_blob_test)

"""
8.5 Creating a training loop and testing loop for a multi-class pytorch model
"""
# Fit the multi-class model to the data
torch.manual_seed(42)
torch.cuda.manual_seed(42)
# Set number of epochs
epochs = 100
# Put data to the target device
X_blob_train, y_blob_train = X_blob_train.to(device), y_blob_train.to(device)
X_blob_test, y_blob_test = X_blob_test.to(device), y_blob_test.to(device)
# Loop through data
for epoch in range(epochs):
    # Training
    model_4.train()

    y_logits = model_4(X_blob_train)
    # Not really necessary the softmax, argmax is enough
    y_preds = torch.softmax(y_logits, dim=1).argmax(dim=1)

    loss = loss_fn(y_logits, y_blob_train)
    acc = accuracy_fn(y_true=y_blob_train,
                      y_pred=y_preds)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Testing
    model_4.eval()
    with torch.inference_mode():
        test_logits = model_4(X_blob_test)
        test_preds = torch.softmax(test_logits, dim=1).argmax(dim=1)

        test_loss = loss_fn(test_logits, y_blob_test)
        test_acc = accuracy_fn(y_true=y_blob_test, y_pred=test_preds)

    # Print out what's happening
    if epoch % 10 == 0:
        print(f'Epoch: {epoch} | Loss: {loss:.4f}  '
              f'Acc: {acc:.2f}% | Test loss: {test_loss:.4f}  '
              f'Test acc: {test_acc:.2f}%')

"""
8.6 Making and evaluating predictions with a Pytorch multi-class model
"""
# Make predictions
model_4.eval()
with torch.inference_mode():
    y_logits = model_4(X_blob_test)
# View the first 10 predictions
print(y_logits[:10])
# Go from logits to prediction probabilities
y_pred_probs = torch.softmax(y_logits, dim=1)
print(y_pred_probs[:10])
# Go from pred probs to pred labels
y_preds = torch.argmax(y_pred_probs, dim=1)
print(y_preds[:10])
print(y_blob_test)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Train')
plot_decision_boundary(model_4, X_blob_train, y_blob_train)
plt.subplot(1, 2, 2)
plt.title('Test')
plot_decision_boundary(model_4, X_blob_test, y_blob_test)
plt.show()

# 1. Create models directory
MODEL_PATH = Path('/home/michel/models')
MODEL_PATH.mkdir(parents=True, exist_ok=True)
# 2. Create model save path
MODEL_NAME = 'nn_classification_model_4.pt'
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
# 3. Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_4.state_dict(),
           f=MODEL_SAVE_PATH)

"""
9. A few more classification metrics (to evaluate our classification model)
* Accuracy - out of 100 examples how many the model gets right
* Precision - higher means less false positives
* Recall - higher means less false negative
- There is a trade off between precision and recall
* F1-score - a combination between precision and recall
* Confusion matrix
* Classification report
"""
# Setup the metric
torchmetrics_accuracy = Accuracy('multiclass', num_classes=4).to(device)
# Calculate accuracy
print(torchmetrics_accuracy(y_preds, y_blob_test))
