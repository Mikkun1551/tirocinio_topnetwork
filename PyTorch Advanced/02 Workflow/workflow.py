"""
02. Pytorch Workflow

Let's explore an example PyTorch end-to-end workflow.
Resources:
    Ground truth notebook: https://github.com/mrdbourke/pytorch-deep-learning/blob/main/01_pytorch_workflow.ipynb
    Book version of notebook: https://www.learnpytorch.io/01_pytorch_workflow/
    Ask a question: https://github.com/mrdbourke/pytorch-deep-learning/discussions
"""
what_were_covering = {1: 'data (prepare and load)',
                      2: 'build model',
                      3: 'fitting the model to data (training)',
                      4: 'making predictions and evaluating a model (inference)',
                      5: 'saving and loading a model',
                      6: 'putting it all together'}
import torch
from torch import nn # contains all pytorch building blocks for neural networks
import matplotlib.pyplot as plt
from pathlib import Path

# Check pytorch version
print(torch.__version__)

"""
1. Data (preparing and loading)
Data can be almost anything... in machine learning
* Excel spredsheet
* Images of any kind
* Videos (Youtube has lots of data...)
* Audio like songs or podcasts
* DNA
* Text
Machine learning is a game of two parts:
1. Get data into a numerical representation
2. Build a model to learn patterns in that numerical representation
To showcase this, let's create some "known" data using linear representation formula
We'll use a linear regression formula to make a straight line with 'know' "parameters"
"""
# Create known parameters
weight = 0.7
bias = 0.3

# Create
start = 0
end = 1
step = 0.02
# A capital rapresents a matrix or a tensor, lowercase a vector
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

print(X[:10], y[:10])
print(len(X), len(y)) # 50 50

"""
Splitting data into training and test sets (one of the most important concepts in machine learning in general)
Let's create a training and test test with our data.
"""
# Create a train/test split of 80-20
train_split = int(0.8 * len(X))
print(train_split) # 40
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]
print(len(X_train), len(y_train), len(X_test), len(y_test)) # 40 40 10 10

"""
How might we better visualize our data?
This is where the data explorer's motto comes in!
"Visualize, visualize, visualize!"
"""
def plot_predictions(train_data=X_train, train_labels=y_train,
                     test_data=X_test, test_labels=y_test,
                     predictions=None):
    """
    Plots training and test data and compares predictions
    """
    plt.figure(figsize=(10, 7))
    # Plot training data in blue
    plt.scatter(train_data, train_labels, c='b', s=4, label='Training data')
    # Plot test data in green
    plt.scatter(test_data, test_labels, c='g', s=4, label='Testing data')
    # Are there predictions?
    if predictions is not None:
        # Plot the predictions if they exist
        plt.scatter(test_data, predictions, c='r', s=4, label='Predictions')
    # Show the legend
    plt.legend(prop={'size': 14})
    plt.show()

# plot_predictions()

"""
2. Build model
Our first Pytorch model!

What our model does:
* Start with random values (weight & bias)
* Look at training data and adjust the random values to better
represent (or get closer to) the ideal values (the weight & bias values
we used to create the data)

How does it do so?
Through two main algorithms:
1. Gradient descent https://youtu.be/IHZwWFHWa-w
2. Backpropagation https://youtu.be/tIeHLnjs5U8
"""
# Create a linear regression model class
class LinearRegressionModel(nn.Module): # <- almost everything in pytorch inherits from nn.Module
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.rand(1, # start with a random weight and try to adjust it to the ideal weight
                                               requires_grad=True, # can this parameter be updated via gradient descent?
                                               dtype=torch.float)) # PyTorch loves the datatype torch.float32
        self.bias = nn.Parameter(torch.rand(1, # start with a random bias and try to adjust it to the ideal bias
                                            requires_grad=True, # can this parameter be updated via gradient descent?
                                            dtype=torch.float)) # PyTorch loves the datatype torch.float32

    # Forward method to define the computation in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor: # x is the input data
        return self.weights * x + self.bias # this is the linear regression formula

"""
Pytorch model building essentials

* torch.nn - contain all of the building block for computational graphs
(neural networks can be a computational graph)
* torch.nn.Parameter - what parameters should our model try and learn,
often a Pytorch layer from torch.nn will set these for us
* torch.nn.Module - the base class for all neural network module, 
if you subclass it, you should overwrite forward()
* torch.optim - this where the optimizers in pytorch live, they will
help with gradient descent
* def forward() - all nn.module subclasses require you to overwrite
forward(), this method decides what happens in the forward computation



Checking the contents of our Pytorch model

How we've created a model, let's see what's inside...
We can check out model parameters or what's inside with .parameters()
"""
# Set a random seed
torch.manual_seed(42)
# Create an instance of the model (this is a subcalss of the nn.module)
model_0 = LinearRegressionModel()
print(model_0) # LinearRegressionModel()
# Check out the parameters
print(model_0.parameters()) # <generator object Module.parameters at 0x7bf0a5babdf0>
print(list(model_0.parameters()))
# List named parameters
print(model_0.state_dict()) # OrderedDict([('weights', tensor([0.8823])), ('bias', tensor([0.9150]))])
# We want our model weights and bias be like the ones we set at the start

"""
Making prediction using 'torch.inference_mode()'
To check our model's predictive power, let's see how well it predicts y_test based on x_test
When we pass data trought our model, it's going to run it through the forward() method
"""
# Make predictions with model
with torch.inference_mode(): # a better .no_grad()
    y_preds = model_0(X_test)
print(y_preds)
# plot_predictions(predictions=y_preds)

"""
3. Train model

The whole idea of training is for a model from some unknown parameters (may be random) to some known parameters
Or in other words from poor representation of the data to a better representation of the data
One way to measure how poor or how wrong your model predictions are is to use a loss function
* Note: Loss function may also be called cost function or criterion in different areas.
For our case, we're going to refer to it as a loss function.

Things we need to train:
* Loss function: a function to measure how wrong your model's predictions are
to the ideal outputs, lower is better.
* Optimizer: takes the loss of the model and adjusts the model's parameters
(weights & bias) to improve the loss function
- Inside the optimizer you'll often have to set two parameters:
    * params - the model parameters you'd like to optimize, for example:
    params=model_0.parameters
    * lr (learning rate) - the learning rate is a hyperparameter that defines
    how big/small the optimizer changes the parameters with each step
    (a small lr results in small changes, a large lr results in large changes)

hyperparameters - we data scientist set them
parameters - the neural network will set and modify them
And specifically for pytorch we need:
* A training loop
* A testing loop
"""
# Checkout our model's parameters
print(model_0.state_dict()) # OrderedDict([('weights', tensor([0.8823])), ('bias', tensor([0.9150]))])
# Setup a loss function
loss_fn = nn.L1Loss()
# Setup a optimizer (stocasthic gradient descent)
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.01) # learning rate = possibly the most important hyperparameter you can set

"""
Q: Which loss function and optimizer should i use?
A: This will be problem specific. But with experience you'll get an idea of
what works and doesn't in particular problem set
For example, for a regression problem (like ours), a loss of function of
nn.L1Loss() (MAE, mean absolute error) and a optimizer like torch.optimi.SGD() will suffice
But for classification problem like classifying whtever a photo is of a dog
or a cat, you'll likely want to use a loss function of nn.BCELoss() (binary cross entropy loss)



Building a training (and testing) loop in pytorch
A couple of things we need in a training loop:
0. Loop through the data
1. Forward pass (this involves data moving through our model's forward())
to make predictions on data - also called "forward propagation"
2. Calculate the loss (compare forward prediction to ground truth label)
3. Optimizer zero grad
4. Loss backward - move backwards through the network to calculate the gradients
of each parameters of our model respect to the loss (backpropagation)
5. Optimizer step - use the optimizer to adjust our model's parameters to try and
improve the loss (gradient descent)



Training
"""
torch.manual_seed(42)
# An epoch is one loop through the data, this is a hyperparameter because we set it
epochs = 101 # 86 is when stops learn
# Track different values
epoch_count = []
loss_values = []
test_loss_values = []
# 0. Loop through the data
for epoch in range(epochs):
    # Set the model to training mode
    model_0.train() # train mode in pytorch sets all parameters that require gradients to require gradients

    # 1. Forward pass
    y_pred = model_0(X_train)

    # 2. Calculate the loss
    loss = loss_fn(y_pred, y_train)
    # print(f'Loss: {loss}')

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Perform backprogation on the loss with respect to the parameters of the model
    loss.backward()

    # 5. Step the optimizer (perform gradient descent)
    optimizer.step()
    # by default how the optimizer changes will accumulate through the loop,
    # so we have to zero them above in step 3 for the next iteration of the loop

    """
    Testing
    """
    model_0.eval() # turns off different settings in the model not needed for evaluation/testing (dropout/batch norm layers)
    with torch.inference_mode(): # turn off gradient tracking & a couple of more things
        # 1. Do the forward pass
        test_pred = model_0(X_test)

        # 2. Calculate the loss
        test_loss = loss_fn(test_pred, y_test)

    # Print out what's happening
    if epoch % 10 == 0:
        epoch_count.append(epoch)
        loss_values.append(loss.item())
        test_loss_values.append(test_loss.item())
        print(f'Epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}')
        # Print out model state_dict()
        print(model_0.state_dict())

# Plot loss curve
plt.plot(epoch_count, loss_values, label='Train loss')
plt.plot(epoch_count, test_loss_values, label='Test loss')
plt.title('Training and test loss curves')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()
plt.show()
with torch.inference_mode():
    y_preds_new = model_0(X_test)
# plot_predictions(predictions=y_preds_new)

"""
Saving a model in Pytorch
There are three main methods for saving and loading models in pytorch
1. torch.save() - allows you save a pytorch object in python pickle format
2. torch.load() - allows you to load a saved pytorch obejct
3. torch.nn.Module.load_state_dict() - this allows to load a model's saved state dictionary
"""
# Saving our pytorch model
# 1. Create models directory
MODEL_PATH = Path('/home/michel/models')
MODEL_PATH.mkdir(parents=True, exist_ok=True)
# 2. Create model save path
MODEL_NAME = 'workflow_model_0.pt'
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
# 3. Save the model state_dict
print(f'Saving model to: {MODEL_SAVE_PATH}')
torch.save(obj=model_0.state_dict(), f=MODEL_SAVE_PATH)

"""
Loading a Pytorch model
Since we saver out model's state_dict() rather than the entire model, we'll create a new instante of our
model class and load the saved state_dict() into that
"""
# To load a saved state_dict we have to instantiate a new instance of our model class
loaded_model = LinearRegressionModel()
# Load the saved state_dict of model_0 (this will update the new instance with updated parameters)
print(loaded_model.state_dict()) # a new model with random values
loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
print(loaded_model.state_dict()) # now this is the model we saved
# Make some predictions with our loaded model
loaded_model.eval()
with torch.inference_mode():
    loaded_model_preds = loaded_model(X_test)
# Compare loaded model preds with original model preds
print(y_preds == loaded_model_preds)
# Make some model preds (troubleshooting)
model_0.eval()
with torch.inference_mode():
    y_preds = model_0(X_test)
print(y_preds == loaded_model_preds)

"""
Create device-agnostic code
This means if we've got access to a GPU, our code will use it (for potentially faster computing)
If no GPU is available, the code will default to using CPU
"""
# Setup device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

"""
6.1. Data
"""
# Create some data using the linear regression formula of y = weight * X + bias
weight = 0.7
bias = 0.3
# Create range values
start = 0
end = 1
step = 0.02
# Create X and y (features and labels)
X = torch.arange(start, end, step).unsqueeze(dim=1) # without unsqueeze, errors will pop up
y = weight * X + bias
print(X[:10], y[:10])
# Split data
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]
len(X_train), len(y_train), len(X_test), len(y_test)
def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=None):
    """
    Plots training data, test data and compares predictions.
    """
    plt.figure(figsize=(10, 7))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    # Are there predictions?
    if predictions is not None:
        # Plot the predictions if they exist
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # Show the legend
    plt.legend(prop={"size": 14})

# Plot the data
# Note: if you don't have the plot_predictions() function loaded, this will error
plot_predictions(X_train, y_train, X_test, y_test)

"""
6.2 Building a PyTorch Linear model
"""
# Create a linear model by subclassing nn.Module
class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        # Use nn.Linear() for creating the model parameters / also called: linear transform, probing layer, fully connected layer, dense layer
        self.linear_layer = nn.Linear(in_features=1,
                                      out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)

# Set the manual seed
torch.manual_seed(42)
model_1 = LinearRegressionModelV2()
model_1, model_1.state_dict()
print(model_1.state_dict())
print(X_train[:5], y_train[:5])
# Check the model current device
print(next(model_1.parameters()).device)
# Set the model to use the target device
model_1.to(device)
print(next(model_1.parameters()).device)
print(model_1.state_dict())

"""
6.3 Training
For training we need:
* Loss function
* Optimizer
* Training loop
* Testing loop
"""
# Setup loss function
loss_fn = nn.L1Loss() # same as MAE
# Setup our optimizer
optimizer = torch.optim.SGD(params=model_1.parameters(),
                            lr=0.01)
# Let's write a training loop
torch.manual_seed(42)
epochs = 200
# Put data on the target device (device agnostic code for data)
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)
for epoch in range(epochs):
  model_1.train()

  # 1. Forward pass
  y_pred = model_1(X_train)

  # 2. Calculate the loss
  loss = loss_fn(y_pred, y_train)

  # 3. Optimizer zero grad
  optimizer.zero_grad()

  # 4. Perform backpropagation
  loss.backward()

  # 5. Optimizer step
  optimizer.step()

  ### Testing
  model_1.eval()
  with torch.inference_mode():
    test_pred = model_1(X_test)
    test_loss = loss_fn(test_pred, y_test)

  # Print out what's happening
  if epoch % 10 == 0:
    print(f"Epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}")

print(model_1.state_dict())
print(weight, bias)

"""
6.4 Making and evaluating predictions
"""
# Turn model into evaluation mode
model_1.eval()
# Make predictions on the test data
with torch.inference_mode():
  y_preds = model_1(X_test)

print(y_preds)
# Check out our model predictions visually
plot_predictions(predictions=y_preds.cpu())

"""
6.5 Saving & loading a trained model
"""
# 1. Create models directory
MODEL_PATH = Path('/home/michel/models')
MODEL_PATH.mkdir(parents=True, exist_ok=True)
# 2. Create model save path
MODEL_NAME = 'workflow_model_1.pt'
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
# 3. Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_1.state_dict(),
           f=MODEL_SAVE_PATH)
print(model_1.state_dict())
# Load a PyTorch model
# Create a new instance of lienar regression model V2
loaded_model_1 = LinearRegressionModelV2()
# Load the saved model_1 state_dict
loaded_model_1.load_state_dict(torch.load(MODEL_SAVE_PATH))
# Put the loaded model to device
loaded_model_1.to(device)
print(next(loaded_model_1.parameters()).device)
loaded_model_1.state_dict()
# Evaluate loaded model
loaded_model_1.eval()
with torch.inference_mode():
  loaded_model_1_preds = loaded_model_1(X_test)
print(y_preds == loaded_model_1_preds)
