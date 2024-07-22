"""
Exercises & Extra-curriculum

See exercises for this notebook here: https://www.learnpytorch.io/03_pytorch_computer_vision/#exercises
"""
import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from timeit import default_timer as timer
from tqdm.auto import tqdm
import random
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
from helper_functions import accuracy_fn

# Setup device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

"""
1. What are 3 areas in industry where computer vision is currently being used?
"""
# Self-driving cars, such as Tesla using computer vision to percieve
# what's happening on the road. See Tesla AI day for more - https://youtu.be/j0z4FweCy4M

# Healthcare imaging, such as using computer vision to help interpret X-rays.
# Google also uses computer vision for detecting polyps in the intenstines -
# https://ai.googleblog.com/2021/08/improved-detection-of-elusive-polyps.html

# Security, computer vision can be used to detect whether someone is invading your
# home or not - https://store.google.com/au/product/nest_cam_battery?hl=en-GB

"""
2. Search "what is overfitting in machine learning" and write down a sentence about what you find.
"""
# Overfitting is the fenomena where the neural network performs very well during
# the training phase but poorly during the testing phase, so with unseen data

"""
3. Search "ways to prevent overfitting in machine learning", 
write down 3 of the things you find and a sentence about each. 
Note: there are lots of these, so don't worry too much about all of them, just pick 3 and start with those.
"""
# Regularization techniques - You could use dropout on your neural networks), dropout involves randomly
# removing neurons in different layers so that the remaining neurons hopefully learn more robust weights/patterns.

# Use a different model - maybe the model you're using for a specific problem is too complicated, as in,
# it's learning the data too well because it has so many layers. You could remove some layers to simplify your model.
# Or you could pick a totally different model altogether, one that may be more suited to your particular problem.
# Or... you could also use transfer learning (taking the patterns from one model and applying them to your own problem)

# Reduce noise in data/cleanup dataset/introduce data augmentation techniques - If the model is learning the data
# too well, it might be just memorizing the data, including the noise. One option would be to remove the noise/clean up
# the dataset or if this doesn't, you can introduce artificial noise through the use of data augmentation to
# artificially increase the diversity of your training dataset.

"""
4. Spend 20-minutes reading and clicking through the CNN Explainer website.
"""
# Upload your own example image using the "upload" button and see what happens
# in each layer of a CNN as your image passes through it.
# https://poloclub.github.io/cnn-explainer/

"""
5. Load the torchvision.datasets.MNIST() train and test datasets.
"""
train_data = datasets.MNIST(
    root='/home/michel/Desktop/TopNetwork/08: PyTorch/PYTORCH_NOTEBOOKS/Data', # where to download data to?
    train=True, # do we want the training set? If false will be the test set
    download=True, # do we want to download yes/no?
    transform=torchvision.transforms.ToTensor(), # how do we want to transform the data?
    target_transform=None) # how do we want to transform the labels/targets?

test_data = datasets.MNIST(
    root='/home/michel/Desktop/TopNetwork/08: PyTorch/PYTORCH_NOTEBOOKS/Data',
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform=None)

"""
6. Visualize at least 5 different samples of the MNIST training dataset.
"""
for i in range(5):
  img = train_data[i][0]
  img_squeeze = img.squeeze()
  label = train_data[i][1]
  # plt.figure(figsize=(3, 3))
  # plt.imshow(img_squeeze, cmap="gray")
  # plt.title(label)
  # plt.axis(False)
  # plt.show()

"""
7. Turn the MNIST train and test datasets into dataloaders using 
torch.utils.data.DataLoader, set the batch_size=32
"""
BATCH_SIZE = 32
train_dataloader = DataLoader(dataset=train_data, # which dataset to use
                              batch_size=BATCH_SIZE, # batch size
                              shuffle=True) # if we want the data shuffle every time
test_dataloader = DataLoader(dataset=test_data,
                              batch_size=BATCH_SIZE,
                              shuffle=False)

"""
8. Recreate model_2 used in this notebook (the same model from the CNN Explainer website,
also known as TinyVGG) capable of fitting on the MNIST dataset.
"""
class MNIST(nn.Module):
    def __init__(self, input_shape:int,
                 hidden_units: int,
                 output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*7*7,
                      out_features=output_shape))

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x

class_names = train_data.classes
model = MNIST(1, 10, len(class_names))

"""
9. Train the model you built in exercise 8. on CPU and GPU and see how long it takes on each.
"""
def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
    """Performs a training with model trying to learn on data_loader"""
    train_loss, train_acc = 0, 0
    # Put the model into training mode
    model.train()
    # Add a loop to through the training batches
    for batch, (X, y) in enumerate(data_loader):
        # Put data on target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass (outputs the raw logits from the model)
        y_pred = model(X)

        # 2. Calculate loss and accuracy (per batch)
        loss = loss_fn(y_pred, y)
        train_loss += loss # accumulate train loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred.argmax(dim=1)) # from logits to prediction labels

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5.Optimizer step
        optimizer.step()

    # Divide total train loss and acc by lenght of train dataloader
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f'Train loss: {train_loss:.5f} | Train acc: {train_acc:.2f}%')

def test_step(model: torch.nn.Module,
             data_loader: torch.utils.data.DataLoader,
             loss_fn: torch.nn.Module,
             accuracy_fn,
             device: torch.device = device):
    """Performs a testing loop step on model going over data_loader"""
    test_loss, test_acc = 0, 0
    # Put the model in eval mode
    model.eval()
    # Turn on inference mode context manager
    with torch.inference_mode():
        for X, y in data_loader:
            # Send the data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred = model(X)

            # 2. Calculate the loss/acc
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y,
                                    y_pred=test_pred.argmax(dim=1)) # from logits to prediction labels

        # Adjust metrics and print out
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f'Test loss: {test_loss:.5f} | Test acc: {test_acc:.2f}%\n')

model = model.to('cpu')
epochs = 3
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
time_start_cpu = timer()
for epoch in tqdm(range(epochs)):
    print(f'Epoch: {epoch}\n------')
    train_step(model=model,
               data_loader=train_dataloader,
               loss_fn=loss_fn,
               optimizer=optimizer,
               accuracy_fn=accuracy_fn,
               device='cpu')
    test_step(model=model,
               data_loader=test_dataloader,
               loss_fn=loss_fn,
               accuracy_fn=accuracy_fn,
               device='cpu')

time_end_cpu = timer()
total_cpu_time = time_end_cpu - time_start_cpu

model_gpu = MNIST(1, 10, len(class_names)).to(device)
epochs = 3
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_gpu.parameters(), lr=0.1)
time_start_gpu = timer()
for epoch in tqdm(range(epochs)):
    print(f'Epoch: {epoch}\n------')
    train_step(model=model_gpu,
               data_loader=train_dataloader,
               loss_fn=loss_fn,
               optimizer=optimizer,
               accuracy_fn=accuracy_fn,
               device=device)
    test_step(model=model_gpu,
               data_loader=test_dataloader,
               loss_fn=loss_fn,
               accuracy_fn=accuracy_fn,
               device=device)

time_end_gpu = timer()
total_gpu_time = time_end_gpu - time_start_gpu

print(f'Time CPU: {total_cpu_time}')
print(f'Time GPU: {total_gpu_time}')

"""
10. Make predictions using your trained model and visualize at least 5 
of them comparing the prediciton to the target label.
"""
def make_predictions(model: torch.nn.Module,
                     data: list,
                     device: torch.device = device):
  pred_probs = []
  model.eval()
  with torch.inference_mode():
    for sample in data:
      sample = torch.unsqueeze(sample, dim=0).to(device)
      pred_logits = model(sample)
      pred_prob = torch.softmax(pred_logits.squeeze(), dim=0)
      pred_probs.append(pred_prob.cpu())
  return torch.stack(pred_probs)

test_samples = []
test_labels = []
for sample, label in random.sample(list(test_data), k=5):
  test_samples.append(sample)
  test_labels.append(label)

pred_probs = make_predictions(model=model,
                              data=test_samples)
pred_classes = pred_probs.argmax(dim=1)
plt.figure(figsize=(4, 10))
nrows = 5
ncols = 1
for i, sample in enumerate(test_samples):
    plt.subplot(nrows, ncols, i+1)
    plt.imshow(sample.squeeze(), cmap='gray')
    pred_label = class_names[pred_classes[i]]
    truth_label = class_names[test_labels[i]]
    title_text = f'Pred: {pred_label} | Truth: {truth_label}'
    if pred_label == truth_label:
        plt.title(title_text, fontsize=10, c='g')
    else:
        plt.title(title_text, fontsize=10, c='r')
    plt.axis(False)
plt.show()

"""
11. Plot a confusion matrix comparing your model's predictions to the truth labels.
"""
y_preds = []
model.eval()
with torch.inference_mode():
    for X, y in tqdm(test_dataloader, desc='Making predictions...'):
        X, y = X.to(device), y.to(device)
        y_logit = model(X)
        y_pred = torch.softmax(y_logit.squeeze(), dim=0).argmax(dim=1)
        y_preds.append(y_pred.cpu())
    y_pred_tensor = torch.cat(y_preds)
    # print(y_pred_tensor, len(y_pred_tensor))

confmat = ConfusionMatrix(num_classes=len(class_names),
                          task="multiclass")
confmat_tensor = confmat(preds=y_pred_tensor,
                         target=test_data.targets)
fig, ax = plot_confusion_matrix(conf_mat=confmat_tensor.numpy(),
                                class_names=class_names,
                                figsize=(10, 7))
plt.show()

"""
12. Create a random tensor of shape [1, 3, 64, 64] and pass it through a nn.Conv2d() 
layer with various hyperparameter settings (these can be any settings you choose), 
what do you notice if the kernel_size parameter goes up and down?
"""
rand_tensor= torch.rand(1, 3, 64, 64)
conv_layer = nn.Conv2d(in_channels=3,
                      out_channels=64,
                      kernel_size=3,
                      stride=2,
                      padding=1)
output = conv_layer(rand_tensor)
print(rand_tensor.shape)
print(conv_layer)
print(output.shape)
# The shape of the width and height changes according to the kernel size, higher it is, lower the height and width

"""
13. Use a model similar to the trained model_2 from this notebook to make predictions 
on the test torchvision.datasets.FashionMNIST dataset.
"""
test_data = datasets.FashionMNIST(
    root='/home/michel/Desktop/TopNetwork/08: PyTorch/PYTORCH_NOTEBOOKS/Data',
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform=None)

class_names = test_data.classes

test_samples = []
test_labels = []
for sample, label in random.sample(list(test_data), k=5):
  test_samples.append(sample)
  test_labels.append(label)

pred_probs = make_predictions(model=model,
                              data=test_samples)
pred_classes = pred_probs.argmax(dim=1)
plt.figure(figsize=(4, 10))
nrows = 5
ncols = 1
# Then plot some predictions where the model was wrong alongside what the label of the image should've been.
for i, sample in enumerate(test_samples):
    plt.subplot(nrows, ncols, i+1)
    plt.imshow(sample.squeeze(), cmap='gray')
    pred_label = class_names[pred_classes[i]]
    truth_label = class_names[test_labels[i]]
    title_text = f'Pred: {pred_label} | Truth: {truth_label}'
    if pred_label == truth_label:
        plt.title(title_text, fontsize=10, c='g')
    else:
        plt.title(title_text, fontsize=10, c='r')
    plt.axis(False)
plt.show()

# After visualing these predictions do you think it's more of a modelling error or a data error?
#
# The model isn't performing well because it wasn't train for recognize images of clothing, it's a modelling error


# As in, could the model do better or are the labels of the data too close
# to each other (e.g. a "Shirt" label is too close to "T-shirt/top")?
#
# It could, but it also performs poorly on different types of clothing so is not the case
