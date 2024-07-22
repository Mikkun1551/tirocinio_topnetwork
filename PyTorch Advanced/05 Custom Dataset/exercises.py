"""
Exercises & Extra-curriculum

See exercises for this notebook here: https://www.learnpytorch.io/04_pytorch_custom_datasets/#exercises
"""
import torch
from torch import nn
import requests
import zipfile
from pathlib import Path
import os
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from typing import Tuple, Dict, List, Any
from torch.utils.data import Dataset
from torchinfo import summary
from tqdm.auto import tqdm
from timeit import default_timer as timer
import pandas as pd
import torchvision.io

# Setup device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

"""
1. Our models are underperforming (not fitting the data well). 
What are 3 methods for preventing underfitting? 
Write them down and explain each with a sentence.
"""
# 1. Add more layers - your model it could be too simple for the data you are giving it
# 2. Train for longer - give more time to your model for training and learn more
# 3. Transfer learning - it helps by leveraging already existing patterns
# from one model and using them for your problems

"""
2. Recreate the data loading functions we built in sections 1, 2, 3 and 4.
You should have train and test DataLoader's ready to use.
"""
# 1. Get data
# Setup path to data folder
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

# If the image folder doesn't exist, download it and prepare it...
if image_path.is_dir():
  print(f"{image_path} directory exists.")
else:
  print(f"Did not find {image_path} directory, creating...")
  image_path.mkdir(parents=True, exist_ok=True)

# Download pizza, steak, sushi data (images from GitHub)
with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
  request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
  print("Downloading pizza, steak, sushi data...")
  f.write(request.content)

# Unzip pizza, steak, sushi data
with zipfile.ZipFile(data_path/"pizza_steak_sushi.zip", "r") as zip_ref:
  print(f"Unzipping pizza, steak, suhsi data to {image_path}")
  zip_ref.extractall(image_path)

# 2. Become one with the data
def walk_through_dir(dir_path):
    """Walks through dir_path returning its contents"""
    for dirpath, dirnames, filesnames in os.walk(dir_path):
        print(f'There are {len(dirnames)} directories and {len(filesnames)} images in "{dirpath}"')

# Setup train and testing paths
train_dir = image_path / "train"
test_dir = image_path / "test"

# 1. Get all image paths (* means "any combination")
image_path_list = list(image_path.glob("*/*/*.jpg"))
print(image_path_list[:3])

# 2. Get random image path
random_image_path = random.choice(image_path_list)
print(random_image_path)

# 3. Get image class from path name
image_class = random_image_path.parent.stem
print(image_class)

# 4. Open image
img = Image.open(random_image_path)

# Print metadata
print(f"Random image path: {random_image_path}")
print(f"Image class: {image_class}")
print(f"Image height: {img.height}")
print(f"Image width: {img.width}")

# Turn the image into an array
img_as_array = np.asarray(img)

# Plot the image
plt.figure(figsize=(10, 7))
plt.imshow(img_as_array)
plt.title(f"Image class: {image_class} | Image shape: {img_as_array.shape} -> [height, width, color_channels]")
plt.axis(False)

# 3.1 Transforming data with torchvision.transforms
# Write transform for turning images into tensors
data_transform = transforms.Compose([
  # Resize the images to 64x64x3 (64 height, 64 width, 3 color channels)
  transforms.Resize(size=(64, 64)),
  # Flip the images randomly on horizontal
  transforms.RandomHorizontalFlip(p=0.5),
  # Turn the image into a torch.Tensor
  transforms.ToTensor()])  # converts all pixel values from 0-255 to be between 0-1

def plot_transformed_images(image_paths, transform, n=3, seed=None):
  """Selects random images from a path of images and load/tranforms
  them then plots the original vs the transformed version"""
  if seed:
    random.seed(seed)
  random_image_paths = random.sample(image_paths, k=n)
  for image_path in random_image_paths:
    with Image.open(image_path) as f:
      fig, ax = plt.subplots(nrows=1, ncols=2)
      ax[0].imshow(f)
      ax[0].set_title(f'Original\nSize: {f.size}')
      ax[0].axis(False)

      # Transform and plot target image
      tranformed_image = transform(f).permute(1, 2, 0)
      # Transform puts the colors first but plt wants them last
      ax[1].imshow(tranformed_image)
      ax[1].set_title(f'Transformed\nShape: {tranformed_image.shape}')
      ax[1].axis('off')
      fig.suptitle(f'Class: {image_path.parent.stem}', fontsize=16)
      plt.show()

plot_transformed_images(image_path_list, transform=data_transform, n=3)

# Use ImageFolder to create dataset(s)
train_data = datasets.ImageFolder(root=str(train_dir), # target folder of images
                                  transform=data_transform, # transforms to perform on data (images)
                                  target_transform=None) # transforms to perform on labels (if necessary)

test_data = datasets.ImageFolder(root=str(test_dir),
                                 transform=data_transform,
                                 target_transform=None)

# Get class names as a list
class_names = train_data.classes

# Can also get class names as a dict
class_dict = train_data.class_to_idx

# Check the lengths
len(train_data), len(test_data)

# Turn train and test Datasets into DataLoaders
from torch.utils.data import DataLoader
BATCH_SIZE = 1
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              num_workers=os.cpu_count(),
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             num_workers=os.cpu_count(),
                             shuffle=False)

# How many batches of images are in our data loaders?
len(train_dataloader), len(test_dataloader)

img, label = next(iter(train_dataloader))

# Batch size will now be 1, try changing the batch_size parameter above and see what happens
print(f"Image shape: {img.shape} -> [batch_size, color_channels, height, width]")
print(f"Label shape: {label.shape}")

"""
3. Recreate model_0 we built in section 7
"""
class TinyVGG(nn.Module):
  """
  Model architecture that replicates the TinyVGG
  model from CNN explainer website
  """
  def __init__(self, input_shape: int,
               hidden_units: int,
               output_shape: int) -> None:
    super().__init__()
    #  Create a convolutional block
    self.conv_block_1 = nn.Sequential(
      nn.Conv2d(in_channels=input_shape,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,  # stride 1 removes 2 pixels from the image
                padding=0),
      nn.ReLU(),
      nn.Conv2d(in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=0),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2,
                   stride=2))  # default is same like kernel_size
    self.conv_block_2 = nn.Sequential(
      nn.Conv2d(in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=0),
      nn.ReLU(),
      nn.Conv2d(in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=0),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2,  # max pool of 2 halves the pixel in the image
                   stride=2))
    self.classifier = nn.Sequential(
      nn.Flatten(),
      # check output shape of conv_block_2, or use torchinfo
      nn.Linear(in_features=hidden_units * 16 * 16,
                out_features=output_shape))

  def forward(self, x):
    # Benefits from operator fusion, aka speeds up gpu performance
    # https://horace.io/brrr_intro.html
    return self.classifier(self.conv_block_2(self.conv_block_1(x)))

model_0 = TinyVGG(3, 10, len(class_names)).to(device)

"""
4. Create training and testing functions for model_0.
"""
def train_step_no_acc_fn(model: torch.nn.Module,
                         data_loader: torch.utils.data.DataLoader,
                         loss_fn: torch.nn.Module,
                         optimizer: torch.optim.Optimizer,
                         # accuracy_fn,
                         device: torch.device = device):
  """Performs a training with model trying to learn on data_loader.
  This version doesn't take an accuracy function as parameter.
  It also returns train loss and accuracy"""
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
    train_loss += loss.item()  # accumulate train loss
    # train_acc += accuracy_fn(y_true=y,
    #                         y_pred=y_pred.argmax(dim=1)) # from logits to prediction labels

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backward
    loss.backward()

    # 5.Optimizer step
    optimizer.step()

    # Calculate accuracy metric (without pre-built function)
    y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
    train_acc += (y_pred_class == y).sum().item() / len(y_pred)

  # Divide total train loss and acc by lenght of train dataloader
  train_loss /= len(data_loader)
  train_acc /= len(data_loader)
  return train_loss, train_acc


def test_step_no_acc_fn(model: torch.nn.Module,
                        data_loader: torch.utils.data.DataLoader,
                        loss_fn: torch.nn.Module,
                        # accuracy_fn,
                        device: torch.device = device):
  """Performs a testing loop step on model going over data_loader
  This version doesn't take an accuracy function as parameter.
  It also returns test loss and accuracy"""
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
      test_loss += loss_fn(test_pred, y).item()
      # test_acc += accuracy_fn(y_true=y,
      #                         y_pred=test_pred.argmax(dim=1)) # from logits to prediction labels

      # Calculate the accuracy
      test_pred_labels = test_pred.argmax(dim=1)
      test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)

    # Adjust metrics and print out
    test_loss /= len(data_loader)
    test_acc /= len(data_loader)
    return test_loss, test_acc

"""
5. Try training the model you made in exercise 3 for 
5, 20 and 50 epochs, what happens to the results?
"""
# Use torch.optim.Adam() with a learning rate of 0.001 as the optimizer.
def train(model: torch.nn.Module,
          train_dataloader,
          test_dataloader,
          optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5,
          device=device):
  """Performs training and evaluation of a model if given
  a train and test function"""

  # 2. Create empty results dictionary
  results = {'train_loss': [],
             'train_acc': [],
             'test_loss': [],
             'test_acc': []}

  # 3. Loop through training and testing steps for a number of epochs
  for epoch in tqdm(range(epochs)):
    train_loss, train_acc = train_step_no_acc_fn(model=model,
                                                 data_loader=train_dataloader,
                                                 loss_fn=loss_fn,
                                                 optimizer=optimizer,
                                                 device=device)
    test_loss, test_acc = test_step_no_acc_fn(model=model,
                                              data_loader=test_dataloader,
                                              loss_fn=loss_fn,
                                              device=device)

    # 4. Print out what's happening
    print(f'Epoch: {epoch}\n'
          f'Train loss: {train_loss:.4f} Train acc: {train_acc * 100:.2f}%\n'
          f'Test loss: {test_loss:.4f} Test acc: {test_acc * 100:.2f}%\n')

    # 5. Update results dictionary
    results['train_loss'].append(train_loss)
    results['train_acc'].append(train_acc)
    results['test_loss'].append(test_loss)
    results['test_acc'].append(test_acc)

  # 6. Return the filled results at the end of the epochs
  return results

model_5 = TinyVGG(3, 10, len(class_names)).to(device)

# Set number of epochs
NUM_EPOCHS = 5

# Setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.001)

# Start the timer
start_time = timer()

# Train model_0
model_5_results = train(model=model_5,
                        train_dataloader=train_dataloader,
                        test_dataloader=test_dataloader,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        epochs=NUM_EPOCHS)

# End timer and print out how long it took
end_time = timer()
total_time = end_time - start_time
print(f'Total training time: {total_time:.2f} seconds for 5 epochs')

model_20 = TinyVGG(3, 10, len(class_names)).to(device)
NUM_EPOCHS = 20
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.001)
start_time = timer()
model_20_results = train(model=model_20,
                        train_dataloader=train_dataloader,
                        test_dataloader=test_dataloader,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        epochs=NUM_EPOCHS)
end_time = timer()
total_time = end_time - start_time
print(f'Total training time: {total_time:.2f} seconds for 20 epochs')

model_50 = TinyVGG(3, 10, len(class_names)).to(device)
NUM_EPOCHS = 50
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.001)
start_time = timer()
model_50_results = train(model=model_50,
                        train_dataloader=train_dataloader,
                        test_dataloader=test_dataloader,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        epochs=NUM_EPOCHS)
end_time = timer()
total_time = end_time - start_time
print(f'Total training time: {total_time:.2f} seconds for 50 epochs')

"""
6. Double the number of hidden units in your model 
and train it for 20 epochs, what happens to the results?
"""

"""
7. Double the data you're using with your model and train it 
for 20 epochs, what happens to the results?
* Note: You can use the custom data creation notebook to scale up your Food101 dataset.
* You can also find the already formatted double data 
(20% instead of 10% subset) dataset on GitHub,
you will need to write download code like in exercise 2 to get it into this notebook.
"""

"""
8. Make a prediction on your own custom image of pizza/steak/sushi 
(you could even download one from the internet) and share your prediction.
"""
# Does the model you trained in exercise 7 get it right?

# If not, what do you think you could do to improve it?
