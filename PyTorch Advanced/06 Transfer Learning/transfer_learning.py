"""
06. PyTorch Transfer Learning
What is transfer learning?
Transfer learning involves taking the parameters of what one model has
learned on another dataset and applying to our own problem
* Pre-trained model = foundation models

After getting the right versions let's import the code we've written
in previous sections so that we don't have to write it all again
"""
import torch # version -> 2.2.2+cu121 | we wanted 1.12+
import torchvision # version -> '0.17.2+cu121' | we wanted 0.13+
import torchinfo
from torch import nn
from torchvision import transforms
from torchinfo import summary
import matplotlib.pyplot as plt
import os
import requests
import zipfile
from pathlib import Path
from timeit import default_timer as timer
from typing import List, Tuple
from PIL import Image
import random
# Import python files with useful functions
import data_setup
import engine
from helper_functions import plot_loss_curves


# Setup device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
1. Get data
We need our pizza, steak, sushi data to build a transfer learning model on
"""
# Setup path to a data folder
data_path = Path('/home/michel/datasets')
image_path = data_path / 'pizza_steak_sushi'

# If the image folder doesn't exist, download it and prepare it...
if image_path.is_dir():
    print(f'The given directory already exists, skipping download')
else:
    print(f"The given directory doesn't exist, downloading now...")
    image_path.mkdir(parents=True, exist_ok=True)
    # Download pizza, steak and sushi data
    with open(data_path / 'pizza_steak_sushi.zip', 'wb') as f:
        request = requests.get('https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip')
        print('Downloading pizza, steak, sushi data...')
        f.write(request.content)
    # Unzip pizza, steak, sushi data
    with zipfile.ZipFile(data_path / 'pizza_steak_sushi.zip', 'r') as zip_ref:
        print('Unzipping pizza, steak and sushi data...')
        zip_ref.extractall(image_path)
    # Remove .zip file
    os.remove(data_path / 'pizza_steak_sushi.zip')

# Setup directory path
train_dir = image_path / 'train'
test_dir = image_path / 'test'

"""
2. Create Datasets and Dataloaders
Now we got some data, want to turn it into Pytorch DataLoaders.
To do so, we can use data_setup.py and the create_dataloaders() function
There's one thing we have to think about when loading: how to transform it?
And with torchvision there's two ways to do this:
1. Manually create transforms - we define what transforms you want
you data to go through
2. Automatically created transforms - the transforms your data are
defined by the model you'd like to use

Important point: when using a pretrained model, it's important that the
data (including your custom data) that you pass through it is transformed
in the same way that the data the model was trained on



2.1 Creating a transform for torchvision.models (manual creation)
torchvision.model contains pretrained models (ready for transfer learning)
We need to normalize the images with a standard, 
in this case we use the standard of ImageNet
"""
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
manual_transoforms = transforms.Compose([
    transforms.Resize((224, 224)), # resize the image
    transforms.ToTensor(), # get images into range 0 to 1
    normalize]) # make sure images have the same distribution
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=str(train_dir),
                                                                               test_dir=str(test_dir),
                                                                               transform=manual_transoforms,
                                                                               batch_size=32)
# print(train_dataloader, test_dataloader, class_names)

"""
2.2 Creating a transform from torchvision.models (auto creation)
There is the possibility for automatic data transform creation based
on the pretrained model weights you're using
"""
# Get a set of pretrained model weights
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT # default = best available weights

# Get the transforms used to create our pretrained weights
auto_transforms = weights.transforms()
print(auto_transforms)

# Create Dataloaders usign automatic transform
# train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=str(train_dir),
#                                                                                test_dir=str(test_dir),
#                                                                                transform=auto_transforms,
#                                                                                batch_size=32)
# print(train_dataloader, test_dataloader, class_names)

"""
3. Getting a pretrained model
There are various places to get a pretrained model, such as:
-  PyTorch domains libraries (torchvision, torchtext, torchaudio, torchrec)
-  Torch image models (timm library)
-  HuggingFace Hub
-  Paperswithcode SOTA



3.1 Which pretrained model shoul you use?
Experiment, experiment... experiment!
The whole idea of transfer learning: take an already well-performing
model from a problem space similar to your own and then customize to
your own problem

Thing to consider are:
1. Speed - how fast does it need to runs?
2. Size - how big is the model?
3. Performance - how well does it go on your chosen problem?

Where does the model live? On the device? (self driving car)
Or on a server? Which model should we chose?
For our case (deploying FoodVision Mini on a mobile device),
EffNetB0 is one of our best options in terms of performance vs size
However, in light of the Bitter Lesson, if we had infinite compute,
we'd likely pick the biggest model + most parameters + most general
we could - http://www.incompleteideas.net/IncIdeas/BitterLesson.html



3.2 Setting up a pretrained model
Want to create an instance of pretrained EffNetB0 -
https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b0.html
When setting up a pretrained model we only modify the input and output layer,
this is called feature extraction, usefull for small amount of data
If you have large amount of data you will do also fine-tuning, you might
change the input layer but will change some of the last layers
"""
# How to setup a pretrained model
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
model = torchvision.models.efficientnet_b0(weights=weights)

"""
3.3 Getting a summary of our model with torchinfo.summary() for feature extraction
"""
summary(model=model,
        input_size=(1, 3, 224, 224), # BS, C, H, W
        col_names=['input_size', 'output_size', 'num_params', 'trainable'],
        col_width=20,
        row_settings=['var_names'])

"""
3.4 Freezing the base model and changing the output layer to suit our needs
With a feature extractor model, typically you will 'freeze' the base layers of a
pretrained/foundation model and update the output layers to suit your own problem
"""
# Freeze all the base layers in EffNetB0
for param in model.features.parameters():
    # print(param)
    param.requires_grad = False

# Update the classifier head of our model to suit our problem
print(model.classifier)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True), # the same of the original
    # In our case we need to change only the out_features to the 3 foods
    nn.Linear(in_features=1280, # feature vector coming in
              out_features=len(class_names))) # how many classes do we have (3)
print(model.classifier)

"""
4. Train model
"""
# Define loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 0.001)

# Set manual seed
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Start the timer
start_time = timer()

# Setup training and save the results
results = engine.train(model=model,
                       train_dataloader=train_dataloader,
                       test_dataloader=test_dataloader,
                       optimizer=optimizer,
                       loss_fn=loss_fn,
                       epochs=5, # it stabilize around 5 epochs
                       device=device)

# Calculate total time
end_time = timer()
total_time = end_time - start_time
# On CPU the model is slow, should pass it on the GPU
print(f'Training took: {total_time:.2f} seconds')

"""
5. Evaluate model by plotting loss curves
"""
# For use some pre-build functions
try:
    from helper_functions import plot_loss_curves
except:
    print(f"Couldn't find helper functions.py, downloading...")
    with open('helper_functions.py', 'wb') as f:
        import requests
        request = requests.get('https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py')
        f.write(request.content)
    from helper_functions import plot_loss_curves

# Plot the loss curves of our model
# plot_loss_curves(results)

"""
6. Make predictions on images from the test set
Some things to keep in mind when making predictions/inference on test
data/custom data
We have to make sure that our test/custom data is:
* Same shape - images need to be same shape as model was trained on
* Same datatype - custom data should be on the same datatype
* Same device - data/custom data should be on the same device as model
* Same transform - if you've transformed your custom data, ideally you
will transform the test data and custom data the same

To do all of this automatically, let's call the function pred_and_plot_image
1. Take in a trained model, a list of class names, a filepath to a target image,
an image size, a transform and a target device
2. Open the image with PIL.Image.Open()
3. Create a transform if one doesn't exist
4. Make sure the model is on the target device
5. Turn the model to model.eval() mode to make sure it's ready for inference
(this will turn off things like nn.Dropout())
6. Transform the target image and make sure its dimensionality is suited for the
model (this mainly relates to batch size)
7. Make prediction on the image by passing to the model
8. Convert the model's output logits to prediction probabilities using
torch.softmax()
9. Convert model's prediction probabilities to prediction labels using
torch.argmax()
10. Plot the image with matplotlib and set the title to the prediction
label from step 9 and prediction probability from step 8
"""
# 1. Take in a trained model
def pred_and_plot_image(model: torch.nn.Module,
                        image_path: str,
                        class_names: List[str],
                        image_size: Tuple[int, int] = (224, 224),
                        transform: torchvision.transforms = None,
                        device: torch.device=device):
    # 2. Open the image with PIL
    img = Image.open(image_path)

    # 3. Create a transform if one doesn't exist
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

    # Predict on image #
    # 4. Make sure the model is on the target device
    model.to(device)

    # 5. Turn on inference mode and eval mode
    model.eval()
    with torch.inference_mode():
        # 6. Transform the image and add an extra batch dimension
        transformed_image = image_transform(img).unsqueeze(dim=0) # BS, C, H, W

        # 7. Make a prediction on the transformed image by passing it to the model
        # also pass it on the same device
        target_image_pred = model(transformed_image.to(device))

    # 8. Convert the model's output logits to pred probs
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 9. Convert the model's pred probs to pred labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # 10. Plot image with predicted label and probability
    plt.figure()
    plt.imshow(img)
    plt.title(f'Pred: {class_names[target_image_pred_label]} | '
              f'Prob: {target_image_pred_probs.max():.3f}')
    plt.axis(False)
    plt.show()

# Get a random list of image paths from the test set
num_images_to_plot = 5
test_image_path_list = list(Path(test_dir).glob('*/*.jpg'))
test_image_path_sample = random.sample(population=test_image_path_list,
                                       k=num_images_to_plot)

# Make prediction on and plot images
for image_path in test_image_path_sample:
    pred_and_plot_image(model=model,
                        image_path=image_path,
                        class_names=class_names,
                        image_size=(224, 224))

"""
6.1 Making preditions on a custom image
Let's make a prediction on pizza dad image
"""
# Download custom image, setup custom image path
custom_image = '04-pizza-dad.jpeg'
custom_image_path = data_path / custom_image

# Download the image if it doesn't already exist
if not custom_image_path.is_file():
    with open(custom_image_path, 'wb') as f:
        # When downloading from GitHub, need to use the 'raw' file link
        request = requests.get('https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/04-pizza-dad.jpeg')
        print(f'Downloading {custom_image}...')
        f.write(request.content)
else:
    print(f'Image already downloaded, skipping')

# Make the predition on our custom image
pred_and_plot_image(model=model,
                    image_path=str(custom_image_path),
                    class_names=class_names,
                    image_size=(224, 224))

# 1. Create models directory
MODEL_PATH = Path('/home/michel/models')
MODEL_PATH.mkdir(parents=True, exist_ok=True)
# 2. Create model save path
MODEL_NAME = 'transfer_learning_model.pt'
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
# 3. Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model.state_dict(),
           f=MODEL_SAVE_PATH)
