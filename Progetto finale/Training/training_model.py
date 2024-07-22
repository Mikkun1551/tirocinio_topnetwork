"""
Programma per allenare un modello efficientnet b2 sul dataset 5 foods
"""
import torch
import torchvision
from torch import nn
from torchvision import transforms
from torchinfo import summary
import matplotlib.pyplot as plt
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

# Setup path to a data folder
data_path = Path('/home/michel/datasets')
image_path = data_path / '5_foods_20_percent'

# Setup directory path
train_dir = image_path / 'train'
test_dir = image_path / 'test'

# Normalize with ImageNet for the model efficientnet
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
manual_transforms = transforms.Compose([
    transforms.Resize((224, 224)), # resize the image
    transforms.ToTensor(), # get images into range 0 to 1
    normalize]) # make sure images have the same distribution
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=str(train_dir),
                                                                               test_dir=str(test_dir),
                                                                               transform=manual_transforms,
                                                                               batch_size=8)

# Get a set of pretrained model weights
weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT # default = best available weights
# Instantiate a model
model = torchvision.models.efficientnet_b2(weights=weights)

# Checking our model architecture
summary(model=model,
        input_size=(1, 3, 224, 224), # BS, C, H, W
        col_names=['input_size', 'output_size', 'num_params', 'trainable'],
        col_width=20,
        row_settings=['var_names'])

# Freeze all the base layers in EffNetB2
for param in model.features.parameters():
    param.requires_grad = False

# Update the classifier head of our model to suit our problem
print(model.classifier)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3, inplace=True), # the same of the original
    # In our case we need to change only the out_features to the number of classes of the dataset
    nn.Linear(in_features=1408, # feature vector coming in
              out_features=len(class_names), # how many classes do we have
              bias=True))
print(model.classifier)



# Begin training
# Define loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 0.001)

# Start the timer
start_time = timer()

# Setup training and save the results
results = engine.train(model=model,
                       train_dataloader=train_dataloader,
                       test_dataloader=test_dataloader,
                       optimizer=optimizer,
                       loss_fn=loss_fn,
                       epochs=30,
                       device=device)

# Calculate total time
end_time = timer()
total_time = end_time - start_time
# On CPU the model is slow (35s per epoch)
print(f'Training took: {total_time:.2f} seconds')

# Plot the loss curves of our model
plot_loss_curves(results)

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

# Make the predition on our custom image
custom_image_path = "/home/michel/Pictures/test_images/ice_cream_1.jpg"
pred_and_plot_image(model=model,
                    image_path=str(custom_image_path),
                    class_names=class_names,
                    image_size=(224, 224))

# 1. Create models directory
MODEL_PATH = Path('/home/michel/models')
MODEL_PATH.mkdir(parents=True, exist_ok=True)
# 2. Create model save path
MODEL_NAME = 'effnetb2_5_foods.pt'
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
# 3. Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model.state_dict(),
           f=MODEL_SAVE_PATH)
