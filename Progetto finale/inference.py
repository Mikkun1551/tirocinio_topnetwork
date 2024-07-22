"""
Programma per fare inference con un modello customizzato di efficientnet b2
su un dataset di 5 classi su una o più immagini
"""
import torch
from torch import nn
import torchvision
from torchvision import transforms
from typing import Tuple
from PIL import Image
from pathlib import Path

# Setup device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def pred_image(class_names: Path, # path to the file with the class names
               image_path: str, # path to the images to pred
               path_to_weights: str, # path to the weights of the model
               image_size: Tuple[int, int] = (224, 224),
               transform: torchvision.transforms = None,
               device: torch.device=device):

    # 1. Instantiate classes by giving path of the file with the classes
    with open(class_names, 'r') as read_class:
        class_names = [classes.strip() for classes in read_class.readlines()]


    # 2. Open the image with PIL
    img = Image.open(image_path)


    # 3. Let's instantiate a effnet_b2 model and customize it like our custom_model
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    # Instantiate a model
    model = torchvision.models.efficientnet_b2(weights=weights)
    # Modify the output layer like our model
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),  # the same of the original
        # In our case we need to change only the out_features to the number of classes of the dataset
        nn.Linear(in_features=1408,  # feature vector coming in
                  out_features=len(class_names), # how many classes do we have
                  bias=True))
    # Select the model to upload giving a path
    model.load_state_dict(torch.load(path_to_weights, map_location=torch.device(device)))


    # 4. Create a transform if one doesn't exist
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

    # 5. Make sure the model is on the target device
    model.to(device)

    # 6. Turn on inference mode and eval mode
    model.eval()
    with torch.inference_mode():
        # 7. Transform the image and add an extra batch dimension
        transformed_image = image_transform(img).unsqueeze(dim=0) # BS, C, H, W

        # 8. Make a prediction on the transformed image by passing it to the model
        # also pass it on the same device
        target_image_pred = model(transformed_image.to(device))

    # 9. Convert the model's output logits to pred probs
    pred_probs = torch.softmax(target_image_pred, dim=1)

    # 10. Convert the model's pred probs to pred labels
    pred_label = torch.argmax(pred_probs, dim=1)

    return f'Class label: {str(class_names[pred_label])} | Class prob: {pred_probs.max():.3f}'
