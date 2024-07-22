"""
Programma per prendere il dataset Food101 e crearci un dataset più piccolo
Al momento è settato per prendere 5 classi diverse e il 20% dei loro contenuti
"""
import torchvision.datasets as datasets
import random
import shutil

# Setup data directory
import pathlib
data_dir = pathlib.Path("../data")

# Get training data
train_data = datasets.Food101(root=str(data_dir),
                              split="train",
                              # transform=transforms.ToTensor(),
                              download=True)

# Get testing data
test_data = datasets.Food101(root=str(data_dir),
                             split="test",
                             # transform=transforms.ToTensor(),
                             download=True)

# Get random 20% of training images
# Setup data paths
data_path = data_dir / "food-101" / "images"
# Write the classes you want to take
target_classes = ["donuts", "ice_cream", "pizza", "sushi", "waffles"]

# Change amount of data to get (e.g. 0.1 = random 10%, 0.2 = random 20%)
amount_to_get = 0.2


# Create function to separate a random amount of data
def get_subset(image_path=data_path,
               data_splits=["train", "test"],
               target_classes=target_classes,
               amount=amount_to_get):
    label_splits = {}

    # Get labels
    for data_split in data_splits:
        print(f"[INFO] Creating image split for: {data_split}...")
        label_path = data_dir / "food-101" / "meta" / f"{data_split}.txt"
        with open(label_path, "r") as f:
            labels = [line.strip("\n") for line in f.readlines() if line.split("/")[0] in target_classes]

        # Get random subset of target classes image ID's
        number_to_sample = round(amount * len(labels))
        print(f"[INFO] Getting random subset of {number_to_sample} images for {data_split}...")
        sampled_images = random.sample(labels, k=number_to_sample)

        # Apply full paths
        image_paths = [pathlib.Path(str(image_path / sample_image) + ".jpg") for sample_image in sampled_images]
        label_splits[data_split] = image_paths
    return label_splits


label_splits = get_subset(amount=amount_to_get)
print(label_splits["train"][:10])

# Create target directory path
target_dir_name = f"../data/5_foods_{str(int(amount_to_get*100))}_percent"
print(f"Creating directory: '{target_dir_name}'")

# Setup the directories
target_dir = pathlib.Path(target_dir_name)

# Make the directories
target_dir.mkdir(parents=True, exist_ok=True)

for image_split in label_splits.keys():
    for image_path in label_splits[str(image_split)]:
        dest_dir = target_dir / image_split / image_path.parent.stem / image_path.name
        if not dest_dir.parent.is_dir():
            dest_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Copying {image_path} to {dest_dir}...")
        shutil.copy2(image_path, dest_dir)

# Check lengths of directories
def walk_through_dir(dir_path):
    """
    Walks through dir_path returning its contents.
    Args:
      dir_path (str): target directory

    Returns:
      A print out of:
        number of subdiretories in dir_path
        number of images (files) in each subdirectory
        name of each subdirectory
    """
    import os
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


walk_through_dir(str(target_dir))

# Zip images
zip_file_name = data_dir / f"5_foods_{str(int(amount_to_get*100))}_percent"
shutil.make_archive(str(zip_file_name),
                    format="zip",
                    root_dir=target_dir)
