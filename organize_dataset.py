import pandas as pd
import os
import shutil

# Read metadata CSV
metadata = pd.read_csv("HAM10000_metadata.csv")

# Image folders
image_dirs = [
    "HAM10000_images_part_1",
    "HAM10000_images_part_2"
]

# Create dataset folders
os.makedirs("dataset/benign", exist_ok=True)
os.makedirs("dataset/malignant", exist_ok=True)

# Cancer labels
malignant_labels = ["mel", "bcc", "akiec"]

for index, row in metadata.iterrows():

    image_id = row["image_id"]
    label = row["dx"]

    filename = image_id + ".jpg"

    source_path = None

    # Find image
    for folder in image_dirs:

        temp_path = os.path.join(folder, filename)

        if os.path.exists(temp_path):
            source_path = temp_path
            break

    if source_path is None:
        continue

    # Decide destination
    if label in malignant_labels:
        target_folder = "dataset/malignant"
    else:
        target_folder = "dataset/benign"

    shutil.copy(source_path, target_folder)

print("Dataset organized successfully!")