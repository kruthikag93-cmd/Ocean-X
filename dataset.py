from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset


class SonarDataset(Dataset):

    def __init__(self, image_dir, label_dir):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)

        # Find all PNG sonar images
        self.images = sorted(self.image_dir.glob("*.png"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        # Get image path
        image_path = self.images[index]

        # The label has the same filename
        label_path = self.label_dir / image_path.name

        # Load sonar image as grayscale
        image = Image.open(image_path).convert("L")

        # Load ground-truth mask
        label = Image.open(label_path).convert("L")

        # ------------------------------------------------
        # Convert PIL image directly to PyTorch tensor
        # This avoids the NumPy compatibility problem
        # ------------------------------------------------

        image = torch.frombuffer(
            image.tobytes(),
            dtype=torch.uint8
        ).reshape(image.height, image.width)

        label = torch.frombuffer(
            label.tobytes(),
            dtype=torch.uint8
        ).reshape(label.height, label.width)

        # ------------------------------------------------
        # Normalize sonar image
        # 0-255  →  0-1
        # ------------------------------------------------

        image = image.float() / 255.0

        # Convert label to float
        label = label.float()

        # ------------------------------------------------
        # Add channel dimension
        #
        # Before:
        #   [height, width]
        #
        # After:
        #   [1, height, width]
        # ------------------------------------------------

        image = image.unsqueeze(0)
        label = label.unsqueeze(0)

        return image, label


# ======================================================
# DATASET PATHS
# ======================================================

image_dir = "../train/images"
label_dir = "../train/labels"


# ======================================================
# CREATE DATASET
# ======================================================

dataset = SonarDataset(
    image_dir,
    label_dir
)


# ======================================================
# TEST DATASET
# ======================================================

print("Number of images:", len(dataset))


# Get the third image
image, label = dataset[2]


print("Image shape:", image.shape)
print("Label shape:", label.shape)

print("Image data type:", image.dtype)
print("Label data type:", label.dtype)

print("Image minimum:", image.min().item())
print("Image maximum:", image.max().item())

print("Label minimum:", label.min().item())
print("Label maximum:", label.max().item())