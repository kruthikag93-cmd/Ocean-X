import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from pathlib import Path


# ======================================================
# SETTINGS
# ======================================================

TRAIN_IMAGE_DIR = Path("../dataset/train/images")
TRAIN_LABEL_DIR = Path("../dataset/train/labels")

VAL_IMAGE_DIR = Path("../dataset/val/images")
VAL_LABEL_DIR = Path("../dataset/val/labels")
BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 0.001

MODEL_PATH = "unet_shipwreck.pth"


# ======================================================
# DEVICE
# ======================================================

if torch.backends.mps.is_available():

    device = torch.device("mps")

elif torch.cuda.is_available():

    device = torch.device("cuda")

else:

    device = torch.device("cpu")


print("Using device:", device)


# ======================================================
# FAST IMAGE TO TENSOR FUNCTION
# ======================================================

def image_to_tensor(image):

    # Convert PIL image into raw bytes
    data = image.tobytes()

    # Create tensor directly from bytes
    tensor = torch.frombuffer(
        data,
        dtype=torch.uint8
    )

    # Convert to float
    tensor = tensor.float()

    # Reshape to 1 x 256 x 256
    tensor = tensor.reshape(
        1,
        256,
        256
    )

    return tensor


# ======================================================
# DATASET
# ======================================================

class SonarPatchDataset(Dataset):

    def __init__(self, image_dir, label_dir):

        self.image_dir = Path(image_dir)

        self.label_dir = Path(label_dir)

        self.images = sorted(
            self.image_dir.glob("*.png")
        )


    def __len__(self):

        return len(self.images)


    def __getitem__(self, index):

        image_path = self.images[index]

        label_path = (
            self.label_dir /
            image_path.name
        )


        # Load sonar image
        image = Image.open(
            image_path
        ).convert("L")


        # Load ground truth mask
        label = Image.open(
            label_path
        ).convert("L")


        # Fast conversion
        image = image_to_tensor(image)

        label = image_to_tensor(label)


        # Normalize sonar image
        image = image / 255.0


        # Convert mask to 0 or 1
        label = (label > 0).float()


        return image, label


# ======================================================
# CREATE DATASETS
# ======================================================

train_dataset = SonarPatchDataset(
    TRAIN_IMAGE_DIR,
    TRAIN_LABEL_DIR
)

val_dataset = SonarPatchDataset(
    VAL_IMAGE_DIR,
    VAL_LABEL_DIR
)
# TEMPORARY SMALL DATASET FOR TESTING
#train_dataset = Subset(
 #   train_dataset,
  #  range(min(20, len(train_dataset)))
#)

#val_dataset = Subset(
 #   val_dataset,
  #  range(min(5, len(val_dataset)))
#)

print("Training set:", len(train_dataset))
print("Validation set:", len(val_dataset))
print(
    "Training samples:",
    len(train_dataset)
)

print(
    "Validation samples:",
    len(val_dataset)
)


# ======================================================
# DATA LOADERS
# ======================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ======================================================
# IMPORT U-NET
# ======================================================

from model import UNet


model = UNet()

model = model.to(device)


# ======================================================
# LOSS FUNCTION
# ======================================================

def dice_loss(prediction, target):
    prediction = torch.sigmoid(prediction)

    prediction = prediction.view(-1)
    target = target.view(-1)

    smooth = 1.0

    intersection = (prediction * target).sum()

    dice = (2 * intersection + smooth) / (
        prediction.sum() + target.sum() + smooth
    )

    return 1 - dice


def combined_loss(prediction, target):
    bce = nn.BCEWithLogitsLoss()(prediction, target)
    dice = dice_loss(prediction, target)

    return bce + dice


loss_function = combined_loss


# ======================================================
# OPTIMIZER
# ======================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ======================================================
# TRAINING
# ======================================================

for epoch in range(EPOCHS):

    model.train()

    total_train_loss = 0.0


    for batch_number, (images, labels) in enumerate(train_loader):

        images = images.to(device)

        labels = labels.to(device)


        # Clear previous gradients
        optimizer.zero_grad()


        # Forward pass
        predictions = model(images)


        # Calculate loss
        loss = loss_function(
            predictions,
            labels
        )


        # Backpropagation
        loss.backward()


        # Update weights
        optimizer.step()


        total_train_loss += loss.item()


    average_train_loss = (
        total_train_loss /
        len(train_loader)
    )


    # ==================================================
    # VALIDATION
    # ==================================================

    model.eval()

    total_val_loss = 0.0


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)

            labels = labels.to(device)


            predictions = model(images)


            loss = loss_function(
                predictions,
                labels
            )


            total_val_loss += loss.item()


    average_val_loss = (
        total_val_loss /
        len(val_loader)
    )


    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {average_train_loss:.4f} | "
        f"Val Loss: {average_val_loss:.4f}"
    )


# ======================================================
# SAVE MODEL
# ======================================================

torch.save(
    model.state_dict(),
    MODEL_PATH
)


print()
print("========================================")
print("TRAINING COMPLETED")
print("========================================")

print(
    "Model saved to:",
    MODEL_PATH
)