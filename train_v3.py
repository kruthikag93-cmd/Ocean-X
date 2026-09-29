import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import numpy as np

from old_model import UNet


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_IMAGE_DIR = "../dataset/train/images"
TRAIN_LABEL_DIR = "../dataset/train/labels"

VAL_IMAGE_DIR = "../dataset/val/images"
VAL_LABEL_DIR = "../dataset/val/labels"

HARD_NEGATIVE_FILE = "hard_negative_list.txt"

BASE_MODEL = "unet_shipwreck_hard_negative.pth"
OUTPUT_MODEL = "unet_shipwreck_v3.pth"

BATCH_SIZE = 4
EPOCHS = 8

LEARNING_RATE = 0.00005

# Increase importance of positive target pixels.
POSITIVE_WEIGHT = 2.0

# Give hard-negative examples extra sampling probability.
HARD_NEGATIVE_WEIGHT = 2.5

# Validation threshold discovered from our previous analysis.
THRESHOLD = 0.25

DEVICE = torch.device("cpu")


# ============================================================
# DATASET
# ============================================================

class SonarDataset(Dataset):

    def __init__(self, image_dir, label_dir):

        self.image_dir = image_dir
        self.label_dir = label_dir

        self.images = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        filename = self.images[index]

        image_path = os.path.join(
            self.image_dir,
            filename
        )

        label_path = os.path.join(
            self.label_dir,
            filename
        )

        image = Image.open(image_path).convert("L")
        label = Image.open(label_path).convert("L")

        image_array = np.array(image, dtype=np.float32) / 255.0
        label_array = np.array(label, dtype=np.float32)

        label_array = (label_array > 0).astype(np.float32)

        image_tensor = torch.from_numpy(
            image_array
        ).unsqueeze(0)

        label_tensor = torch.from_numpy(
            label_array
        ).unsqueeze(0)

        return image_tensor, label_tensor


# ============================================================
# LOAD DATA
# ============================================================

train_dataset = SonarDataset(
    TRAIN_IMAGE_DIR,
    TRAIN_LABEL_DIR
)

val_dataset = SonarDataset(
    VAL_IMAGE_DIR,
    VAL_LABEL_DIR
)

print()
print("==========================================")
print("        OCEAN X AI V3 TRAINING")
print("==========================================")
print()

print("Training patches:", len(train_dataset))
print("Validation patches:", len(val_dataset))


# ============================================================
# HARD NEGATIVE LIST
# ============================================================

hard_negatives = set()

if os.path.exists(HARD_NEGATIVE_FILE):

    with open(HARD_NEGATIVE_FILE, "r") as f:

        for line in f:

            name = line.strip()

            if name:
                hard_negatives.add(name)

print("Hard negatives:", len(hard_negatives))


# ============================================================
# CREATE BALANCED SAMPLING WEIGHTS
# ============================================================

sample_weights = []

print()
print("Preparing training weights...")

for filename in train_dataset.images:

    weight = 1.0

    # --------------------------------------------------------
    # Hard negative patches
    # --------------------------------------------------------

    if filename in hard_negatives:
        weight = HARD_NEGATIVE_WEIGHT

    else:

        # ----------------------------------------------------
        # Positive patches
        # ----------------------------------------------------

        label_path = os.path.join(
            TRAIN_LABEL_DIR,
            filename
        )

        label = Image.open(label_path).convert("L")

        label_array = np.array(label)

        positive_pixels = np.count_nonzero(
            label_array > 0
        )

        if positive_pixels > 0:
            weight = 1.5

    sample_weights.append(weight)


sample_weights = torch.DoubleTensor(sample_weights)

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# MODEL
# ============================================================

model = UNet().to(DEVICE)

print()
print("Loading base model:")
print(BASE_MODEL)

checkpoint = torch.load(
    BASE_MODEL,
    map_location=DEVICE
)

model.load_state_dict(checkpoint)

print("Base model loaded successfully!")


# ============================================================
# LOSS
# ============================================================

loss_function = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor(
        [POSITIVE_WEIGHT],
        device=DEVICE
    )
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# VALIDATION METRICS
# ============================================================

def calculate_validation_metrics():

    model.eval()

    total_intersection = 0
    total_union = 0

    total_dice_numerator = 0
    total_dice_denominator = 0

    total_tp = 0
    total_fp = 0
    total_fn = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            probabilities = torch.sigmoid(outputs)

            predictions = (
                probabilities >= THRESHOLD
            ).float()

            # -----------------------------------------------
            # IoU
            # -----------------------------------------------

            intersection = (
                predictions * labels
            ).sum().item()

            union = (
                predictions.sum()
                + labels.sum()
                - intersection
            )

            total_intersection += intersection
            total_union += union

            # -----------------------------------------------
            # Dice
            # -----------------------------------------------

            total_dice_numerator += (
                2 * intersection
            )

            total_dice_denominator += (
                predictions.sum().item()
                + labels.sum().item()
            )

            # -----------------------------------------------
            # Precision / Recall
            # -----------------------------------------------

            tp = (
                predictions * labels
            ).sum().item()

            fp = (
                predictions * (1 - labels)
            ).sum().item()

            fn = (
                (1 - predictions) * labels
            ).sum().item()

            total_tp += tp
            total_fp += fp
            total_fn += fn

    # --------------------------------------------------------
    # Final metrics
    # --------------------------------------------------------

    if total_union > 0:
        iou = total_intersection / total_union
    else:
        iou = 0

    if total_dice_denominator > 0:
        dice = (
            total_dice_numerator
            / total_dice_denominator
        )
    else:
        dice = 0

    precision = (
        total_tp
        / (total_tp + total_fp + 1e-8)
    )

    recall = (
        total_tp
        / (total_tp + total_fn + 1e-8)
    )

    return iou, dice, precision, recall


# ============================================================
# TRAINING
# ============================================================

best_iou = 0.0

print()
print("==========================================")
print("Starting training")
print("==========================================")
print()

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = loss_function(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    average_loss = (
        running_loss / len(train_loader)
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    iou, dice, precision, recall = (
        calculate_validation_metrics()
    )

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {average_loss:.4f} | "
        f"IoU: {iou:.4f} | "
        f"Dice: {dice:.4f} | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f}"
    )

    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if iou > best_iou:

        best_iou = iou

        torch.save(
            model.state_dict(),
            OUTPUT_MODEL
        )

        print(
            f"  >>> New best model saved! "
            f"IoU = {best_iou:.4f}"
        )


# ============================================================
# FINISHED
# ============================================================

print()
print("==========================================")
print("        TRAINING COMPLETE")
print("==========================================")
print()

print("Best validation IoU:", round(best_iou, 4))
print("Model saved as:", OUTPUT_MODEL)
print()