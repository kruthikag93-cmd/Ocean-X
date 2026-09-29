import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

from model import UNet


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "unet_shipwreck.pth"

IMAGE_DIR = "../patches/images"
LABEL_DIR = "../patches/labels"

# Choose one patch to test
TEST_IMAGE = "DM_Wilson_03_x896_y3712.png"


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)


# ============================================================
# LOAD MODEL
# ============================================================

model = UNet()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )
)

model = model.to(device)

model.eval()

print("Model loaded successfully!")


# ============================================================
# LOAD TEST IMAGE
# ============================================================

image_path = Path(IMAGE_DIR) / TEST_IMAGE
label_path = Path(LABEL_DIR) / TEST_IMAGE


if not image_path.exists():
    print("ERROR: Test image not found!")
    print("Looking for:", image_path)
    exit()


if not label_path.exists():
    print("ERROR: Test label not found!")
    print("Looking for:", label_path)
    exit()


# ============================================================
# READ IMAGE
# ============================================================

image = Image.open(image_path).convert("L")

image_array = np.array(image, dtype=np.float32)

# Normalize image from 0-255 to 0-1
image_array = image_array / 255.0


# ============================================================
# READ GROUND TRUTH
# ============================================================

label = Image.open(label_path).convert("L")

label_array = np.array(label, dtype=np.float32)

# Convert label to binary
ground_truth = (label_array > 0.5).astype(np.float32)


# ============================================================
# CONVERT IMAGE TO TENSOR
# ============================================================

test_input = torch.from_numpy(image_array)

# Add channel dimension
test_input = test_input.unsqueeze(0)

# Add batch dimension
test_input = test_input.unsqueeze(0)

# Move to CPU/MPS
test_input = test_input.to(device)


print("Input shape:", test_input.shape)


# ============================================================
# RUN U-NET
# ============================================================

with torch.no_grad():

    prediction = model(test_input)

    # Convert model output to probability
    probability = torch.sigmoid(prediction)

    # Convert probability to binary mask
    predicted_mask = (probability > 0.5).float()


# ============================================================
# CONVERT BACK TO NUMPY
# ============================================================

predicted_mask = predicted_mask.squeeze().cpu().numpy()

probability = probability.squeeze().cpu().numpy()


print("Prediction shape:", predicted_mask.shape)

print(
    "Maximum prediction probability:",
    probability.max()
)

print(
    "Predicted shipwreck pixels:",
    predicted_mask.sum()
)

print(
    "Ground truth shipwreck pixels:",
    ground_truth.sum()
)


# ============================================================
# CALCULATE IoU
# ============================================================

intersection = np.logical_and(
    predicted_mask == 1,
    ground_truth == 1
).sum()

union = np.logical_or(
    predicted_mask == 1,
    ground_truth == 1
).sum()


if union > 0:
    iou = intersection / union
else:
    iou = 0


print("IoU:", round(iou, 4))


# ============================================================
# DISPLAY RESULTS
# ============================================================

plt.figure(figsize=(15, 5))


# Original sonar image
plt.subplot(1, 3, 1)

plt.imshow(
    image_array,
    cmap="gray"
)

plt.title("Original Sonar")

plt.axis("off")


# Ground truth
plt.subplot(1, 3, 2)

plt.imshow(
    ground_truth,
    cmap="gray"
)

plt.title("Ground Truth")

plt.axis("off")


# AI prediction
plt.subplot(1, 3, 3)

plt.imshow(
    predicted_mask,
    cmap="gray"
)

plt.title(
    f"AI Prediction\nIoU = {iou:.4f}"
)

plt.axis("off")


plt.tight_layout()

plt.show()
