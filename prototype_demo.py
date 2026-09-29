import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os

from old_model import UNet


# =========================================================
# SETTINGS
# =========================================================

MODEL_PATH = "unet_shipwreck_hard_negative.pth"

IMAGE_DIR = "../dataset/test/images"
LABEL_DIR = "../dataset/test/labels"

THRESHOLD = 0.40


# =========================================================
# LOAD MODEL
# =========================================================

print("========================================")
print("       OCEAN X PROTOTYPE DEMO")
print("========================================")

model = UNet()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu"
    )
)

model.eval()

print("Model loaded successfully.")


# =========================================================
# FIND A GOOD DEMO IMAGE
# =========================================================

image_files = glob.glob(
    os.path.join(IMAGE_DIR, "*.png")
)

selected_image = None
selected_label = None

print("Searching for suitable demonstration image...")


for image_path in image_files:

    filename = os.path.basename(image_path)

    label_path = os.path.join(
        LABEL_DIR,
        filename
    )

    if not os.path.exists(label_path):
        continue

    label = np.array(
        Image.open(label_path).convert("L")
    )

    positive_pixels = np.sum(label > 0)

    total_pixels = label.size

    ratio = positive_pixels / total_pixels

    # Select a reasonably sized positive target
    if (
        positive_pixels > 500
        and
        ratio < 0.30
    ):
        selected_image = image_path
        selected_label = label_path
        break


# =========================================================
# CHECK
# =========================================================

if selected_image is None:

    print("No suitable demo image found.")

    print("Try changing the selection range.")

    raise SystemExit


print()
print("Selected demonstration image:")
print(os.path.basename(selected_image))


# =========================================================
# LOAD SONAR IMAGE
# =========================================================

image = Image.open(
    selected_image
).convert("L")

image_array = np.array(image)


# =========================================================
# PREPARE INPUT
# =========================================================

input_tensor = torch.from_numpy(
    image_array.astype(np.float32) / 255.0
)

input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)


# =========================================================
# AI PREDICTION
# =========================================================

with torch.no_grad():

    output = model(input_tensor)

    probability = torch.sigmoid(
        output
    )[0, 0].numpy()


# =========================================================
# DETECTION
# =========================================================

prediction = (
    probability >= THRESHOLD
).astype(np.uint8)


predicted_pixels = np.sum(prediction)

maximum_probability = probability.max()

mean_probability = probability.mean()


# =========================================================
# RESULTS
# =========================================================

print()
print("========================================")
print("         OCEAN X AI RESULT")
print("========================================")

print(
    "Maximum probability:",
    round(float(maximum_probability), 4)
)

print(
    "Mean probability:",
    round(float(mean_probability), 4)
)

print(
    "Detected pixels:",
    int(predicted_pixels)
)

print(
    "Threshold:",
    THRESHOLD
)

print("========================================")


# =========================================================
# VISUALIZATION
# =========================================================

plt.figure(
    figsize=(15, 5)
)


# ---------------------------------------------------------
# ORIGINAL
# ---------------------------------------------------------

plt.subplot(1, 3, 1)

plt.imshow(
    image_array,
    cmap="gray"
)

plt.title(
    "Original Side-Scan Sonar"
)

plt.axis("off")


# ---------------------------------------------------------
# AI PROBABILITY
# ---------------------------------------------------------

plt.subplot(1, 3, 2)

plt.imshow(
    probability,
    cmap="hot",
    vmin=0,
    vmax=1
)

plt.colorbar(
    label="Detection Probability"
)

plt.title(
    f"Ocean X AI Probability\n"
    f"Max = {maximum_probability:.3f}"
)

plt.axis("off")


# ---------------------------------------------------------
# FINAL DETECTION
# ---------------------------------------------------------

plt.subplot(1, 3, 3)

plt.imshow(
    image_array,
    cmap="gray"
)

plt.imshow(
    prediction,
    cmap="Reds",
    alpha=0.45
)

plt.title(
    f"Ocean X Detection\n"
    f"Detected Pixels = {predicted_pixels}"
)

plt.axis("off")


# =========================================================
# SAVE PROTOTYPE IMAGE
# =========================================================

plt.tight_layout()

output_file = "ocean_x_prototype_demo.png"

plt.savefig(
    output_file,
    dpi=200,
    bbox_inches="tight"
)

print()
print(
    "Prototype image saved as:",
    output_file
)

plt.show()