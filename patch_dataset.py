from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt


# ======================================================
# SETTINGS
# ======================================================

IMAGE_DIR = Path("../train/images")
LABEL_DIR = Path("../train/labels")

PATCH_SIZE = 256

# We want patches that contain at least this much
# shipwreck information.
MIN_SHIP_PIXELS = 100


# ======================================================
# FIND SHIPWRECK PATCHES
# ======================================================

def find_shipwreck_patches(image_path, label_path):

    image = Image.open(image_path).convert("L")
    label = Image.open(label_path).convert("L")

    width, height = image.size

    patches = []

    # Move through the image in 256x256 blocks
    for y in range(0, height - PATCH_SIZE + 1, PATCH_SIZE):

        for x in range(0, width - PATCH_SIZE + 1, PATCH_SIZE):

            # Crop sonar image
            image_patch = image.crop(
                (x, y, x + PATCH_SIZE, y + PATCH_SIZE)
            )

            # Crop corresponding ground truth
            label_patch = label.crop(
                (x, y, x + PATCH_SIZE, y + PATCH_SIZE)
            )

            # Count shipwreck pixels
            label_pixels = list(label_patch.getdata())

            ship_pixels = sum(
                1 for pixel in label_pixels if pixel > 0
            )

            # Keep patch if it contains enough shipwreck
            if ship_pixels >= MIN_SHIP_PIXELS:

                patches.append(
                    (
                        x,
                        y,
                        image_patch,
                        label_patch,
                        ship_pixels
                    )
                )

    return patches


# ======================================================
# TEST WITH DM_WILSON_03
# ======================================================

image_path = IMAGE_DIR / "DM_Wilson_03.png"
label_path = LABEL_DIR / "DM_Wilson_03.png"


patches = find_shipwreck_patches(
    image_path,
    label_path
)


print("Image:", image_path.name)
print("Number of shipwreck patches:", len(patches))


# ======================================================
# DISPLAY THE FIRST PATCH
# ======================================================

if len(patches) > 0:

    x, y, image_patch, label_patch, ship_pixels = patches[0]

    print("Patch location:")
    print("X:", x)
    print("Y:", y)

    print("Shipwreck pixels:", ship_pixels)

    plt.figure(figsize=(10, 5))

    # Sonar patch
    plt.subplot(1, 2, 1)

    plt.imshow(
        image_patch,
        cmap="gray"
    )

    plt.title("Sonar Patch")

    plt.axis("off")


    # Ground truth patch
    plt.subplot(1, 2, 2)

    plt.imshow(
        label_patch,
        cmap="gray"
    )

    plt.title("Ground Truth Patch")

    plt.axis("off")


    plt.tight_layout()

    plt.show()


else:

    print("No shipwreck patches found.")