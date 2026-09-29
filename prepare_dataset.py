from pathlib import Path
from PIL import Image
import random


# ======================================================
# SETTINGS
# ======================================================

IMAGE_DIR = Path("../train/images")
LABEL_DIR = Path("../train/labels")

OUTPUT_DIR = Path("../patches")

PATCH_SIZE = 256
STRIDE = 128

# Minimum number of shipwreck pixels required
# for a patch to be considered a positive patch.
MIN_SHIP_PIXELS = 50

# Maximum number of background patches per image.
MAX_BACKGROUND_PATCHES = 10


# ======================================================
# CREATE OUTPUT FOLDERS
# ======================================================

positive_image_dir = OUTPUT_DIR / "images"
positive_label_dir = OUTPUT_DIR / "labels"

positive_image_dir.mkdir(parents=True, exist_ok=True)
positive_label_dir.mkdir(parents=True, exist_ok=True)


# ======================================================
# PROCESS ALL SONAR IMAGES
# ======================================================

all_images = sorted(IMAGE_DIR.glob("*.png"))

print("Total sonar images:", len(all_images))

total_positive = 0
total_background = 0


for image_number, image_path in enumerate(all_images):

    label_path = LABEL_DIR / image_path.name

    if not label_path.exists():
        print("WARNING: Missing label:", image_path.name)
        continue

    # Load images
    image = Image.open(image_path).convert("L")
    label = Image.open(label_path).convert("L")

    width, height = image.size

    background_candidates = []

    positive_count_for_image = 0

    # --------------------------------------------------
    # Sliding window
    # --------------------------------------------------

    for y in range(0, height - PATCH_SIZE + 1, STRIDE):

        for x in range(0, width - PATCH_SIZE + 1, STRIDE):

            image_patch = image.crop(
                (x, y, x + PATCH_SIZE, y + PATCH_SIZE)
            )

            label_patch = label.crop(
                (x, y, x + PATCH_SIZE, y + PATCH_SIZE)
            )

            # Count shipwreck pixels
            ship_pixels = sum(
                1
                for pixel in label_patch.getdata()
                if pixel > 0
            )

            # --------------------------------------------------
            # POSITIVE PATCH
            # --------------------------------------------------

            if ship_pixels >= MIN_SHIP_PIXELS:

                filename = (
                    f"{image_path.stem}"
                    f"_x{x}_y{y}.png"
                )

                image_patch.save(
                    positive_image_dir / filename
                )

                label_patch.save(
                    positive_label_dir / filename
                )

                total_positive += 1
                positive_count_for_image += 1

            # --------------------------------------------------
            # BACKGROUND CANDIDATE
            # --------------------------------------------------

            else:

                background_candidates.append(
                    (x, y, image_patch, label_patch)
                )


    # --------------------------------------------------
    # RANDOMLY SELECT BACKGROUND PATCHES
    # --------------------------------------------------

    random.shuffle(background_candidates)

    selected_background = background_candidates[
        :MAX_BACKGROUND_PATCHES
    ]


    for x, y, image_patch, label_patch in selected_background:

        filename = (
            f"{image_path.stem}"
            f"_background_x{x}_y{y}.png"
        )

        image_patch.save(
            positive_image_dir / filename
        )

        label_patch.save(
            positive_label_dir / filename
        )

        total_background += 1


    print(
        f"[{image_number + 1}/{len(all_images)}] "
        f"{image_path.name} → "
        f"{positive_count_for_image} positive patches"
    )


# ======================================================
# FINAL SUMMARY
# ======================================================

print()
print("=" * 50)
print("PATCH DATASET CREATED")
print("=" * 50)

print("Positive patches:", total_positive)
print("Background patches:", total_background)
print("Total patches:", total_positive + total_background)

print()
print("Saved images to:")
print(positive_image_dir)

print()
print("Saved labels to:")
print(positive_label_dir)