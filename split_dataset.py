from pathlib import Path
import random
import shutil


# ======================================================
# SETTINGS
# ======================================================

SOURCE_DIR = Path("../patches")

IMAGE_SOURCE = SOURCE_DIR / "images"
LABEL_SOURCE = SOURCE_DIR / "labels"

OUTPUT_DIR = Path("../dataset")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42


# ======================================================
# CHECK RATIOS
# ======================================================

if abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) > 0.001:
    raise ValueError("Train/Val/Test ratios must add up to 1.0")


# ======================================================
# CREATE OUTPUT FOLDERS
# ======================================================

for split in ["train", "val", "test"]:

    (OUTPUT_DIR / split / "images").mkdir(
        parents=True,
        exist_ok=True
    )

    (OUTPUT_DIR / split / "labels").mkdir(
        parents=True,
        exist_ok=True
    )


# ======================================================
# GET ALL PATCHES
# ======================================================

image_files = sorted(IMAGE_SOURCE.glob("*.png"))

print("Total patches found:", len(image_files))


# ======================================================
# FIND ORIGINAL SONAR IMAGE FOR EACH PATCH
# ======================================================

groups = {}

for image_file in image_files:

    name = image_file.stem

    # Background patch:
    # DM_Wilson_03_background_x1024_y3840
    if "_background_x" in name:

        original_name = name.split("_background_x")[0]

    # Positive patch:
    # DM_Wilson_03_x1024_y3840
    else:

        original_name = name.rsplit("_x", 1)[0]


    if original_name not in groups:
        groups[original_name] = []

    groups[original_name].append(image_file)


# ======================================================
# SHOW ORIGINAL IMAGE GROUPS
# ======================================================

print("Original sonar images represented:", len(groups))


# ======================================================
# SHUFFLE ORIGINAL IMAGES
# ======================================================

original_images = sorted(groups.keys())

random.seed(RANDOM_SEED)

random.shuffle(original_images)


# ======================================================
# SPLIT ORIGINAL IMAGES
# ======================================================

total_original = len(original_images)

train_count = int(total_original * TRAIN_RATIO)
val_count = int(total_original * VAL_RATIO)

train_images = original_images[:train_count]

val_images = original_images[
    train_count:
    train_count + val_count
]

test_images = original_images[
    train_count + val_count:
]


print()
print("Original image split:")
print("Training:", len(train_images))
print("Validation:", len(val_images))
print("Testing:", len(test_images))


# ======================================================
# FUNCTION TO COPY PATCHES
# ======================================================

def copy_split(split_name, original_names):

    image_count = 0

    for original_name in original_names:

        patches = groups[original_name]

        for image_file in patches:

            label_file = LABEL_SOURCE / image_file.name

            if not label_file.exists():

                print(
                    "WARNING: Missing label:",
                    image_file.name
                )

                continue


            destination_image = (
                OUTPUT_DIR
                / split_name
                / "images"
                / image_file.name
            )

            destination_label = (
                OUTPUT_DIR
                / split_name
                / "labels"
                / label_file.name
            )


            shutil.copy2(
                image_file,
                destination_image
            )

            shutil.copy2(
                label_file,
                destination_label
            )

            image_count += 1


    return image_count


# ======================================================
# COPY DATA
# ======================================================

train_patches = copy_split(
    "train",
    train_images
)

val_patches = copy_split(
    "val",
    val_images
)

test_patches = copy_split(
    "test",
    test_images
)


# ======================================================
# FINAL RESULTS
# ======================================================

print()
print("=" * 50)
print("DATASET SPLIT COMPLETE")
print("=" * 50)

print("Training patches:", train_patches)
print("Validation patches:", val_patches)
print("Testing patches:", test_patches)

print(
    "Total:",
    train_patches + val_patches + test_patches
)

print()
print("Dataset created at:")
print(OUTPUT_DIR)