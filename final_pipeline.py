import os
import json
import csv

import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image
from scipy import ndimage

from old_model import UNet


# ============================================================
# OCEAN X — FINAL FULL-IMAGE PIPELINE
# ============================================================

MODEL_PATH = "unet_shipwreck_v3.pth"

IMAGE_PATH = os.environ.get(
    "OCEAN_X_IMAGE_PATH",
    "../train/images/DM_Wilson_03.png"
)

JSON_OUTPUT = "ocean_x_results.json"
CSV_OUTPUT = "ocean_x_results.csv"
IMAGE_OUTPUT = "ocean_x_final_detection.png"

DEVICE = torch.device("cpu")


# ============================================================
# AI CONFIGURATION
# ============================================================

PATCH_SIZE = 256
STRIDE = 128
BATCH_SIZE = 8

PIXEL_THRESHOLD = 0.35
MIN_AREA = 10
MEAN_CONFIDENCE_THRESHOLD = 0.45

# Distance used to merge nearby candidate regions
MERGE_DISTANCE = 80


# ============================================================
# LOAD MODEL
# ============================================================

model = UNet().to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(checkpoint)
model.eval()


print()
print("==========================================")
print("       OCEAN X FINAL PIPELINE")
print("==========================================")
print()

print("Model:", MODEL_PATH)
print("Input:", IMAGE_PATH)
print("Pixel threshold:", PIXEL_THRESHOLD)
print("Confidence threshold:", MEAN_CONFIDENCE_THRESHOLD)
print("Merge distance:", MERGE_DISTANCE)


# ============================================================
# LOAD IMAGE
# ============================================================

image = Image.open(
    IMAGE_PATH
).convert("L")

image_array = (
    np.array(
        image,
        dtype=np.float32
    ) / 255.0
)

height, width = image_array.shape

print()
print(
    "Image size:",
    width,
    "x",
    height
)


# ============================================================
# PAD IMAGE
# ============================================================

pad_h = (
    PATCH_SIZE
    - (height - PATCH_SIZE) % STRIDE
) % STRIDE

pad_w = (
    PATCH_SIZE
    - (width - PATCH_SIZE) % STRIDE
) % STRIDE

padded = np.pad(
    image_array,
    (
        (0, pad_h),
        (0, pad_w)
    ),
    mode="reflect"
)

padded_height, padded_width = padded.shape


# ============================================================
# CREATE PATCH POSITIONS
# ============================================================

positions = []

for y in range(
    0,
    padded_height - PATCH_SIZE + 1,
    STRIDE
):

    for x in range(
        0,
        padded_width - PATCH_SIZE + 1,
        STRIDE
    ):

        positions.append(
            (y, x)
        )


print(
    "Total patches:",
    len(positions)
)


# ============================================================
# PROBABILITY ACCUMULATORS
# ============================================================

probability_sum = np.zeros(
    (padded_height, padded_width),
    dtype=np.float32
)

prediction_count = np.zeros(
    (padded_height, padded_width),
    dtype=np.float32
)


# ============================================================
# AI INFERENCE
# ============================================================

print()
print("Running AI inference...")


for start in range(
    0,
    len(positions),
    BATCH_SIZE
):

    batch_positions = positions[
        start:start + BATCH_SIZE
    ]

    patches = []

    for y, x in batch_positions:

        patch = padded[
            y:y + PATCH_SIZE,
            x:x + PATCH_SIZE
        ]

        patches.append(
            patch
        )

    batch = np.stack(
        patches
    )

    batch_tensor = torch.from_numpy(
        batch
    ).unsqueeze(1).to(DEVICE)

    with torch.no_grad():

        outputs = model(
            batch_tensor
        )

        probabilities = torch.sigmoid(
            outputs
        ).cpu().numpy()[:, 0]

    for index, (y, x) in enumerate(
        batch_positions
    ):

        probability_sum[
            y:y + PATCH_SIZE,
            x:x + PATCH_SIZE
        ] += probabilities[index]

        prediction_count[
            y:y + PATCH_SIZE,
            x:x + PATCH_SIZE
        ] += 1

    completed = min(
        start + BATCH_SIZE,
        len(positions)
    )

    print(
        f"\rProcessed "
        f"{completed}/{len(positions)} patches",
        end=""
    )


print()
print("Inference complete!")


# ============================================================
# STITCH PROBABILITY MAP
# ============================================================

full_probability = (
    probability_sum
    / np.maximum(
        prediction_count,
        1
    )
)

full_probability = full_probability[
    :height,
    :width
]


# ============================================================
# PIXEL THRESHOLD
# ============================================================

raw_prediction = (
    full_probability
    >= PIXEL_THRESHOLD
)


# ============================================================
# CONNECTED COMPONENTS
# ============================================================

labeled, number_of_objects = ndimage.label(
    raw_prediction
)

print()
print(
    "Initial regions:",
    number_of_objects
)


# ============================================================
# CONFIDENCE FILTER
# ============================================================

candidate_detections = []

for object_id in range(
    1,
    number_of_objects + 1
):

    component = (
        labeled == object_id
    )

    area = component.sum()

    if area < MIN_AREA:
        continue

    mean_confidence = (
        full_probability[component].mean()
    )

    max_confidence = (
        full_probability[component].max()
    )

    if (
        mean_confidence
        < MEAN_CONFIDENCE_THRESHOLD
    ):
        continue

    ys, xs = np.where(
        component
    )

    candidate_detections.append({
        "area": int(area),
        "mean_confidence": float(
            mean_confidence
        ),
        "max_confidence": float(
            max_confidence
        ),
        "x_min": int(xs.min()),
        "y_min": int(ys.min()),
        "x_max": int(xs.max()),
        "y_max": int(ys.max())
    })


print(
    "After confidence filtering:",
    len(candidate_detections)
)


# ============================================================
# MERGING FUNCTIONS
# ============================================================

def boxes_are_close(
    box1,
    box2,
    distance=MERGE_DISTANCE
):

    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    if x1_max < x2_min:
        horizontal_gap = (
            x2_min - x1_max
        )

    elif x2_max < x1_min:
        horizontal_gap = (
            x1_min - x2_max
        )

    else:
        horizontal_gap = 0

    if y1_max < y2_min:
        vertical_gap = (
            y2_min - y1_max
        )

    elif y2_max < y1_min:
        vertical_gap = (
            y1_min - y2_max
        )

    else:
        vertical_gap = 0

    return (
        horizontal_gap <= distance
        and vertical_gap <= distance
    )


def merge_two_detections(
    det1,
    det2
):

    area = (
        det1["area"]
        + det2["area"]
    )

    mean_confidence = (
        det1["mean_confidence"]
        * det1["area"]
        +
        det2["mean_confidence"]
        * det2["area"]
    ) / max(area, 1)

    return {
        "area": int(area),

        "mean_confidence": float(
            mean_confidence
        ),

        "max_confidence": float(
            max(
                det1["max_confidence"],
                det2["max_confidence"]
            )
        ),

        "x_min": int(
            min(
                det1["x_min"],
                det2["x_min"]
            )
        ),

        "y_min": int(
            min(
                det1["y_min"],
                det2["y_min"]
            )
        ),

        "x_max": int(
            max(
                det1["x_max"],
                det2["x_max"]
            )
        ),

        "y_max": int(
            max(
                det1["y_max"],
                det2["y_max"]
            )
        )
    }


# ============================================================
# MERGE NEARBY REGIONS
# ============================================================

detections = [
    dict(detection)
    for detection in candidate_detections
]

changed = True

while changed:

    changed = False

    merged = []

    used = set()

    for i in range(
        len(detections)
    ):

        if i in used:
            continue

        current = detections[i]

        for j in range(
            i + 1,
            len(detections)
        ):

            if j in used:
                continue

            candidate = detections[j]

            box1 = (
                current["x_min"],
                current["y_min"],
                current["x_max"],
                current["y_max"]
            )

            box2 = (
                candidate["x_min"],
                candidate["y_min"],
                candidate["x_max"],
                candidate["y_max"]
            )

            if boxes_are_close(
                box1,
                box2
            ):

                current = merge_two_detections(
                    current,
                    candidate
                )

                used.add(j)
                changed = True

        merged.append(
            current
        )

        used.add(i)

    detections = merged


# ============================================================
# ADD TARGET IDs
# ============================================================

final_detections = []

for index, detection in enumerate(
    detections,
    start=1
):

    detection["target_id"] = index

    detection["image"] = os.path.basename(
        IMAGE_PATH
    )

    final_detections.append(
        detection
    )


# ============================================================
# SORT BY AREA
# ============================================================

final_detections.sort(
    key=lambda item: item["area"],
    reverse=True
)

# Re-number after sorting
for index, detection in enumerate(
    final_detections,
    start=1
):

    detection["target_id"] = index


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print()
print("==========================================")
print("         FINAL TARGET DETECTIONS")
print("==========================================")

print(
    "Final candidate regions:",
    len(final_detections)
)

for detection in final_detections:

    print()

    print(
        "Target:",
        detection["target_id"]
    )

    print(
        "Area:",
        detection["area"],
        "pixels"
    )

    print(
        "Mean confidence:",
        f"{detection['mean_confidence']:.3f}"
    )

    print(
        "Maximum confidence:",
        f"{detection['max_confidence']:.3f}"
    )

    print(
        "Bounding box:",
        (
            detection["x_min"],
            detection["y_min"],
            detection["x_max"],
            detection["y_max"]
        )
    )


# ============================================================
# SAVE JSON
# ============================================================

json_data = {
    "system": "Ocean X",
    "problem_statement": "SIH26057",
    "input_image": os.path.basename(
        IMAGE_PATH
    ),
    "image_width": int(width),
    "image_height": int(height),

    "configuration": {
        "model": MODEL_PATH,
        "patch_size": PATCH_SIZE,
        "stride": STRIDE,
        "pixel_threshold": PIXEL_THRESHOLD,
        "minimum_area": MIN_AREA,
        "mean_confidence_threshold":
            MEAN_CONFIDENCE_THRESHOLD,
        "merge_distance": MERGE_DISTANCE
    },

    "detections": final_detections
}


with open(
    JSON_OUTPUT,
    "w"
) as file:

    json.dump(
        json_data,
        file,
        indent=4
    )


# ============================================================
# SAVE CSV
# ============================================================

csv_fields = [
    "target_id",
    "image",
    "area",
    "mean_confidence",
    "max_confidence",
    "x_min",
    "y_min",
    "x_max",
    "y_max"
]

with open(
    CSV_OUTPUT,
    "w",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=csv_fields
    )

    writer.writeheader()

    for detection in final_detections:

        writer.writerow(
            {
                field: detection[field]
                for field in csv_fields
            }
        )


# ============================================================
# CREATE FINAL VISUALIZATION
# ============================================================

overlay = np.stack(
    [image_array] * 3,
    axis=-1
)

# Draw each final bounding box
for detection in final_detections:

    x_min = detection["x_min"]
    y_min = detection["y_min"]

    x_max = detection["x_max"]
    y_max = detection["y_max"]

    # Bounding box
    overlay[
        y_min:y_min + 3,
        x_min:x_max + 1
    ] = [1, 0, 0]

    overlay[
        y_max - 2:y_max + 1,
        x_min:x_max + 1
    ] = [1, 0, 0]

    overlay[
        y_min:y_max + 1,
        x_min:x_min + 3
    ] = [1, 0, 0]

    overlay[
        y_min:y_max + 1,
        x_max - 2:x_max + 1
    ] = [1, 0, 0]


# ============================================================
# SAVE FIGURE
# ============================================================

plt.figure(
    figsize=(18, 6)
)

plt.subplot(
    1,
    3,
    1
)

plt.imshow(
    image_array,
    cmap="gray"
)

plt.title(
    "Original Side-Scan Sonar"
)

plt.axis("off")


plt.subplot(
    1,
    3,
    2
)

plt.imshow(
    full_probability,
    cmap="viridis"
)

plt.colorbar(
    fraction=0.046,
    pad=0.04
)

plt.title(
    "Ocean X AI Probability Map"
)

plt.axis("off")


plt.subplot(
    1,
    3,
    3
)

plt.imshow(
    overlay
)

plt.title(
    "Ocean X Final Target Detection"
)

plt.axis("off")


plt.tight_layout()

plt.savefig(
    IMAGE_OUTPUT,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# COMPLETE
# ============================================================

print()
print("==========================================")
print("       OCEAN X PIPELINE COMPLETE")
print("==========================================")

print()
print(
    "JSON saved:",
    JSON_OUTPUT
)

print(
    "CSV saved:",
    CSV_OUTPUT
)

print(
    "Visualization saved:",
    IMAGE_OUTPUT
)

print()