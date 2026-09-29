import torch
import glob
import numpy as np
from PIL import Image

from old_model import UNet


# --------------------------------------------------
# 1. Load trained model
# --------------------------------------------------

model = UNet()

model.load_state_dict(
    torch.load(
        "unet_shipwreck_v3.pth",
        map_location="cpu"
    )
)

model.eval()

print("Model loaded successfully!")


# --------------------------------------------------
# 2. Validation images
# --------------------------------------------------

files = glob.glob(
    "../dataset/val/images/*.png"
)

print("Validation images:", len(files))


# --------------------------------------------------
# 3. Store predictions and labels
# --------------------------------------------------

all_predictions = []
all_labels = []


for image_path in files:

    # Load sonar image
    image = Image.open(
        image_path
    ).convert("L")

    image_array = np.array(
        image,
        dtype=np.float32
    ) / 255.0


    # Convert image to tensor
    x = torch.from_numpy(
        image_array
    ).unsqueeze(0).unsqueeze(0)


    # Prediction
    with torch.no_grad():

        prediction = torch.sigmoid(
            model(x)
        ).squeeze().numpy()


    # Ground truth
    label_path = image_path.replace(
        "/images/",
        "/labels/"
    )

    label = (
        np.array(
            Image.open(label_path)
        ) > 0
    )


    all_predictions.append(
        prediction
    )

    all_labels.append(
        label
    )


# --------------------------------------------------
# 4. Threshold evaluation function
# --------------------------------------------------

def calculate_metrics(threshold):

    total_intersection = 0
    total_union = 0

    total_tp = 0
    total_fp = 0
    total_fn = 0


    for prediction, label in zip(
        all_predictions,
        all_labels
    ):

        predicted = prediction >= threshold


        # Intersection and union
        intersection = np.logical_and(
            predicted,
            label
        ).sum()

        union = np.logical_or(
            predicted,
            label
        ).sum()


        total_intersection += intersection
        total_union += union


        # Confusion values
        tp = np.logical_and(
            predicted,
            label
        ).sum()

        fp = np.logical_and(
            predicted,
            ~label
        ).sum()

        fn = np.logical_and(
            ~predicted,
            label
        ).sum()


        total_tp += tp
        total_fp += fp
        total_fn += fn


    # IoU
    if total_union == 0:
        iou = 0

    else:
        iou = (
            total_intersection /
            total_union
        )


    # Dice
    denominator = (
        2 * total_tp +
        total_fp +
        total_fn
    )

    if denominator == 0:
        dice = 0

    else:
        dice = (
            2 * total_tp /
            denominator
        )


    # Precision
    if total_tp + total_fp == 0:
        precision = 0

    else:
        precision = (
            total_tp /
            (total_tp + total_fp)
        )


    # Recall
    if total_tp + total_fn == 0:
        recall = 0

    else:
        recall = (
            total_tp /
            (total_tp + total_fn)
        )


    return (
        iou,
        dice,
        precision,
        recall
    )


# --------------------------------------------------
# 5. Test different thresholds
# --------------------------------------------------

thresholds = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.60,
    0.70
]


print()
print("==============================================")
print("          THRESHOLD EVALUATION")
print("==============================================")

print(
    f"{'Threshold':<12}"
    f"{'IoU':<12}"
    f"{'Dice':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
)

print("----------------------------------------------")


for threshold in thresholds:

    iou, dice, precision, recall = (
        calculate_metrics(threshold)
    )

    print(
        f"{threshold:<12.2f}"
        f"{iou:<12.4f}"
        f"{dice:<12.4f}"
        f"{precision:<12.4f}"
        f"{recall:<12.4f}"
    )


print("==============================================")