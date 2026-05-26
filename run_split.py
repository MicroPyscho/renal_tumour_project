import os
import shutil
import random
from pathlib import Path

SOURCE = Path("artifacts/data_ingestion/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone")
DEST   = Path("artifacts/data_ingestion/CT-KIDNEY-DATASET-split")
SPLIT  = 0.80
SEED   = 42

random.seed(SEED)

for class_dir in SOURCE.iterdir():
    if not class_dir.is_dir():
        continue

    images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
    random.shuffle(images)

    split_idx  = int(len(images) * SPLIT)
    train_imgs = images[:split_idx]
    val_imgs   = images[split_idx:]

    for subset, imgs in [("train", train_imgs), ("val", val_imgs)]:
        dest_dir = DEST / subset / class_dir.name
        dest_dir.mkdir(parents=True, exist_ok=True)
        for img in imgs:
            shutil.copy(img, dest_dir / img.name)

    print(f"{class_dir.name}: {len(train_imgs)} train | {len(val_imgs)} val")

print("\nDone.")