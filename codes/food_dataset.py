import cv2
import numpy as np
import os
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import matplotlib.pyplot as plt


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

"""Minority classes should be known after calculating class weights on the food dataset. 
For 251 food types inside the dataset folders, the minority classes are those with significantly fewer samples compared to the majority classes. 
These classes may require special attention during training, such as using a heavier augmentation pipeline or applying class weighting to address 
 the imbalance."""




def compute_class_weights(dataset):
    class_counts = {}
    for _, label in dataset:
        if label in class_counts:
            class_counts[label] += 1
        else:
            class_counts[label] = 1
    total_samples = sum(class_counts.values())
    weights = {label: total_samples / count for label, count in class_counts.items()}
    return weights

def weighted_random_sampler(dataset):
    class_weights = compute_class_weights(dataset)
    sample_weights = [class_weights[label] for _, label in dataset]
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler
def get_transforms(image_size: int = 224, augment: bool = True) -> A.Compose:
    """
    Standard augmentation pipeline.

    Used for majority classes during training and for validation / inference.
    Each transform is justified for cassava leaf classification:

    - RandomResizedCrop  : random scale + crop simulates varying distances from
                           the leaf and improves spatial invariance.
    - HorizontalFlip     : cassava leaves have no preferred left/right orientation.
    - Rotate             : field photos are rarely perfectly upright.
    - BrightnessContrast : lighting conditions vary significantly across Africa.
    - ColorJitter        : handles white-balance differences in cheap smartphones.
    - Affine             : small translations and scale changes add spatial diversity.
    - Normalize          : ImageNet statistics match the pretrained backbone priors.
    """
    if augment:
        return A.Compose([
            A.RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=25, border_mode=0, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.Affine(translate_percent=0.05, scale=(0.90, 1.10), rotate=(-15, 15), p=0.5),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    # Validation / inference: deterministic resize + normalise only.
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

def get_robust_transforms(image_size: int = 224) -> A.Compose:
    """
    Heavy augmentation pipeline for minority classes (CBB, CBSD, CGM).

    Extra operations vs the standard pipeline, each motivated by real-world
    African field conditions:

    - VerticalFlip       : cassava leaves have no canonical top/bottom orientation.
    - GaussianBlur       : mimics motion blur and low-quality smartphone optics.
    - ImageCompression   : simulates JPEG artefacts from cheap devices with small
                           internal storage that aggressively compress images.
    - CoarseDropout      : forces the model to use partial information, improving
                           robustness when lesions are partially obscured by dust,
                           fingers, or overlapping leaves.
    - RandomShadow       : simulates partial shadows cast by surrounding vegetation
                           — common in field photography under a canopy.
    """
    return A.Compose([
        A.RandomResizedCrop(size=(image_size, image_size), scale=(0.5, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.4),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=35, border_mode=0, p=0.6),
        A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.6),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.7),
        A.Affine(translate_percent=0.08, scale=(0.85, 1.15), rotate=(-20, 20), p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.4),
        A.ImageCompression(quality_range=(60, 95), p=0.3),
        A.CoarseDropout(
            num_holes_range=(4, 8),
            hole_height_range=(16, 32),
            hole_width_range=(16, 32),
            p=0.4,
        ),
        A.RandomShadow(p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

class FoodDataset(Dataset):
    def __init__( self, dataframe:  pd.DataFrame, images_dir: str | Path, augment:    bool = True, image_size: int  = 224, minority_classes: list = None) -> None:
        self.df         = dataframe.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.augment    = augment
        self.image_size = image_size
        self.minority_classes = minority_classes if minority_classes is not None else []

        # Pre-build both pipelines once — avoids rebuilding per sample.
        self._robust_tf   = get_robust_transforms(image_size)
        self._standard_tf = get_transforms(image_size, augment=augment)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row   = self.df.iloc[idx]
        label = int(row["label"])

        image_path = self.images_dir / row["image_id"]
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Route minority classes to the heavier augmentation pipeline.
        if self.augment and label in self.minority_classes:
            image = self._robust_tf(image=image)["image"]
        else:
            image = self._standard_tf(image=image)["image"]

        return image, torch.tensor(label, dtype=torch.long)
    



def audit_images(df, img_dir, min_std=5.0, min_bytes=1500):
    issues = []


    for idx, row in df.iterrows():
        path = os.path.join(img_dir, str(row['img_name']))

        # 1. File exists?
        if not os.path.exists(path):
            issues.append({'idx': idx, 'img_name': row['img_name'],
                        'label': row['label'], 'reason': 'missing'})
            continue

        # 2. Truncated file?
        if os.path.getsize(path) < min_bytes:
            issues.append({'idx': idx, 'img_name': row['img_name'],
                        'label': row['label'], 'reason': 'truncated'})
            continue

        # 3. Readable?
        try:
            img = Image.open(path).convert('RGB')
            arr = np.array(img, dtype=np.float32)
        except Exception as ex:
            issues.append({'idx': idx, 'img_name': row['img_name'],
                        'label': row['label'], 'reason': f'corrupt:{ex}'})
            continue

        # 4. Blank / near-black / near-white?
        mean, std = arr.mean(), arr.std()
        if std < min_std:
            issues.append({'idx': idx, 'img_name': row['img_name'],
                        'label': row['label'],
                        'reason': f'low_variance(std={std:.1f},mean={mean:.1f})'})
            

    return pd.DataFrame(issues)




