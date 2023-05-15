import os
from glob import glob
from typing import Callable, Tuple

import pytorch_lightning as pl
import torch
import torch.utils.data as data
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import (ColorJitter, Compose, GaussianBlur,
                                    Normalize, RandomApply, RandomGrayscale,
                                    RandomHorizontalFlip, RandomResizedCrop,
                                    RandomVerticalFlip, Resize, RandomRotation,
                                    ToTensor)

from torchvision.io import read_image


class CANDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        batch_size: int = 256,
        workers: int = 20,
        num_val_samples: int = 1000,
        crop_size: int = 224,
    ):
        """Duel view data module

        Args:
            root: Path to image directory
            batch_size: Number of batch samples
            workers: Number of data workers
            num_val_samples: Number of samples to leave out for a validation set
            crop_size: Size of image crop
            min_scale: Minimum crop scale
            max_scale: Maximum crop scale
            brightness: Brightness intensity
            contrast: Contast intensity
            saturation: Saturation intensity
            hue: Hue intensity
            color_jitter_prob: Probability of applying color jitter
            gray_scale_prob: Probability of converting to grayscale
            flip_prob: Probability of applying horizontal flip
            gaussian_prob: Probability of applying Gaussian blurring
            mean: Image normalization channel means
            std: Image normalization channel standard deviations
        """
        super().__init__()
        self.save_hyperparameters()
        self.root = root
        self.batch_size = batch_size
        self.workers = workers
        self.num_val_samples = num_val_samples
        self.crop_size = crop_size

        self.init_transforms = InitialTransforms(size=self.crop_size)

    def setup(self, stage: str = "fit"):
        if stage == "fit":
            dataset = SimpleDataset(self.root, self.init_transforms)

            # Randomly take num_val_samples images for a validation set
            self.train_dataset, self.val_dataset = data.random_split(
                dataset,
                [len(dataset) - self.num_val_samples, self.num_val_samples],
                generator=torch.Generator().manual_seed(42),  # Fixed seed
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            #prefetch_factor=8
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,
        )


class SimpleDataset(data.Dataset):
    def __init__(self, root: str, init_transforms: Callable):
        """Image dataset from nested directory

        Args:
            root: Path to directory
            transforms: Image augmentations
        """
        super().__init__()
        self.root = root
        # print number of children of directory
        print(f"Number of children: {len(os.listdir(root))}")
        self.paths = [
            f for f in glob(f"{root}/**/*", recursive=True) if os.path.isfile(f)
        ]
        self.init_transforms = init_transforms

        print(f"Loaded {len(self.paths)} images from {root}")

    def __getitem__(self, index: int):

        img = read_image(self.paths[index]).type(torch.float32)
        img = self.init_transforms(img)
        return img

    def __len__(self):
        return len(self.paths)


class InitialTransforms:
    def __init__(
        self,
        size: int = 224,
    ):
        """Initial transform for image homogeneity (to be executed on CPU)

        Args:
            size: Size of image resize
        """
        self.transforms = Compose([
            Resize((size, size)),
        ])

    def __call__(self, img: torch.Tensor):
        return self.transforms(img)



class Transforms:
    def __init__(
        self,
        crop_size: int = 224,
        min_scale: float = 0.08,
        max_scale: float = 1.0,
        brightness: float = 0.8,
        contrast: float = 0.8,
        saturation: float = 0.8,
        hue: float = 0.2,
        color_jitter_prob: float = 0.8,
        gray_scale_prob: float = 0.2,
        gaussian_prob: float = 0.5,
        flip_prob: float = 0.5,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.228, 0.224, 0.225),
    ):
        """Augmentation pipeline for contrastive learning (to be executed on accelerator)

        Args:
            crop_size: Size of image crop
            min_scale: Minimum crop scale
            max_scale: Maximum crop scale
            brightness: Brightness intensity
            contast: Contast intensity
            saturation: Saturation intensity
            hue: Hue intensity
            color_jitter_prob: Probability of applying color jitter
            gray_scale_prob: Probability of converting to grayscale
            gaussian_prob: Probability of applying Gausian blurring
            flip_prob: Probability of applying horizontal flip
            mean: Image normalization means
            std: Image normalization standard deviations
        """
        super().__init__()

        self.transforms = Compose(
            [
                RandomResizedCrop(size=crop_size, scale=(min_scale, max_scale)),
                RandomApply(
                    [
                        ColorJitter(
                            brightness=brightness,  # type:ignore
                            contrast=contrast,  # type:ignore
                            saturation=saturation,  # type:ignore
                            hue=hue,  # type:ignore
                        )
                    ],
                    p=color_jitter_prob,
                ),
                RandomGrayscale(p=gray_scale_prob),
                RandomApply([GaussianBlur(kernel_size=23)], p=gaussian_prob),
                RandomHorizontalFlip(p=flip_prob),
                RandomVerticalFlip(p=flip_prob),
                #RandomRotation(180, expand=False),

                #Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img: torch.Tensor):
        return self.transforms(img)
