import os
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
import matplotlib
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import glob
matplotlib.use('TKAgg')


class DogCats(Dataset):
    def __init__(self, size=(224, 224), dataset_path='/media/bonilla/My Book/DogsCats/data/train/*.jpg', augment=True):
        self.size = size
        self.transform = transforms.Compose([self.ToTensor(augment)])
        self.images_list = glob.glob(dataset_path)
        self.seq = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Sometimes(
                0.5,
                iaa.GaussianBlur(sigma=(0, 0.5))
            ),
            iaa.LinearContrast((0.75, 1.5)),
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
            iaa.Multiply((0.8, 1.2), per_channel=0.2),
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-25, 25),
                shear=(-8, 8)
            )
        ], random_order=True)

    @staticmethod
    class ToTensor(object):
        def __init__(self, augment):
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self.augment_pipeline = None
            if augment:
                self.augment_pipeline = iaa.Sequential([
                    iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),
                    iaa.Fliplr(0.5),
                    iaa.Affine(rotate=(-20, 20), mode='symmetric'),
                    iaa.Sometimes(0.25, iaa.OneOf(
                        [
                            iaa.Dropout(p=(0, 0.1)),
                            iaa.CoarseDropout(0.1, size_percent=0.5)
                         ])),
                    iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)
                ])

        def __call__(self, sample):
            image = sample
            if self.augment_pipeline is not None:
                image = self.augment_pipeline.augment_image(image)
            image = image.astype('float32') / 255.
            image = image.transpose((2, 0, 1))
            image = torch.from_numpy(image)
            image = self.normalize(image)
            return image

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image = cv2.imread(self.images_list[idx])
        image = cv2.resize(image, self.size)[:, :, ::-1]
        augmented = self.seq.augment_image(image)
        image = self.transform(image)
        augmented = self.transform(augmented)
        return image, augmented


if __name__ == '__main__':
    dl = DogCats()
    a = dl[0]
    print()
