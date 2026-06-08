import torch
import numpy as np


class SimpleIntensityAugmentation:
    def __init__(self,
                 noise_std=0.02,
                 brightness_range=0.3,
                 contrast_range=0.3):

        self.noise_std = noise_std
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

    def __call__(self, pre_ct, post_ct):
        pre_np = pre_ct.numpy()
        post_np = post_ct.numpy()

        pre_np = self._intensity_augmentation(pre_np)
        post_np = self._intensity_augmentation(post_np)

        if self.noise_std > 0:
            pre_np = self._add_gaussian_noise(pre_np, self.noise_std)
            post_np = self._add_gaussian_noise(post_np, self.noise_std)

        pre_ct = torch.from_numpy(pre_np).float()
        post_ct = torch.from_numpy(post_np).float()

        return pre_ct, post_ct

    def _intensity_augmentation(self, volume):
        if self.brightness_range > 0:
            brightness_factor = 1 + np.random.uniform(
                -self.brightness_range, self.brightness_range
            )
            volume = volume * brightness_factor

        if self.contrast_range > 0:
            mean = volume.mean()
            contrast_factor = 1 + np.random.uniform(
                -self.contrast_range, self.contrast_range
            )
            volume = (volume - mean) * contrast_factor + mean

        volume = np.clip(volume, 0, 1)

        return volume

    def _add_gaussian_noise(self, volume, std):
        noise = np.random.normal(0, std, volume.shape)
        volume = volume + noise
        volume = np.clip(volume, 0, 1)
        return volume


class NoAugmentation:
    def __call__(self, pre, post):
        return pre, post
