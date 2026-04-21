from typing import List, Tuple, Union, Optional
import cv2
import numpy as np
import albumentations as A
from torchvision import transforms
import torch
import pywt

class WaveletSuppression:
    def __init__(self, suppression_type='texture', wavelet='haar', level=1):
        """
        suppression_type: 'texture' (keeps shape/LL) or 'shape' (keeps texture/details)
        """
        self.suppression_type = suppression_type
        self.wavelet = wavelet
        self.level = level

    def __call__(self, img_tensor):
        # 1. Convert PyTorch Tensor (C, H, W) to Numpy (H, W, C)
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        reconstructed_channels = []
        
        # 2. Process each color channel independently
        for i in range(img_np.shape[2]):
            channel = img_np[:, :, i]
            
            # Decompose into components: LL (Shape) and details (Texture)
            coeffs = pywt.wavedec2(channel, self.wavelet, level=self.level)
            
            if self.suppression_type == 'texture':
                # Keep LL (Shape), zero out high-frequency details
                new_coeffs = [coeffs[0]]
                for detail_tuple in coeffs[1:]:
                    new_coeffs.append(tuple(np.zeros_like(d) for d in detail_tuple))
            
            elif self.suppression_type == 'shape':
                # Zero out LL (Shape), keep high-frequency details
                new_coeffs = [np.zeros_like(coeffs[0])]
                new_coeffs.extend(coeffs[1:])
            else:
                raise ValueError(f"Unknown suppression_type: {self.suppression_type}")

            # Reconstruct the channel
            rec_channel = pywt.waverec2(new_coeffs, self.wavelet)
            
            # Fix any slight size mismatches from padding
            rec_channel = rec_channel[:channel.shape[0], :channel.shape[1]]
            reconstructed_channels.append(rec_channel)

        # 3. Stack channels, clip valid pixel values, and return as Tensor
        img_rec = np.stack(reconstructed_channels, axis=2)
        img_rec = np.clip(img_rec, 0, 1) 
        return torch.from_numpy(img_rec).permute(2, 0, 1).float()

class CenterCrop(object):
    def __init__(self, size: Tuple[int, int], p: float = 1):
        self.CenterCrop = A.CenterCrop(size, size[2], p=p)

class HorizontalFlip(object):
    def __init__(self, p: float = 0.5):
        self.HorizontalFlip = A.HorizontalFlip(p=p)

class RandomResizedCrop:
    def __init__(self, resize_size: Union[int, Tuple[int, int]] = (120, 120), scale: Tuple[float, float] = (0.08, 1.0), ratio: Tuple[float, float] = (0.75, 1.3333333333333333), p: float = 1.0):
        h, w = (resize_size, resize_size) if type(resize_size) != tuple else resize_size
        self.RandomResizedCrop = A.RandomResizedCrop(h, w, scale=scale, ratio=ratio, p=p)

class Resize:
    def __init__(self, size: Tuple[int, int]):
        self.Resize = A.Resize(size, size[2])

class ContinousGrayScale:
    def __init__(self, alpha: float = 1.0, p: float = 1.0):
        assert 0.0 <= alpha <= 1.0, "Alpha must be in the range [2]."
        self.alpha = alpha
        self.p = p

class ChannelShuffle:
    def __init__(self, p: float = 1.0):
        self.p = p

class BilateralFilter:
    def __init__(self, d: int = 5, sigma_color: int = 75, sigma_space: int = 75, p: float = 1.0):
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
        self.p = p

class FastNLMeansDenoising:
    def __init__(self, h: int = 5, template_window_size: int = 7, search_window_size: int = 21, p: float = 1.0):
        self.h = h
        self.template_window_size = template_window_size
        self.search_window_size = search_window_size
        self.p = p

class GaussianBlur:
    def __init__(self, k: int = 5, sigma: float = 1.0, p: float = 1.0):
        self.k = k
        self.sigma = sigma
        self.p = p

class PatchShuffle:
    def __init__(self, grid_size: int = 3, p: float = 1.0):
        self.GridShuffle = A.RandomGridShuffle(grid=(grid_size, grid_size), p=p)
        self.p = p

class PatchRotation:
    def __init__(self, grid_size: int = 3, p: float = 1.0, output_size: tuple = (224, 224), interpolation=cv2.INTER_LINEAR):
        self.grid_size = grid_size
        self.p = p
        self.output_size = output_size
        self.interpolation = interpolation

class CutOut(object):
    def __init__(self, max_edge: float = 0.7, min_edge: float = 0.2, p: float = 0.5):
        self.CutOut = A.CoarseDropout(max_holes=1, max_height=max_edge, max_width=max_edge, min_height=min_edge, min_width=min_edge, p=p)

class CutMix:
    def __init__(self, alpha: float = 1.0, p: float = 1.0):
        self.alpha = alpha
        self.p = p

class MixUp:
    def __init__(self, alpha: float = 1.0, p: float = 1.0):
        self.alpha = alpha
        self.p = p

def get_dataset_statistics(dataset):
    if dataset in ['imagenet', 'oxfordiiitpet', 'caltech101', 'flowers102', 'stl10', 'imagenet16']:
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
    elif dataset == 'bloodmnist':
        mean = [0.796, 0.659, 0.696]
        std  = [0.226, 0.259, 0.096]
    elif dataset == 'chestmnist':
        mean = [0.497, 0.497, 0.497]
        std  = [0.247, 0.247, 0.247]
    elif dataset == 'dermamnist':
        mean = [0.763, 0.538, 0.561]
        std  = [0.136, 0.158, 0.176]
    elif dataset == 'pathmnist':
        mean = [0.740, 0.532, 0.705]
        std  = [0.165, 0.217, 0.157]
    elif dataset == 'retinamnist':
        mean = [0.394, 0.241, 0.145]
        std  = [0.323, 0.210, 0.151]
    elif dataset == 'aid':
        mean = [0.397, 0.408, 0.368]
        std  = [0.216, 0.194, 0.191]
    elif dataset == 'patternnet':
        mean = [0.359, 0.360, 0.319]
        std  = [0.195, 0.185, 0.178]
    elif dataset == 'rsd46whu':
        mean = [0.378, 0.419, 0.373]
        std  = [0.207, 0.178, 0.171]
    elif dataset == 'ucmerced':
        mean = [0.483, 0.489, 0.450]
        std  = [0.217, 0.201, 0.195]
    elif dataset == 'deepglobe':
        mean = [0.407, 0.380, 0.283]
        std  = [0.150, 0.118, 0.108]
    return mean, std

def get_transform(
    train_augmentations: str, test_augmentations: str, p: float, p_list: Optional[List[int]],
    resize_size: Optional[int], grid_size: Optional[int], gray_alpha: Optional[float],
    bilateral_d: Optional[int], sigma_color: Optional[int], sigma_space: Optional[int],
    nlmeans_h: Optional[int], template_window_size: Optional[int], search_window_size: Optional[int],
    gaussian_k: Optional[int], gaussian_sigma: Optional[float], split: str = 'train', dataset: str = 'imagenet',
    wavelet_type: Optional[str] = 'haar', wavelet_level: Optional[int] = 1
) -> List[object]:

    mean, std = get_dataset_statistics(dataset)
    compose = []

    size = int(resize_size) if resize_size is not None else 224
    # Native image sizes differ across samples (e.g. Caltech-101); batching requires fixed H×W before ToTensor.
    compose.append(transforms.Resize((size, size)))

    compose.append(transforms.ToTensor())

    # 2. Apply our DWT Suppression to the Tensor
    if 'wavelettexture' in test_augmentations:
        compose.append(WaveletSuppression(
            suppression_type='texture', wavelet=wavelet_type, level=wavelet_level
        ))
    elif 'waveletshape' in test_augmentations:
        compose.append(WaveletSuppression(
            suppression_type='shape', wavelet=wavelet_type, level=wavelet_level
        ))

    # 3. Normalize the final Tensor using dataset stats
    compose.append(transforms.Normalize(mean, std))

    return transforms.Compose(compose)