import torch
import numpy as np
import cv2
from PIL import Image
import torch.nn.functional as f
from .monitoring import plot_img
import math

class Binarize(object):
    """ Binarize the input given a threshold on the pixel value.

    Args:
        binary_threshold:  all pixels with a value lower than the binary threshold are going to be 0, all others are 1.
    """

    def __init__(self, binary_threshold=0.5):
        self.binary_threshold = binary_threshold

    def __call__(self, sample):
        mask = torch.zeros_like(sample)
        mask[sample > self.binary_threshold] = 1

        return mask

class Invert(object):
    """ Binarize the input given a threshold on the pixel value.

    Args:
        binary_threshold:  all pixels with a value lower than the binary threshold are going to be 0, all others are 1.
    """

    def __init__(self, binary_threshold=0.5):
        self.binary_threshold = binary_threshold

    def __call__(self, sample):
        mask = torch.zeros_like(sample)
        mask[sample != 1] = 1 - sample[sample != 1]
        mask /= mask.max()

        return mask

class Scale_0_1(object):
    """ Binarize the input given a threshold on the pixel value.

    Args:
        binary_threshold:  all pixels with a value lower than the binary threshold are going to be 0, all others are 1.
    """

    def __init__(self):
        self.te = 1

    def __call__(self, sample):
        sample = sample - sample.min()
        sample /= sample.max()
        return sample


class Binarize_batch(object):
    """ Binarize the input given a threshold on the pixel value.

    Args:
        binary_threshold:  all pixels with a value lower than the binary threshold are going to be 0, all others are 1.
    """

    def __init__(self, binary_threshold=0.5):
        self.binary_threshold = binary_threshold

    def __call__(self, sample):
        mask = torch.zeros_like(sample)
        mask[sample > self.binary_threshold] = 1

        return mask


class Scale_0_1_batch(object):
    """ Binarize the input given a threshold on the pixel value.

    Args:
        binary_threshold:  all pixels with a value lower than the binary threshold are going to be 0, all others are 1.
    """

    def __init__(self):
        self.te = 1

    def __call__(self, sample):
        mini,_ = sample.view(sample.size(0), -1).min(dim=1)
        sample = sample - mini.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        maxi, _ = sample.view(sample.size(0), -1).max(dim=1)
        sample /= maxi.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return sample

class Dilate(object):
    def __init__(self, kernel_size):
        self.kernel_size=kernel_size
        self.kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)

    def __call__(self, sample):
        cv_img = np.array(sample)
        dilated_image = cv2.dilate(cv_img, self.kernel, iterations=1)
        return Image.fromarray(dilated_image)


class ScaleCenter(object):
    def __init__(self, padding=10):
        self.coverage_per_category = torch.load('utils/coverage_per_category.pkl')
        self.global_mean = self.coverage_per_category.mean()
        self.padding = padding

    def __call__(self, image, label):
        ratio = self.global_mean/self.coverage_per_category[label]
        #plot_img(image.unsqueeze(0))
        non_zeros = (image[0, :, :] != 0).nonzero()
        padded_data = f.pad(image.unsqueeze(0), (self.padding, self.padding, self.padding, self.padding), 'constant', value=0.0)
        transformed_data = torch.zeros_like(image)
        #plot_img(padded_data)
        h_min, h_max = non_zeros[:, 0].min(), non_zeros[:, 0].max()
        w_min, w_max = non_zeros[:, 1].min(), non_zeros[:, 1].max()
        height, width = h_max - h_min, w_max - w_min
        max_square = max(height, width)
        center = h_min + height // 2 + self.padding, w_min + width // 2 + self.padding
        # center = h_min + height // 2, w_min + width // 2
        if max_square % 2 != 0:
            offset_sq = 1
        else:
            offset_sq = 0
        crop = padded_data[:, :, center[0] - max_square // 2: center[0] + max_square // 2 + offset_sq + 1,
               center[1] - max_square // 2: center[1] + max_square // 2 + offset_sq + 1]

        #plot_img(crop)

        output_size = min(int(math.sqrt(ratio) * crop.size(-1)), image.size(-1))

        resized_crop = f.interpolate(crop, size=(output_size, output_size))
        #plot_img(resized_crop)
        if resized_crop.size(2) > image.size(-1):
            resized_crop = f.interpolate(crop, size=(image.size(-2), image.size(-1)))

        image_center = image.size(-2) // 2, image.size(-1) // 2
        if resized_crop.size(-1) % 2 == 0:
            length = resized_crop.size(-2) // 2
        else:
            length = (resized_crop.size(-2) // 2) + 1

        transformed_data[:, image_center[0] - resized_crop.size(-2) // 2: image_center[0] + length,
                image_center[1] - resized_crop.size(-1) // 2: image_center[1] + length] = resized_crop[0]
        #plot_img(transformed_data.unsqueeze(0))
        return transformed_data



