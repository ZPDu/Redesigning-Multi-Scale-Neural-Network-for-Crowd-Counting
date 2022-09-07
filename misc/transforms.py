import numbers
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch
import torchvision.transforms as standard_transforms
from torchvision.transforms import functional as F
# ===============================img tranforms============================

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask

class FiveCrop(standard_transforms.FiveCrop):
    def forward(self, img, mask):
        return F.five_crop(img, self.size), F.five_crop(mask, self.size)

class Lambda(standard_transforms.Lambda):
    """Apply a user-defined lambda as a transform. This transform does not support torchscript.
    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd1, lambd2):
        # if not callable(lambd):
        #     raise TypeError(f"Argument lambd should be callable, got {repr(type(lambd).__name__)}")
        self.lambd1 = lambd1
        self.lambd2 = lambd2

    def __call__(self, img, mask):
        if self.lambd1 is not None:
          img = self.lambd1(img)
        if self.lambd2 is not None:
          mask = self.lambd2(mask)
        return img, mask

class RandomHorizontallyFlip(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask

class RandomHorizontallyFlip3(object):
    def __call__(self, img, mask, roi):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT), roi.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask, roi

class RandomResizedCrop(standard_transforms.RandomResizedCrop):  
    def __call__(self, img, mask):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation), F.resized_crop(mask, i, j, h, w, self.size, self.interpolation)

class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)
        # print('img')
        # print(img.size)
        # print('mask')
        # print(mask.size)
        assert img.size == mask.size
        w, h = img.size

        th, tw  = self.size
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            return img.resize((tw, th), Image.BILINEAR), mask.resize((tw, th), Image.NEAREST)

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))

# ===============================label tranforms============================

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor
        
class LabelNormalize(object):
    def __init__(self, para):
        self.para = para

    def __call__(self, tensor):
        # tensor = 1./(tensor+self.para).log()
        tensor = torch.from_numpy(np.array(tensor))
        tensor = tensor*self.para
        return tensor

class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()

class GTScaleDown(object):
    def __init__(self, factor=8):
        self.factor = factor

    def __call__(self, img):
        w, h = img.size
        if self.factor==1:
            return img
        tmp = np.array(img.resize((w//self.factor, h//self.factor), Image.BICUBIC))*self.factor*self.factor
        img = Image.fromarray(tmp)
        return img
