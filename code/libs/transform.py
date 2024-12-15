import torch
import torch.nn as nn
from torchvision.transforms import functional as F

class Compose:
    """
    Compose a set of transforms that are jointly applied to
    input image and its corresponding detection annotations (e.g., boxes)
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(nn.Module):
    """
    Convert an image (PIL or np.array) to tensor.
    This function will additional perform normalization so that each pixel value
    is a floating point number in the range of [0, 1].
    """

    def forward(self, image, target):
        image = F.to_tensor(image)
        return image, target


# TODO: Delete if unused
def _resize_image(image, img_min_size, img_max_size):
    """
    Resize an image such that its shortest side = img_min_size
    and its largest side is <= img_max_size
    """
    im_shape = torch.tensor(image.shape[-2:])
    min_size = torch.min(im_shape).to(dtype=torch.float32)
    max_size = torch.max(im_shape).to(dtype=torch.float32)
    scale = torch.min(img_min_size / min_size, img_max_size / max_size)
    scale_factor = scale.item()

    image = torch.nn.functional.interpolate(
        image[None],
        size=None,
        scale_factor=scale_factor,
        mode="bilinear",
        recompute_scale_factor=True,
        align_corners=False,
    )[0]

    return image
