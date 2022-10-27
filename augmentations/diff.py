# Differentiable Augmentation for Data-Efficient GAN Training
# Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han
# https://arxiv.org/pdf/2006.10738

import torch
import torch.nn.functional as F


class DiffAug(object):

    def __call__(self, sample, ground_truth, channels_first=True):
        if not channels_first:
            sample = sample.permute(0, 3, 1, 2)
            ground_truth = ground_truth.permute(0, 3, 1, 2)

        sample, ground_truth = _augment(sample, ground_truth)

        if not channels_first:
            sample = sample.permute(0, 2, 3, 1)
            ground_truth = ground_truth.permute(0, 2, 3, 1)

        return sample.contiguous(), ground_truth.contiguous()


def _augment(sample, ground_truth):
    brightness = torch.rand(sample.size(0), 1, 1, 1, dtype=sample.dtype, device=sample.device)
    sample = rand_brightness(sample, brightness=brightness)
    ground_truth = rand_brightness(ground_truth, brightness=brightness)

    saturation = torch.rand(sample.size(0), 1, 1, 1, dtype=sample.dtype, device=sample.device)
    sample = rand_saturation(sample, saturation=saturation)
    ground_truth = rand_saturation(ground_truth, saturation=saturation)

    contrast = torch.rand(sample.size(0), 1, 1, 1, dtype=sample.dtype, device=sample.device)
    sample = rand_contrast(sample, contrast=contrast)
    ground_truth = rand_contrast(ground_truth, contrast=contrast)

    translation_ratio = 0.125
    shift_x, shift_y = int(sample.size(2) * translation_ratio + 0.5), int(sample.size(3) * translation_ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[sample.size(0), 1, 1], device=sample.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[sample.size(0), 1, 1], device=sample.device)
    sample = rand_translation(sample, translation_x=translation_x, translation_y=translation_y)
    ground_truth = rand_translation(ground_truth, translation_x=translation_x, translation_y=translation_y)

    resize_scale = np.random.rand()
    sample = rand_resize(sample, resize_scale=resize_scale)
    ground_truth = rand_resize(ground_truth, resize_scale=resize_scale)

    cutout_ratio = 0.25
    cutout_size = int(sample.size(2) * cutout_ratio + 0.5), int(x.size(3) * cutout_ratio + 0.5)
    offset_x = torch.randint(0, sample.size(2) + (1 - cutout_size[0] % 2), size=[sample.size(0), 1, 1],
                             device=sample.device)
    offset_y = torch.randint(0, sample.size(3) + (1 - cutout_size[1] % 2), size=[sample.size(0), 1, 1],
                             device=sample.device)
    sample = rand_cutout(sample, cutout_size=cutout_size, offset_x=offset_x, offset_y=offset_y)
    ground_truth = rand_cutout(ground_truth, cutout_size=cutout_size, offset_x=offset_x, offset_y=offset_y)

    return sample, ground_truth


def rand_brightness(x, brightness=None):
    x = x + (brightness - 0.5)
    return x


def rand_saturation(x, saturation=None):
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (saturation * 2) + x_mean
    return x


def rand_contrast(x, contrast=None):
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * (contrast + 0.5) + x_mean
    return x


def rand_translation(x, translation_x, translation_y):  # ratio: org: 0.125
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2).contiguous()
    return x


def rand_resize(x, resize_scale, min_ratio=0.8, max_ratio=1.2):  # ratio: org: 0.125
    resize_ratio = resize_scale * (max_ratio - min_ratio) + min_ratio
    resized_img = F.interpolate(x, size=int(resize_ratio * x.shape[3]), mode='bilinear')
    org_size = x.shape[3]

    if int(resize_ratio * x.shape[3]) < x.shape[3]:
        left_pad = (x.shape[3] - int(resize_ratio * x.shape[3])) / 2.
        left_pad = int(left_pad)
        right_pad = x.shape[3] - left_pad - resized_img.shape[3]
        x = F.pad(resized_img, (left_pad, right_pad, left_pad, right_pad), "constant", 0.)
    else:
        left = (int(resize_ratio * x.shape[3]) - x.shape[3]) / 2.
        left = int(left)
        x = resized_img[:, :, left:(left + x.shape[3]), left:(left + x.shape[3])]
    assert x.shape[2] == org_size
    assert x.shape[3] == org_size

    return x


def rand_cutout(x, cutout_size, offset_x, offset_y):
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x
