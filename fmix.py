#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author :    yongle.wang
@Email :  wangyl306@163.com
@Time  : 2021/08/16 13:45:27
"""

# * FMix: Enhancing Mixed Sample Data Augmentation [https://arxiv.org/abs/2002.12047]

import math
import random
import os
import numpy as np
from scipy.stats import beta
import cv2


"""
FMix方法流程如下：

    从beta分布中采样得到lambda值
    从傅里叶空间中获取低频图像
    利用低频图像获取二值图像mask
    随机选择两张图像Image1,Image2
    分别计算 x_1​ = Image2 * mask 和 x_2= Image1 X（1-mask）
    输出图像Image = x_1 + x_2
"""


def fftfreqnd(h, w=None, z=None):
    """ Get bin values for discrete fourier transform of size (h, w, z)

    :param h: Required, first dimension size
    :param w: Optional, second dimension size
    :param z: Optional, third dimension size
    """
    fz = fx = 0
    fy = np.fft.fftfreq(h)

    if w is not None:
        fy = np.expand_dims(fy, -1)

        if w % 2 == 1:
            fx = np.fft.fftfreq(w)[: w // 2 + 2]
        else:
            fx = np.fft.fftfreq(w)[: w // 2 + 1]

    if z is not None:
        fy = np.expand_dims(fy, -1)
        if z % 2 == 1:
            fz = np.fft.fftfreq(z)[:, None]
        else:
            fz = np.fft.fftfreq(z)[:, None]

    return np.sqrt(fx * fx + fy * fy + fz * fz)


def get_spectrum(freqs, decay_power, ch, h, w=0, z=0):
    """ 获取傅里叶图像 with given size and frequencies decayed by decay power

    :param freqs: Bin values for the discrete fourier transform
    :param decay_power: Decay power for frequency decay prop 1/f**d
    :param ch: Number of channels for the resulting mask
    :param h: Required, first dimension size
    :param w: Optional, second dimension size
    :param z: Optional, third dimension size
    """
    scale = np.ones(1) / (np.maximum(freqs, np.array([1.0 / max(w, h, z)])) ** decay_power)

    param_size = [ch] + list(freqs.shape) + [2]
    param = np.random.randn(*param_size)

    scale = np.expand_dims(scale, -1)[None, :]

    return scale * param


def make_low_freq_image(decay, shape, ch=1):
    """ 从傅里叶空间采样一个低频图像 from fourier space

    :param decay_power: Decay power for frequency decay prop 1/f**d
    :param shape: Shape of desired mask, list up to 3 dims
    :param ch: Number of channels for desired mask
    """
    freqs = fftfreqnd(*shape)
    spectrum = get_spectrum(freqs, decay, ch, *shape)  # .reshape((1, *shape[:-1], -1))
    spectrum = spectrum[:, 0] + 1j * spectrum[:, 1]
    mask = np.real(np.fft.irfftn(spectrum, shape))

    if len(shape) == 1:
        mask = mask[:1, : shape[0]]
    if len(shape) == 2:
        mask = mask[:1, : shape[0], : shape[1]]
    if len(shape) == 3:
        mask = mask[:1, : shape[0], : shape[1], : shape[2]]

    mask = mask
    mask = mask - mask.min()
    mask = mask / mask.max()
    return mask


def sample_lam(alpha, reformulate=False):
    """ Sample a lambda from symmetric beta distribution with given alpha

    :param alpha: Alpha value for beta distribution
    :param reformulate: If True, uses the reformulation of [1].
    """
    if reformulate:
        lam = beta.rvs(alpha + 1, alpha)
    else:
        lam = beta.rvs(alpha, alpha)

    return lam


def binarise_mask(mask, lam, in_shape, max_soft=0.0):
    """ 二值化低频图像，使得有平均lambda值

    :param mask: Low frequency image, usually the result of `make_low_freq_image`
    :param lam: Mean value of final mask
    :param in_shape: Shape of inputs
    :param max_soft: Softening value between 0 and 0.5 which smooths hard edges in the mask.
    :return:
    """
    idx = mask.reshape(-1).argsort()[::-1]
    mask = mask.reshape(-1)
    num = math.ceil(lam * mask.size) if random.random() > 0.5 else math.floor(lam * mask.size)

    eff_soft = max_soft
    if max_soft > lam or max_soft > (1 - lam):
        eff_soft = min(lam, 1 - lam)

    soft = int(mask.size * eff_soft)
    num_low = num - soft
    num_high = num + soft

    mask[idx[:num_high]] = 1
    mask[idx[num_low:]] = 0
    mask[idx[num_low:num_high]] = np.linspace(1, 0, (num_high - num_low))

    mask = mask.reshape((1, *in_shape))
    return mask


def sample_mask(alpha, decay_power, shape, max_soft=0.0, reformulate=False):
    """ Samples a mean lambda from beta distribution parametrised by alpha, creates a low frequency image and binarises
    it based on this lambda

    :param alpha: Alpha value for beta distribution from which to sample mean of mask
    :param decay_power: Decay power for frequency decay prop 1/f**d
    :param shape: Shape of desired mask, list up to 3 dims
    :param max_soft: Softening value between 0 and 0.5 which smooths hard edges in the mask.
    :param reformulate: If True, uses the reformulation of [1].
    """
    if isinstance(shape, int):
        shape = (shape,)

    # Choose lambda
    lam = sample_lam(alpha, reformulate)

    # Make mask, get mean / std
    mask = make_low_freq_image(decay_power, shape)
    mask = binarise_mask(mask, lam, shape, max_soft)

    # * save mask
    new_mask = mask.repeat(3, axis=0)[np.newaxis, :, :, :]
    new_mask = convert_img_to_save(new_mask)
    cv2.imwrite("./mask.jpg", new_mask)

    return lam, mask


"""
# * MASK
soft_masks_np = [make_low_freq_image(decay_power, [SHAPE, SHAPE]) for _ in range(NUM_IMAGES)]

masks_np = [binarise_mask(mask, LAMBDA, [SHAPE, SHAPE]) for mask in soft_masks_np]
masks = torch.from_numpy(np.stack(masks_np, axis=0)).float().repeat(1, 3, 1, 1)
"""


def sample_and_apply(sample1, sample2, alpha, decay_power, max_soft=0.0, reformulate=False):
    """

    :param x: Image batch on which to apply fmix of shape [b, c, shape*]
    :param alpha: Alpha value for beta distribution from which to sample mean of mask
    :param decay_power: Decay power for frequency decay prop 1/f**d
    :param shape: Shape of desired mask, list up to 3 dims
    :param max_soft: Softening value between 0 and 0.5 which smooths hard edges in the mask.
    :param reformulate: If True, uses the reformulation of [1].
    :return: mixed input, permutation indices, lambda value of mix,
    """
    # 判断shape
    width = sample1.shape[1]
    height = sample1.shape[0]
    shape = (width, height)
    print("sample_shape:", sample1.shape)
    if sample1.shape != sample2.shape:
        sample2 = cv2.resize(sample2, shape)

    lam, mask = sample_mask(alpha, decay_power, sample1.shape[:2], max_soft, reformulate)

    sample1 = sample1[np.newaxis, :, :, :]
    sample1 = np.transpose(sample1, (0, 3, 1, 2))
    sample2 = sample2[np.newaxis, :, :, :]
    sample2 = np.transpose(sample2, (0, 3, 1, 2))

    x1, x2 = sample1 * mask, sample2 * (1 - mask)

    return x1 + x2, x1, x2, lam


def convert_img_to_save(original_img):
    """[summary]

    Arguments:
        original_img {[ndarray]} -- [b,c,h,w]

    Returns:
        [ndrarray] -- [h,w,c]
    """
    img_save = np.transpose(original_img, (0, 2, 3, 1))
    img_save = np.squeeze(img_save)

    return img_save


if __name__ == "__main__":
    data_dir = "./data"
    alpha = 1
    decay_power = 3.0
    max_soft = 0.0
    reformulate = False

    samples = random.sample(os.listdir(data_dir), 2)
    sample1 = cv2.imread(os.path.join(data_dir, samples[0]))
    sample2 = cv2.imread(os.path.join(data_dir, samples[1]))
    print("samples:", samples)
    fmix_img, x1, x2, _ = sample_and_apply(sample1, sample2, alpha, decay_power, max_soft, reformulate)

    cv2.imwrite("./fmix.jpg", convert_img_to_save(fmix_img))
    cv2.imwrite("./x1.jpg", convert_img_to_save(x1))
    cv2.imwrite("./x2.jpg", convert_img_to_save(x2))

