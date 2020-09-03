import torch
from numpy.random import randint


def squash(x, dim=-1, eps=1e-8):
    squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * x / torch.sqrt(squared_norm + eps)


def put_mask(x):
    """
    :param x: image tensor of shape [batch_size, channels, height, width]
    :return: the same image with a random bounding box set to zero in all channels
    """
    height = x.shape[2]
    width = x.shape[3]
    x1, x2, y1, y2 = randint(0, height), randint(0, height), randint(0, width), randint(0, width)
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    masked_x = x.clone().detach()
    masked_x[:, :, x1:x2, y1:y2] = 0
    return masked_x
