import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

"""
добавить функции из Нетологии cv!!!
"""


def gauss_kernel(ksize=5, sigma=2.5):
    """ Возвращает гауссовское ядро размера ksize и дисперсией sigma """
    # ksize - размер ядра
    # sigma - дисперсия (ширина фильтра)
    ax = np.arange(-ksize // 2 + 1.0, ksize // 2 + 1.0)
    xx, yy = np.meshgrid(ax, ax)
    e = np.float32((xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel = 1.0 / (sigma * np.sqrt(2.0 * np.pi)) * np.exp(-e)
    return kernel / np.sum(kernel)


def laplace_kernel(ksize=5, sigma=2.5):
    """ Возвращает ядро Лапласа размера ksize и дисперсией sigma """
    # ksize - размер ядра
    # sigma - дисперсия (ширина фильтра)
    ax = np.arange(-ksize // 2 + 1.0, ksize // 2 + 1.0)
    xx, yy = np.meshgrid(ax, ax)
    e = np.float32((xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel = 1.0 / (np.pi * sigma ** 4) * (1.0 - e) * np.exp(-e)
    return kernel / np.sum(kernel)


def imshow(img, **kwargs):
    """Демонстрация изображения при помощи matplotlib"""
    plt.figure(figsize=kwargs.get('figsize', (12, 12)))
    plt.imshow(img.numpy().astype("uint8"))
    plt.title(kwargs.get('title', 'image'))
    plt.axis("off")
    plt.show()
