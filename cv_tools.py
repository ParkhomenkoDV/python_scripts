import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

"""
добавить функции из Нетологии cv!!!
"""


def imshow(img, **kwargs):
    """Демонстрация изображения при помощи matplotlib"""
    plt.figure(figsize=kwargs.get('figsize', (12, 12)))
    plt.imshow(img.numpy().astype("uint8"))
    plt.title(kwargs.get('title', 'image'))
    plt.axis("off")
    plt.show()
