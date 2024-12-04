import matplotlib.pyplot as plt
import functools
import numpy as np
import cv2

img = cv2.imread('img.png', 0)
    #img = np.float32(img) / 255.0
    #kernel1 = np.array([[-1, 0, 1]])
    #kernel2 = np.array([[-1], [0], [1]])

gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)

mag, angle = cv2.cartToPolar(gx, gy, angle=(), angleInDegrees=True)

cv2.imwrite('result.jpg', mag)

print(np.int32(angle))