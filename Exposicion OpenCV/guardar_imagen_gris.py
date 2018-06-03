import cv2
import numpy as np

img = cv2.imread('img/bill_gates.jpg', cv2.IMREAD_COLOR)
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('img/con_gris.png',gray_image)