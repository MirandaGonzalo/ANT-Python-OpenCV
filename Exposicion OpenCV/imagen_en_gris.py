import cv2
import numpy as np

img = cv2.imread('img/bill_gates.jpg', cv2.IMREAD_COLOR)
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Bill en gris', gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()