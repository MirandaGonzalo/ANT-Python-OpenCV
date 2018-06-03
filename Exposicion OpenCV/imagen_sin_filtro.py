import numpy as np
import cv2

img = cv2.imread('img/bill_gates.jpg', cv2.IMREAD_COLOR)
cv2.imshow('Bill a color', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
    