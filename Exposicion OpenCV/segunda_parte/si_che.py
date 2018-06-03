import numpy as np
import cv2

# Load two images
img2 = cv2.imread('si_che.png')
img1 = cv2.imread('destrozado.jpeg')
rows,cols,channels = img2.shape
roi = img1[0:rows, 0:cols ]
img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)
img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
img2_fg = cv2.bitwise_and(img2,img2,mask = mask)
dst = cv2.add(img1_bg,img2_fg)
img1[0:rows, 0:cols ] = dst

cv2.rectangle(img1,(600,500),(175,200),(0,0,255),10)
cv2.line(img1,(600,500),(175,200),(0,0,255),10)
cv2.line(img1,(175,500),(600,200),(0,0,255),10)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img1,'GG ANT',(175,625), font, 4,(255,255,255),2)
cv2.imshow('res',img1)
cv2.imwrite('destrozado_con_opencv.png', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
