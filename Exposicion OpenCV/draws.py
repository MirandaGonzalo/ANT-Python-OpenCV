import cv2
import numpy as np

img = np.zeros((512,512,3), np.uint8)
cv2.rectangle(img,(0,0),(300,200),(0,0,255),10)
cv2.line(img,(25,25),(25,400),(0,255,255),10)
#cv2.line(img,(0,0),(175,200),(0,0,0),50)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'Texto',(50,300), font, 4,(255,255,255),2)
cv2.imshow('Draw01',img)
cv2.waitKey(0)