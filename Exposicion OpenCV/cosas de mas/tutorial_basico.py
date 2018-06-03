import numpy as np
import cv2
from matplotlib import pyplot as plt

def image_without_cvtcolor():
    img = cv2.imread('mono.png', cv2.IMREAD_COLOR)
    plt.imshow(img)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

def grey():
    img = cv2.imread('mono.png', cv2.IMREAD_COLOR)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray_image)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

def save_gray_image():
    img = cv2.imread('mono.png', cv2.IMREAD_COLOR)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if cv2.imwrite('con_gris.png',gray_image):
        return True
    else:
        return False
	
def ver_colores():
	flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
	print flags

def draw_line_diagonal():
	# Draw a diagonal blue line with thickness of 5 px
    img = np.zeros((512,512,3), np.uint8)
    cv2.line(img,(0,0),(511,511),(255,0,0),5)
    cv2.imshow('Draw01',img)
    cv2.waitKey(0)
    
def camara():
	# Define the codec and create VideoWriter object
    fourcc = cv2.cv.CV_FOURCC(*'XVID')
    out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
    cap = cv2.VideoCapture(-1)
    while(cap.isOpened()):
        print "llega"
        ret, frame = cap.read()
        time.sleep(2)
        if ret:
            out.write(frame)
            cv2.imshow('frame',frame)
        else:
            return False
    cap.release()
    out.release()
    cv2.destroyAllWindows()

#draw_line_diagonal();
#grey_with_color(img)
#camara()
#probando(img)
