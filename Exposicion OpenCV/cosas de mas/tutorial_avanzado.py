import numpy as np
import cv2
from matplotlib import pyplot as plt

def denoising():
	print "Falta Hacer Este"

def one_feature():
	x = np.random.randint(25,100,25)
	y = np.random.randint(175,255,25)
	z = np.hstack((x,y))
	z = z.reshape((50,1))
	z = np.float32(z)
	plt.hist(z,256,[0,256])
	plt.show()

#one_feature()

def knn():
	# Feature set containing (x,y) values of 25 known/training data
	trainData = np.random.randint(0,100,(25,2)).astype(np.float32)

	# Labels each one either Red or Blue with numbers 0 and 1
	responses = np.random.randint(0,2,(25,1)).astype(np.float32)

	# Take Red families and plot them
	red = trainData[responses.ravel()==0]
	plt.scatter(red[:,0],red[:,1],80,'r','^')

	# Take Blue families and plot them
	blue = trainData[responses.ravel()==1]
	plt.scatter(blue[:,0],blue[:,1],80,'b','s')

	plt.show()

#knn()

def foreground():
	img = cv2.imread('messi.jpg')
	mask = np.zeros(img.shape[:2],np.uint8)

	bgdModel = np.zeros((1,65),np.float64)
	fgdModel = np.zeros((1,65),np.float64)

	rect = (50,50,450,290)
	cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

	mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
	img = img*mask2[:,:,np.newaxis]

	plt.imshow(img),plt.colorbar(),plt.show()

#foreground()

def hough_circle():
	img = cv2.imread('sudoku.png',0)
	img = cv2.medianBlur(img,5)
	cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
	circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20, param1=50,param2=30,minRadius=0,maxRadius=0)
	circles = np.uint16(np.around(circles))
	for i in circles[0,:]:
		# draw the outer circle
		cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
		# draw the center of the circle
		cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

	cv2.imshow('detected circles',cimg)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

#hough_circle()

def hough_transform():
	img = cv2.imread('sudoku.png')
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	edges = cv2.Canny(gray,50,150,apertureSize = 3)
	minLineLength = 100
	maxLineGap = 10
	lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
	for x1,y1,x2,y2 in lines[0]:
		cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

	cv2.imwrite('houghlines5.png',img)
	img2 = cv2.imread('houghlines5.png')
	cv2.imshow('asdf',img2)	
	cv2.waitKey(0)
	cv2.destroyAllWindows()

#hough_transform()

def blending():
	img = cv2.imread('messi.jpg',0)
	img2 = img.copy()
	template = cv2.imread('messi_face.jpg',0)
	w, h = template.shape[::-1]

	# All the 6 methods for comparison in a list
	methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
		        'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

	for meth in methods:
		img = img2.copy()
		method = eval(meth)

		# Apply template Matching
		res = cv2.matchTemplate(img,template,method)
		min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

		# If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
		if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
		    top_left = min_loc
		else:
		    top_left = max_loc
		bottom_right = (top_left[0] + w, top_left[1] + h)

		cv2.rectangle(img,top_left, bottom_right, 255, 2)

		plt.subplot(121),plt.imshow(res,cmap = 'gray')
		plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
		plt.subplot(122),plt.imshow(img,cmap = 'gray')
		plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
		plt.suptitle(meth)

		plt.show()

#blending()

def canny_edge_detection():
	img = cv2.imread('macri.png',0)
	edges = cv2.Canny(img,100,200)

	plt.subplot(121),plt.imshow(img)
	plt.title('Original Image'), plt.xticks([]), plt.yticks([])
	plt.subplot(122),plt.imshow(edges)
	plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

	plt.show()

#canny_edge_detection()

def laplacian():
	img = cv2.imread('macri.png',0)
	laplacian = cv2.Laplacian(img,cv2.CV_64F)
	sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
	sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

	plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
	plt.title('Original'), plt.xticks([]), plt.yticks([])
	plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
	plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
	plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
	plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
	plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
	plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

	plt.show()

#laplacian()

def blur():
	img = cv2.imread('macri.png')
	blur = cv2.blur(img,(5,5))

	plt.subplot(121),plt.imshow(img),plt.title('Original')
	plt.xticks([]), plt.yticks([])
	plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
	plt.xticks([]), plt.yticks([])
	plt.show()

#blur()

def convolution():
	img = cv2.imread('macri.png')
	kernel = np.ones((5,5),np.float32)/25
	dst = cv2.filter2D(img,-1,kernel)

	plt.subplot(121),plt.imshow(img),plt.title('Original')
	plt.xticks([]), plt.yticks([])
	plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
	plt.xticks([]), plt.yticks([])
	plt.show()

#convolution()

def perspective_transformation():
	img = cv2.imread('macri.png')
	rows,cols,ch = img.shape

	pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
	pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])

	M = cv2.getPerspectiveTransform(pts1,pts2)

	dst = cv2.warpPerspective(img,M,(300,300))

	plt.subplot(121),plt.imshow(img),plt.title('Input')
	plt.subplot(122),plt.imshow(dst),plt.title('Output')
	plt.show()

#perspective_transformation()

def scaling():
	img = cv2.imread('macri.png',0)
	rows,cols = img.shape

	M = np.float32([[1,0,100],[0,1,50]])
	dst = cv2.warpAffine(img,M,(cols,rows))

	cv2.imshow('img',dst)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

#scaling()

def rotation():
	img = cv2.imread('mono.png')
	rows,cols,ch = img.shape

	pts1 = np.float32([[50,50],[200,50],[50,200]])
	pts2 = np.float32([[10,100],[200,50],[100,250]])

	M = cv2.getAffineTransform(pts1,pts2)

	dst = cv2.warpAffine(img,M,(cols,rows))

	plt.subplot(121),plt.imshow(img),plt.title('Input')
	plt.subplot(122),plt.imshow(dst),plt.title('Output')
	plt.show()

#rotation()

def thresholding():
	img = cv2.imread('mono.png',0)
	ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
	ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
	ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
	ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
	ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)

	titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
	images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

	for i in xrange(6):
		plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
		plt.title(titles[i])
		plt.xticks([]),plt.yticks([])

	plt.show()

#thresholding()

def adaptive_thresholding():
	img = cv2.imread('mono.png',0)
	img = cv2.medianBlur(img,5)

	ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
	th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
		        cv2.THRESH_BINARY,11,2)
	th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
		        cv2.THRESH_BINARY,11,2)

	titles = ['Original Image', 'Global Thresholding (v = 127)',
		        'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
	images = [img, th1, th2, th3]

	for i in xrange(4):
		plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
		plt.title(titles[i])
		plt.xticks([]),plt.yticks([])
	plt.show()

#adaptive_thresholding()


def ver_colores():
	flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
	print flags

#ver_colores()
cap = cv2.VideoCapture(0)

def capas_camaras():
	while(1):

		# Take each frame
		_, frame = cap.read()

		# Convert BGR to HSV
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

		# define range of blue color in HSV
		lower_blue = np.array([110,50,50])
		upper_blue = np.array([130,255,255])

		# Threshold the HSV image to get only blue colors
		mask = cv2.inRange(hsv, lower_blue, upper_blue)

		# Bitwise-AND mask and original image
		res = cv2.bitwise_and(frame,frame, mask= mask)

		cv2.imshow('frame',frame)
		cv2.imshow('mask',mask)
		cv2.imshow('res',res)
		k = cv2.waitKey(5) & 0xFF
		if k == 27:
		    break
	cv2.destroyAllWindows()

#capas_camaras()
