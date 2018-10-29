import math
from PIL import Image
import numpy as numpy
from matplotlib import pyplot as plt

# Import INPUT IMAGE
# Translate into indexed array via numpy
try:
	input_image = Image.open("/Users/GAO/Documents/NYU Fall 2018/Computer Vision/CannyEdgeDetector/CASSAVA_sm.bmp")
except IOerror:    #Error case
	pass
indexed_image = numpy.array(input_image)


# FUNCTIONS/SUBPROGRAMS
def gaussian(x):
	gaussianmask = numpy.array([[1,1,2,2,2,1,1],
								[1,2,2,4,2,2,1],
								[2,2,4,8,4,2,2],
								[2,4,8,16,8,4,2],
								[2,2,4,8,4,2,2],
								[1,2,2,4,2,2,1],
								[1,1,2,2,2,1,1]])
	gaussianresult = numpy.array(x)


	for i in range(3,(x.shape[1]-3)):			#iterates according to # of horizontal pixels in input image (shape[1] = # columns / horizontal dimension)

		for j in range(3,(x.shape[0]-3)):		#iterates according to # of vertical pixels in input image  (shape[0] = # rows / vertical dimension)
			print(j,i)
			img_submatrix = x[j-3:j+4,i-3:i+4]

			convolution_sum = 0
			for k in range(0,7):
				for l in range(0,7):
					convolution_sum += img_submatrix[k,l]*gaussianmask[k,l]
			convolution = convolution_sum / 140
			gaussianresult[j,i] = convolution
	return gaussianresult




def gradientify(y):
	#Implement Prewitt's operator to determine X and Y magnitude of gradient
	gradient_magnitude_arr = numpy.array(y)
	gradient_angle_arr = numpy.array(y)


	for j in range(1,(y.shape[1]-1)):			#iterates according to # of horizontal pixels in input image (shape[1] = # columns / horizontal dimension)

		for i in range(1,(y.shape[0]-1)):		#iterates according to # of vertical pixels in input image  (shape[0] = # rows / vertical dimension)
		
			gx = y[i-1,j+1] + y[i,j+1] + y[i+1,j+1] - y[i-1,j-1] - y[i,j-1] - y[i+1,j-1]
			gy = y[i-1,j-1] + y[i-1,j] + y[i-1,j+1] - y[i+1,j-1] - y[i+1,j] - y[i+1,j+1]
			gradient_magnitude = int(math.sqrt(gx^2 + gy^2))
			gradient_angle = math.atan(gy/gx)

			gradient_magnitude_arr[i,j] = gradient_magnitude
			gradient_angle_arr[i,j] = gradient_angle
	return (gradient_magnitude_arr, gradient_angle_arr)

def nonmaximasuppress((magnitude, angle)):
		
	nms_result = numpy.array(magnitude)
	for j in range(1,(magnitude.shape[1]-1)):			#iterates according to # of horizontal pixels in input image (shape[1] = # columns / horizontal dimension)

		for i in range(1,(magnitude.shape[0]-1)):

			gradient_angle = (angle[i,j])
			if gradient_angle > -22.5 and gradient_angle <= 22.5:
				region = 0
			elif gradient_angle > 22.5 and gradient_angle <= 67.5:
				region = 1
			elif gradient_angle > 67.5 and gradient_angle <= 112.5:
				region = 2
			elif gradient_angle > 112.5 and gradient_angle <= 157.5:
				region = 3
			elif gradient_angle > 157.5 and gradient_angle <= 202.5:
				region = 0
			if gradient_angle > 202.5 and gradient_angle <= 247.5:
				region = 1
			elif gradient_angle > 247.5 and gradient_angle <= 292.5:
				region = 2
			elif gradient_angle > 292.5 and gradient_angle <= 337.5:
				region = 3
			
			if region == 0:
				if magnitude[i,j] > magnitude[i,j+1] and magnitude[i,j] > magnitude[i,j-1]:
					nms_result[i,j] = magnitude[i,j]
				else:
					nms_result[i,j] = 0
			elif region == 1:
				if magnitude[i,j] > magnitude[i-1,j+1] and magnitude[i,j] > magnitude[i+1,j-1]:
					nms_result[i,j] = magnitude[i,j]
				else:
					nms_result[i,j] = 0
			elif region == 2:
				if magnitude[i,j] > magnitude[i-1,j] and magnitude[i,j] > magnitude[i+1,j]:
					nms_result[i,j] = magnitude[i,j]
				else:
					nms_result[i,j] = 0
			elif region == 3:
				if magnitude[i,j] > magnitude[i-1,j-1] and magnitude[i,j] > magnitude[i+1,j+1]:
					nms_result[i,j] = magnitude[i,j]
				else:
					nms_result[i,j] = 0
	return nms_result





def cannyedges(indexed_image):
	step1 = gaussian(indexed_image)
	step2 = gradientify(step1)
	step3 = nonmaximasuppress(step2)
	return step3




output_image = cannyedges(indexed_image)





plt.imshow(output_image, cmap='gray')
plt.show()
