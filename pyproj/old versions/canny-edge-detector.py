import math
from PIL import Image
import numpy as numpy
from matplotlib import pyplot as plt

# Import INPUT IMAGE
# Translate into indexed array via numpy
try:
	input_image = Image.open("/Users/GAO/Documents/NYU Fall 2018/Computer Vision/CannyEdgeDetector/zebra-crossing-1.bmp")
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


	for j in range(3,(x.shape[1]-3)):			#iterates according to # of horizontal pixels in input image (shape[1] = # columns / horizontal dimension)

		for i in range(3,(x.shape[0]-3)):		#iterates according to # of vertical pixels in input image  (shape[0] = # rows / vertical dimension)
			img_submatrix = x[i-3:i+4,j-3:j+4]

			convolution_sum = 0
			for k in range(0,7):
				for l in range(0,7):
					convolution_sum += img_submatrix[k,l]*gaussianmask[k,l]
			convolution = convolution_sum / 140
			gaussianresult[i,j] = convolution

	plt.imshow(gaussianresult, cmap='gray')
	plt.show()
	return gaussianresult



def gradientify(y):
	#Implement Prewitt's operator to determine X and Y magnitude of gradient
	dim = y.shape
	gradient_magnitude_horiz_arr = numpy.zeros(dim, dtype=numpy.int)
	gradient_magnitude_vert_arr = numpy.zeros(dim, dtype=numpy.int)
	gradient_magnitude_arr = numpy.zeros(dim, dtype=numpy.int)
	gradient_angle_arr = numpy.zeros(dim)

	for i in range(1,(y.shape[0]-1)):			#iterates according to # of horizontal pixels in input image (shape[1] = # columns / horizontal dimension)
		for j in range(1,(y.shape[1]-1)):		#iterates according to # of vertical pixels in input image  (shape[0] = # rows / vertical dimension)
		
			gx = ((y[i-1,j+1] + y[i,j+1] + y[i+1,j+1]) - (y[i-1,j-1] + y[i,j-1] + y[i+1,j-1]))
			abs_gx = abs(gx)
			normal_gx = abs_gx/6
			print(i,j)
			print(gx)
			gradient_magnitude_horiz_arr[i,j] = normal_gx

			gy = ((y[i-1,j-1] + y[i-1,j] + y[i-1,j+1]) - (y[i+1,j-1] + y[i+1,j] + y[i+1,j+1]))
			abs_gy = abs(gy)
			normal_gy = abs_gy/6

			print(i,j)
			print(gy)
			gradient_magnitude_vert_arr[i,j] = normal_gy

			gradient_magnitude = math.sqrt(normal_gx**2 + normal_gy**2)
			#print(i,j)
			#print(gx,gy)
			gradient_angle = math.degrees(math.atan2(gy,gx))
			#print(gradient_angle)

			gradient_magnitude_arr[i,j] = int(gradient_magnitude)
			gradient_angle_arr[i,j] = gradient_angle

	#print(gradient_magnitude_arr)

	plt.imshow(gradient_magnitude_horiz_arr, cmap='gray')
	plt.show()

	plt.imshow(gradient_magnitude_vert_arr, cmap='gray') 
	plt.show()

	plt.imshow(gradient_magnitude_arr, cmap='gray')
	plt.show()

	plt.imshow(gradient_angle_arr, cmap='gray')
	plt.show()


	return (gradient_magnitude_arr, gradient_angle_arr)



def nms((magnitude, angle)):
		
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
			elif gradient_angle > 202.5 and gradient_angle <= 247.5:
				region = 1
			elif gradient_angle > 247.5 and gradient_angle <= 292.5:
				region = 2
			elif gradient_angle > 292.5 and gradient_angle <= 337.5:
				region = 3
			
			if region == 0:
				if magnitude[i,j] >= magnitude[i,j+1] and magnitude[i,j] >= magnitude[i,j-1]:
					nms_result[i,j] = magnitude[i,j]
				else:
					nms_result[i,j] = 0
			elif region == 1:
				if magnitude[i,j] >= magnitude[i-1,j+1] and magnitude[i,j] >= magnitude[i+1,j-1]:
					nms_result[i,j] = magnitude[i,j]
				else:
					nms_result[i,j] = 0
			elif region == 2:
				if magnitude[i,j] >= magnitude[i-1,j] and magnitude[i,j] >= magnitude[i+1,j]:
					nms_result[i,j] = magnitude[i,j]
				else:
					nms_result[i,j] = 0
			elif region == 3:
				if magnitude[i,j] >= magnitude[i-1,j-1] and magnitude[i,j] >=  magnitude[i+1,j+1]:
					nms_result[i,j] = magnitude[i,j]
				else:
					nms_result[i,j] = 0

	#print(nms_result)
	plt.imshow(nms_result, cmap='gray')
	plt.show()
	return nms_result



def threshold(r):
	histogram = [0 for c in range(0,256)]
	total_px_count = 0

	for i in range(0,(r.shape[0])):
		for j in range(0,(r.shape[1])):
			if r[i,j] > 0:
				histogram[r[i,j]] += 1
				total_px_count += 1

	r10 = numpy.array(r)
	r30 = numpy.array(r)
	r50 = numpy.array(r)
	r70 = numpy.array(r)
	r90 = numpy.array(r)


	########
	#P = 10%
	########	

	ptile10 = total_px_count / 10
	count10 = 0
	h10 = 255
	while count10 <= ptile10:
		count10 += histogram[h10]
		h10 -= 1
	t10 = h10
	print(t10)
	for i in range(0,(r10.shape[0]-1)):
		for j in range(0,(r10.shape[1]-1)):
			if r10[i,j] < t10:
				r10[i,j] = 0
	plt.imshow(r10, cmap='gray')
	plt.show()

	########
	#P = 30%
	########
	
	ptile30 = total_px_count * 0.3
	count30 = 0
	h30 = 255
	while count30 <= ptile30:
		count30 += histogram[h30]
		h30 -= 1
	t30 = h30
	print(t30)
	for i in range(0,(r30.shape[0]-1)):
		for j in range(0,(r30.shape[1]-1)):
			if r30[i,j] < t30:
				r30[i,j] = 0
	plt.imshow(r30, cmap='gray')
	plt.show()

	########
	#P = 50%
	########

	ptile50 = total_px_count * 0.5
	count50 = 0
	h50 = 255
	while count50 <= ptile50:
		count50 += histogram[h50]
		h50 -= 1
	t50 = h50
	print(t50)
	for i in range(0,(r50.shape[0]-1)):
		for j in range(0,(r50.shape[1]-1)):
			if r50[i,j] < t50:
				r50[i,j] = 0
	plt.imshow(r50, cmap='gray')
	plt.show()

	########
	#P = 70%
	########

	ptile70 = total_px_count * 0.7
	count70 = 0
	h70 = 255
	while count70 <= ptile70:
		count70 += histogram[h70]
		h70 -= 1
	t70 = h70
	print(t70)
	for i in range(0,(r70.shape[0]-1)):
		for j in range(0,(r70.shape[1]-1)):
			if r70[i,j] < t70:
				r70[i,j] = 0
	plt.imshow(r70, cmap='gray')
	plt.show()

	########
	#P = 90%
	########

	ptile90 = total_px_count * 0.9
	count90 = 0
	h90 = 255
	while count90 <= ptile90:
		count90 += histogram[h90]
		h90 -= 1
	t90 = h90
	print(t90)
	for i in range(0,(r90.shape[0]-1)):
		for j in range(0,(r90.shape[1]-1)):
			if r90[i,j] < t90:
				r90[i,j] = 0
	plt.imshow(r90, cmap='gray')
	plt.show()

	return r70

#gradientify(indexed_image)


def cannyedges(q):
	step1 = gaussian(q)
	step2 = gradientify(step1)
	step3 = nms(step2)
	step4 = threshold(step3)
	return step4
#print(indexed_image)

output_image = cannyedges(indexed_image)
print(output_image[1].shape)
