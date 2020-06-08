import numpy
import cv2
from scipy.fftpack import dct
from scipy.fftpack import idct
from time import time

BWimage=cv2.imread('the_mandalorian.png',cv2.IMREAD_GRAYSCALE)
cv2.imwrite('the_mandalorian_bn.png',BWimage)
cv2.imshow('imagenBW', BWimage) # dtype=uint8
cv2.waitKey(0)

#Conversion a int16 (signado) y restando la componente de directa
BWimage = BWimage.astype( numpy.int16 )-128

#Codificacion
rows,cols=BWimage.shape
jpgImage=numpy.zeros([rows,cols],dtype=numpy.int8) #Debieran ser 16 bits
factor=(0.22/64)
start_time=time()
for r in range(0,rows,8):
	for c in range(0,cols,8):
		block=BWimage [r:r+8,c:c+8]
		dctblock= (factor * dct(dct(block,axis=0),axis=1)).astype(numpy.int8)
		jpgImage[r:r+8,c:c+8]=dctblock
elpased_time=time()-start_time
print ("Elapsed time : ", elpased_time)
print ("Frecuency : ", 1/elpased_time)


#Decodificacion
rndrImage=numpy.zeros([rows,cols],dtype=numpy.uint8)
for r in range(0,rows,8):
	for c in range(0,cols,8):
		dctblock=jpgImage[r:r+8,c:c+8]
		block=(idct(idct(dctblock,axis=0),axis=1)+128).astype(numpy.uint8)
		rndrImage[r:r+8,c:c+8]=block

cv2.imshow('render', rndrImage) # dtype=uint8
cv2.imwrite('the_mandalorian_render.jpg',rndrImage)
cv2.waitKey(0)

cv2.destroyAllWindows()

