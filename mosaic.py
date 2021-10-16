""" python3 mosaic.py /home/axelauza/Descargas/FIRMEZA/SENSITIVO1711_2/corteRgb/Arandano\ 1-2.jpg  /home/axelauza/Descargas/FIRMEZA/SENSITIVO1711_2/corteRgb/Arandano\ 1-3.jpg  /home/axelauza/Descargas/FIRMEZA/SENSITIVO1711_2/corteRgb/Arandano\ 1-4.jpg  /home/axelauza/Descargas/FIRMEZA/SENSITIVO1711_2/corteRgb/Arandano\ 1-5.jpg"""
import numpy as np
import cv2
import sys,os


img1 = cv2.imread(sys.argv[1])
img2 = cv2.imread(sys.argv[2])
img3 = cv2.imread(sys.argv[3])
img4 = cv2.imread(sys.argv[4])

rows,cols,d=img1.shape
rows2,cols2,d=img2.shape
rows3,cols3,d=img3.shape
rows4,cols4,d=img4.shape
#mosaic = np.full((rows * 2, cols * 2, d), 114, dtype=np.uint8)  # base image with 4 tiles
rowsbox=0
colsbox=0
if(rows<rows2):
    rowsbox+=rows2
else:
    rowsbox+=rows

if(rows3<rows4):
    rowsbox+=rows4
else:
    rowsbox+=rows3

if(cols<cols3):
    colsbox+=cols3
else:
    colsbox+=cols

if(cols2<cols4):
    colsbox+=cols4
else:
    colsbox+=cols2

mosaic = np.full((rowsbox,colsbox, d), 114, dtype=np.uint8)  # base image with 4 tiles
print(img1.shape)
print(img2.shape)
print(img3.shape)
print(img4.shape)
print(mosaic.shape)

mosaic[0:rows,0:cols]=img1
mosaic[0:rows2,cols: cols+cols2 ]=img2
mosaic[rows:rows+rows3,0: cols3]=img3
mosaic[rows2:rows2+rows4,cols3:cols3+cols4]=img4
"""mosaic[0:rows,0:cols]=img1
mosaic[0:rows,cols: cols*2 ]=img2
mosaic[rows:rows*2,0: cols]=img3
mosaic[rows:rows*2,cols:cols*2]=img4"""

cv2.imshow('img1',img1)
cv2.imshow('img2',img2)
cv2.imshow('img3',img3)
cv2.imshow('img4',img4)
cv2.imshow('img',mosaic)

cv2.waitKey(0)