import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import cv2 as cv
import math
import matplotlib.cm as cm
from scipy.signal import convolve2d
import random


# function to calculate 1st order derivative of gaussian
def gaussian(sigma,x,y):
    a= 1/(np.sqrt(2*np.pi)*sigma)
    b=math.exp(-(x**2+y**2)/(2*(sigma**2)))
    c = a*b
    return a*b


## getting kernel from  gaussian for [-1,0,1]
def gaussian_kernel():
    G=np.zeros((5,5))
    for i in range(-2,3):
        for j in range(-2,3):
            G[i+1,j+1]=gaussian(1.5,i,j)
    return G

def Lucas_Kanade_Expand(image):

    h,w = image.shape
    newWidth = int(w * 2)
    newHei = int(h * 2)
    newImage = np.zeros((newHei,newWidth))  # interpolate of image i.e. inserting making image by inserting 0 alternate to every original pixels
    newImage[::2, ::2] = image
    G = gaussian_kernel()
    for i in range(2, newImage.shape[0] - 2, 2):
        for j in range(2, newImage.shape[1] - 2, 2):
            newImage[i, j] = np.sum(newImage[i - 2:i + 3, j - 2:j + 3] * G)  # convolving with gaussian mask

    return newImage

def LK_Expand_Iterative(Img,Level):
    if Level==0:#level 0 means current level i.e. no change
        return cv.imread(Img,0)
    i=0
    newImage=cv.imread(Img,0)
    while(i<Level):
        newImage=Lucas_Kanade_Expand(newImage)
        i=i+1
    return newImage

def Lucas_Kanade_Reduce(I1):
    h,w = I1.shape
    newWidth = int(w / 2)
    newHei = int(h / 2)
    G = gaussian_kernel()
    newImage = np.ones((newHei,newWidth))
    for i in range(2, I1.shape[0] - 2, 2):  # making image of half size by skiping alternate pixels
        for j in range(2, I1.shape[1] - 2, 2):
            newImage[int(i / 2), int(j / 2)] = np.sum(I1[i - 2:i + 3, j - 2:j + 3] * G)  # convolving with gaussian mask

    return newImage

def LK_Reduce_Iterative(Img,Level):
    if Level==0:#level 0 means current level i.e. no change
        return cv.imread(Img,0)
    i=0
    newImage=cv.imread(Img,0)
    while(i<Level):
        newImage=Lucas_Kanade_Reduce(newImage)
        i=i+1

    return newImage

def LK_Reduce_Iterative_cvinput(Img,Level):
    if Level==0:#level 0 means current level i.e. no change
        return Img
    i=0
    newImage=Img
    while(i<Level):
        newImage=Lucas_Kanade_Reduce(newImage)
        i=i+1

    return newImage

def Laplacian_pyramid(Img,Level):
    pyramids=list()
    lap_pyramids=list()
    shape=cv.imread(Img,0).shape
    for i in range(Level):
        img=LK_Expand_Iterative(Img,i)
        img_resize=cv.resize(img,shape)
        pyramids.append(img_resize)
    print("guas size:",len(pyramids))
    for i in range(len(pyramids)):
        print("i:",i)
        if (i==(len(pyramids)-1)):
            lap_pyramids.append(pyramids[i])
            print('a')
        else:
            lap_pyramids.append(pyramids[i]-pyramids[i+1])
            print('b')
    print("lap size:",len(lap_pyramids))
    return pyramids,lap_pyramids


def Expand(image):
    h,w = image.shape
    newWidth = int(w * 2)
    newHei = int(h * 2)
    newImage = np.zeros((newHei,newWidth))  # interpolate of image i.e. inserting making image by inserting 0 alternate to every original pixels
    newImage[::2, ::2] = image
    G = gaussian_kernel()
    for i in range(2, newImage.shape[0] - 2, 2):
        for j in range(2, newImage.shape[1] - 2, 2):
            newImage[i, j] = np.sum(newImage[i - 2:i + 3, j - 2:j + 3] * G)  # convolving with gaussian mask

    return newImage