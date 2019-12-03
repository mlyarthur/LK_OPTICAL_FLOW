
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import math
import matplotlib.cm as cm
import random

def warp_back(image1,image2,u,v):
    I1=image1
    I2=image2
    I1_warp=np.zeros(I1.shape)
    ###
    for i in range(I2.shape[0]):
        for j in range(I2.shape[1]):
            #print(i,j,u.shape,I2.shape)
            warp_i=i+u[i,j]
            warp_j=j+v[i,j]
            if warp_i<0:
                warp_i=0
            if warp_j<0:
                warp_j=0
            if not (warp_i)<I2.shape[0]:
                warp_i=I2.shape[0]-1
            if not (warp_j)<I2.shape[1]:
                warp_j=I2.shape[1]-1
            I1_warp[i,j]=I2[np.int32(warp_i),np.int32(warp_j)]
    return I1_warp

def warp_forward(image1,u,v):
    I1=image1
    I2_warp=np.zeros(I1.shape)
    ###
    for i in range(I1.shape[0]):
        for j in range(I1.shape[1]):
            warp_i=i-u[i,j]
            warp_j=j-v[i,j]
            if warp_i<0:
                warp_i=0
            if warp_j<0:
                warp_j=0
            if not (warp_i)<I1.shape[0]:
                warp_i=I1.shape[0]-1
            if not (warp_j)<I1.shape[1]:
                warp_j=I1.shape[1]-1
            #print("i:%d,j:%d",warp_i,warp_j)
            I2_warp[i,j]=I1[np.int32(warp_i),np.int32(warp_j)]
    return I2_warp