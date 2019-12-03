import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import cv2
import math
import matplotlib.cm as cm
from scipy.signal import convolve2d
import random

def Lucas_Kanade_all(image1,image2,bkernel='Gaussian',bksize=0):
    oldframe = cv2.imread(image1)
    newframe = cv2.imread(image2)

    if bkernel=='Gaussian':
        oldframe=cv2.GaussianBlur(oldframe,(bksize,bksize),cv2.BORDER_DEFAULT)
        newframe=cv2.GaussianBlur(newframe,(bksize,bksize),cv2.BORDER_DEFAULT)
    elif bkernel=='Average':
        oldframe=cv2.blur(oldframe,(bksize,bksize))
        newframe=cv2.blur(newframe,(bksize,bksize)) 

    I1 = cv2.cvtColor(oldframe, cv2.COLOR_BGR2GRAY)
    I1=I1.astype(np.float64)
    I2 = cv2.cvtColor(newframe, cv2.COLOR_BGR2GRAY)

    Gx = np.reshape(np.asarray([[-1, 1], [-1, 1]]), (2, 2))  # for image 1 and image 2 in x direction
    Gy = np.reshape(np.asarray([[-1, -1], [1, 1]]), (2, 2))  # for image 1 and image 2 in y direction
    Gt1 = np.reshape(np.asarray([[-1, -1], [-1, -1]]), (2, 2))  # for 1st image
    Gt2 = np.reshape(np.asarray([[1, 1], [1, 1]]), (2, 2))  # for 2nd image


    Ix = (convolve2d(I1, Gx) + convolve2d(I2, Gx)) / 2 #smoothing in x direction
    Iy = (convolve2d(I1, Gy) + convolve2d(I2, Gy)) / 2 #smoothing in y direction
    It1 = convolve2d(I1, Gt1) + convolve2d(I2, Gt2)   #taking difference of two images using gaussian mask of all -1 and all 1


    ## IF we are reducing than fetures will be reduced by 2**Level
    h,w=I1.shape
    newImage=np.zeros((h,w,2))


    #print("ix.shape",Ix.shape)
    u = np.zeros(Ix.shape)
    v = np.zeros(Ix.shape)
    status=np.zeros(Ix.shape) # this will tell change in x,y
    A = np.zeros((2, 2))
    B = np.zeros((2, 1))

    for y in range(I1.shape[0]):
        for x in range(I1.shape[1]):
            A[0, 0] = np.sum((Ix[y - 1:y + 2, x - 1:x + 2]) ** 2)
            A[1, 1] = np.sum((Iy[y - 1:y + 2, x - 1:x + 2]) ** 2)
            A[0, 1] = np.sum(Ix[y - 1:y + 2, x - 1:x + 2] * Iy[y - 1:y + 2, x - 1:x + 2])
            A[1, 0] = np.sum(Ix[y - 1:y + 2, x - 1:x + 2] * Iy[y - 1:y + 2, x - 1:x + 2])
            Ainv = np.linalg.pinv(A)

            B[0, 0] = -np.sum(Ix[y - 1:y + 2, x - 1:x + 2] * It1[y - 1:y + 2, x - 1:x + 2])
            B[1, 0] = -np.sum(Iy[y - 1:y + 2, x - 1:x + 2] * It1[y - 1:y + 2, x - 1:x + 2])
            prod = np.matmul(Ainv, B)

            u[y, x] = prod[0]
            v[y, x] = prod[1]
            #print("u:%d,v:%d:",u[y,x],v[y,x])

            if np.int32(x+u[y,x])==x and np.int32(y+v[y,x])==y:    # this means that there is no change(x+dx==x,y+dy==y) so marking it as 0 else
                status[y,x]=0
            else:
                status[y,x]=1 # this tells us that x+dx , y+dy is not equal to x and y         
    return I1,u,v

def Lucas_Kanade_all(image1,image2,bkernel='Gaussian',bksize=0):
    oldframe = image1
    newframe = image2

    if bkernel=='Gaussian':
        oldframe=cv2.GaussianBlur(oldframe,(bksize,bksize),cv2.BORDER_DEFAULT)
        newframe=cv2.GaussianBlur(newframe,(bksize,bksize),cv2.BORDER_DEFAULT)
    elif bkernel=='Average':
        oldframe=cv2.blur(oldframe,(bksize,bksize))
        newframe=cv2.blur(newframe,(bksize,bksize)) 

    I1 = oldframe
    I1=I1.astype(np.float64)
    I2 = newframe

    Gx = np.reshape(np.asarray([[-1, 1], [-1, 1]]), (2, 2))  # for image 1 and image 2 in x direction
    Gy = np.reshape(np.asarray([[-1, -1], [1, 1]]), (2, 2))  # for image 1 and image 2 in y direction
    Gt1 = np.reshape(np.asarray([[-1, -1], [-1, -1]]), (2, 2))  # for 1st image
    Gt2 = np.reshape(np.asarray([[1, 1], [1, 1]]), (2, 2))  # for 2nd image


    Ix = (convolve2d(I1, Gx,mode='same') + convolve2d(I2, Gx,mode='same')) / 2 #smoothing in x direction
    Iy = (convolve2d(I1, Gy,mode='same') + convolve2d(I2, Gy,mode='same')) / 2 #smoothing in y direction
    It1 = convolve2d(I1, Gt1,mode='same') + convolve2d(I2, Gt2,mode='same')   #taking difference of two images using gaussian mask of all -1 and all 1


    ## IF we are reducing than fetures will be reduced by 2**Level
    h,w=I1.shape
    newImage=np.zeros((h,w,2))


    #print("ix.shape",Ix.shape)
    u = np.zeros(Ix.shape)
    v = np.zeros(Ix.shape)
    status=np.zeros(Ix.shape) # this will tell change in x,y
    A = np.zeros((2, 2))
    B = np.zeros((2, 1))

    for y in range(I1.shape[0]):
        for x in range(I1.shape[1]):
            A[0, 0] = np.sum((Ix[y - 1:y + 2, x - 1:x + 2]) ** 2)
            A[1, 1] = np.sum((Iy[y - 1:y + 2, x - 1:x + 2]) ** 2)
            A[0, 1] = np.sum(Ix[y - 1:y + 2, x - 1:x + 2] * Iy[y - 1:y + 2, x - 1:x + 2])
            A[1, 0] = np.sum(Ix[y - 1:y + 2, x - 1:x + 2] * Iy[y - 1:y + 2, x - 1:x + 2])
            Ainv = np.linalg.pinv(A)

            B[0, 0] = -np.sum(Ix[y - 1:y + 2, x - 1:x + 2] * It1[y - 1:y + 2, x - 1:x + 2])
            B[1, 0] = -np.sum(Iy[y - 1:y + 2, x - 1:x + 2] * It1[y - 1:y + 2, x - 1:x + 2])
            prod = np.matmul(Ainv, B)

            u[y, x] = prod[0]
            v[y, x] = prod[1]
            #print("u:%d,v:%d:",u[y,x],v[y,x])

            if np.int32(x+u[y,x])==x and np.int32(y+v[y,x])==y:    # this means that there is no change(x+dx==x,y+dy==y) so marking it as 0 else
                status[y,x]=0
            else:
                status[y,x]=1 # this tells us that x+dx , y+dy is not equal to x and y         
    return I1,u,v


def Lucas_Kanade(image1, image2,bkernel='Gaussian',bksize=0):

    oldframe = cv2.imread(image1)
    I1 = cv2.cvtColor(oldframe, cv2.COLOR_BGR2GRAY)

    newframe = cv2.imread(image2)
    I2 = cv2.cvtColor(newframe, cv2.COLOR_BGR2GRAY)
    if bkernel=='Gaussian':
        print("bksize:",bksize)
        oldframe=cv2.GaussianBlur(oldframe,(bksize,bksize),cv2.BORDER_DEFAULT)
        newframe=cv2.GaussianBlur(newframe,(bksize,bksize),cv2.BORDER_DEFAULT)
    elif bkernel=='Average':
        oldframe=cv2.blur(oldframe,(bksize,bksize))
        newframe=cv2.blur(newframe,(bksize,bksize))   
    color = np.random.randint(0, 255, (100000, 3))
    Gx = np.reshape(np.asarray([[-1, 1], [-1, 1]]), (2, 2))  # for image 1 and image 2 in x direction
    Gy = np.reshape(np.asarray([[-1, -1], [1, 1]]), (2, 2))  # for image 1 and image 2 in y direction
    Gt1 = np.reshape(np.asarray([[-1, -1], [-1, -1]]), (2, 2))  # for 1st image
    Gt2 = np.reshape(np.asarray([[1, 1], [1, 1]]), (2, 2))  # for 2nd image


    Ix = (convolve2d(I1, Gx) + convolve2d(I2, Gx)) / 2 #smoothing in x direction

    Iy = (convolve2d(I1, Gy) + convolve2d(I2, Gy)) / 2 #smoothing in y direction
    It1 = convolve2d(I1, Gt1) + convolve2d(I2, Gt2)   #taking difference of two images using gaussian mask of all -1 and all 1
    
    # parameter to get features
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    features= cv2.goodFeaturesToTrack(I1, mask = None, **feature_params)  #using opencv2 function to get feature for which we are plotting flow
    feature = np.int32(features)
    feature = np.reshape(feature, newshape=[-1, 2])
    # drawprint(feature.shape)
    '''
    feature_list=[]
    for i in range(I1.shape[0]):
        for j in range(I1.shape[1]):
            feature_list.append([j,i])##opencv2 coordination
    feature=np.asarray(feature_list)
    feature = np.reshape(feature, newshape=[-1, 2])
    
    # print(feature)
    '''
    u = np.zeros(Ix.shape)
    v = np.zeros(Ix.shape)
    status=np.zeros(feature.shape[0]) # this will tell change in x,y
    A = np.zeros((2, 2))
    B = np.zeros((2, 1))
    mask = np.zeros_like(oldframe)

    newImage=np.zeros_like(feature)
    """Assumption is  that all the neighbouring pixels will have similar motion. 
    Lucas-Kanade method takes a 3x3 patch around the point. So all the 9 points have the same motion.
    We can find (fx,fy,ft) for these 9 points. So now our problem becomes solving 9 equations with two unknown variables which is over-determined. 
    A better solution is obtained with least square fit method.
    Below is the final solution which is two equation-two unknown problem and solve to get the solution.
                               U=Ainverse*B 
    where U is matrix of 1 by 2 and contains change in x and y direction(x==U[0] and y==U[1])
    we first calculate A matrix which is 2 by 2 matrix of [[fx**2, fx*fy],[ fx*fy fy**2] and now take inverse of it
    and B is -[[fx*ft1],[fy,ft2]]"""

    for a,i in enumerate(feature):
        x, y = i

        A[0, 0] = np.sum((Ix[y - 1:y + 2, x - 1:x + 2]) ** 2)

        A[1, 1] = np.sum((Iy[y - 1:y + 2, x - 1:x + 2]) ** 2)
        A[0, 1] = np.sum(Ix[y - 1:y + 2, x - 1:x + 2] * Iy[y - 1:y + 2, x - 1:x + 2])
        A[1, 0] = np.sum(Ix[y - 1:y + 2, x - 1:x + 2] * Iy[y - 1:y + 2, x - 1:x + 2])
        Ainv = np.linalg.pinv(A)

        B[0, 0] = -np.sum(Ix[y - 1:y + 2, x - 1:x + 2] * It1[y - 1:y + 2, x - 1:x + 2])
        B[1, 0] = -np.sum(Iy[y - 1:y + 2, x - 1:x + 2] * It1[y - 1:y + 2, x - 1:x + 2])
        prod = np.matmul(Ainv, B)

        u[y, x] = prod[0]
        v[y, x] = prod[1]

        #print(u[y,x],v[y,x])
        newImage[a]=[np.int32(x+u[y,x]),np.int32(y+v[y,x])]
        if np.int32(x+u[y,x])==x and np.int32(y+v[y,x])==y:    # this means that there is no change(x+dx==x,y+dy==y) so marking it as 0 else
            status[a]=0
        else:
            status[a]=1 # this tells us that x+dx , y+dy is not equal to x and y

    um=np.flipud(u)
    vm=np.flipud(v)

    good_new=newImage[status==1] #status will tell the position where x and y are changed so for plotting getting only that points
    good_old = feature[status==1]
    #print(good_new.shape)
    #print(good_old.shape)

    # draw the tracks
    '''
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        newframe = cv2.circle(newframe, (a, b), 5, color[i].tolist(), -1)
    img = cv2.add(newframe, mask)
    '''
    return oldframe,good_old,u,v

'''
def Lucas_Kanade_Pyramid(image1,I1, image2,I2,Level,Reduce_Expand):
    oldframe = cv2.imread(image1)
    I1 = cv.cvtColor(oldframe, cv.COLOR_BGR2GRAY)
    I1=I1.astype(np.float64)
    newframe = cv2.imread(image2)
    I2 = cv2.cv2tColor(newframe, cv2.COLOR_BGR2GRAY)
    Gx = np.reshape(np.asarray([[-1, 1], [-1, 1]]), (2, 2))  # for image 1 and image 2 in x direction
    Gy = np.reshape(np.asarray([[-1, -1], [1, 1]]), (2, 2))  # for image 1 and image 2 in y direction
    Gt1 = np.reshape(np.asarray([[-1, -1], [-1, -1]]), (2, 2))  # for 1st image
    Gt2 = np.reshape(np.asarray([[1, 1], [1, 1]]), (2, 2))  # for 2nd image


    Ix = (convolve2d(I1, Gx) + convolve2d(I2, Gx)) / 2 #smoothing in x direction
    Iy = (convolve2d(I1, Gy) + convolve2d(I2, Gy)) / 2 #smoothing in y direction
    It1 = convolve2d(I1, Gt1) + convolve2d(I2, Gt2)   #taking difference of two images using gaussian mask of all -1 and all 1


    ## IF we are reducing than fetures will be reduced by 2**Level
    h,w=I1.shape
    if Reduce_Expand=="Reduce":
        newImage=np.zeros((h,w,2))/(2**Level)
    else:
        newImage=np.zeros((h,w,2))*(2*Level)
    # print(feature)

    print("ix.shape",Ix.shape)
    u = np.zeros(Ix.shape)
    v = np.zeros(Ix.shape)
    status=np.zeros(Ix.shape) # this will tell change in x,y
    A = np.zeros((2, 2))
    B = np.zeros((2, 1))
    #print("i1.SHAPE:",I1.shape[0],I1.shape[1])
    for y in range(I1.shape[0]):
        for x in range(I1.shape[1]):
            #print(y,x)
            A[0, 0] = np.sum((Ix[y - 1:y + 2, x - 1:x + 2]) ** 2)
            A[1, 1] = np.sum((Iy[y - 1:y + 2, x - 1:x + 2]) ** 2)
            A[0, 1] = np.sum(Ix[y - 1:y + 2, x - 1:x + 2] * Iy[y - 1:y + 2, x - 1:x + 2])
            A[1, 0] = np.sum(Ix[y - 1:y + 2, x - 1:x + 2] * Iy[y - 1:y + 2, x - 1:x + 2])
            Ainv = np.linalg.pinv(A)
            #print(A[0,0])
            B[0, 0] = -np.sum(Ix[y - 1:y + 2, x - 1:x + 2] * It1[y - 1:y + 2, x - 1:x + 2])
            B[1, 0] = -np.sum(Iy[y - 1:y + 2, x - 1:x + 2] * It1[y - 1:y + 2, x - 1:x + 2])
            prod = np.matmul(Ainv, B)

            u[y, x] = prod[0]
            v[y, x] = prod[1]
            #print("u:%d,v:%d:",u[y,x],v[y,x])

            if np.int32(x+u[y,x])==x and np.int32(y+v[y,x])==y:    # this means that there is no change(x+dx==x,y+dy==y) so marking it as 0 else
                status[y,x]=0
            else:
                status[y,x]=1 # this tells us that x+dx , y+dy is not equal to x and y

    if Reduce_Expand=="Reduce":# multiplying by 2**Level to get position in original image
        u=np.int32(u*(2**Level))
        v=np.int32(v*(2**Level))

    else:#divding by 2**level to get original position back
        u=np.int32(u/(2**Level))
        v=np.int32(v/(2**Level))            
    return oldframe,u,v
'''