import numpy as np
import cv2

from Lk import Lucas_Kanade_all
from pyramid import LK_Reduce_Iterative,Expand,LK_Reduce_Iterative_cvinput
from warp import warp_forward

def LK_Hie(img1,img2,max_level=0):
	I1=cv2.imread(img1,0)
	I2=cv2.imread(img2,0)
	(w,h)=I1.shape
	print("ix.shape:",I1.shape)
	h_pow2=h//(2**max_level) * (2**max_level)
	w_pow2=w//(2**max_level) * (2**max_level)
	I1=cv2.resize(I1,(h_pow2,w_pow2))
	I2=cv2.resize(I2,(h_pow2,w_pow2))
	
	n=max_level
	level=list([n-k for k in range(n+1)])
	for k in level:
		I1_reduce_k=LK_Reduce_Iterative_cvinput(I1,k)
		I2_reduce_k=LK_Reduce_Iterative_cvinput(I2,k)
		if k==n:
			U=np.zeros(I1_reduce_k.shape)
			V=np.zeros(I2_reduce_k.shape)
		else:
			U=2*Expand(U)
			V=2*Expand(V)
		print("B I1_reduce_k:{},U.SHAPE:{}".format(I1_reduce_k.shape,U.shape))

		Wk=warp_forward(I1_reduce_k,U,V)
		_,Dx,Dy=Lucas_Kanade_all(Wk,I2_reduce_k,bkernel='Gaussian',bksize=0)
		print("A I1_reduce_k:{},U.SHAPE:{},Dx shape:{},wk shape{}:".format(I1_reduce_k.shape,U.shape,Dx.shape,Wk.shape))
		U=U+Dx
		V=V+Dy
	return I1,I2,U,V
		
	
