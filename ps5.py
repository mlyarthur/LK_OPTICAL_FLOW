
import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils import flow2image
from  Lk import Lucas_Kanade,Lucas_Kanade_all
from pyramid import LK_Reduce_Iterative,Laplacian_pyramid
from warp import warp_back
from LK_hier import LK_Hie


output_root="output/"

#Question 1.a
def LK_test_a(bkernel,bksize):
    image1='input/TestSeq/Shift0.png'
    image2='input/TestSeq/ShiftR2.png'
    image3='input/TestSeq/ShiftR5U5.png'
    _,old_pts_0,u,v=Lucas_Kanade(image2,image1)
    oldframe,old_pts_1,u1,v1=Lucas_Kanade(image3,image1,bksize=bksize)
    plt.clf()
    plt.imshow(cv2.imread(image1)),plt.title("right 2")
    for pts in old_pts_0:
        x,y=pts.ravel()
        vx=u[y,x]
        vy=v[y,x]
        plt.arrow(x,y, vx, vy, head_width=5, head_length=5, color='b')
    plt.savefig(output_root+'ps5-1-a-1.png')
    #plt.subplot(121),plt.imshow(cv2.imread(image1)),plt.title("Right5 upper5")
    #plt.subplot(122)
    plt.imshow(cv2.imread(image1)),plt.title("ShiftR5U5")
    for pts in old_pts_1:
        x,y=pts.ravel()
        vx=u1[y,x]
        vy=v1[y,x]
        plt.arrow(x,y, vx, vy, head_width=5, head_length=5, color='b') 
    plt.savefig(output_root+'ps5-1-a-2.png')
    plt.show()

#Question 1.b
def LK_test_b(bkernel,bksize):
    image1='input/TestSeq/Shift0.png'
    image2='input/TestSeq/ShiftR10.png'
    image3='input/TestSeq/ShiftR20.png'
    image4='input/TestSeq/ShiftR40.png'
    images=[image2,image3,image4]
    for i,image in enumerate(images):
        oldframe,old_pts_1,u1,v1=Lucas_Kanade(image,image1,bksize=bksize)
        plt.imshow(cv2.imread(image1)),plt.title(image.split('/')[-1])
        for pts in old_pts_1:
            x,y=pts.ravel()
            vx=u1[y,x]
            vy=v1[y,x]
            plt.arrow(x,y, vx, vy, head_width=5, head_length=5, color='b') 
        plt.savefig(output_root+'ps5-1-b-%d.png'%i)

#Question 2.a
def Gaussain_pyramid_reduce():
    image1='input/DataSeq1/yos_img_01.jpg'
    for level in range(4):
        image_return=LK_Reduce_Iterative(image1,level)
        #plt.clf()
        num=141+level
        plt.subplot(num),plt.imshow(image_return),plt.title("level %d"%level)
    plt.savefig(output_root+'ps5-2-a-1.png')

#Question 2.b
def Gaussain_pyramid_expand():
    image1='input/DataSeq1/yos_img_01.jpg'
    Gau_pyramids,lap_pyramids=Laplacian_pyramid(image1,4)
    for level in range(4):
        #plt.clf()
        num=141+level
        plt.subplot(num),plt.imshow(lap_pyramids[level]),plt.title("level %d"%level)
    plt.savefig(output_root+'ps5-2-b-1.png')

    for level in range(4):
        #plt.clf()
        num=141+level
        plt.subplot(num),plt.imshow(Gau_pyramids[level]),plt.title("level %d"%level)
    plt.savefig(output_root+'gua.png')

#Question 3.a.1
def LK_Warp_3a1():
    image1='input/DataSeq1/yos_img_01.jpg'
    image2='input/DataSeq1/yos_img_02.jpg'
    image3='input/DataSeq1/yos_img_03.jpg'
    _,old_pts_0,u,v=Lucas_Kanade(image2,image1)
    bksize=5
    _,old_pts_1,u1,v1=Lucas_Kanade(image3,image2,bksize=bksize)
    _,old_pts_0,u,v=Lucas_Kanade(image2,image1,bksize=bksize)
    plt.clf()
    plt.subplot(121),plt.imshow(cv2.imread(image1)),plt.title("1-2")
    for pts in old_pts_0:
        x,y=pts.ravel()
        vx=u[y,x]
        vy=v[y,x]
        plt.arrow(x,y, vx, vy, head_width=5, head_length=5, color='b')
    plt.subplot(122),plt.imshow(cv2.imread(image2)),plt.title("2-3")
    for pts in old_pts_1:
        x,y=pts.ravel()
        vx=u[y,x]
        vy=v[y,x]
        plt.arrow(x,y, vx, vy, head_width=5, head_length=5, color='b')
    plt.savefig(output_root+'ps5-3-a-1.png')

def LK_Warp_3a2():
    #image1='input/TestSeq/Shift0.png'
    #image2='input/TestSeq/ShiftR2.png'
    image1='input/DataSeq1/yos_img_01.jpg'
    image2='input/DataSeq1/yos_img_02.jpg'
    bksize=5
    I1,u,v=Lucas_Kanade_all(image1,image2,bksize=bksize)

    '''
    print("oldframe.shape:{},u:{},v{}".format(oldframe.shape,u.shape,v.shape))
    plt.imshow(u,'gray'),plt.savefig(output_root+'vx.png')
    plt.imshow(cv2.imread(image1)),plt.title(image1.split('/')[-1]+'to'+image2.split('/')[-1])
    h,w,_=oldframe.shape
    for x in range(h):
        for y in range(w):
            vx=u[x,y]
            vy=v[x,y]
            plt.arrow(x,y, vx, vy, head_width=1, head_length=1, color='b') 
    plt.savefig(output_root+'ps5-3-a-0.png')
    '''

    image1_warp=warp_back(image1,image2,u,v)
    image1_differ=I1-image1_warp
    plt.imshow(image1_differ,'gray'),plt.title('image1_differ')
    plt.savefig(output_root+'ps5-3-a-2.png')


def LK_Warp_3a3():
    image1='input/DataSeq2/0.png'
    image2='input/DataSeq2/1.png'
    image3='input/DataSeq2/2.png'
    _,old_pts_0,u,v=Lucas_Kanade(image2,image1)
    bksize=5
    _,old_pts_1,u1,v1=Lucas_Kanade(image3,image2,bksize=bksize)
    _,old_pts_0,u,v=Lucas_Kanade(image2,image1,bksize=bksize)
    plt.clf()
    plt.subplot(121),plt.imshow(cv2.imread(image1)),plt.title("1-2")
    for pts in old_pts_0:
        x,y=pts.ravel()
        vx=u[y,x]
        vy=v[y,x]
        plt.arrow(x,y, vx, vy, head_width=5, head_length=5, color='b')
    plt.subplot(122),plt.imshow(cv2.imread(image2)),plt.title("2-3")
    for pts in old_pts_1:
        x,y=pts.ravel()
        vx=u[y,x]
        vy=v[y,x]
        plt.arrow(x,y, vx, vy, head_width=5, head_length=5, color='b')
    plt.savefig(output_root+'ps5-3-a-3.png')

def LK_Warp_3a4():
    image1='input/DataSeq2/1.png'
    image2='input/DataSeq2/2.png'
    bksize=5
    I1,u,v=Lucas_Kanade_all(image1,image2,bksize=bksize)

    image1_warp=warp_back(image1,image2,u,v)
    image1_differ=I1-image1_warp
    plt.imshow(image1_differ,'gray'),plt.title('image1_differ')
    plt.savefig(output_root+'ps5-3-a-4.png')

#Question 4.a
def hiearchical_LK_4a():
    image1='input/TestSeq/ShiftR10.png'
    image2='input/TestSeq/ShiftR20.png'
    image3='input/TestSeq/ShiftR40.png'
    I10,I210,U10,V10=LK_Hie(image1,image2,max_level=4)#log10=3.3
    I20,I220,U20,V20=LK_Hie(image2,image3,max_level=4)#log20=4.3
    I30,I230,U30,V30=LK_Hie(image1,image3,max_level=4)#log30=4.9
    warp_10_=warp_back(I10,I210,U10,V10)
    warp_20_=warp_back(I20,I220,U20,V20)
    warp_30_=warp_back(I30,I230,U30,V30)
    differ_10_=I10-warp_10_
    differ_20_=I20-warp_20_
    differ_30_=I30-warp_30_
    plt.subplot(131),plt.imshow(differ_10_,'gray'),plt.title('differ10')
    plt.subplot(132),plt.imshow(differ_20_,'gray'),plt.title('differ20')
    plt.subplot(133),plt.imshow(differ_30_,'gray'),plt.title('differ30')
    plt.savefig(output_root+'ps5-4-a-2.png')

    plt.subplot(131),plt.imshow(flow2image(U10,V10)),plt.title('flow10')
    plt.subplot(132),plt.imshow(flow2image(U20,V20)),plt.title('flow20')
    plt.subplot(133),plt.imshow(flow2image(U30,V30)),plt.title('flow30')
    plt.savefig(output_root+'ps5-4-a-1.png')

#Question 4.b
def hiearchical_LK_4b():
    image1='input/DataSeq1/yos_img_01.jpg'
    image2='input/DataSeq1/yos_img_02.jpg'
    image3='input/DataSeq1/yos_img_03.jpg'
    I10,I210,U10,V10=LK_Hie(image1,image2,max_level=4)#log10=3.3
    I20,I220,U20,V20=LK_Hie(image2,image3,max_level=4)#log20=4.3
    I30,I230,U30,V30=LK_Hie(image1,image3,max_level=4)#log30=4.9
    warp_10_=warp_back(I10,I210,U10,V10)
    warp_20_=warp_back(I20,I220,U20,V20)
    warp_30_=warp_back(I30,I230,U30,V30)
    differ_10_=I10-warp_10_
    differ_20_=I20-warp_20_
    differ_30_=I30-warp_30_
    plt.subplot(131),plt.imshow(differ_10_,'gray'),plt.title('differ10')
    plt.subplot(132),plt.imshow(differ_20_,'gray'),plt.title('differ20')
    plt.subplot(133),plt.imshow(differ_30_,'gray'),plt.title('differ30')
    plt.savefig(output_root+'ps5-4-b-2.png')

    plt.subplot(131),plt.imshow(flow2image(U10,V10)),plt.title('flow10')
    plt.subplot(132),plt.imshow(flow2image(U20,V20)),plt.title('flow20')
    plt.subplot(133),plt.imshow(flow2image(U30,V30)),plt.title('flow30')
    plt.savefig(output_root+'ps5-4-b-1.png')

#Question 4.c
def hiearchical_LK_4c():
    image1='input/DataSeq2/0.png'
    image2='input/DataSeq2/1.png'
    image3='input/DataSeq2/2.png'
    I10,I210,U10,V10=LK_Hie(image1,image2,max_level=4)#log10=3.3
    I20,I220,U20,V20=LK_Hie(image2,image3,max_level=4)#log20=4.3
    I30,I230,U30,V30=LK_Hie(image1,image3,max_level=4)#log30=4.9
    warp_10_=warp_back(I10,I210,U10,V10)
    warp_20_=warp_back(I20,I220,U20,V20)
    warp_30_=warp_back(I30,I230,U30,V30)
    differ_10_=I10-warp_10_
    differ_20_=I20-warp_20_
    differ_30_=I30-warp_30_
    plt.subplot(131),plt.imshow(differ_10_,'gray'),plt.title('differ10')
    plt.subplot(132),plt.imshow(differ_20_,'gray'),plt.title('differ20')
    plt.subplot(133),plt.imshow(differ_30_,'gray'),plt.title('differ30')
    plt.savefig(output_root+'ps5-4-c-2.png')

    plt.subplot(131),plt.imshow(flow2image(U10,V10)),plt.title('flow10')
    plt.subplot(132),plt.imshow(flow2image(U20,V20)),plt.title('flow20')
    plt.subplot(133),plt.imshow(flow2image(U30,V30)),plt.title('flow30')
    plt.savefig(output_root+'ps5-4-c-1.png')

if __name__=="__main__":
    for i in range (5,50,5):
        if i%2==0:
            i=i+1
    #LK_test_a('Average',15)
    #LK_test_b('Average',15)
    #Gaussain_pyramid_reduce()
    #Gaussain_pyramid_expand()
    #LK_Warp_3a1()
    #LK_Warp_3a2()
    #LK_Warp_3a3()
    #LK_Warp_3a4()
    #hiearchical_LK_4a()
    hiearchical_LK_4b()
    hiearchical_LK_4c()



    