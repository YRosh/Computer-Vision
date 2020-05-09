import numpy as np
import cv2
from PIL import Image
import random

def match_points(kp1, kp2, des1, des2, threshold):
    new_kp1 = [] 
    new_kp2 = []
    for i, feat in enumerate(des2, 0):
        dist = np.sqrt(np.sum(np.square(np.subtract(des1,feat)),axis=1))
        sort_dis = np.argsort(dist)[:2]
        a = np.sqrt(np.sum(np.square(np.subtract(des1[sort_dis[0]],feat))))
        b = np.sqrt(np.sum(np.square(np.subtract(des1[sort_dis[1]],feat))))
        if a < threshold*b:
            new_kp2.append(kp2[i])
            new_kp1.append(kp1[sort_dis[0]])
    return (new_kp1, new_kp2)

def RANSAC(p1, p2):
    if len(p1)>4:
        final_h = np.array([])
        temp = -1
        for i in range(len(p1)):
            rand_pts = np.array(random.sample(list(range(len(p1))),4))
            x1 = [] ; x2=[]
            for j in rand_pts:
                x1.append( [ p1[j].pt[0] , p1[j].pt[1] , 1 ] )
                x2.append( [ p2[j].pt[0] , p2[j].pt[1] , 1 ] )
            x1,x2 = np.array(x2),np.array(x1)
            shape = x1.shape[0]
    
            A = np.zeros((2*shape,9))
    
            for i in range(shape):
                A[2*i] = [-x1[i][0] , -x1[i][1] , -1 , 0 , 0 , 0 , x1[i][0]*x2[i][0] , x1[i][1]*x2[i][0] , x2[i][0] ]
                A[2*i +1] = [ 0 , 0 , 0 , -x1[i][0] , -x1[i][1] , -1 , x1[i][0]*x2[i][1] , x1[i][1]*x2[i][1] , x2[i][1] ]
            
            U, S, Vh = np.linalg.svd(A)
            H = Vh[8].reshape((3,3))
            h = H/H[2,2]
            new = np.matmul(h, points2.T)
            error = np.linalg.norm(np.subtract(points1 , new.T),axis=1) < 3
            if temp<np.count_nonzero(error):
                temp = np.count_nonzero(error)
                final_h = h
    return final_h
    

if __name__ == '__main__':
    img_l = Image.open(r'E:\sem-6\CV\Assignment-2\Data\uttower_left.jpg')
    img_r = Image.open(r'E:\sem-6\CV\Assignment-2\Data\uttower_right.jpg')
    img_l = np.array(img_l)
    img_r = np.array(img_r)

    sift = cv2.xfeatures2d.SIFT_create()
    kp1, ds1 = sift.detectAndCompute(img_l, None)
    kp2, ds2 = sift.detectAndCompute(img_r, None)

    (p1,p2) = match_points(kp1, kp2, ds1, ds2, threshold=0.7)
    
    newImg = np.zeros((max(len(img_l), len(img_r)), img_l.shape[1] + img_r.shape[1],3),dtype='uint8')
    
    newImg[:, :len(img_l[0]),:] = img_l
    newImg[:, len(img_l[0]):,:] = img_r
    
    points1 = np.array([[p.pt[0],p.pt[1],1] for p in p1 ])
    points2 = np.array([[p.pt[0],p.pt[1],1] for p in p2 ])
    
    H = RANSAC(p1, p2)

    (x1,y1) = img_l.shape[:2]
    (x2,y2) = img_r.shape[:2]
    result = cv2.warpPerspective(img_r, H, (y1+y2, x1))
    result[:x1,:y2] = img_l
    img = Image.fromarray(result)
    img.show()
