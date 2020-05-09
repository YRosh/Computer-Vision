from PIL import Image
import numpy as np
from numpy import linalg as LA

def convolve(img, kernal):
    imgtemp = np.zeros((img.shape[0], img.shape[1]))
    kernal_width = kernal.shape[0]
    pad = kernal_width//2
    img = np.pad(img, ((pad, pad), (pad, pad)), mode='edge')

    for h in range(imgtemp.shape[0]):
        for w in range(imgtemp.shape[1]):
            img_slice = img[h:h+kernal_width, w:w+kernal_width]
            imgtemp[h][w] = np.sum(np.multiply(img_slice, kernal))
    
    return imgtemp

def gradient_y(img):
    img = np.array(img)
    
    kernal = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    imgcov = convolve(img, kernal)
    
    return imgcov

def gradient_x(img):
    img = np.array(img)
    
    kernal = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    imgcov = convolve(img, kernal)
    
    return imgcov

def harrisCorner(img, Ixx, Ixy, Iyy):
    img = np.array(img) 
    responces = []
    k = 0.04
    offset = 3//2
    
    for h in range(offset, img.shape[0]-offset):
        for w in range(offset, img.shape[1]-offset):
            Sxx = np.sum(Ixx[h-offset:h+offset+1, w-offset:w+offset+1])
            Sxy = np.sum(Ixy[h-offset:h+offset+1, w-offset:w+offset+1])
            Syy = np.sum(Iyy[h-offset:h+offset+1, w-offset:w+offset+1])
            
            det = Sxx*Syy - Sxy**2
            trace = Sxx + Syy
            
            responce = det - k*(trace**2)
            #print(responce)
            if responce > 5000000000:

                responces.append((h,w))
              
    img = np.dstack((img, img, img))
    
    for responce in responces:
        img[responce[0], responce[1], : ] = [255, 0, 0]
        
    img = Image.fromarray(np.uint8(img))
    img.show(title="Harris Corner Detection")

def shiTomasi(img, Ixx, Ixy, Iyy):
    img = np.array(img)
    responces = []
    offset = 3//2
    
    for h in range(offset, img.shape[0]-offset):
        for w in range(offset, img.shape[1]-offset):
            #print(Ixx[h-offset:h+offset+1, w-offset:w+offset+1])
            Sxx = np.sum(Ixx[h-offset:h+offset+1, w-offset:w+offset+1])
            Sxy = np.sum(Ixy[h-offset:h+offset+1, w-offset:w+offset+1])
            Syy = np.sum(Iyy[h-offset:h+offset+1, w-offset:w+offset+1])
            
            hessian = np.array([[Sxx, Sxy], [Sxy, Syy]])
            eigenval, eigenvec = LA.eig(hessian)
            
            if min(eigenval) > 80000:
                responces.append((h, w))
                
    img = np.dstack((img, img, img))
    
    for responce in responces:
        img[responce[0], responce[1], : ] = [255, 0, 0]
        
    img = Image.fromarray(np.uint8(img))
    img.show(title="Shi-Thomasi Corner Detection")
        
            
if __name__ == '__main__':
    imgpath = 'E:/sem-6/CV/Assignment-1/HW1_Q2/Image2.jpg'
    img = Image.open(imgpath)
    #print(np.array(img))
    img.show()
    
    I_x = gradient_x(img)
    I_y = gradient_y(img)
    
    Ixx = I_x**2
    Ixy = I_x*I_y
    Iyy = I_y**2
    
    #print(Ixx)
    
    shiTomasi(img, Ixx, Ixy, Iyy)
    
    harrisCorner(img, Ixx, Ixy, Iyy)