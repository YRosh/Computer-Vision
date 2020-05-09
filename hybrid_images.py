import numpy as np
from PIL import Image 
import sys

def generateKernel(sigma, width):
    kernel = np.zeros((width,width))
    temp = width//2
    for i in range(width):
        for j in range(width):
            x = -temp+i
            y = -temp+j
            kernel[i,j] = (1/(2*np.pi*(sigma**2)))*np.exp(-1*(x**2+y**2)/(2*sigma**2))
    return kernel

def lowpass(path, kernal):
    kernal_width = kernal.shape[0]
    pad = kernal_width//2
    img = Image.open(path)
    img = np.array(img)
    
    imgout = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype="uint8")
    
    img = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
    
    for h in range(imgout.shape[0]):
        for w in range(imgout.shape[1]):
            for c in range(imgout.shape[2]):
                imgSlice = img[h:h+kernal_width, w:w+kernal_width, c]
                conv = np.multiply(imgSlice, kernal)
                imgout[h][w][c] = np.sum(conv)

    return imgout

if __name__ == '__main__':
    img1path = sys.argv[1]
    img2path = sys.argv[2]
    
    kernal = generateKernel(3,11)
    img1 = lowpass(img1path, kernal)
    # img1.show()
    
    kernal = generateKernel(1, 7)
    img2 = lowpass(img2path, kernal)
    img2real = Image.open(img2path)
    img2real = np.array(img2real)
    img2 = img2real - img2
    img2 = Image.fromarray(np.uint8(img2))
    img2 = img2.resize((img1.shape[1], img1.shape[0]))
    img2 = np.array(img2)
    
    img = img1+img2
    img = Image.fromarray(np.uint8(img))
    img.show()