from PIL import Image
import numpy as np


def norcorr(img1,img2):
    window_size=7
    pad=window_size//2

    img1 = np.array(img1)
    img2 = np.array(img2)
    
    imgout = np.zeros((img1.shape[0], img1.shape[1]), dtype="uint8")
    for h in range(pad, imgout.shape[0]-pad):
      for w in range(pad, imgout.shape[1]-pad):
        d = 0
        preNormCorr = -999999
        for k in range(30):
            normCorr = 0
            tempNormCorr = 0
            c_1 = img1[h-pad:h+pad,w-pad:w+pad]
            c_2 = img2[h-pad:h+pad,w-pad-k:w+pad-k]
            mean1 = np.mean(c_1)
            mean2 = np.mean(c_2)
            sd1 = np.std(c_1)
            sd2 = np.std(c_2)
            for m in range(-int(pad), int(pad)):
                for n in range(-int(pad), int(pad)):
                    tempNormCorr = (int(img1[h+m, w+n])-mean1)*(int(img2[h+m, (w+n) - k])-mean2)
                    tempNormCorr /= (sd1*sd2)
                    normCorr +=tempNormCorr
            normCorr /= (window_size*window_size)
            if normCorr>preNormCorr:
                preNormCorr = normCorr
                d = k
        imgout[h, w] = d*(255/30)
      print(h)
    img = Image.fromarray(np.uint8(imgout))
    img.show(title="norcorr")

def ssd(img1,img2):

    window_size=7
    pad=window_size//2

    img1 = np.array(img1)
    img2 = np.array(img2)
    
    imgout = np.zeros((img1.shape[0], img1.shape[1]), dtype="uint8")

    min=[]

    for h in range(pad,imgout.shape[0]-pad):
        for w in range(pad,imgout.shape[1]-pad):
              d=0
              min_ssd=99999999
              for k in range(30):
                ssd = 0
                tempSsd = 0
                for m in range(-int(pad), int(pad)):
                  for n in range(-int(pad), int(pad)):
                    tempSsd = int(img1[h+m, w+n]) - int(img2[h+m, (w+n) - k])
                    ssd += tempSsd*tempSsd
                if ssd<min_ssd:
                  min_ssd = ssd
                  d = k

              imgout[h][w]=(d)*(255/30)
        print(h)
    img = Image.fromarray(np.uint8(imgout))
    img.show(title="ssd")
                    



if __name__ == '__main__':
    imgpath1 = 'C:/Users/Amulya/Desktop/sem6/cv/assign2/HW2_Data/tsukuba1.ppm'
    img1 = Image.open(imgpath1).convert('L')
    #print(np.array(img))
    img1.show()
    imgpath2 = 'C:/Users/Amulya/Desktop/sem6/cv/assign2/HW2_Data/tsukuba2.ppm'
    img2 = Image.open(imgpath2).convert('L')
    img2.show()
    norcorr(img1,img2)
    ssd(img1,img2)





