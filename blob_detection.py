from PIL import Image, ImageDraw
import numpy as np
import math
from scipy import spatial

def find_area_intersection(distance, r1, r2):
    x = (distance ** 2 + r1 ** 2 - r2 ** 2) / (2 * distance)
    y = math.sqrt(r1**2 - x**2)
    
    theta_A = 2*math.asin(y/r1)
    theta_B = 2*math.asin(y/r2)
    
    sect_area_A = r1**2*(theta_A/2)
    sect_area_B = r2**2*(theta_B/2)
    
    tri_area_A = y*math.sqrt(r1**2 - y**2)
    tri_area_B = y*math.sqrt(r2**2 - y**2)
    
    area_overlap = sect_area_A + sect_area_B - tri_area_A - tri_area_B
    return area_overlap/ (math.pi * (min(r1, r2) ** 2))

def overlap_two_circles(blob1, blob2):
    
    n_dim = 2
    root_ndim = 1.414

    r1 = blob1[-1] * root_ndim
    r2 = blob2[-1] * root_ndim

    distance = math.sqrt(np.sum((blob1[:-1] - blob2[:-1])**2))
    if distance > r1 + r2:
        return 0
    
    if distance <= abs(r1 - r2):
        return 1

    if n_dim == 2:
        return find_area_intersection(distance, r1, r2)

def remove_redundant_blobs(blobs_array, overlap):
    sigma = blobs_array[:, -1].max()
    
    distance = 2 * sigma * math.sqrt(2)
    
    tree = spatial.cKDTree(blobs_array[:, :-1])
    
    pairs = np.array(list(tree.query_pairs(distance)))
    if len(pairs) == 0:
        return blobs_array
    else:
        for (i, j) in pairs:
            blob1, blob2 = blobs_array[i],blobs_array[j]
            if overlap_two_circles(blob1,blob2) > overlap:
                if blob1[-1] > blob2[-1]:
                    blob2[-1] = 0
                else:
                    blob1[-1] = 0

    return np.array([b for b in blobs_array if b[-1] > 0])

def convolution(image, kernel, kernel_size):
    pad_index = int(kernel_size/2)
    temp = np.pad(image, pad_index, mode="edge")
    res = np.zeros(image.shape, dtype=float)
    size = temp.shape
    leave_index = int(kernel_size - 1)
    
    for i in range(size[0]-leave_index):
        for j in range(size[1]-leave_index):
            window = temp[i:i+kernel_size,j:j+kernel_size]
            product = window*kernel
            res[i][j] = np.sum(product)
    return res

# def generate_log(sigma):
#     n = np.ceil(np.ceil(sigma*6)/2)
#     y,x = np.ogrid[-n:n+1,-n:n+1]
#     log = (1/6.28)*(1/sigma**4)*(x**2 + y**2 - 2*sigma**2)*np.exp(-(x**2 + y**2)/(2*sigma**2))
#     return log

def generate_log(sigma):
    #window size 
    n = np.ceil(np.ceil(sigma*6)/2)
    y,x = np.ogrid[-n:n+1,-n:n+1]
    y_filter = np.exp(-(y*y/(2.*sigma*sigma)))
    x_filter = np.exp(-(x*x/(2.*sigma*sigma)))
    final_filter = (-(2*sigma**2) + (x*x + y*y) ) *  (x_filter*y_filter) * (1/(2*np.pi*sigma**4))
    return final_filter


if __name__ == "__main__":
    imgPath = 'E:/sem-6/CV/Assignment-1/HW1_Q3/butterfly.jpg'
    img = Image.open(imgPath).convert('L')
    img = np.array(img)
    img = img/255

    stack_mat = np.zeros((10,img.shape[0],img.shape[1]))

    k = 1.25
    for i in range(0,10):
        log = generate_log(1.25*(k**i))
        stack_mat[i] = convolution(img,log,log.shape[0])
    #stack_mat = stack_mat**2
    
    height = stack_mat.shape[1]
    width = stack_mat.shape[2]
    
    temp_mat = np.zeros((10,9,9))
    final_mat = []
    for h in range(height-9):
        for w in range(width-9):
            vert_start = h
            vert_end =  h + 9
            horiz_start =  w
            horiz_end =  w + 9
            temp_mat = stack_mat[:,vert_start:vert_end,horiz_start:horiz_end]
            layer , x_cord , y_cord = np.unravel_index(temp_mat.argmax(), temp_mat.shape)
            if(temp_mat[layer,x_cord,y_cord] > 0.05):
                sigma = 1.25*(1.25**layer)
                final_mat.append((vert_start + x_cord, horiz_start + y_cord,sigma))
    
    final_mat = list(set(final_mat))
    final_mat = np.array(final_mat)
    final_mat = remove_redundant_blobs(final_mat,0.3)
        
    img = Image.open(imgPath)
    for blob in final_mat:
        draw = ImageDraw.Draw(img)
        y, x, sigma = blob
        r = sigma*1.414
        draw.arc((x-r, y-r, x+r, y+r), start=0, end=360, fill=(0, 0, 255)) 
    img.show()      