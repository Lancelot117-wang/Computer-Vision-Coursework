# This code is part of:
#
#   CMPSCI 670: Computer Vision
#   University of Massachusetts, Amherst
#   Instructor: Subhransu Maji
#

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage.color import rgb2gray

threshold = 50

def getBlobs(sp,n):
    temp = []
    
    for i in range(1,sp.shape[0]-1):
        for j in range(1,sp.shape[1]-1):
            for k in range(1,sp.shape[2]-1):
                b = True
                for x in [i-1,i,i+1]:
                    for y in [j-1,j,j+1]:
                        for z in [k-1,k,k+1]:
                            if sp[i][j][k]<sp[x][y][z]:
                                b = False
                if b==True:
                    c = True
                    for m in range(len(temp)):
                        if temp[m][0]==j and temp[m][1]==i:
                            c = False
                    if c==True and sp[i][j][k]>threshold:
                        temp.append([j,i,(3*(n**k)-1)/2,sp[i][j][k]])

    t = np.asarray(temp)

    return t

def detectBlobs(im, n=15, k=1.2):
    # Input:
    #   IM - input image
    #
    # Ouput:
    #   BLOBS - n x 4 array with blob in each row in (x, y, radius, score)
    #
    # Dummy - returns a blob at the center of the image

    gray = rgb2gray(im)
    h = gray.shape[0]
    w = gray.shape[1]

    space = np.zeros((h,w,n),dtype=np.float)

    ws = float(3)
    s = float(ws-2)/8
    k = float(k)
    for i in range(n):
        result = ndimage.gaussian_laplace(gray, sigma=s)
        
        for m in range(h):
            for n in range(w):
                space[m][n][i]=(result[m][n]**2)*(ws**4)
        '''
        plt.imshow(space[:,:,i], cmap='gray')
        plt.title(ws)
        plt.show()
        '''

        ws *= k
        s = (ws-2)/8

    blobs = getBlobs(space,k)

    print(blobs.shape)

    return blobs
