import math
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

# EXTRACTDIGITFEATURES extracts features from digit images
#   features = extractDigitFeatures(x, featureType) extracts FEATURES from images
#   images X of the provided FEATURETYPE. The images are assumed to the of
#   size [W H 1 N] where the first two dimensions are the width and height.
#   The output is of size [D N] where D is the size of each feature and N
#   is the number of images. 
def extractDigitFeatures(x, featureType):
    
    if featureType == 'pixel':
        features=np.reshape(x,(x.shape[0]*x.shape[1],x.shape[2]))
    elif featureType == 'hog':
        temp=np.zeros((8,x.shape[0]/4,x.shape[1]/4,x.shape[2]),dtype=int)
        features=np.zeros((x.shape[0]*x.shape[1]/16*8,x.shape[2]),dtype=int)
        for i in range(x.shape[2]):
            sx=ndimage.sobel(x[:,:,i],axis=0,mode='constant')
            sy=ndimage.sobel(x[:,:,i],axis=1,mode='constant')
            d=np.zeros(sx.shape,dtype=float)
            for j in range(sx.shape[0]):
                for k in range(sx.shape[1]):
                    if sy[j][k]==0:
                        if sx[j][k]>=0:
                            d[j][k]=float('inf')
                        else:
                            d[j][k]=float('-inf')
                    else:
                        d[j][k]=sx[j][k]/sy[j][k]
            d=np.arctan(d)
            for j in range(sx.shape[0]):
                for k in range(sx.shape[1]):
                    if sy[j][k]<0:
                        if d[j][k]>0:
                            d[j][k]-=math.pi
                        else:
                            d[j][k]+=math.pi
            for j in range(x.shape[0]/4):
                for k in range(x.shape[1]/4):
                    for m in range(4):
                        for n in range(4):
                            t=int((d[j*4+m][k*4+n]+math.pi)/(2*math.pi/8))-1
                            temp[t][j][k][i]+=1
                    for o in range(8):
                        features[(j*4+k)*8+o][i]=temp[o][j][k][i]
    elif featureType == 'lbp':
        features=np.zeros(((x.shape[0]-2)*(x.shape[1]-2),x.shape[2]),dtype=int)
        for i in range(x.shape[2]):
            for j in range(1,x.shape[0]-1):
                for k in range(1,x.shape[1]-1):
                    count=0
                    if x[j-1][k-1][i]>x[j][k][i]:
                        count+=1
                    if x[j-1][k][i]>x[j][k][i]:
                        count+=2
                    if x[j-1][k+1][i]>x[j][k][i]:
                        count+=4
                    if x[j][k+1][i]>x[j][k][i]:
                        count+=8
                    if x[j+1][k+1][i]>x[j][k][i]:
                        count+=16
                    if x[j+1][k][i]>x[j][k][i]:
                        count+=32
                    if x[j+1][k-1][i]>x[j][k][i]:
                        count+=64
                    if x[j][k-1][i]>x[j][k][i]:
                        count+=128
                    features[(j-1)*(x.shape[1]-2)+k-1][i]=count
    '''
    features=np.sqrt(features)
    '''
    for i in range(features.shape[1]):
        count=0
        for j in range(features.shape[0]):
            count+=features[j][i]**2
        count=math.sqrt(count)
        for j in range(features.shape[0]):
            features[j][i]/=count
    
    return features

