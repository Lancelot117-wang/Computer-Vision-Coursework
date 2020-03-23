import os
import time
import numpy as np
import matplotlib.pyplot as plt 
import sys
import math
import scipy.ndimage.filters as f
import scipy.signal as s
from utils import gaussian

def gFilter(img, kernel):
    res = f.convolve(img, kernel)
    return res

def mFilter(img, n = 3):
    res = s.medfilt2d(img, n)
    return res

def nlmFilter(img, p, w, y):
    height = img.shape[0]
    width = img.shape[1]
    res = np.zeros((height,width))
    res[:][:]=img[:][:]
    center = np.zeros((height,width,2))
    q = (p-1)/2
    v = (w-1)/2

    for i in range(q,height-q):
        for j in range(q,width-q):
            xs=0.
            ys=0.
            s=0.
            for ii in range(i-q,i+q+1):
                for jj in range(j-q,j+q+1):
                    s+=img[ii][jj]
                    xs+=ii*img[ii][jj]
                    ys+=jj*img[ii][jj]
            if s!= 0:
                xs/=s
                ys/=s
            else:
                xs=i
                ys=j

            center[i][j][0]=xs
            center[i][j][1]=ys
    
    for i in range(v,height-v):
        for j in range(v,width-v):
            xf=0.
            yf=0.
            
            xs=center[i][j][0]
            ys=center[i][j][1]
            
            for ii in range(i-v,i+v+1):
                if ii-q<0 or ii+q>=height:
                    continue
                for jj in range(j-v,j+v+1):
                    if jj-q<0 or jj+q>=width:
                        continue
                    xss=center[ii][jj][0]
                    yss=center[ii][jj][1]

                    xf+=math.exp(-y*((xs-xss)**2+(ys-yss)*2))*img[ii][jj]
                    yf+=math.exp(-y*((xs-xss)**2+(ys-yss)*2))

            res[i][j]=xf/yf

    return res
            
