import numpy as np
import cv2
import time
import random
import scipy

# This code is part of:
#
#   CMPSCI 670: Computer Vision, Fall 2019
#   University of Massachusetts, Amherst
#   Instructor: Subhransu Maji
#
#   Mini-project 4

def ransac(matches, blobs1, blobs2):
    a=0
    b=0
    c=0

    n=blobs1.shape[0]
    m=blobs2.shape[0]
    
    threshold=300

    times=1000

    co=0
    inliers=[]
    for i in range(times):      
        a=random.randint(0,n-1)
        while matches[a]==-1:
            a=random.randint(0,n-1)
        b=random.randint(0,n-1)
        while b==a or matches[b]==-1:
            b=random.randint(0,n-1)
        c=random.randint(0,n-1)
        while c==a or c==b or matches[c]==-1:
            c=random.randint(0,n-1)

        A=np.zeros((6,6),dtype=int)
        B=np.zeros((6),dtype=int)

        A[0]=[blobs2[matches[a]][0],blobs2[matches[a]][1],0,0,1,0]
        A[1]=[0,0,blobs2[matches[a]][0],blobs2[matches[a]][1],0,1]
        A[2]=[blobs2[matches[b]][0],blobs2[matches[b]][1],0,0,1,0]
        A[3]=[0,0,blobs2[matches[b]][0],blobs2[matches[b]][1],0,1]
        A[4]=[blobs2[matches[c]][0],blobs2[matches[c]][1],0,0,1,0]
        A[5]=[0,0,blobs2[matches[c]][0],blobs2[matches[c]][1],0,1]

        B=[[blobs1[a][0]],\
           [blobs1[a][1]],\
           [blobs1[b][0]],\
           [blobs1[b][1]],\
           [blobs1[c][0]],\
           [blobs1[c][1]]]

        temp,t1,t2,t3 = scipy.linalg.lstsq(A,B)

        count=0
        inl=[]
        for j in range(n):
            if matches[j]==-1:
                continue
            C=[[blobs2[matches[j]][0],blobs2[matches[j]][1],0,0,1,0],\
               [0,0,blobs2[matches[j]][0],blobs2[matches[j]][1],0,1]]
            D=np.matmul(C,temp)

            differ=(D[0]-blobs1[j][0])**2+\
                    (D[1]-blobs1[j][1])**2
            if differ<threshold:
                count+=1
                inl.append(j)
        if count>co:
            co=count
            inliers=inl

    A=[]
    B=[]
    for i in inliers:
        A.append([blobs2[matches[i]][0],blobs2[matches[i]][1],0,0,1,0])
        A.append([0,0,blobs2[matches[i]][0],blobs2[matches[i]][1],0,1])

        B.append([blobs1[i][0]])
        B.append([blobs1[i][1]])

    tt,t1,t2,t3 = scipy.linalg.lstsq(A,B)

    transf=[[tt[0][0],tt[1][0],tt[4][0]],\
            [tt[2][0],tt[3][0],tt[5][0]]]

    return inliers, transf
    
                                                   


