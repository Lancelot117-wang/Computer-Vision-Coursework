import numpy as np
from scipy.spatial.distance import cdist

# This code is part of:
#
#   CMPSCI 670: Computer Vision, Fall 2019
#   University of Massachusetts, Amherst
#   Instructor: Subhransu Maji
#
#   Mini-project 4

def computeMatches(f1, f2):
    """ Match two sets of SIFT features f1 and f2 """
    print('computing matches')
    result = np.zeros((f1.shape[0]),dtype=int)

    d = f1.shape[1]
    n = f1.shape[0]
    m = f2.shape[0]

    for i in range(n):
        print(i)
        s=0
        t=0
        for j in range(m):
            temp=0
            for k in range(d):
                temp+=(f1[i][k]-f2[j][k])**2
            if s==0 or s>temp:
                t=s
                s=temp
                result[i]=j
            if t==0:
                t=temp

        if s/t>0.8:
            result[i]=-1

    return result

