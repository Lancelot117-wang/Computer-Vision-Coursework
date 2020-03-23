import math
import numpy as np

def multiclassLRTrain(x, y, param):
    eta=param['eta']
    maxiter=param['maxiter']
    lambda_=param['lambda']

    classLabels = np.unique(y)
    numClass = classLabels.shape[0]
    numFeats = x.shape[0]
    numData = x.shape[1]

    # Initialize weights randomly (Implement gradient descent)
    model = {}
    model['w'] = np.random.randn(numClass, numFeats)*0.01
    model['classLabels'] = classLabels

    for i in range(maxiter):
        g=np.zeros((numClass, numFeats),dtype=float)
        for j in range(numData):
            g[y[j],:]+=gradient(model['w'],x[:,j],y[j],lambda_)
        g/=numData
        model['w']-=g*eta
        #print('iteration:',i+1)

    return model

def gradient(w,f,k,lam):
    if np.dot(w[k],f)>700:
        a=-f*math.exp(700)
    else:
        a=-f*math.exp(np.dot(w[k],f))
    n=0.
    for i in range(w.shape[0]):
        if np.dot(w[i],f)>700:
            n+=math.exp(700) 
        else:
            n+=math.exp(np.dot(w[i],f)) 
    b=-a/n
    c=2*lam*w[k]
    return a+b+c
