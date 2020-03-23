import numpy as np

def multiclassLRPredict(model, x):
    numData = x.shape[1]

    classLabels = model['classLabels']
    numClass = classLabels.shape[0]

    w = model['w']
    
    # Simply predict the first class (Implement this)
    ypred = np.ones(numData)

    for i in range(numData):
        temp = np.zeros((numClass),dtype=float)
        for j in range(numClass):
            temp[j] = np.dot(x[:,i],w[j])
        ypred[i] *= np.argmax(temp)
    
    return ypred
