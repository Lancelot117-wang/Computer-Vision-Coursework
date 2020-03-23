import numpy as np

def prepareData(imArray, ambientImage):
    for i in range(64):
        imArray[:,:,i]-=ambientImage[:,:]
        imArray[:,:,i]=imArray[:,:,i]/255.

    np.clip(imArray[:,:,:],0,1)

    return imArray
    
    #raise NotImplementedError("You should implement this.")
