import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

def photometricStereo(imarray, lightdirs):
    albedo=np.zeros((len(imarray),len(imarray[0])))
    normals=np.zeros((len(imarray),len(imarray[0]),3))
                            
    matrix=np.zeros((len(imarray[0][0])))

    for i in range(len(imarray)):
        for j in range(len(imarray[i])):
            matrix[:]=imarray[i,j,:]
            (temp,a,b,c)=sp.linalg.lstsq(lightdirs,matrix)
            albedo[i][j]=np.linalg.norm(temp)
            normals[i,j,:]=temp[:]/albedo[i][j]

    return albedo, normals
    
    #raise NotImplementedError("You should implement this.")
