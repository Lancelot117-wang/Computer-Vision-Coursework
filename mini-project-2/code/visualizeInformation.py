import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cl

def visualizeInformation(img, r):
    k=0.03
    t = np.zeros((img.shape[0],img.shape[1]))
    t[:,:] = (img[:,:,0]+img[:,:,1]+img[:,:,2])/3
    x = np.zeros((img.shape[0]-1,img.shape[1]-1))
    y = np.zeros((img.shape[0]-1,img.shape[1]-1))
    x[:,:] = t[1:,:-1] - t[:-1,:-1]
    y[:,:] = t[:-1,1:] - t[:-1,:-1]

    height = img.shape[0]-1
    width = img.shape[1]-1

    m = np.zeros((height-2*r,width-2*r,2,2))
    res = np.zeros((height-2*r,width-2*r))
    for i in range(r,height-r):
        for j in range(r,width-r):
            for a in range(-r,r+1):
                for b in range(-r,r+1):
                    m[i-r][j-r][0][0]+=x[i+a][j+b]**2
                    m[i-r][j-r][0][1]+=x[i+a][j+b]*y[i+a][j+b]
                    m[i-r][j-r][1][0]+=x[i+a][j+b]*y[i+a][j+b]
                    m[i-r][j-r][1][1]+=y[i+a][j+b]**2
            res[i-r][j-r]=m[i-r][j-r][0][0]*m[i-r][j-r][1][1]-\
                       m[i-r][j-r][0][1]*m[i-r][j-r][1][0]-\
                       k*((m[i-r][j-r][0][0]+m[i-r][j-r][1][1])**2)

    plt.imshow(res, cmap='coolwarm', norm=cl.Normalize(-0.1,0.1))
    plt.colorbar()
    plt.show()
    
    return
