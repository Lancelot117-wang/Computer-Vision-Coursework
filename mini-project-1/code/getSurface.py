import numpy as np
import random

def getSurface(surfaceNormals, method):
    height=np.zeros((len(surfaceNormals),len(surfaceNormals[0])))
    derivative=np.zeros((len(surfaceNormals),len(surfaceNormals[0]),2))
    for i in range(2):
        derivative[:,:,i]=surfaceNormals[:,:,i]/surfaceNormals[:,:,2]

    if method == 'column':
        height[0]=np.cumsum(derivative[0,:,0])
        for j in range(len(height[0])):
            height[:,j]=np.cumsum(derivative[:,j,1])+height[0,j]

        return height
        #raise NotImplementedError("You should implement this.")
    if method == 'row':
        height[:,0]=np.cumsum(derivative[:,0,1])
        for i in range(len(height)):
            height[i,:]=np.cumsum(derivative[i,:,0])+height[i,0]

        return height
        #raise NotImplementedError("You should implement this.")
    if method == 'average':
        height1=np.zeros((len(surfaceNormals),len(surfaceNormals[0])))
        height2=np.zeros((len(surfaceNormals),len(surfaceNormals[0])))
        
        height1[0]=np.cumsum(derivative[0,:,0])
        for j in range(len(height1[0])):
            height1[:,j]=np.cumsum(derivative[:,j,1])+height1[0,j]

        height2[:,0]=np.cumsum(derivative[:,0,1])
        for i in range(len(height2)):
            height2[i,:]=np.cumsum(derivative[i,:,0])+height2[i,0]

        height[:,:]=(height1[:,:]+height2[:,:])/2.

        return height
        #raise NotImplementedError("You should implement this.")
    if method == 'random':
        height1=np.zeros((len(surfaceNormals),len(surfaceNormals[0])))

        height[0]=np.cumsum(derivative[0,:,0])
        height[:,0]=np.cumsum(derivative[:,0,1])

        for i in range(1,len(height)):
            for j in range(1,len(height[0])):
                temp=0
                k=[i,j]
                while k[0]!=0 and k[1]!=0:
                    if 0==random.randint(0,2):
                        temp+=derivative[k[0],k[1],0]
                        k[0]-=1
                    else:
                        temp+=derivative[k[0],k[1],1]
                        k[1]-=1
                temp+=height[k[0]][k[1]]
                height[i][j]=temp

        return height

        #raise NotImplementedError("You should implement this.")
