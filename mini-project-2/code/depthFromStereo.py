import numpy as np

def depthFromStereo(img1, img2, ws):
    n=(ws-1)/2
    height=img1.shape[0]
    width=img1.shape[1]
    t1 = np.zeros((height,width))
    t2 = np.zeros((height,width))
    t1[:,:] = (img1[:,:,0]+img1[:,:,1]+img1[:,:,2])/3
    t2[:,:] = (img2[:,:,0]+img2[:,:,1]+img2[:,:,2])/3
    disparity = np.zeros((height-2*n,width-2*n))
    depth = np.zeros((height-2*n,width-2*n))
    for i in range(n,height-n):
        for j in range(n,width-n):
            t=0
            tres=n**2
            tsum=0
            for k in range(n,width-n):
                tsum=0
                for x in range(-n,n+1):
                    for y in range(-n,n+1):
                        tsum+=(t1[i+x][j+y]-t2[i+x][k+y])**2
                tsum/=(2*n+1)**2
                if tsum < tres:
                    tres=tsum
                    t=(k-j)
                
            disparity[i-n][j-n]=t
            
    for i in range(height-2*n):
        for j in range(width-2*n):
            if disparity[i][j] != 0:
                depth[i][j]=1/disparity[i][j]
    
    return depth
