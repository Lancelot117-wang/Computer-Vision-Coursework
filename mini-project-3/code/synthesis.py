import random
import numpy as np
import matplotlib.pyplot as plt

def synthRandomPatch(img, tileSize, numTiles, outSize):
    res = np.zeros((outSize,outSize),dtype=np.int)

    for i in range(numTiles):
        for j in range(numTiles):
            x=random.randint(0,img.shape[0]-tileSize)
            y=random.randint(0,img.shape[1]-tileSize)

            for m in range(i*tileSize,(i+1)*tileSize):
                for n in range(j*tileSize,(j+1)*tileSize):
                   res[m][n]=img[m-i*tileSize+x][n-j*tileSize+y]


    plt.imshow(res,cmap='gray')
    plt.colorbar()
    plt.show()

    return res

def synthQuilting(img, tileSize, numTiles, outSize):
    height = img.shape[0]
    width = img.shape[1]
    
    q=int(tileSize/6)
    
    res = np.zeros((outSize,outSize),dtype=np.int)

    for i in range(numTiles):
        for j in range(numTiles):
            x=0
            y=0

            bestMatches = []
            
            if i==0 and j==0:
                x=random.randint(0,img.shape[0]-tileSize)
                y=random.randint(0,img.shape[1]-tileSize)
            elif i==0:
                ssd = int((tileSize*q)*(255**2))
        
                for m in range(height-tileSize):
                    for n in range(width-tileSize):
                        t=0.
                        for ii in range(tileSize):
                            for jj in range(q):
                                t+=(img[m+ii][n+jj]-res[ii][j*tileSize-q+jj])**2
                        
                        if t<ssd:
                            ssd=t
        
                for m in range(height-tileSize):
                    for n in range(width-tileSize):
                        t=0.
                        for ii in range(tileSize):
                            for jj in range(q):
                                t+=(img[m+ii][n+jj]-res[ii][j*tileSize-q+jj])**2

                        if t<=ssd*1.1:
                            bestMatches.append([m, n])
            elif j==0:
                ssd = int((tileSize*q)*(255**2))
        
                for m in range(height-tileSize):
                    for n in range(width-tileSize):
                        t=0.
                        for ii in range(q):
                            for jj in range(tileSize):
                                t+=(img[m+ii][n+jj]-res[i*tileSize-q+ii][jj])**2
                        
                        if t<ssd:
                            ssd=t
        
                for m in range(height-tileSize):
                    for n in range(width-tileSize):
                        t=0.
                        for ii in range(q):
                            for jj in range(tileSize):
                                t+=(img[m+ii][n+jj]-res[i*tileSize-q+ii][jj])**2

                        if t<=ssd*1.1:
                            bestMatches.append([m, n])
            else:
                ssd = int((tileSize*tileSize)*(255**2))
        
                for m in range(height-tileSize):
                    for n in range(width-tileSize):
                        t=0.
                        for ii in range(tileSize):
                            for jj in range(q):
                                t+=(img[m+ii][n+jj]-res[ii][j*tileSize-q+jj])**2
                        for ii in range(q):
                            for jj in range(tileSize):
                                t+=(img[m+ii][n+jj]-res[i*tileSize-q+ii][jj])**2
                        if t<ssd:
                            ssd=t
        
                for m in range(height-tileSize):
                    for n in range(width-tileSize):
                        t=0.
                        for ii in range(tileSize):
                            for jj in range(q):
                                t+=(img[m+ii][n+jj]-res[ii][j*tileSize-q+jj])**2
                        for ii in range(q):
                            for jj in range(tileSize):
                                t+=(img[m+ii][n+jj]-res[i*tileSize-q+ii][jj])**2
                        
                        if t<=ssd*1.1:
                            bestMatches.append([m, n])

            if len(bestMatches)==1:
                r=0
            elif len(bestMatches)!=0:
                r=random.randint(0,len(bestMatches)-1)

            if i!=0 or j!=0:
                x=bestMatches[r][0]
                y=bestMatches[r][1]

            for m in range(i*tileSize,(i+1)*tileSize):
                for n in range(j*tileSize,(j+1)*tileSize):
                    res[m][n]=img[m-i*tileSize+x][n-j*tileSize+y]


    plt.imshow(res,cmap='gray')
    plt.colorbar()
    plt.show()

    return res

def synthEfrosLeung(img, winSize, outSize):
    res = np.zeros((outSize,outSize),dtype=np.int16)
    visit = np.zeros((outSize,outSize))

    height = img.shape[0]
    width = img.shape[1]
    
    s=(winSize-1)/2

    x=random.randint(0,img.shape[0]-3)
    y=random.randint(0,img.shape[1]-3)

    c=int(outSize/2-1)

    for i in range(3):
        for j in range(3):
            res[c+i][c+j]=img[x+i][y+j]

            visit[c+i][c+j]=1

    pixelList = [[c-1, c, 1], [c-1, c+1, 1], [c-1, c+2, 1],\
                [c+3, c, 1], [c+3, c+1, 1], [c+3, c+2, 1],\
                [c, c-1, 1], [c+1, c-1, 1], [c+2, c-1, 1],\
                [c, c+3, 1], [c+1, c+3, 1], [c+2, c+3, 1]]

    while len(pixelList)!=0:
        pixelList.sort(key = lambda x: x[2], reverse = True)

        a = pixelList[0][0]
        b = pixelList[0][1]
        
        mask = np.zeros((winSize,winSize))

        for i in range(winSize):
            for j in range(winSize):
                if a-s+i<0 or a-s+i>=outSize:
                    continue
                if b-s+j<0 or b-s+j>=outSize:
                    continue
                if visit[a-s+i][b-s+j]==1:
                    mask[i][j]=1

        ssd = int((winSize**2)*(255**2))
        
        for i in range(height-2*s):
            for j in range(width-2*s):
                t=0.
                for ii in range(winSize):
                    for jj in range(winSize):
                        if a-s+ii<0 or a-s+ii>=outSize:
                            continue
                        if b-s+jj<0 or b-s+jj>=outSize:
                            continue
                        t+=mask[ii][jj]*((img[i+ii][j+jj]-res[a-s+ii][b-s+jj])**2)
                        
                if t<ssd:
                    ssd=t

        bestMatches = []
        
        for i in range(height-2*s):
            for j in range(width-2*s):
                t=0.
                for ii in range(winSize):
                    for jj in range(winSize):
                        if a-s+ii<0 or a-s+ii>=outSize:
                            continue
                        if b-s+jj<0 or b-s+jj>=outSize:
                            continue
                        t+=mask[ii][jj]*((img[i+ii][j+jj]-res[a-s+ii][b-s+jj])**2)

                if t<=ssd*1.1:
                    bestMatches.append([i+s, j+s])

        if len(bestMatches)==1:
            r=0
        else:
            r=random.randint(0,len(bestMatches)-1)
            
        c=bestMatches[r][0]
        d=bestMatches[r][1]

        res[a][b]=img[c][d]

        visit[a][b]=1

        del pixelList[0]

        aa=False
        bb=False
        cc=False
        dd=False

        if a-1<0:
            aa=True
        if a+1>=outSize:
            bb=True
        if b-1<0:
            cc=True
        if b+1>=outSize:
            dd=True

        if aa==False:
            if visit[a-1][b]==1:
                aa=True
        if bb==False:
            if visit[a+1][b]==1:
                bb=True
        if cc==False:
            if visit[a][b-1]==1:
                cc=True
        if dd==False:
            if visit[a][b+1]==1:
                dd=True
        
        for i in range(len(pixelList)):
            if aa==False:
                if pixelList[i][0]==a-1 and pixelList[i][1]==b:
                    pixelList[i][2]+=1
                    aa=True
            if bb==False:
                if pixelList[i][0]==a+1 and pixelList[i][1]==b:
                    pixelList[i][2]+=1
                    bb=True
            if cc==False:
                if pixelList[i][0]==a and pixelList[i][1]==b-1:
                    pixelList[i][2]+=1
                    cc=True
            if dd==False:
                if pixelList[i][0]==a and pixelList[i][1]==b+1:
                    pixelList[i][2]+=1
                    dd=True

        if aa==False:
            pixelList.append([a-1,b,1])
        if bb==False:
            pixelList.append([a+1,b,1])
        if cc==False:
            pixelList.append([a,b-1,1])
        if dd==False:
            pixelList.append([a,b+1,1])

    plt.imshow(res,cmap='gray')
    plt.colorbar()
    plt.show()

    return res
