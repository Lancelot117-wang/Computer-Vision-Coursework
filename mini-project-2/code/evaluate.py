import numpy as np

def evaluate(img, res, count):
    image_height, image_width, c = img.shape

    temp = np.zeros((image_height,image_width))
    
    for i in range(image_height):
        for j in range(image_width):
            if img[i][j][0]==1 or img[i][j][1]==1 or img[i][j][2]==1:
                temp[i][j] = -1
            elif img[i][j][0]==0 or img[i][j][1]==0 or img[i][j][2]==0:
                temp[i][j] = -1
            elif img[i][j][0] > img[i][j][1] and img[i][j][0] > img[i][j][2]:
                temp[i][j] = 0
            elif img[i][j][1] > img[i][j][0] and img[i][j][1] > img[i][j][2]:
                temp[i][j] = 1
            elif img[i][j][2] > img[i][j][0] and img[i][j][2] > img[i][j][1]:
                temp[i][j] = 2
            else:
                temp[i][j] = -1
    t=0
    for i in range(0,image_height-1,2):
        for j in range(0,image_width-1,2):
            if temp[i][j]==-1 or temp[i+1][j]==-1 or temp[i][j+1]==-1 or temp[i+1][j+1]==-1:
                continue
            a = int(temp[i][j])
            b = int(temp[i+1][j])
            c = int(temp[i][j+1])
            d = int(temp[i+1][j+1])
            #if a==b and b==c and c==d:
                #continue
            res[a][b][c][d] = res[a][b][c][d] + 1
            t += 1

    return res, count+t
