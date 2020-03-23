# This code is part of:
#
#   CMPSCI 370: Computer Vision, Spring 2018
#   University of Massachusetts, Amherst
#   Instructor: Subhransu Maji
#
#   Homework 1 

import numpy as np

def alignChannels(img, max_shift):
    shift_i=[-max_shift[0],-max_shift[0]]
    shift_j=[-max_shift[1],-max_shift[1]]
    SSD1 = 0
    SSD2 = 0
    
    for m in range(-max_shift[0], max_shift[0]+1):
        for n in range(-max_shift[1], max_shift[1]+1):   
            img_temp=img.copy()
            img_temp[:, :, 1] = np.roll(img[:, :, 1], [m, n], axis=[0, 1])
            img_temp[:, :, 2] = np.roll(img[:, :, 2], [m, n], axis=[0, 1])
            if m<0 and n<0:
                result1 = np.sum((img_temp[:m,:n,0]-img_temp[:m,:n,1])**2)
                result2 = np.sum((img_temp[:m,:n,0]-img_temp[:m,:n,2])**2)
            elif m>=0 and n<0:
                result1 = np.sum((img_temp[m:,:n,0]-img_temp[m:,:n,1])**2)
                result2 = np.sum((img_temp[m:,:n,0]-img_temp[m:,:n,2])**2)
            elif m<0 and n>=0:
                result1 = np.sum((img_temp[:m,n:,0]-img_temp[:m,n:,1])**2)
                result2 = np.sum((img_temp[:m,n:,0]-img_temp[:m,n:,2])**2)
            else:
                result1 = np.sum((img_temp[m:,n:,0]-img_temp[m:,n:,1])**2)
                result2 = np.sum((img_temp[m:,n:,0]-img_temp[m:,n:,2])**2)
            if m == -max_shift[0] and n == -max_shift[1]:
                SSD1 = result1
                SSD2 = result2
            else:
                if SSD1 > result1:
                    SSD1 = result1
                    shift_i[0] = m
                    shift_j[0] = n
                if SSD2 > result2:
                    SSD2 = result2
                    shift_i[1] = m
                    shift_j[1] = n

    img[:, :, 1] = np.roll(img[:, :, 1], [shift_i[0], shift_j[0]], axis=[0, 1])
    img[:, :, 2] = np.roll(img[:, :, 2], [shift_i[1], shift_j[1]], axis=[0, 1])
    
    pred_shift = np.array([shift_i, shift_j]).T

    return img, pred_shift
    #raise NotImplementedError("You should implement this.")

