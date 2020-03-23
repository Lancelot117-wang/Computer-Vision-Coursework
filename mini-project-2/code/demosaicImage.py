# This code is part of:
#
#   CMPSCI 670: Computer Vision, Fall 2018
#   University of Massachusetts, Amherst
#   Instructor: Subhransu Maji
#
#   Homework 1 

import numpy as np

def demosaicImage(image, method):
    ''' Demosaics image.

    Args:
        img: np.array of size NxM.
        method: demosaicing method (baseline or nn).

    Returns:
        Color image of size NxMx3 computed using method.
    '''

    if method.lower() == "baseline":
        return demosaicBaseline(image.copy())
    elif method.lower() == 'nn':
        return demosaicNN(image.copy()) # Implement this
    elif method.lower() == 'linear':
        return demosaicLinear(image.copy()) # Implement this
    elif method.lower() == 'adagrad':
        return demosaicAdagrad(image.copy()) # Implement this
    else:
        raise ValueError("method {} unkown.".format(method))


def demosaicBaseline(img):
    '''Baseline demosaicing.
    
    Replaces missing values with the mean of each color channel.
    
    Args:
        img: np.array of size NxM.

    Returns:
        Color image of sieze NxMx3 demosaiced using the baseline 
        algorithm.
    '''
    mos_img = np.tile(img[:, :, np.newaxis], [1, 1, 3])
    image_height, image_width = img.shape

    red_values = img[0:image_height:2, 0:image_width:2]
    mean_value = red_values.mean()
    mos_img[:, :, 0] = mean_value
    mos_img[0:image_height:2, 0:image_width:2, 0] = img[0:image_height:2, 0:image_width:2]

    blue_values = img[1:image_height:2, 1:image_width:2]
    mean_value = blue_values.mean()
    mos_img[:, :, 2] = mean_value
    mos_img[1:image_height:2, 1:image_width:2, 2] = img[1:image_height:2, 1:image_width:2]

    mask = np.ones((image_height, image_width))
    mask[0:image_height:2, 0:image_width:2] = -1
    mask[1:image_height:2, 1:image_width:2] = -1
    green_values = mos_img[mask > 0]
    mean_value = green_values.mean()

    green_channel = img
    green_channel[mask < 0] = mean_value
    mos_img[:, :, 1] = green_channel

    return mos_img


def demosaicNN(img):
    '''Nearest neighbor demosaicing.
    
    Args:
        img: np.array of size NxM.
    '''
    mos_img = np.tile(img[:, :, np.newaxis], [1, 1, 3])
    image_height, image_width = img.shape

    for i in range(0,image_height,2):
        for j in range(0,image_width,2):
            mos_img[i][j][0] = img[i][j]
            if i+1 < image_height:
                mos_img[i+1][j][0] = img[i][j]
            if j+1 < image_width:
                mos_img[i][j+1][0] = img[i][j]
            if i+1 < image_height and j+1 < image_width:
                mos_img[i+1][j+1][0] = img[i][j]

    for i in range(1,image_height,2):
        for j in range(1,image_width,2):
            mos_img[i][j][2] = img[i][j]
            mos_img[i-1][j][2] = img[i][j]
            mos_img[i][j-1][2] = img[i][j]
            mos_img[i-1][j-1][2] = img[i][j]
            if i+1 < image_height:
                mos_img[i+1][j-1][2] = img[i][j]
                mos_img[i+1][j][2] = img[i][j]
            if j+1 < image_width:
                mos_img[i-1][j+1][2] = img[i][j]
                mos_img[i][j+1][2] = img[i][j]
            if i+1 < image_height and j+1 < image_width:
                mos_img[i+1][j+1][2] = img[i][j]

    for i in range(1,image_height,2):
        for j in range(0,image_width,2):
            mos_img[i][j][1] = img[i][j]
            mos_img[i-1][j][1] = img[i][j]
            if i+1 < image_height:
                mos_img[i+1][j][1] = img[i][j]

    for i in range(0,image_height,2):
        for j in range(1,image_width,2):
            mos_img[i][j][1] = img[i][j]
            mos_img[i][j-1][1] = img[i][j]
            if j+1 < image_width:
                mos_img[i][j+1][1] = img[i][j]
    
    return mos_img


def demosaicLinear(img):
    '''Nearest neighbor demosaicing.
    
    Args:
        img: np.array of size NxM.
    '''
    mos_img = np.tile(img[:, :, np.newaxis], [1, 1, 3])
    image_height, image_width = img.shape
    
    for i in range(0,image_height,2):
        for j in range(0,image_width,2):
            mos_img[i][j][0] = img[i][j]
            if i+2 < image_height and j+2 < image_width:
                mos_img[i+1][j+1][0] = (img[i][j]+img[i+2][j]+img[i][j+2]+img[i+2][j+2])/4
                mos_img[i+1][j][0] = (img[i][j]+img[i+2][j])/2
                mos_img[i][j+1][0] = (img[i][j]+img[i][j+2])/2
            elif i+1 < image_height and j+2 < image_width:
                mos_img[i+1][j][0] = img[i][j]
                mos_img[i][j+1][0] = (img[i][j]+img[i][j+2])/2
                mos_img[i+1][j+1][0] = (img[i][j]+img[i][j+2])/2
            elif i+2 < image_height and j+1 < image_width:
                mos_img[i][j+1][0] = img[i][j]
                mos_img[i+1][j][0] = (img[i][j]+img[i+2][j])/2
                mos_img[i+1][j+1][0] = (img[i][j]+img[i+2][j])/2
            elif i+1 < image_height and j+1 < image_width:
                mos_img[i+1][j][0] = img[i][j]
                mos_img[i][j+1][0] = img[i][j]
                mos_img[i+1][j+1][0] = img[i][j]
            elif i+1 < image_height:
                mos_img[i+1][j][0] = img[i][j]
            elif j+1 < image_width:
                mos_img[i][j+1][0] = img[i][j]

    mos_img[0][0][2] = img[1][1]
    
    for i in range(1,image_height,2):
        mos_img[i][0][2] = img[i][1]
        if i+2 < image_height:
            mos_img[i+1][0][2] = (img[i][1]+img[i+2][1])/2
        elif i+1 < image_height:
            mos_img[i+1][0][2] = img[i][1]

    for j in range(1,image_width,2):
        mos_img[0][j][2] = img[1][j]
        if j+2 < image_width:
            mos_img[0][j+1][2] = (img[1][j]+img[1][j+2])/2
        elif j+1 < image_width:
            mos_img[0][j+1][2] = img[1][j]

    for i in range(1,image_height,2):
        for j in range(1,image_width,2):
            mos_img[i][j][2] = img[i][j]
            if i+2 < image_height and j+2 < image_width:
                mos_img[i+1][j+1][2] = (img[i][j]+img[i+2][j]+img[i][j+2]+img[i+2][j+2])/4
                mos_img[i+1][j][2] = (img[i][j]+img[i+2][j])/2
                mos_img[i][j+1][2] = (img[i][j]+img[i][j+2])/2
            elif i+1 < image_height and j+2 < image_width:
                mos_img[i+1][j][2] = img[i][j]
                mos_img[i][j+1][2] = (img[i][j]+img[i][j+2])/2
                mos_img[i+1][j+1][2] = (img[i][j]+img[i][j+2])/2
            elif i+2 < image_height and j+1 < image_width:
                mos_img[i][j+1][2] = img[i][j]
                mos_img[i+1][j][2] = (img[i][j]+img[i+2][j])/2
                mos_img[i+1][j+1][2] = (img[i][j]+img[i+2][j])/2
            elif i+1 < image_height and j+1 < image_width:
                mos_img[i+1][j][2] = img[i][j]
                mos_img[i][j+1][2] = img[i][j]
                mos_img[i+1][j+1][2] = img[i][j]
            elif i+1 < image_height:
                mos_img[i+1][j][2] = img[i][j]
            elif j+1 < image_width:
                mos_img[i][j+1][2] = img[i][j]
    
    mask = np.ones((image_height, image_width))
    mask[0:image_height:2, 0:image_width:2] = -1
    mask[1:image_height:2, 1:image_width:2] = -1

    green_channel = img
    for i in range(image_height):
        for j in range(image_width):
            if mask[i][j] < 0:
                if i-1>-1 and j-1>-1 and i+1<image_height and j+1<image_width:
                    green_channel[i][j]=(green_channel[i-1][j]+green_channel[i][j-1]+green_channel[i+1][j]+green_channel[i][j+1])/4
                elif j-1>-1 and i+1<image_height and j+1<image_width:
                    green_channel[i][j]=(green_channel[i][j-1]+green_channel[i+1][j]+green_channel[i][j+1])/3
                elif i-1>-1 and i+1<image_height and j+1<image_width:
                    green_channel[i][j]=(green_channel[i-1][j]+green_channel[i+1][j]+green_channel[i][j+1])/3
                elif i-1>-1 and j-1>-1 and j+1<image_width:
                    green_channel[i][j]=(green_channel[i-1][j]+green_channel[i][j-1]+green_channel[i][j+1])/3
                elif i-1>-1 and j-1>-1 and i+1<image_height:
                    green_channel[i][j]=(green_channel[i-1][j]+green_channel[i][j-1]+green_channel[i+1][j])/3
                elif i+1<image_height and j+1<image_width:
                    green_channel[i][j]=(green_channel[i+1][j]+green_channel[i][j+1])/2
                elif i-1>-1 and j-1>-1:
                    green_channel[i][j]=(green_channel[i-1][j]+green_channel[i][j-1])/2
                elif j-1>-1 and i+1<image_height:
                    green_channel[i][j]=(green_channel[i][j-1]+green_channel[i+1][j])/2
                elif i-1>-1 and j+1<image_width:
                    green_channel[i][j]=(green_channel[i-1][j]+green_channel[i][j+1])/2                   
    mos_img[:, :, 1] = green_channel
                
    return mos_img


def demosaicAdagrad(img):
    '''Nearest neighbor demosaicing.
    
    Args:
        img: np.array of size NxM.
    '''
    mos_img = np.tile(img[:, :, np.newaxis], [1, 1, 3])
    image_height, image_width = img.shape
    
    for i in range(0,image_height,2):
        for j in range(0,image_width,2):
            mos_img[i][j][0] = img[i][j]
            if i+2 < image_height and j+2 < image_width:
                a = abs(img[i][j]-img[i+2][j+2])
                b = abs(img[i+2][j]-img[i][j+2])
                if a > b:
                    mos_img[i+1][j+1][0] = (img[i+2][j]+img[i][j+2])/2
                else:
                    mos_img[i+1][j+1][0] = (img[i][j]+img[i+2][j+2])/2
                mos_img[i+1][j][0] = (img[i][j]+img[i+2][j])/2
                mos_img[i][j+1][0] = (img[i][j]+img[i][j+2])/2
            elif i+1 < image_height and j+2 < image_width:
                mos_img[i+1][j][0] = img[i][j]
                mos_img[i][j+1][0] = (img[i][j]+img[i][j+2])/2
                mos_img[i+1][j+1][0] = (img[i][j]+img[i][j+2])/2
            elif i+2 < image_height and j+1 < image_width:
                mos_img[i][j+1][0] = img[i][j]
                mos_img[i+1][j][0] = (img[i][j]+img[i+2][j])/2
                mos_img[i+1][j+1][0] = (img[i][j]+img[i+2][j])/2
            elif i+1 < image_height and j+1 < image_width:
                mos_img[i+1][j][0] = img[i][j]
                mos_img[i][j+1][0] = img[i][j]
                mos_img[i+1][j+1][0] = img[i][j]
            elif i+1 < image_height:
                mos_img[i+1][j][0] = img[i][j]
            elif j+1 < image_width:
                mos_img[i][j+1][0] = img[i][j]

    mos_img[0][0][2] = img[1][1]
    
    for i in range(1,image_height,2):
        mos_img[i][0][2] = img[i][1]
        if i+2 < image_height:
            mos_img[i+1][0][2] = (img[i][1]+img[i+2][1])/2
        elif i+1 < image_height:
            mos_img[i+1][0][2] = img[i][1]

    for j in range(1,image_width,2):
        mos_img[0][j][2] = img[1][j]
        if j+2 < image_width:
            mos_img[0][j+1][2] = (img[1][j]+img[1][j+2])/2
        elif j+1 < image_width:
            mos_img[0][j+1][2] = img[1][j]

    for i in range(1,image_height,2):
        for j in range(1,image_width,2):
            mos_img[i][j][2] = img[i][j]
            if i+2 < image_height and j+2 < image_width:
                a = abs(img[i][j]-img[i+2][j+2])
                b = abs(img[i+2][j]-img[i][j+2])
                if a > b:
                    mos_img[i+1][j+1][2] = (img[i+2][j]+img[i][j+2])/2
                else:
                    mos_img[i+1][j+1][2] = (img[i][j]+img[i+2][j+2])/2
                mos_img[i+1][j][2] = (img[i][j]+img[i+2][j])/2
                mos_img[i][j+1][2] = (img[i][j]+img[i][j+2])/2
            elif i+1 < image_height and j+2 < image_width:
                mos_img[i+1][j][2] = img[i][j]
                mos_img[i][j+1][2] = (img[i][j]+img[i][j+2])/2
                mos_img[i+1][j+1][2] = (img[i][j]+img[i][j+2])/2
            elif i+2 < image_height and j+1 < image_width:
                mos_img[i][j+1][2] = img[i][j]
                mos_img[i+1][j][2] = (img[i][j]+img[i+2][j])/2
                mos_img[i+1][j+1][2] = (img[i][j]+img[i+2][j])/2
            elif i+1 < image_height and j+1 < image_width:
                mos_img[i+1][j][2] = img[i][j]
                mos_img[i][j+1][2] = img[i][j]
                mos_img[i+1][j+1][2] = img[i][j]
            elif i+1 < image_height:
                mos_img[i+1][j][2] = img[i][j]
            elif j+1 < image_width:
                mos_img[i][j+1][2] = img[i][j]
    
    mask = np.ones((image_height, image_width))
    mask[0:image_height:2, 0:image_width:2] = -1
    mask[1:image_height:2, 1:image_width:2] = -1

    green_channel = img
    for i in range(image_height):
        for j in range(image_width):
            if mask[i][j] < 0:
                if i-1>-1 and j-1>-1 and i+1<image_height and j+1<image_width:
                    a = abs(green_channel[i-1][j]-green_channel[i+1][j])
                    b = abs(green_channel[i][j-1]-green_channel[i][j+1])
                    if a > b:
                        green_channel[i][j] = (green_channel[i][j-1]+green_channel[i][j+1])/2
                    else:
                        green_channel[i][j] = (green_channel[i-1][j]+green_channel[i+1][j])/2
                elif j-1>-1 and i+1<image_height and j+1<image_width:
                    a = abs(green_channel[i][j-1]-green_channel[i+1][j])
                    b = abs(green_channel[i][j+1]-green_channel[i+1][j])
                    if a > b:
                        green_channel[i][j] = (green_channel[i][j+1]+green_channel[i+1][j])/2
                    else:
                        green_channel[i][j] = (green_channel[i][j-1]+green_channel[i+1][j])/2
                elif i-1>-1 and i+1<image_height and j+1<image_width:
                    a = abs(green_channel[i][j+1]-green_channel[i-1][j])
                    b = abs(green_channel[i][j+1]-green_channel[i+1][j])
                    if a > b:
                        green_channel[i][j] = (green_channel[i][j+1]+green_channel[i+1][j])/2
                    else:
                        green_channel[i][j] = (green_channel[i][j+1]+green_channel[i-1][j])/2
                elif i-1>-1 and j-1>-1 and j+1<image_width:
                    a = abs(green_channel[i][j-1]-green_channel[i-1][j])
                    b = abs(green_channel[i][j+1]-green_channel[i-1][j])
                    if a > b:
                        green_channel[i][j] = (green_channel[i][j+1]+green_channel[i-1][j])/2
                    else:
                        green_channel[i][j] = (green_channel[i][j-1]+green_channel[i-1][j])/2
                elif i-1>-1 and j-1>-1 and i+1<image_height:
                    a = abs(green_channel[i][j-1]-green_channel[i-1][j])
                    b = abs(green_channel[i][j-1]-green_channel[i+1][j])
                    if a > b:
                        green_channel[i][j] = (green_channel[i][j-1]+green_channel[i+1][j])/2
                    else:
                        green_channel[i][j] = (green_channel[i][j-1]+green_channel[i-1][j])/2
                elif i+1<image_height and j+1<image_width:
                    green_channel[i][j]=(green_channel[i+1][j]+green_channel[i][j+1])/2
                elif i-1>-1 and j-1>-1:
                    green_channel[i][j]=(green_channel[i-1][j]+green_channel[i][j-1])/2
                elif j-1>-1 and i+1<image_height:
                    green_channel[i][j]=(green_channel[i][j-1]+green_channel[i+1][j])/2
                elif i-1>-1 and j+1<image_width:
                    green_channel[i][j]=(green_channel[i-1][j]+green_channel[i][j+1])/2                   
    mos_img[:, :, 1] = green_channel

    return mos_img
