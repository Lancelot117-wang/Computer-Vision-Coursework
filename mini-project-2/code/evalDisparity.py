# This code is part of:
#
#   CMPSCI 670: Computer Vision
#   University of Massachusetts, Amherst
#   Instructor: Subhransu Maji

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cl
from utils import imread
from depthFromStereo import depthFromStereo
from visualizeInformation import visualizeInformation
import os

read_path = "../data/disparity/"
#im_name1 = "tsukuba_im1.jpg" 
#im_name2 = "tsukuba_im5.jpg"
im_name1 = "poster_im2.jpg" 
im_name2 = "poster_im6.jpg"
#Read test images
img1 = imread(os.path.join(read_path, im_name1))
img2 = imread(os.path.join(read_path, im_name2))

visualizeInformation(img1, 10)

'''
#Compute depth
depth = depthFromStereo(img1, img2, 7)

#Show result
plt.imshow(depth)
plt.imshow(depth, cmap='tab20',norm=cl.Normalize(-0.1,0.1))
plt.colorbar()
plt.show()
save_path = "../output/disparity/"
#save_file = "tsukuba.png"
save_file = "poster.png"
if not os.path.isdir(save_path):
	os.makedirs(save_path)
plt.imsave(os.path.join(save_path, save_file), depth, vmin=-0.1, vmax=0.1, cmap='tab20')
'''

