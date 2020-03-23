import os
import numpy as np
import matplotlib.pyplot as plt
from utils import imread
from detectBlobs import detectBlobs
from drawBlobs import drawBlobs

# Evaluation code for blob detection
# Your goal is to implement scale space blob detection using LoG  
#
# This code is part of:
#
#   CMPSCI 670: Computer Vision
#   University of Massachusetts, Amherst
#   Instructor: Subhransu Maji

#imageName = 'butterfly.jpg'
#imageName = 'einstein.jpg'
#imageName = 'fishes.jpg'
imageName = 'sunflowers.jpg'
numBlobsToDraw = 1000
imName = imageName.split('.')[0]

datadir = os.path.join('..', 'data', 'blobs')
im = imread(os.path.join(datadir, imageName))

blobs = detectBlobs(im,15,1.2)  # dummy placeholder

drawBlobs(im, blobs, numBlobsToDraw)

