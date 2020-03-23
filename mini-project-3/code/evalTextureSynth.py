import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from synthesis import synthRandomPatch
from synthesis import synthQuilting
from synthesis import synthEfrosLeung

# Load images
#img = io.imread('../data/texture/D20.png')
img = io.imread('../data/texture/Texture2.bmp')
#img = io.imread('../data/texture/english.jpg')

image=np.zeros((img.shape[0],img.shape[1]),dtype=np.int)
image[:,:]=img[:,:,0]
'''
# Random patches
tileSize = 40 # specify block sizes
numTiles = 5
outSize = numTiles * tileSize # calculate output image size
# implement the following, save the random-patch output and record run-times
im_patch = synthRandomPatch(image, tileSize, numTiles, outSize)
'''
# Non-parametric Texture Synthesis using Efros & Leung algorithm  
winsize = 15 # specify window size (5, 7, 11, 15)
outSize = 70 # specify size of the output image to be synthesized (square for simplicity)
# implement the following, save the synthesized image and record the run-times
im_synth = synthEfrosLeung(image, winsize, outSize)
'''
# Random patches
tileSize = 30 # specify block sizes
numTiles = 5
outSize = numTiles * tileSize # calculate output image size
# implement the following, save the random-patch output and record run-times
im_patch = synthQuilting(image, tileSize, numTiles, outSize)
'''
