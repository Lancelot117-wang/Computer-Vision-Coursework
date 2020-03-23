# This code is part of:
#
#   CMPSCI 670: Computer Vision
#   University of Massachusetts, Amherst
#   Instructor: Subhransu Maji
#

import os
import time
import numpy as np
import matplotlib.pyplot as plt 
import sys
from utils import imread
from utils import gaussian
from imFilter import gFilter
from imFilter import mFilter
from imFilter import nlmFilter

im = imread('../data/denoising/saturn.png')
noise1 = imread('../data/denoising/saturn-noise2sp.png')
noise2 = imread('../data/denoising/saturn-noise1sp.png')

error1 = ((im - noise1)**2).sum()
error2 = ((im - noise2)**2).sum()

print 'Input, Errors: {:.2f} {:.2f}'.format(error1, error2)

plt.figure(1)

plt.subplot(131)
plt.imshow(im)
plt.title('Input')

plt.subplot(132)
plt.imshow(noise1)
plt.title('SE {:.2f}'.format(error1))

plt.subplot(133)
plt.imshow(noise2)
plt.title('SE {:.2f}'.format(error2))

plt.show()
'''
# Denoising algorithm (Gaussian filtering)
k1 = gaussian(3,0.5)
r1 = gFilter(noise1, k1)

k2 = gaussian(3,1)
r2 = gFilter(noise1, k2)

k3 = gaussian(3,2)
r3 = gFilter(noise1, k3)

error0 = ((im - noise1)**2).sum()
error1 = ((im - r1)**2).sum()
error2 = ((im - r2)**2).sum()
error3 = ((im - r3)**2).sum()

plt.figure(1)

plt.subplot(221)
plt.imshow(noise1)
plt.title('SE {:.2f}'.format(error0))

plt.subplot(222)
plt.imshow(r1)
plt.title('SE {:.2f}'.format(error1))

plt.subplot(223)
plt.imshow(r2)
plt.title('SE {:.2f}'.format(error2))

plt.subplot(224)
plt.imshow(r3)
plt.title('SE {:.2f}'.format(error3))

plt.show()
'''
'''
# Denoising algorithm (Median filtering)
r1 = mFilter(noise1, 3)

r2 = mFilter(noise1, 5)

r3 = mFilter(noise1, 7)

error0 = ((im - noise1)**2).sum()
error1 = ((im - r1)**2).sum()
error2 = ((im - r2)**2).sum()
error3 = ((im - r3)**2).sum()

plt.figure(1)

plt.subplot(221)
plt.imshow(noise1)
plt.title('SE {:.2f}'.format(error0))

plt.subplot(222)
plt.imshow(r1)
plt.title('SE {:.2f}'.format(error1))

plt.subplot(223)
plt.imshow(r2)
plt.title('SE {:.2f}'.format(error2))

plt.subplot(224)
plt.imshow(r3)
plt.title('SE {:.2f}'.format(error3))

plt.show()
'''

# Denoising algorithm (Non-local means)
r1 = nlmFilter(noise1, 5, 11, 1)

r2 = nlmFilter(noise1, 5, 15, 1)

r3 = nlmFilter(noise1, 5, 19, 1)

error0 = ((im - noise1)**2).sum()
error1 = ((im - r1)**2).sum()
error2 = ((im - r2)**2).sum()
error3 = ((im - r3)**2).sum()

plt.figure(1)

plt.subplot(221)
plt.imshow(noise1)
plt.title('SE {:.2f}'.format(error0))

plt.subplot(222)
plt.imshow(r1)
plt.title('SE {:.2f}'.format(error1))

plt.subplot(223)
plt.imshow(r2)
plt.title('SE {:.2f}'.format(error2))

plt.subplot(224)
plt.imshow(r3)
plt.title('SE {:.2f}'.format(error3))

plt.show()

