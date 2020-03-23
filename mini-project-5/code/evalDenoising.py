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
from tensorflow import keras

def imread(path):
    img = plt.imread(path).astype(float)
    #Remove alpha channel if it exists
    if img.ndim > 2 and img.shape[2] == 4:
        img = img[:, :, 0:3]
    #Puts images values in range [0,1]
    if img.max() > 1.0:
        img /= 255.0
    return img

# Load clean and noisy image
im = imread('../data/denoising/saturn.png')
noise1 = imread('../data/denoising/saturn-noisy.png')
#im = imread('../data/denoising/lena.png')
#noise1 = imread('../data/denoising/lena-noisy.png')

error1 = ((im - noise1)**2).sum()
print('Noisy image SE: %8.2f'%(error1))

plt.figure(1)

plt.subplot(121)
plt.imshow(im, cmap='gray')
plt.title('Input')

plt.subplot(122)
plt.imshow(noise1, cmap='gray')
plt.title('Noisy image SE %8.2f'%(error1))

plt.show(block=False)

imShape=im.shape
inputShape=(imShape[0],imShape[1],1)

################################################################################
# Denoising algorithm (Deep Image Prior)
################################################################################
###

model = keras.Sequential([
    keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=(2, 2), padding='same',\
                        data_format='channels_last',\
                        input_shape=inputShape,activation='relu'),
    keras.layers.BatchNormalization(),\
    keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(2, 2),padding='same',\
                        data_format='channels_last',\
                        activation='relu'),
    keras.layers.BatchNormalization(),\
    keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(2, 2),padding='same',\
                        data_format='channels_last',\
                        activation='relu'),
    keras.layers.BatchNormalization(),\
    keras.layers.UpSampling2D(data_format='channels_last'),\
    keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding='same',\
                        data_format='channels_last',\
                        activation='relu'),
    keras.layers.BatchNormalization(),\
    keras.layers.UpSampling2D(data_format='channels_last'),\
    keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='same',\
                        data_format='channels_last',\
                        activation='relu'),
    keras.layers.BatchNormalization(),\
    keras.layers.UpSampling2D(data_format='channels_last'),\
    keras.layers.Conv2D(filters=16, kernel_size=(3,3), padding='same',\
                        data_format='channels_last',\
                        activation='relu'),
    keras.layers.BatchNormalization(),\
    keras.layers.Conv2D(filters=1, kernel_size=(3,3), padding='same',\
                        data_format='channels_last',\
                        activation='sigmoid')
])
Adam = keras.optimizers.Adam(lr=0.01)
model.compile(optimizer=Adam,
              loss='mean_squared_error',
              metrics=['accuracy'])
noise1_in = np.reshape(noise1,(1,)+inputShape)
gaussian = np.random.normal(size=imShape)
im_in = noise1+gaussian
im_in = np.reshape(im_in,(1,)+inputShape)
model.fit(im_in, noise1_in, epochs=5000)

###

out_img = model.predict(im_in)
out_img = np.reshape(out_img,imShape)

error1 = ((im - noise1)**2).sum()
error2 = ((im - out_img)**2).sum()

plt.figure(3)
plt.axis('off')

plt.subplot(131)
plt.imshow(im, cmap='gray')
plt.title('Input')

plt.subplot(132)
plt.imshow(noise1, cmap='gray')
plt.title('SE %8.2f'%(error1))

plt.subplot(133)
plt.imshow(out_img, cmap='gray')
plt.title('SE %8.2f'%(error2))

plt.show()

