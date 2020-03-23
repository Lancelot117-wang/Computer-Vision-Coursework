import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import imread
from utils import showMatches
from detectBlobs import detectBlobs
from compute_sift import compute_sift
from computeMatches import computeMatches
from ransac import ransac
from mergeImages import mergeImages
from skimage.feature import blob_log

#Image directory
dataDir = os.path.join('..', 'data', 'stitching')

#Read input images
testExamples = ['hill', 'field', 'ledge', 'pier', 'river', 'roofs', 'building', 'uttower']
exampleIndex = 5
imageName1 = '{}1_r.jpg'.format(testExamples[exampleIndex])
imageName2 = '{}2_r.jpg'.format(testExamples[exampleIndex])

#im1 = imread(os.path.join(dataDir, imageName1))
#im2 = imread(os.path.join(dataDir, imageName2))

im1 = cv2.imread(os.path.join(dataDir, imageName1))
im2 = cv2.imread(os.path.join(dataDir, imageName2))

im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp1 = sift.detect(im1, None)
kp2 = sift.detect(im2, None)

blobs1 = [[kp1[i].pt[0], kp1[i].pt[1], kp1[i].size] for i in xrange(len(kp1))]
blobs1 = np.array(blobs1)
blobs2 = [[kp2[i].pt[0], kp2[i].pt[1], kp2[i].size] for i in xrange(len(kp2))]
blobs2 = np.array(blobs2)

print(blobs1.shape)
print(blobs2.shape)

sift1 = compute_sift(im1, blobs1)
sift2 = compute_sift(im2, blobs2)

#Detect keypoints
#blobs1 = blob_log(im1)
#blobs2 = blob_log(im2)

#print('Start computing SIFT')
#Compute SIFT features
#sift1 = compute_sift(im1, blobs1[:, 0:4])
#sift2 = compute_sift(im2, blobs2[:, 0:4])
#print('SIFT computing done')

#Find the matching between features
matches = computeMatches(sift1, sift2)
showMatches(im1, im2, blobs1, blobs2, matches)

#Ransac to find correct matches and compute transformation
inliers, transf = ransac(matches, blobs1, blobs2)

goodMatches = np.ones_like(matches)
goodMatches[inliers] = matches[inliers]

showMatches(im1, im2, blobs1, blobs2, goodMatches)

#Merge two images and display the output
I1 = cv2.imread(os.path.join(dataDir, imageName1))
I2 = cv2.imread(os.path.join(dataDir, imageName2))
stitchIm = mergeImages(I1, I2, transf)
plt.figure()
plt.imshow(stitchIm)
plt.title('stitched image: {}'.format(testExamples[exampleIndex]))
plt.show()
