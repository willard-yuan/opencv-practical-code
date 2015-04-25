# USAGE
# python driver.py

# import the necessary packages
from rootsift import RootSIFT
import cv2
import os
from numpy import *
import pylab

# load the image we are going to extract descriptors from and convert
# it to grayscale
image = cv2.imread("all_souls_000035.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect Difference of Gaussian keypoints in the image
detector = cv2.FeatureDetector_create("SIFT")
kps = detector.detect(image)

# extract normal SIFT descriptors
extractor = cv2.DescriptorExtractor_create("SIFT")
(kps_sift, descs_sift) = extractor.compute(image, kps)
print "SIFT: kps=%d, descriptors=%s " % (len(kps_sift), descs_sift.shape)

# extract RootSIFT descriptors
rs = RootSIFT()
(kps_rootsift, descs_rootsift) = rs.compute(image, kps)
print "RootSIFT: kps=%d, descriptors=%s " % (len(kps_rootsift), descs_rootsift.shape)

pylab.figure()
rgbImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

img_SIFT = cv2.drawKeypoints(rgbImage,kps_sift,None,(255,0,255),4)
pylab.gray()
pylab.subplot(2,1,1)
pylab.imshow(img_SIFT)
pylab.axis('off')

img_rootsift = cv2.drawKeypoints(rgbImage,kps_rootsift,None,(255,0,255),4)
pylab.gray()
pylab.subplot(2,1,2)
pylab.imshow(img_rootsift)
pylab.gray()
pylab.axis('off')

pylab.show()
