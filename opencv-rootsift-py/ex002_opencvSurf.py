# -*- coding: utf-8 -*-
import cv2
from pylab import *

img = cv2.imread('F:/dropbox/Dropbox/translation/pcv-notebook/data/alcatraz1.jpg')

#OpenCV读取的图像默认通道为BRG
img_RGB= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

surfObj=cv2.SURF()
surfObj.hessianThreshold = 20000

#上面等价于
#surfObj=cv2.SURF(20000)

kp = surfObj.detect(img_RGB,None)
print "kp is a %s, the length of kp is %s"%(type(kp),len(kp))
kp,des = surfObj.compute(img_RGB,kp)

#可以将上面检测和计算合并在一起完成
#kp, des = surf.detectAndCompute(img_RGB,None)

img2 = cv2.drawKeypoints(img_RGB,kp,None,(255,0,255),4)

imshow(img2)
show()