# -*- coding: utf-8 -*-
import cv2
from pylab import *

img = cv2.imread('F:/dropbox/Dropbox/translation/pcv-notebook/data/alcatraz1.jpg')
img_RGB= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#img_RGB= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

siftDetector=cv2.SIFT()
kp = siftDetector.detect(img_RGB,None)
kp,des = siftDetector.compute(img_RGB,kp)
# 关键点列表
print type(kp),len(kp)
# des是一个大小为关键点数目*128的数组
print type(des),des.shape
im=cv2.drawKeypoints(img_RGB,kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#用pylab的imshow()显示
figure()
gray()
subplot(111)
axis('off')
imshow(im)

#用Opencv管理窗口显示
#cv2.imshow('Sift detect',im);  
#cv2.waitKey(0)  
#cv2.destroyAllWindows()

show()



