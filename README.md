##OpenCV小项目

这是一个个人在使用OpenCV过程中写的一些小项目，以及一些非常有用的OpenCV代码，有些代码是对某论文中的部分实现。

##opencv-rootsift-py
用python和OpenCV写的一个rootsift实现，其中RootSIFT部分的代码参照[Implementing RootSIFT in Python and OpenCV](https://www.pyimagesearch.com/2015/04/13/implementing-rootsift-in-python-and-opencv/)这篇文章所写，通过这个你可以了解**Three things everyone should know to improve object retrieval**这篇文章中RootSIFT是怎么实现的。

##sift(asift)-match-with-ransac-cpp
用C++和OpenCV写的一个图像匹配实现，里面包含了采用1NN匹配可视化、1NN匹配后经RANSAC剔除错配点可视化、1NN/2NN<0.8匹配可视化、1NN/2NN<0.8经
RANSAC剔除错配点可视化四个过程，其中1NN/2NN<0.8匹配过程是Lowe的Raw feature match，具体可以阅读Lowe的**Distinctive image features from scale-invariant keypoints**这篇文章。这个对图像检索重排非常有用。另外里面还有用OpenCV写的ASIFT，这部分来源于[OPENCV ASIFT C++ IMPLEMENTATION](http://www.mattsheckells.com/opencv-asift-c-implementation/)，ASIFT还可以到官网页面下载，ASIFT提取的关键点
比SIFT要多得多，速度非常慢，不推荐在对要求实时性的应用中使用。

更多详细的分析可以阅读博文[SIFT(ASIFT) Matching with RANSAC](http://yongyuan.name/blog/SIFT(ASIFT)-Matching-with-RANSAC.html)。

##有用链接
[OpenCV3.0文档](http://docs.opencv.org/master/index.html#gsc.tab=0)
