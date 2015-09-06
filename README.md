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

```c++
	// 测试sparse
	unsigned int centersNum  = 10;
	vector<unsigned int> descrNums;
	descrNums.push_back(8);
	descrNums.push_back(12);
	//unsigned int T[] = {1, 2, 1, 3, 2, 5, 4, 3, 10, 5; 4, 2, 6, 5, 2, 5, 4, 6, 2, 4};
	unsigned int T[] = {1, 2, 1, 3, 2, 5, 4, 3, 10, 5, 4, 2, 6, 5, 2, 5, 4, 6, 2, 4};
	sp_mat Hist(descrNums.size(), centersNum);

	static long int count = 0;
	for (int i = 0; i < descrNums.size(); i++){
		unsigned int* desrcElementsTmp = new unsigned int[descrNums[i]];
		memcpy(desrcElementsTmp, T + count, descrNums[i] * sizeof(T[0]));
		//cout << desrcElementsTmp[0] << '\t' << desrcElementsTmp[1] << '\t' << desrcElementsTmp[2] << '\t' << desrcElementsTmp[3] << '\t' << desrcElementsTmp[4] << '\t' <<endl;
		//cout << desrcElementsTmp[5] << '\t' << desrcElementsTmp[6] << '\t' << desrcElementsTmp[7] << '\t' << desrcElementsTmp[8] << '\t' << desrcElementsTmp[9] << '\t' << desrcElementsTmp[10] << '\t' <<endl;
		//cout << endl;

		sp_mat X(1, centersNum);
		X.zeros();
		for (int j = 0; j < descrNums[i]; j++){
			X(0, desrcElementsTmp[j]-1) += 1;
		}
		X.print("X:");
		X = X/norm(X, 2);
		Hist.row(i) = X;
		count = count + descrNums[i];
		delete desrcElementsTmp;
	}
	//Hist.print("Hist:");
```
