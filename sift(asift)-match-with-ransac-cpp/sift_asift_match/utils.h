//
//  utils.h
//  sift_asift_match
//
//  Created by willard on 8/21/15.
//  Copyright (c) 2015 wilard. All rights reserved.
//

#ifndef __sift_asift_match__utils__
#define __sift_asift_match__utils__

#include <stdio.h>
#include <vector>
#include <iostream>

#include "opencv/cv.h"
#include "opencv/highgui.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/nonfree/nonfree.hpp"

using namespace std;
using namespace cv;

void drawMatch(const string &src, const string &obj, vector<Point2f> &srcPoints, vector<Point2f> &dstPoints);
void findInliers(vector<KeyPoint> &qKeypoints, vector<KeyPoint> &objKeypoints, vector<DMatch> &matches, const string &imgfn, const string &objFileName);

#endif /* defined(__sift_asift_match__utils__) */
