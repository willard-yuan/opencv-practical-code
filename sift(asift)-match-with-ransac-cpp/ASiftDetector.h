//
//  ASiftDetector.h
//  sift_asift_match
//
//  Created by willard on 8/18/15.
//  Copyright (c) 2015 wilard. All rights reserved.
//

#include <iostream>
#include <vector>
//#include <opencv2/core/core.hpp>
//#include <opencv2/features2d/features2d.hpp>

#include "opencv/cv.h"
#include "opencv/highgui.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/nonfree/nonfree.hpp"

using namespace cv;

class ASiftDetector
{
public:
    ASiftDetector();

    void detectAndCompute(const Mat& img, std::vector< KeyPoint >& keypoints, Mat& descriptors);

private:
    void affineSkew(double tilt, double phi, Mat& img, Mat& mask, Mat& Ai);
};
