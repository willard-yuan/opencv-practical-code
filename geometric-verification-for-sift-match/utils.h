/*
 * FileName : utils.c
 * Author   : yongyuanstu@gmail.com
 * Version  : v1.0
 * Date     : 29 Aug 2015 08:31:41 PM CST
 * Brief    : 
 * 
 * Copyright (C) MICL,USTB
 */

#include<iostream>
#include<time.h>
#include<windows.h>
#include <vector>

#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/nonfree/nonfree.hpp"

#include "armadillo"

using namespace std;
//using namespace cv;

struct superluOpts{
	int             tolerance1;  // default: true
	int             tolerance2;    // default: false
	int             tolerance3; // default: 1.0
	unsigned int    minInliers;
	unsigned int    numRefinementIterations;
};

/*********************生成随机颜色*****************/
static cv::Scalar randomColor(cv::RNG& rng);

/********************画匹配点**********************/
void plotMatches(const cv::Mat &src, const cv::Mat &obj, vector<cv::Point2f> &srcPoints, vector<cv::Point2f> &dstPoints);

/**************使用OpenCV自带的寻找内点************/
void findInliers(vector<cv::KeyPoint> &qKeypoints, vector<cv::KeyPoint> &objKeypoints, vector<cv::DMatch> &matches, const string &imgfn, const string &objFileName);

/******************自己写的寻找内点*****************/
arma::mat centering(arma::mat &x);
arma::mat toAffinity(arma::mat &f);
arma::uvec geometricVerification(const arma::mat &frames1, const arma::mat &frames2, const arma::mat &matches, const superluOpts &opts);
