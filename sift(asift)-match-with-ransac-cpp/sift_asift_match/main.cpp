//
//  main.cpp
//  sift_asift_match
//
//  Created by willard on 8/18/15.
//  Copyright (c) 2015 wilard. All rights reserved.
//

#include "ASiftDetector.h"
#include "utils.h"

// 提取sift特征
Mat ExtractSIFTFeature(const string &imgfn, vector<KeyPoint> &points)  {
    auto img = imread(imgfn, true);    //imgfn: image file name
    SiftFeatureDetector detector;
    vector<KeyPoint> keypoints;
    detector.detect(img, keypoints);
    SiftDescriptorExtractor extractor;
    Mat descriptors;
    if (!keypoints.size()) {
        return Mat();
    }
    // extract sift feature
    extractor.compute(img, keypoints, descriptors);
    //points.resize(keypoints.size());
    for (int i = 0; i < keypoints.size(); i++){
        points.push_back(keypoints[i]);
    }
    return descriptors;
}

int main(int argc, const char * argv[]) {
    // insert code here...
    string imgfn = "/Users/willard/Pictures/sample_pictures/DSC_2625_resize.JPG";
    Mat queryImage, queryBlackImg, qDescriptor;
    queryImage = imread(imgfn);
    vector<KeyPoint> qKeypoints;
    qDescriptor = ExtractSIFTFeature(imgfn, qKeypoints);
    /*drawKeypoints(queryImage, qKeypoints, queryBlackImg);
    imshow("Sift", queryBlackImg);
    cvWaitKey(0);*/
    
    string objFileName = "/Users/willard/Pictures/sample_pictures/DSC_2624_resize.JPG";
    Mat objectImage, objBlackImg, objDesriptor;
    objectImage = imread(objFileName);
    vector<KeyPoint> objKeypoints;
    objDesriptor = ExtractSIFTFeature(objFileName, objKeypoints);
    /*drawKeypoints(objectImage, objKeypoints, objBlackImg);
    imshow("Sift", objBlackImg);
    cvWaitKey(0);*/
    
    //直接找1近邻，SIFT找匹配点
    FlannBasedMatcher matcher;
    vector< DMatch > matches1;
    matcher.match(qDescriptor, objDesriptor, matches1);
    findInliers(qKeypoints, objKeypoints, matches1, imgfn, objFileName); // 通过1近邻匹配的点对
    
    // 使用Lowe的raw feature方法匹配
    vector<Point2f> qeK, obK;
    vector<vector<DMatch>> matches;
    vector<DMatch> good_matches2;
    matcher.knnMatch(qDescriptor, objDesriptor, matches, 2);
    for (size_t i = 0; i < matches.size(); i++){
        if (matches[i][0].distance < 0.8*matches[i][1].distance){
            good_matches2.push_back(matches[i][0]);
            qeK.push_back(qKeypoints[matches[i][0].queryIdx].pt);
            obK.push_back(objKeypoints[matches[i][1].trainIdx].pt);
        }
    }
    findInliers(qKeypoints, objKeypoints, good_matches2, imgfn, objFileName); // 通过Lowe匹配的点对

    
    /*ASiftDetector asiftDetector;
    vector<KeyPoint> asiftKeypoints_query;
    Mat asiftDescriptors_query;
    asiftDetector.detectAndCompute(queryImage, asiftKeypoints_query, asiftDescriptors_query);
    vector<KeyPoint> asiftKeypoints_object;
    Mat asiftDescriptors_object;
    asiftDetector.detectAndCompute(objectImage, asiftKeypoints_object, asiftDescriptors_object);*/
    
    /*//Matching descriptor vectors using FLANN matcher, ASIFT找匹配点
    std::vector< DMatch > asiftMatches;
    matcher.match(asiftDescriptors_query, asiftDescriptors_object, asiftMatches);
    findInliers(asiftKeypoints_query, asiftKeypoints_object, asiftMatches, imgfn, objFileName);*/
    
    // 使用内置函数画匹配点对
    /*Mat img_matches;
     drawMatches( queryImage, qKeypoints, objectImage, objKeypoints,
     matches, img_matches, Scalar::all(-1), Scalar::all(-1),
     vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
     imshow("Good Matches & Object detection", img_matches);
     waitKey(0);*/
    
    return 0;
}
