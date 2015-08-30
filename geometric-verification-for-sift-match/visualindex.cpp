//
//  main.cpp
//  vlfeat-bow-with-reranking
//
//  Created by willard on 8/24/15.
//  Copyright (c) 2015 wilard. All rights reserved.
// http://stackoverflow.com/questions/28606011/vlfeat-kdtree-setup-and-query
// http://mcreader-indep.googlecode.com/svn/!svn/bc/13/MCRindep/jni/

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string.h>
#include <cstdio>
#include <math.h>
#include <string.h>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>

#include "file_io.h"
#include "feature_set.h"
#include "siftmatcher.h"
#include "utils.h"

extern "C" {
#include "vl/generic.h"
#include "vl/sift.h"
#include "vl/kdtree.h"
#include "vl/kmeans.h"
#include "vl/host.h"
}

using namespace std;

int main(int argc, const char * argv[]) {
    
    string filename = "/Users/willard/codes/cpp/opencv-computer-vision/vlfeat-bow-with-reranking/vlfeat-bow-with-reranking/imageNamesList.txt";
    vector<string> vocabularyFiles;
    string dirName;
    ifstream file(filename);
    std::string temp;
    while(std::getline(file, temp)) {
        cout << temp << endl;
        /*const size_t found = temp.find_last_of("/\\");
        dirName = temp.substr(0,found) + "/";
        vocabularyFiles.push_back(temp.substr(found+1));*/
        vocabularyFiles.push_back(temp);
    }
    
    string descriptor_Query_path = "/Users/willard/codes/cpp/opencv-computer-vision/cpp/videosearch/indexer/local_features/test_images/img1.siftb";
    string descriptor_Object_path = "/Users/willard/codes/cpp/opencv-computer-vision/cpp/videosearch/indexer/local_features/test_images/img2.siftb";
    
    // 测试siftmatcher效果
    /*string name = "/Users/willard/Pictures/img1.jpg";
    siftmatcher* matcher;
    Image image;
    matcher->initializeImage(name, image);*/
    
    uint sift_desc_dim = 128;
    uint sift_keypoint_dim = 4;
    
    FeatureSet* fQuerySet = readSIFTFile(descriptor_Query_path, sift_keypoint_dim, sift_desc_dim);
    FeatureSet* fObjectSet = readSIFTFile(descriptor_Object_path, sift_keypoint_dim, sift_desc_dim);
    
    // 获取所有描述子的数目
    long int numSIFT = 0;
    for(int i = 0; i < vocabularyFiles.size(); ++i){
        FeatureSet* fSet = readSIFTFile(vocabularyFiles.at(i), sift_keypoint_dim, sift_desc_dim);
        numSIFT += (*fSet).m_vDescriptors.size();
        cout << "第 " << i << "幅图像描述子数目： " << (*fSet).m_vDescriptors.size() << endl;
    }
    
    cout << "所有图像总的描述子数目： " << numSIFT << endl;
    
    //double* dataSIFT = (double *)vl_malloc(sizeof(double)*numSIFT*sift_desc_dim);
    double dataSIFT[numSIFT*sift_desc_dim]; // 使用数组
    static long int tmpNum = 0;
    for(int i = 0; i < vocabularyFiles.size(); ++i){
        FeatureSet* fSet = readSIFTFile(vocabularyFiles.at(i), sift_keypoint_dim, sift_desc_dim);
        for (int j = 0; j < (*fSet).m_vDescriptors.size(); j++)
            for (int k = 0; k < 128; ++k){
                //cout << k+128*j + tmpNum << endl;
                dataSIFT[k+128*j + tmpNum] = (*fSet).m_vDescriptors[j][k];
            }
        tmpNum = tmpNum + (*fSet).m_vDescriptors.size()*128;
        //cout << tmpNum << endl;
    }
    
    // 测试是否写入准确写入数组里面
    cout << "===============测试描述子写入是否准确======================" << endl;
    FeatureSet* fSet1 = readSIFTFile(vocabularyFiles.at(vocabularyFiles.size()-1), sift_keypoint_dim, sift_desc_dim);
    cout << "+++++++++++++++++未写入之前的描述子+++++++++++++++++++++++" << endl;
    for(int i = 0; i < 128; ++i)
        cout << (*fSet1).m_vDescriptors[321][i] << '\t';
    cout << endl;
    cout << "++++++++++++++++++写入之后的描述子+++++++++++++++++++++++" << endl;
    for(int i = 549120; i < 549248; ++i)
        cout << dataSIFT[i] << '\t';
    cout << endl;
    
    // 进行kmeans聚类
    VlKMeans *km = vl_kmeans_new(VL_TYPE_FLOAT, VlDistanceL2);  // 初始化
    vl_kmeans_set_algorithm(km, VlKMeansAlgorithm::VlKMeansANN); // 参数设置
    vl_kmeans_set_initialization(km, VlKMeansInitialization::VlKMeansPlusPlus);
    vl_kmeans_cluster(km, dataSIFT, 128, numSIFT, 1000); // 1000类
    const float *centers = (const float *)vl_kmeans_get_centers(km); // 获取类中心
    
    cout << "=========聚类中心=========" << endl;
    for(int i = 0; i < 128; ++i)
        cout << centers[i] << '\t';
    cout << endl;
    
    cout << "验证聚类中心数目： " << vl_kmeans_get_num_centers(km) << endl;
    
    // 创建kd树，做近似最近邻搜索，查询图像
    float* data = new float[128*(*fQuerySet).m_vDescriptors.size()];
    for (int i = 0; i < (*fQuerySet).m_vDescriptors.size(); i++)
        for (int j = 0; j < 128; j++)
            data[j+128*i] = (*fQuerySet).m_vDescriptors[i][j];
    
    VlKDForest* forest = vl_kdforest_new(VL_TYPE_FLOAT, 128, 8, VlDistanceL2);  //创建kd树对象，128维，8棵树
    forest->thresholdingMethod = VL_KDTREE_MEDIAN;
    vl_kdforest_build(forest, 1000, centers);
    
    // 目标图像
    float* fObjectData = new float[128*(*fObjectSet).m_vDescriptors.size()];
    for (int i = 0; i < (*fObjectSet).m_vDescriptors.size(); i++)
        for (int j = 0; j < 128; j++)
            fObjectData[j+128*i] = (*fObjectSet).m_vDescriptors[i][j];
    
    // 做最近查找 Searcher object
    VlKDForestSearcher* searcher = vl_kdforest_new_searcher(forest);
    VlKDForestNeighbor neighbours[2];
    /* Query the first ten points for now */
    vector<pair<int, int>> matched_index;
    vector<Point2f> queryLoc, objectLoc;
    for(int i=0; i < (*fObjectSet).m_vDescriptors.size(); i++){
        vl_kdforestsearcher_query(searcher, neighbours, 2, fObjectData + 128*i);
        //auto nvisited = vl_kdforestsearcher_query(searcher, neighbours, 2, fObjectData + 128*i);
        cout << "最近邻：" << neighbours[0].index << " 距离：" << neighbours[0].distance << " " "次近邻：" << neighbours[1].index << " 距离：" << neighbours[1].distance << endl;
        if(neighbours[0].distance < 0.8*neighbours[1].distance){
            matched_index.push_back(pair<int, int>(neighbours[0].index, i));
            Point2f tmp1;
            tmp1.x = (*fQuerySet).m_vFrames[neighbours[0].index][0];
            tmp1.y = (*fQuerySet).m_vFrames[neighbours[0].index][1];
            queryLoc.push_back(tmp1);
            Point2f tmp2;
            tmp2.x = (*fObjectSet).m_vFrames[i][0];
            tmp2.y = (*fObjectSet).m_vFrames[i][1];
            objectLoc.push_back(tmp2);
        }
    }
    
    //显示匹配的点对
    string imgfn = "/Users/willard/codes/cpp/opencv-computer-vision/cpp/videosearch/indexer/local_features/test_images/img1.jpg";
    string objFileName = "/Users/willard/codes/cpp/opencv-computer-vision/cpp/videosearch/indexer/local_features/test_images/img2.jpg";
    drawMatch(imgfn, objFileName, queryLoc, objectLoc);
    
    // 计算homography矩阵
    Mat mask;
    vector<Point2f> queryInliers;
    vector<Point2f> sceneInliers;
    //Mat H = findFundamentalMat(queryLoc, objectLoc, mask, CV_FM_RANSAC);
    Mat H = findHomography(queryLoc, objectLoc, CV_RANSAC, 10, mask);
    int inliers_cnt = 0, outliers_cnt = 0;
    for (int j = 0; j < mask.rows; j++){
        if (mask.at<uchar>(j) == 1){
            queryInliers.push_back(queryLoc[j]);
            sceneInliers.push_back(objectLoc[j]);
            inliers_cnt++;
        }else {
            outliers_cnt++;
        }
    }
    //显示剔除误配点对后的匹配点对
    drawMatch(imgfn, objFileName, queryInliers, sceneInliers);
    
    
    fQuerySet->print();
    
    // Clean up
    if (fQuerySet) {
        delete fQuerySet;
    }
    
    return EXIT_SUCCESS;
}
