#include "utils.h"

using namespace arma;

int main(int argc, char** argv){
	// 参数
	superluOpts opts;
	opts.tolerance1 = 20 ;
	opts.tolerance2 = 15 ;
	opts.tolerance3 = 8 ;
	opts.minInliers = 6 ;
	opts.numRefinementIterations = 8 ; //需要更改

	//载入特征点
	mat frames1, frames2, matches;
	frames1.load("C:\\Users\\Administrator\\Desktop\\geometricVerification\\frames11.txt");
	frames2.load("C:\\Users\\Administrator\\Desktop\\geometricVerification\\frames22.txt");
	matches.load("C:\\Users\\Administrator\\Desktop\\geometricVerification\\matches_2nn1.txt");

	arma::uvec inliers_final;
	//arma::mat H_final;

	inliers_final = geometricVerification(frames1, frames2, matches, opts);

	string src = "C:\\Users\\Administrator\\Desktop\\img1.jpg";
	string obj = "C:\\Users\\Administrator\\Desktop\\img2.jpg";

	cv::Mat srcColorImage = cv::imread(src);
    cv::Mat dstColorImage = cv::imread(obj);

	vector<cv::Point2f> srcPoints, dstPoints;
	mat matches_geo = matches.cols(inliers_final);
	//cout << matches_geo.n_rows << "+++++" <<matches_geo.n_cols << endl;
	for (unsigned int i = 0; i < matches_geo.n_cols; ++i){
		cv::Point2f pt1, pt2;
		//cout << matches_geo.at(0, i) << " " << matches_geo.at(1, i) << endl;
		pt1.x = frames1.at(0, matches_geo.at(0, i) - 1);
		pt1.y = frames1.at(1, matches_geo.at(0, i) - 1);
		pt2.x = frames2.at(0, matches_geo.at(1, i) - 1);
		pt2.y = frames2.at(1, matches_geo.at(1, i) - 1);
		srcPoints.push_back(pt1);
		dstPoints.push_back(pt2);
	}

	plotMatches(srcColorImage, dstColorImage, srcPoints, dstPoints);

	system("pause");
	return 0;
}
