#include<iostream>
#include<time.h>
#include<windows.h>
#include "armadillo"
using namespace arma;
using namespace std;

struct superlu_opts{
	int             tolerance1;  // default: true
	int             tolerance2;    // default: false
	int             tolerance3; // default: 1.0
	int             minInliers;
	int             numRefinementIterations;
} opts;

/*-------------------------------------------------------------------
* Returns the (stacking of the) 2x2 matrix A that maps the unit circle
* into the ellipses satisfying the equation x' inv(S) x = 1. Here S
* is a stacked covariance matrix, with elements S11, S12 and S22.
*
mat mapFromS(mat &S){
	tmp = sqrt(S(3,:)) + eps ;
	A(1,1) = sqrt(S(1,:).*S(3,:) - S(2,:).^2) ./ tmp ;
	A(2,1) = zeros(1,length(tmp));
	A(1,2) = S(2,:) ./ tmp ;
	A(2,2) = tmp ;
	return A;
}*/

/* --------------------------------------------------------------------*/
mat centering(mat &x){
	mat tmp = -mean(x.rows(0, 1), 1);
	mat tmp2(2,2);
	tmp2.eye();
	mat tmp1 = join_horiz(tmp2, tmp);
	mat v;
	v << 0 << 0 << 1 << endr;
	mat T = join_vert(tmp1, v);
	//T.print("T =");
	mat xm = T * x;
	//xm.print("xm =");
	
	//at least one pixel apart to avoid numerical problems
	//xm.print("xm =");
	double std11 = stddev(xm.row(0));
	//cout << "std11:" << std11 << endl;
	double std22 = stddev(xm.row(1));
	//cout << "std22:" << std22 << endl;

	double std1 = max(std11, 1.0);
	double std2 = max(std22, 1.0);
	
	mat S;
	S << 1/std1 << 0 << 0 << endr
	  << 0 << 1/std2 << 0 << endr
	  << 0 << 0 << 1 << endr;
	mat C = S * T ;
	//C.print("C =");
	return C;
}

/* --------------------------------------------------------------------*/
mat toAffinity(mat &f){
	mat A;
	mat v;
	v << 0 << 0 << 1 << endr;
	int flag = f.n_rows;
	switch(flag){
		case 6:{ // oriented ellipses
			mat T = f.rows(0, 1);
			mat tmp = join_horiz(f.rows(2, 3), f.rows(4, 5));
			mat tmp1 = join_horiz(tmp, T);
			A = join_vert(tmp1, v);
			break;}
		/*case 3:{    // discs
			mat T = f.rows(0, 1);
			mat s = f.row(2);
			int th = 0 ;
			A = [s*[cos(th) -sin(th) ; sin(th) cos(th)], T ; 0 0 1] ;
			   }
		case 4:{   // oriented discs
			mat T = f.rows(0, 1);
			mat s = f.row(2);
			th = f(4) ;
			A = [s*[cos(th) -sin(th) ; sin(th) cos(th)], T ; 0 0 1] ;
			   }
		case 5:{ // ellipses
			mat T = f.rows(0, 1);
			A = [mapFromS(f(3:5)), T ; 0 0 1] ;
			   }*/
		default:
			cout <<"出错啦！"<<endl;
			break;
	}
	return A;
}

int main(int argc, char** argv){
	// 参数
	opts.tolerance1 = 20 ;
	opts.tolerance2 = 15 ;
	opts.tolerance3 = 8 ;
	opts.minInliers = 6 ;
	opts.numRefinementIterations = 8 ; //需要更改

	//载入特征点
	mat frames1, frames2, matches;
	frames1.load("C:\\Users\\Administrator\\Desktop\\geometricVerification\\frames1.txt");
	frames2.load("C:\\Users\\Administrator\\Desktop\\geometricVerification\\frames2.txt");
	matches.load("C:\\Users\\Administrator\\Desktop\\geometricVerification\\matches_2nn.txt");

	// 测试载入是否准确
	cout<< "element测试: " << " x: " << frames1(0,1) << " y: " << frames1(1,1) <<endl; 
	cout << " 行数： " << frames1.n_rows << " 列数：" << frames1.n_cols << endl;
	cout << "==========================================================" << endl;

	int numMatches = matches.n_cols;
	// 测试匹配数目是否准确
	cout << "为RANSAC前匹配数目： " << numMatches << endl;
	cout << "==========================================================" << endl;

	field<uvec> inliers(1, numMatches);
	field<mat> H(1, numMatches);

	uvec v = linspace<uvec>(0,1,2);
	uvec matchedIndex_Query = conv_to<uvec>::from(matches.row(0)-1);
	uvec matchedIndex_Object = conv_to<uvec>::from(matches.row(1)-1);

	mat x1 = frames1(v, matchedIndex_Query) ;
	mat x2 = frames2(v, matchedIndex_Object);
	cout << " x1查询图像匹配行数： " << x1.n_rows << " 查询图像匹配列数：" << x1.n_cols << endl;
	cout << " x2目标图像匹配行数： " << x2.n_rows << " 目标图像匹配列数：" << x2.n_cols << endl;
	cout<< "x1 element测试: " << " x: " << x1(0,169) << " y: " << x1(1,169) <<endl; 
	cout<< "x2 element测试: " << " x: " << x2(0,1) << " y: " << x2(1,1) <<endl;
	cout << "==========================================================" << endl;

	mat x1hom = join_cols(x1, ones<mat>(1, numMatches));  //在下面添加一行，注意和join_rows的区别
	mat x2hom = join_cols(x2, ones<mat>(1, numMatches));
	cout << " x1hom查询图像匹配行数： " << x1hom.n_rows << " 查询图像匹配列数：" << x1hom.n_cols << endl;
	cout<< "x1hom element测试: " << " x: " << x1hom(0,1) << " y: " << x1hom(1,1) << " z: " << x1hom(2,1) <<endl;
	cout << "==========================================================" << endl;

	mat x1p, H21;  //作用域
	double tol;
	for(int m = 0; m < numMatches; ++m){
		//cout << "m: " << m << endl;
		for(int t = 0; t < opts.numRefinementIterations; ++t){
			//cout << "t: " << t << endl;
			if (t == 0){
				mat tmp1 = frames1.col(matches(0, m)-1);
				mat A1 = toAffinity(tmp1);
				//A1.print("A1 =");
				mat tmp2 = frames2.col(matches(1, m)-1);
				mat A2 = toAffinity(tmp2);
				//A2.print("A2 =");
				H21 = A2 * inv(A1);
				//H21.print("H21 =");
				x1p = H21.rows(0, 1) * x1hom ;
				//x1p.print("x1p =");
				tol = opts.tolerance1;
			}else if(t !=0 && t <= 3){
				mat A1 = x1hom.cols(inliers(0, m));
				mat A2 = x2.cols(inliers(0, m));
				//A1.print("A1 =");
				//A2.print("A2 =");
		        H21 = A2*pinv(A1);
				x1p = H21.rows(0, 1) * x1hom ;
				//x1p.print("x1p =");
				mat v;
				v << 0 << 0 << 1 << endr;
				H21 = join_vert(H21, v);
				//H21.print("H21 =");
				//x1p.print("x1p =");
				tol = opts.tolerance2;
			}else{
				mat x1in = x1hom.cols(inliers(0, m));
				mat x2in = x2hom.cols(inliers(0, m));
				mat S1 = centering(x1in);
				mat S2 = centering(x2in);
				mat x1c = S1 * x1in;
				//x1c.print("x1c =");
				mat x2c = S2 * x2in;
				//x2c.print("x2c =");
				mat A1 = randu<mat>(x1c.n_rows ,x1c.n_cols);
				A1.zeros();
				mat A2 = randu<mat>(x1c.n_rows ,x1c.n_cols);
				A2.zeros();
				mat A3 = randu<mat>(x1c.n_rows ,x1c.n_cols);
				A3.zeros();
				for(int i = 0; i < x1c.n_cols; ++i){
					A2.col(i) = x1c.col(i)*(-x2c.row(0).col(i));
					A3.col(i) = x1c.col(i)*(-x2c.row(1).col(i));
				}
				mat T1 = join_cols(join_horiz(x1c, A1), join_horiz(A1, x1c));
				mat T2 = join_cols(T1, join_horiz(A2, A3));
				//T2.print("T2 =");
				mat U;
				vec s;
				mat V;
				svd_econ(U, s, V, T2);
				//U.print("U =");
				//V.print("V =");
				vec tmm = U.col(U.n_cols-1);
				H21 = reshape(tmm, 3, 3).t();
				H21 = inv(S2) * H21 * S1;
				H21 = H21 / H21(H21.n_rows-1, H21.n_cols-1) ;
				//H21.print("H21 =");
				mat x1phom = H21 * x1hom ;
				mat cc1 = x1phom.row(0) / x1phom.row(2);
				mat cc2 = x1phom.row(1) / x1phom.row(2);
				mat x1p = join_cols(cc1, cc2);
				//x1p.print("x1p =");
				tol = opts.tolerance3;
			}
			mat tmp = square(x2 - x1p); //精度跟matlab相比更高？
			//tmp.print("tmp =");
			mat dist2 = tmp.row(0) + tmp.row(1);
			//dist2.print("dist2 =");
			inliers(0, m) = find(dist2 < pow(tol, 2));
			H(0, m) = H21;
			//H(0, m).print("H(0, m) =");
			//inliers(0, m).print("inliers(0, m) =");
			//cout << inliers(0, m).size() << endl;
			//cout << "==========================================================" << endl;
			if (inliers(0, m).size() < opts.minInliers) break;
			if (inliers(0, m).size() > 0.7 * numMatches) break;
		}
	}
	uvec scores(numMatches);
	for(int i = 0; i < numMatches; ++i){
		scores.at(i) = inliers(0, i).n_rows;
	}
	scores.print("scores = ");
	uword index;
	scores.max(index);
	cout << index << endl;
	uvec inliers_final = inliers(0, index);
	mat H_final = inv(H(0, index));
	H_final.print("H_final = ");
	system("pause");
	return 0;
}