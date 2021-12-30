

#include <opencv2/opencv.hpp>  
#include "highgui/highgui.hpp"    
#include <opencv2/xfeatures2d.hpp>

using namespace cv;
using namespace std;



int main(int argc, char* argv[])
{
	Mat image01 = imread("F:/parallel_computing/opencv_match/建筑1_2.png");
	Mat image02 = imread("F:/parallel_computing/opencv_match/建筑2.png");

	if (image01.data == NULL || image02.data == NULL)
		return 0;
	imshow("前景", image01);
	imshow("背景", image02);

	//灰度图转换  
	Mat image1, image2;
	cvtColor(image01, image1, CV_RGB2GRAY);
	cvtColor(image02, image2, CV_RGB2GRAY);

	//提取特征点    
	Ptr<Feature2D> f2d = xfeatures2d::SURF::create();	//SURF
	vector<KeyPoint> keyPoint1, keyPoint2;
	f2d->detect(image1, keyPoint1);
	f2d->detect(image2, keyPoint2);


	//特征点描述，为下边的特征点匹配做准备    
	Mat imageDesc1, imageDesc2;
	f2d->compute(image1, keyPoint1, imageDesc1);
	f2d->compute(image2, keyPoint2, imageDesc2);

	//获得匹配特征点，并提取最优配对     
	FlannBasedMatcher matcher;
	vector<DMatch> matchePoints;
	matcher.match(imageDesc1, imageDesc2, matchePoints, Mat());
	sort(matchePoints.begin(), matchePoints.end()); //特征点排序 

    //配对结果
    Mat imagematches;
    drawMatches(image01, keyPoint1, image02, keyPoint2, matchePoints, imagematches);
    imshow("Matches", imagematches);
    imwrite("Matches-SURF.jpg", imagematches);

	//获取排在前N个的最优匹配特征点  
	vector<Point2f> imagePoints1, imagePoints2;

	for (int i = 0; i < 3; i++)//挑选的特征点 仿射变换需要3个
	{
		imagePoints1.push_back(keyPoint1[matchePoints[i].queryIdx].pt);
		imagePoints2.push_back(keyPoint2[matchePoints[i].trainIdx].pt);
	}

	//获取图像1到图像2的投影映射矩阵，尺寸为3*3 
	Mat Trans = getAffineTransform(imagePoints1, imagePoints2);//getAffineTransform getPerspectiveTransform
	
	//图像配准  仿射变换
	Mat imageTransform1;
    warpAffine(image01, imageTransform1, Trans, Size(image02.cols, image02.rows));//warpAffine warpPerspective
	imshow("仿射变换-SURF", imageTransform1);
	//imwrite("仿射变换-SURF.jpg", imageTransform1);

    //背景与变换结果融合
	for (int i = 0; i < image02.rows; i++)
	{
		for (int j = 0; j < image02.cols; j++)
		{
			if (imageTransform1.at<Vec3b>(i, j)[0]+ imageTransform1.at<Vec3b>(i, j)[1]+ imageTransform1.at<Vec3b>(i, j)[2]) 
			{
				image02.at<Vec3b>(i, j)[0] = imageTransform1.at<Vec3b>(i, j)[0];
				image02.at<Vec3b>(i, j)[1] = imageTransform1.at<Vec3b>(i, j)[1];
				image02.at<Vec3b>(i, j)[2] = imageTransform1.at<Vec3b>(i, j)[2];
			}
		}
	}
	imshow("融合结果-SURF", image02);
    imwrite("融合结果-SURF.jpg", image02);
	waitKey();
	return 0;
}
