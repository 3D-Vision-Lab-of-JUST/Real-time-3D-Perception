//#pragma once
//
//#include <vector>
//#include <opencv2/core.hpp>
//#include <opencv2/opencv.hpp>
//#include <opencv2/dnn.hpp>
//#include <opencv2/highgui.hpp>
//#include <opencv2/imgproc.hpp>
//
//using namespace std;
//using namespace cv;
//
//
//void GenBoundingBox(float* src, int width, int height, vector<myPoint>& bbox_min, vector<myPoint>& bbox_max)
//{
//	cv::Mat objectMask(height, width, 5, src);
//
//	for (int i = 0; i < 255; i++)
//	{
//		cv::Mat dst(height, width, 5);
//		cv::threshold(objectMask, dst, i, i + 1, THRESH_BINARY);
//		vector<vector<Point>> contours;
//		vector<Vec4i> hierarcy;
//		findContours(dst, contours, hierarcy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
//		vector<Rect> boundRect(contours.size());  //定义外接矩形集合
//		vector<RotatedRect> box(contours.size()); //定义最小外接矩形集合
//		Point2f rect[4];
//		for (int i = 0; i<contours.size(); i++)
//		{
//			box[i] = minAreaRect(Mat(contours[i]));  //计算每个轮廓最小外接矩形
//			boundRect[i] = boundingRect(Mat(contours[i]));
//			box[i].points(rect);  //把最小外接矩形四个端点复制给rect数组
//
//			myPoint min{ boundRect[i].x, boundRect[i].y };
//			myPoint max{ boundRect[i].x + boundRect[i].width, boundRect[i].y + boundRect[i].height };
//			bbox_min.push_back(min);
//			bbox_max.push_back(max);
//		}
//
//	}
//
//}