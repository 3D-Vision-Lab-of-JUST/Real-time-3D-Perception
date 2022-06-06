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
//		vector<Rect> boundRect(contours.size());  //������Ӿ��μ���
//		vector<RotatedRect> box(contours.size()); //������С��Ӿ��μ���
//		Point2f rect[4];
//		for (int i = 0; i<contours.size(); i++)
//		{
//			box[i] = minAreaRect(Mat(contours[i]));  //����ÿ��������С��Ӿ���
//			boundRect[i] = boundingRect(Mat(contours[i]));
//			box[i].points(rect);  //����С��Ӿ����ĸ��˵㸴�Ƹ�rect����
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