#include "stdafx.h"
#include "MaskSensing.h"


using namespace std;
using namespace cv;
using namespace dnn;


CUDAImageManager* m_CUDAImageManager;
RGBDSensor* m_RGBDSensor;


vector<string> classes;
vector<Scalar> colors;
// Initialize the parameters
float confThreshold = 0.5; // Confidence threshold
float maskThreshold = 0.3; // Mask threshold
Net net;

bool isLock = false;
// Draw the predicted bounding box, colorize and show the mask on the image
void drawBox(Mat& frame, int classId, float conf, Rect box, Mat& objectMask)
{
	//Draw a rectangle displaying the bounding box
	rectangle(frame, Point(box.x, box.y), Point(box.x + box.width, box.y + box.height), Scalar(255, 178, 50), 3);

	//Get the label for the class name and its confidence
	string label = format("%.2f", conf);
	if (!classes.empty())
	{
		CV_Assert(classId < (int)classes.size());
		label = classes[classId] + ":" + label;
	}

	//Display the label at the top of the bounding box
	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	box.y = max(box.y, labelSize.height);
	rectangle(frame, Point(box.x, box.y - round(1.5*labelSize.height)), Point(box.x + round(1.5*labelSize.width), box.y + baseLine), Scalar(255, 255, 255), FILLED);
	putText(frame, label, Point(box.x, box.y), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);

	Scalar color = colors[classId%colors.size()];

	// Resize the mask, threshold, color and apply it on the image
	resize(objectMask, objectMask, Size(box.width, box.height));
	//Mat mask = (objectMask > maskThreshold);
	//Mat coloredRoi = (0.3 * color + 0.7 * frame(box));
	//coloredRoi.convertTo(coloredRoi, CV_8UC3);

	//// Draw the contours on the image
	//vector<Mat> contours;
	//Mat hierarchy;
	//mask.convertTo(mask, CV_8U);
	//findContours(mask, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
	//drawContours(coloredRoi, contours, -1, color, 5, LINE_8, hierarchy, 100);
	//coloredRoi.copyTo(frame(box), mask);
	//imwrite("111.jpg", frame);
}
//For each frame, extract the bounding box and mask for each detected object
void postprocess(Mat& frame, const vector<Mat>& outs, uint* maskData, bool& isHasMaskData)
{
	Mat outDetections = outs[0];
	Mat outMasks = outs[1];

	// Output size of masks is NxCxHxW where
	// N - number of detected boxes
	// C - number of classes (excluding background)
	// HxW - segmentation shape
	const int numDetections = outDetections.size[2];
	const int numClasses = outMasks.size[1];

	outDetections = outDetections.reshape(1, outDetections.total() / 7);
	for (int i = 0; i < numDetections; ++i)
	{
		float score = outDetections.at<float>(i, 2);
		if (score > confThreshold)
		{
			// Extract the bounding box
			int classId = static_cast<int>(outDetections.at<float>(i, 1));
			if (classId == 66 || classId == 69)//|| classId == 80
			{
				continue;
			}
			int left = static_cast<int>(frame.cols * outDetections.at<float>(i, 3));
			int top = static_cast<int>(frame.rows * outDetections.at<float>(i, 4));
			int right = static_cast<int>(frame.cols * outDetections.at<float>(i, 5));
			int bottom = static_cast<int>(frame.rows * outDetections.at<float>(i, 6));

			left = max(0, min(left, frame.cols - 1));
			top = max(0, min(top, frame.rows - 1));
			right = max(0, min(right, frame.cols - 1));
			bottom = max(0, min(bottom, frame.rows - 1));

			Rect box = Rect(left, top, right - left + 1, bottom - top + 1);

			// Extract the mask for the object
			Mat objectMask(outMasks.size[2], outMasks.size[3], CV_32F, outMasks.ptr<float>(i, classId));
			// Draw bounding box, colorize and show the mask on the image
			drawBox(frame, classId, score, box, objectMask);
			
			Mat mask = (objectMask > maskThreshold);
			mask.convertTo(mask, CV_8UC1);
			for (int h = 1; h < mask.rows-1; h++)
			{
				for (int w = 1; w < mask.cols-1; w++)
				{
					
					uchar maskNum = mask.at<uchar>(h, w);
					//imshow("Mask", mask);
					//waitKey(0);
					if (maskNum != 0)
					{
						maskData[(h + top) * 640 + w + left] = classId;
						//maskData.at<uchar>(h + top, w + left) = classId;
						
					}
					//maskData.at<uchar>(h, w) = objectMask.at<uchar>(h, w);
				}

			}
			isHasMaskData = true;

		}
		
	}
}

Mat ucharToMat(vec4uc* p2)
{
	//cout<< "length: " << p2-> << endl;
	int img_width = 640;
	int img_height = 480;
	Mat img(Size(img_width, img_height), CV_8UC3);
	for (int i = 0; i < img_width * img_height; i++)
	{
		int b = p2[i].x;
		int g = p2[i].y;
		int r = p2[i].z;
		
		img.at<Vec3b>(i / img_width, (i % img_width))[0] = r;
		img.at<Vec3b>(i / img_width, (i % img_width))[1] = g;
		img.at<Vec3b>(i / img_width, (i % img_width))[2] = b;
		
		
	}
	return img;
}

int startMaskSensing(RGBDSensor* sensor, CUDAImageManager* imageManager)
{
	m_CUDAImageManager = imageManager;
	m_RGBDSensor = sensor;
	
	// Load names of classes
	string classesFile = "./mask_rcnn_inception_v2_coco_2018_01_28/mscoco_labels.names";
	ifstream ifs(classesFile.c_str());
	string line;
	while (getline(ifs, line)) classes.push_back(line);

	// Load the colors
	string colorsFile = "./mask_rcnn_inception_v2_coco_2018_01_28/colors.txt";
	ifstream colorFptr(colorsFile.c_str());
	while (getline(colorFptr, line))
	{
		char* pEnd;
		double r, g, b;
		r = strtod(line.c_str(), &pEnd);
		g = strtod(pEnd, NULL);
		b = strtod(pEnd, NULL);
		Scalar color = Scalar(r, g, b, 255.0);
		colors.push_back(Scalar(r, g, b, 255.0));
	}

	// Give the configuration and weight files for the model
	String textGraph = "./mask_rcnn_inception_v2_coco_2018_01_28/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt";
	String modelWeights = "./mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb";
	String modelConfig = "./mask_rcnn_inception_v2_coco_2018_01_28/pipeline.config";
	// Load the network
	cout << "loading net..." << endl;
	net = readNet(modelWeights, textGraph);
	//Net net = readNetFromTensorflow(modelWeights, textGraph);
	net.setPreferableBackend(DNN_BACKEND_INFERENCE_ENGINE);
	net.setPreferableTarget(DNN_TARGET_CPU);

	// Open a video file or an image file or a camera stream.
	string str, outputFile;

	// Create a window
	static const string kWinName = "Deep learning object detection in OpenCV";

	namedWindow("Mask", WINDOW_NORMAL);
	moveWindow("Mask", 680, 10);
	resizeWindow("Mask", Size(640, 480));
	
	return 0; 
}

void MaskSensing(Mat colorMat, int currentFrameIdx)
{
	uint* maskData = (uint*)malloc(sizeof(uint) * 640 * 480);
	for (int i = 0; i < 640 * 480; i++)
	{
		maskData[i] = 0;
	}
	//uint maskData[640 * 480] = {0};
	
	bool isHasMaskData;
	/*if (isLock)
	{
		while (true)
		{

		}
	}
	isLock = true;
*/
	
	/*Mat colorMat_80_60;
	resize(colorMat_640_480, colorMat_80_60, Size(80, 60), 0, 0, INTER_LINEAR);*/
	
	// get frame from the video
	Mat blob;
	// Create a 4D blob from a frame.
	blobFromImage(colorMat, blob, 1.0, Size(colorMat.cols, colorMat.rows), Scalar(), true, false);
	//blobFromImage(frame, blob);

	//Sets the input to the network
	net.setInput(blob);

	// Runs the forward pass to get output from the output layers
	std::vector<String> outNames(2);
	outNames[0] = "detection_out_final";
	outNames[1] = "detection_masks";
	vector<Mat> outs;
	net.forward(outs, outNames);

	// Extract the bounding box and mask for each of the detected objects
	postprocess(colorMat, outs, maskData, isHasMaskData);
	//  here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	//m_CUDAImageManager->SetMaskData(maskData, currentFrameIdx);
	// Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
	vector<double> layersTimes;
	double freq = getTickFrequency() / 1000;
	double t = net.getPerfProfile(layersTimes) / freq;
	string label = format("Mask-RCNN Inference time for a frame : %0.0f ms", t);
	putText(colorMat, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));


	imshow("Mask", colorMat);
	//isLock = false;
	//imshow("Mask", colorMat);
}

