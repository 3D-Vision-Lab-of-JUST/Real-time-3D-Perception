#pragma once
#include "RGBDSensor.h"
#include "CUDAImageManager.h"

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


int startMaskSensing(RGBDSensor* sensor, CUDAImageManager* imageManager);
void MaskSensing(cv::Mat color, int currentFrameIdx);
cv::Mat ucharToMat(vec4uc* p2);