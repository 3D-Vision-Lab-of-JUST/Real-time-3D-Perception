#pragma once

#include "GlobalAppState.h"

#ifdef REAL_SENSE

#include "RGBDSensor.h"

#include<iostream>
#include <librealsense2/rs.hpp>
#include <string>

using namespace std;
using namespace rs2;

class RealSenseSensor : public RGBDSensor
{
public:
	//! Constructor; allocates CPU memory and creates handles
	RealSenseSensor();

	//! Destructor; releases allocated ressources
	~RealSenseSensor();

	//! Initializes the sensor
	void createFirstConnected();

	//! Processes the depth & color data
	bool processDepth();

	//! processing happends in processdepth()
	bool processColor() {
		return true;
	}

	string getSensorName() const {
		return "RealSense";
	}

private:
	rs2::pipeline pipe;
	rs2::colorizer map;

	unsigned int color_width;
	unsigned int color_height;
	unsigned int depth_width;
	unsigned int depth_height;
	unsigned int frame_rate;
	float m_depth_scale;
};
#endif