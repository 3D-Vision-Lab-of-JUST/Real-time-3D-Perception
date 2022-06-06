#pragma once

#include <cutil_inline.h>
#include <cutil_math.h>
#include <device_functions.h>

#include "cuda_SimpleMatrixUtil.h"
//#include "DepthCameraUtil.h"
#include "VoxelUtilHashSDF.h"

#include "CUDARayCastParams.h"

#ifndef __CUDACC__
#include "mLib.h"
#endif



struct RayCastSample
{
	float sdf;
	float alpha;
	uint weight;
	//uint3 color;
};

#ifndef MINF
#define MINF asfloat(0xff800000)
#endif

extern __constant__ RayCastParams c_rayCastParams;
extern "C" void updateConstantRayCastParams(const RayCastParams& params);


struct RayCastData {

	///////////////
	// Host part //
	///////////////

	__device__ __host__
	RayCastData() {
		d_depth = NULL;
		d_depth4 = NULL;
		d_normals = NULL;
		d_colors = NULL;

		d_vertexBuffer = NULL;

		d_rayIntervalSplatMinArray = NULL;
		d_rayIntervalSplatMaxArray = NULL;
	}

#ifndef __CUDACC__
	__host__
	void allocate(const RayCastParams& params) {
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_depth, sizeof(float) * params.m_width * params.m_height));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_depth4, sizeof(float4) * params.m_width * params.m_height));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_normals, sizeof(float4) * params.m_width * params.m_height));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_colors, sizeof(float4) * params.m_width * params.m_height));
	}

	__host__
	void updateParams(const RayCastParams& params) {
		updateConstantRayCastParams(params);
	}

	__host__
		void free() {
			MLIB_CUDA_SAFE_FREE(d_depth);
			MLIB_CUDA_SAFE_FREE(d_depth4);
			MLIB_CUDA_SAFE_FREE(d_normals);
			MLIB_CUDA_SAFE_FREE(d_colors);
	}
#endif

	/////////////////
	// Device part //
	/////////////////
#ifdef __CUDACC__

	__device__
		const RayCastParams& params() const {
			return c_rayCastParams;
	}

	__device__
	float frac(float val) const {
		return (val - floorf(val));
	}
	__device__
	float3 frac(const float3& virtualWorldPose, bool isRetouched) const {
		if (isRetouched)
		{
			return make_float3(frac(virtualWorldPose.x * 2.0) / 2.0, frac(virtualWorldPose.y * 2.0) / 2.0, frac(virtualWorldPose.z*2.0) / 2.0);
		}
		else {
			return make_float3(frac(virtualWorldPose.x), frac(virtualWorldPose.y), frac(virtualWorldPose.z));
		}
			
	}

	__device__
	bool trilinearInterpolationSimpleFastFast(const HashDataStruct& hash, const float3& pos, float& dist, uchar4& color) const {
		//float oSet = c_hashParams.m_virtualVoxelSize;
		float oSet = c_hashParams.m_virtualVoxelSize;  //0.005
		float3 oSet1 = make_float3(-oSet / 2, -oSet / 2, -oSet / 2);   // 0.0025, 0.005
		float3 oSet2 = make_float3(oSet / 2, -oSet / 2, -oSet / 2);
		float3 oSet3 = make_float3(-oSet / 2, oSet / 2, -oSet / 2);
		float3 oSet4 = make_float3(-oSet / 2, -oSet / 2, oSet / 2);
		float3 oSet5 = make_float3(oSet / 2, oSet / 2, -oSet / 2);
		float3 oSet6 = make_float3(-oSet / 2, oSet / 2, oSet / 2);
		float3 oSet7 = make_float3(oSet / 2, -oSet / 2, oSet / 2);
		float3 oSet8 = make_float3(oSet / 2, oSet / 2, oSet / 2);

		float3 weight1, weight2, weight3, weight4, weight5, weight6, weight7, weight8;
		float distanceBase1, distanceBase2, distanceBase3, distanceBase4, distanceBase5, distanceBase6, distanceBase7, distanceBase8;
		if (hash.isBlockRetouched(hash.worldToVirtualWorldPosFloat(pos)))
		{
			oSet = oSet / 2;
			distanceBase1 = distanceBase2 = distanceBase3 = distanceBase4 = distanceBase5 = distanceBase6 = distanceBase7 = distanceBase8 = 0.5;
			weight1 = weight2 = weight3 = weight4 = weight5 = weight6 = weight7 = weight8 = frac(hash.worldToVirtualWorldPosFloat(pos), true);
			oSet1 = make_float3(-oSet / 2, -oSet / 2, -oSet / 2);   // 0.0025, 0.005
			oSet2 = make_float3(oSet / 2, -oSet / 2, -oSet / 2);
			oSet3 = make_float3(-oSet / 2, oSet / 2, -oSet / 2);
			oSet4 = make_float3(-oSet / 2, -oSet / 2, oSet / 2);
			oSet5 = make_float3(oSet / 2, oSet / 2, -oSet / 2);
			oSet6 = make_float3(-oSet / 2, oSet / 2, oSet / 2);
			oSet7 = make_float3(oSet / 2, -oSet / 2, oSet / 2);
			oSet8 = make_float3(oSet / 2, oSet / 2, oSet / 2);
		}
		else {
			weight1 = weight2 = weight3 = weight4 = weight5 = weight6 = weight7 = weight8 = frac(hash.worldToVirtualWorldPosFloat(pos), false);
			distanceBase1 = distanceBase2 = distanceBase3 = distanceBase4 = distanceBase5 = distanceBase6 = distanceBase7 = distanceBase8 = 1.0;
			if (hash.isBlockRetouched(hash.worldToVirtualWorldPosFloat(pos + oSet1)))
			{
				weight1 = frac(hash.worldToVirtualWorldPosFloat(pos), true);
				distanceBase1 = distanceBase1 / 2;
				oSet1 = make_float3(-oSet / 4, -oSet / 4, -oSet / 4);
			}
			if (hash.isBlockRetouched(hash.worldToVirtualWorldPosFloat(pos + oSet2)))
			{
				weight2 = frac(hash.worldToVirtualWorldPosFloat(pos), true);
				distanceBase2 = distanceBase2 / 2;
				oSet2 = make_float3(oSet / 4, -oSet / 4, -oSet / 4);
			}
			if (hash.isBlockRetouched(hash.worldToVirtualWorldPosFloat(pos + oSet3)))
			{
				weight3 = frac(hash.worldToVirtualWorldPosFloat(pos), true);
				distanceBase3 = distanceBase3 / 2;
				oSet3 = make_float3(-oSet / 4, oSet / 4, -oSet / 4);
			}
			if (hash.isBlockRetouched(hash.worldToVirtualWorldPosFloat(pos + oSet4)))
			{
				weight4 = frac(hash.worldToVirtualWorldPosFloat(pos), true);
				distanceBase4 = distanceBase4 / 2;
				oSet4 = make_float3(-oSet / 4, -oSet / 4, oSet / 4);
			}
			if (hash.isBlockRetouched(hash.worldToVirtualWorldPosFloat(pos + oSet5)))
			{
				weight5 = frac(hash.worldToVirtualWorldPosFloat(pos), true);
				distanceBase5 = distanceBase5 / 2;
				oSet5 = make_float3(oSet / 4, oSet / 4, -oSet / 4);
			}
			if (hash.isBlockRetouched(hash.worldToVirtualWorldPosFloat(pos + oSet6)))
			{
				weight6 = frac(hash.worldToVirtualWorldPosFloat(pos), true);
				distanceBase6 = distanceBase6 / 2;
				oSet6 = make_float3(-oSet / 4, oSet / 4, oSet / 4);
			}
			if (hash.isBlockRetouched(hash.worldToVirtualWorldPosFloat(pos + oSet7)))
			{
				weight7 = frac(hash.worldToVirtualWorldPosFloat(pos), true);
				distanceBase7 = distanceBase7 / 2;
				oSet7 = make_float3(oSet / 4, -oSet / 4, oSet / 4);
			}
			if (hash.isBlockRetouched(hash.worldToVirtualWorldPosFloat(pos + oSet8)))
			{
				weight8 = frac(hash.worldToVirtualWorldPosFloat(pos), true);
				distanceBase8 = distanceBase8 / 2;
				oSet8 = make_float3(oSet / 4, oSet / 4, oSet / 4);
			}
		}
		
		dist = 0.0f;
		float3 colorFloat = make_float3(0.0f, 0.0f, 0.0f);

		//如果这个点右边的block是未被细化的，那么就不同了
		Voxel vMid = hash.getVoxel_World(pos);
		uint maskIdx = vMid.color.w;

		Voxel v = hash.getVoxel_World(pos + oSet1); if(v.weight == 0) return false; float3 vColor = make_float3(v.color.x, v.color.y, v.color.z); dist+= (distanceBase1 -weight1.x)*(distanceBase1 -weight1.y)*(distanceBase1 -weight1.z)*v.sdf; colorFloat+= (distanceBase1 -weight1.x)*(distanceBase1 -weight1.y)*(distanceBase1 -weight1.z)*vColor; 
		      v = hash.getVoxel_World(pos + oSet2); if(v.weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist+=       weight2.x *(distanceBase2 -weight2.y)*(distanceBase2 -weight2.z)*v.sdf; colorFloat+=	   weight2.x *(distanceBase2 -weight2.y)*(distanceBase2 -weight2.z)*vColor;
		      v = hash.getVoxel_World(pos + oSet3); if(v.weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist+= (distanceBase3 -weight3.x)*	   weight3.y *(distanceBase3 -weight3.z)*v.sdf; colorFloat+= (distanceBase3 -weight3.x)*	   weight3.y *(distanceBase3 -weight3.z)*vColor;
		      v = hash.getVoxel_World(pos + oSet4); if(v.weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist+= (distanceBase4 -weight4.x)*(distanceBase4 -weight4.y)*	   weight4.z *v.sdf; colorFloat+= (distanceBase4 -weight4.x)*(distanceBase4 -weight4.y)*	   weight4.z *vColor;
		      v = hash.getVoxel_World(pos + oSet5); if(v.weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist+=	   weight5.x *	   weight5.y *(distanceBase5 -weight5.z)*v.sdf; colorFloat+=	   weight5.x *	   weight5.y *(distanceBase5 -weight5.z)*vColor;
		      v = hash.getVoxel_World(pos + oSet6); if(v.weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist+= (distanceBase6 -weight6.x)*	   weight6.y *	   weight6.z *v.sdf; colorFloat+= (distanceBase6 -weight6.x)*	   weight6.y *	   weight6.z *vColor;
		      v = hash.getVoxel_World(pos + oSet7); if(v.weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist+=	   weight7.x *(distanceBase7 - weight7.y)*	   weight7.z *v.sdf; colorFloat+=	   weight7.x *(distanceBase7 -weight7.y)*	   weight7.z *vColor;
		      v = hash.getVoxel_World(pos + oSet8); if(v.weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist+=	   weight8.x *	   weight8.y *	   weight8.z *v.sdf; colorFloat+=	   weight8.x *	   weight8.y *	   weight8.z *vColor;

		color = make_uchar4(colorFloat.x, colorFloat.y, colorFloat.z, maskIdx);//v.color;
		
		return true;
	}


	//__device__
	//	bool trilinearInterpolationSimpleFastFast(const HashDataStruct& hash, const float3& pos, float& dist, uchar4& color) const {
	//	//float oSet = c_hashParams.m_virtualVoxelSize;
	//	float oSet = c_hashParams.m_virtualVoxelSize / 2;  //0.005

	//	float3 oSet1 = make_float3(-oSet / 2, -oSet / 2, -oSet / 2);   // 0.0025
	//	float3 oSet2 = make_float3(oSet / 2, -oSet / 2, -oSet / 2);
	//	float3 oSet3 = make_float3(-oSet / 2, oSet / 2, -oSet / 2);
	//	float3 oSet4 = make_float3(-oSet / 2, -oSet / 2, oSet / 2);
	//	float3 oSet5 = make_float3(oSet / 2, oSet / 2, -oSet / 2);
	//	float3 oSet6 = make_float3(-oSet / 2, oSet / 2, oSet / 2);
	//	float3 oSet7 = make_float3(oSet / 2, -oSet / 2, oSet / 2);
	//	float3 oSet8 = make_float3(oSet / 2, oSet / 2, oSet / 2);

	//	float3 weight1, weight2, weight3, weight4, weight5, weight6, weight7, weight8;
	//	float distanceBase1, distanceBase2, distanceBase3, distanceBase4, distanceBase5, distanceBase6, distanceBase7, distanceBase8;

	//	if (!hash.isBlockRetouched(hash.worldToVirtualWorldPosFloat(pos + oSet1)))
	//	{
	//		oSet1 = oSet1 * 2;
	//		weight1 = frac(hash.worldToVirtualWorldPosFloat(pos), false);
	//		distanceBase1 = 1.0;
	//	}
	//	else {
	//		weight1 = frac(hash.worldToVirtualWorldPosFloat(pos), true);
	//		distanceBase1 = 0.5;
	//	}
	//	if (!hash.isBlockRetouched(hash.worldToVirtualWorldPosFloat(pos + oSet2)))
	//	{
	//		oSet2 = oSet2 * 2;
	//		weight2 = frac(hash.worldToVirtualWorldPosFloat(pos), false);
	//		distanceBase2 = 1.0;
	//	}
	//	else {
	//		weight2 = frac(hash.worldToVirtualWorldPosFloat(pos), true);
	//		distanceBase2 = 0.5;
	//	}
	//	if (!hash.isBlockRetouched(hash.worldToVirtualWorldPosFloat(pos + oSet3)))
	//	{
	//		oSet3 = oSet3 * 2;
	//		weight3 = frac(hash.worldToVirtualWorldPosFloat(pos), false);
	//		distanceBase3 = 1.0;
	//	}
	//	else {
	//		weight3 = frac(hash.worldToVirtualWorldPosFloat(pos), true);
	//		distanceBase3 = 0.5;
	//	}
	//	if (!hash.isBlockRetouched(hash.worldToVirtualWorldPosFloat(pos + oSet4)))
	//	{
	//		oSet4 = oSet4 * 2;
	//		weight4 = frac(hash.worldToVirtualWorldPosFloat(pos), false);
	//		distanceBase4 = 1.0;
	//	}
	//	else {
	//		weight4 = frac(hash.worldToVirtualWorldPosFloat(pos), true);
	//		distanceBase4 = 0.5;
	//	}
	//	if (!hash.isBlockRetouched(hash.worldToVirtualWorldPosFloat(pos + oSet5)))
	//	{
	//		oSet5 = oSet5 * 2;
	//		weight5 = frac(hash.worldToVirtualWorldPosFloat(pos), false);
	//		distanceBase5 = 1.0;
	//	}
	//	else {
	//		weight5 = frac(hash.worldToVirtualWorldPosFloat(pos), true);
	//		distanceBase5 = 0.5;
	//	}
	//	if (!hash.isBlockRetouched(hash.worldToVirtualWorldPosFloat(pos + oSet6)))
	//	{
	//		oSet6 = oSet6 * 2;
	//		weight6 = frac(hash.worldToVirtualWorldPosFloat(pos), false);
	//		distanceBase6 = 1.0;
	//	}
	//	else {
	//		weight6 = frac(hash.worldToVirtualWorldPosFloat(pos), true);
	//		distanceBase6 = 0.5;
	//	}
	//	if (!hash.isBlockRetouched(hash.worldToVirtualWorldPosFloat(pos + oSet7)))
	//	{
	//		oSet7 = oSet7 * 2;
	//		weight7 = frac(hash.worldToVirtualWorldPosFloat(pos), false);
	//		distanceBase7 = 1.0;
	//	}
	//	else {
	//		weight7 = frac(hash.worldToVirtualWorldPosFloat(pos), true);
	//		distanceBase7 = 0.5;
	//	}
	//	if (!hash.isBlockRetouched(hash.worldToVirtualWorldPosFloat(pos + oSet8)))
	//	{
	//		oSet8 = oSet8 * 2;
	//		weight8 = frac(hash.worldToVirtualWorldPosFloat(pos), false);
	//		distanceBase8 = 1.0;
	//	}
	//	else {
	//		weight8 = frac(hash.worldToVirtualWorldPosFloat(pos), true);
	//		distanceBase8 = 0.5;
	//	}

	//	//const float3 posDual = pos - make_float3(oSetX / 2.0f, oSetY / 2.0f, oSetZ / 2.0f);

	//	//float3 weight = frac(hash.worldToVirtualWorldPosFloat(pos), isRetouched);
	//	dist = 0.0f;
	//	float3 colorFloat = make_float3(0.0f, 0.0f, 0.0f);

	//	//如果这个点右边的block是未被细化的，那么就不同了
	//	Voxel vMid = hash.getVoxel_World(pos);
	//	uint maskIdx = vMid.color.w;

	//	Voxel v = hash.getVoxel_World(pos + oSet1); if (v.weight == 0) return false; float3 vColor = make_float3(v.color.x, v.color.y, v.color.z); dist += (distanceBase1 - weight1.x)*(distanceBase1 - weight1.y)*(distanceBase1 - weight1.z)*v.sdf; colorFloat += (distanceBase1 - weight1.x)*(distanceBase1 - weight1.y)*(distanceBase1 - weight1.z)*vColor;
	//	v = hash.getVoxel_World(pos + oSet2); if (v.weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist += weight2.x *(distanceBase2 - weight2.y)*(distanceBase2 - weight2.z)*v.sdf; colorFloat += weight2.x *(distanceBase2 - weight2.y)*(distanceBase2 - weight2.z)*vColor;
	//	v = hash.getVoxel_World(pos + oSet3); if (v.weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist += (distanceBase3 - weight3.x)*	   weight3.y *(distanceBase3 - weight3.z)*v.sdf; colorFloat += (distanceBase3 - weight3.x)*	   weight3.y *(distanceBase3 - weight3.z)*vColor;
	//	v = hash.getVoxel_World(pos + oSet4); if (v.weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist += (distanceBase4 - weight4.x)*(distanceBase4 - weight4.y)*	   weight4.z *v.sdf; colorFloat += (distanceBase4 - weight4.x)*(distanceBase4 - weight4.y)*	   weight4.z *vColor;
	//	v = hash.getVoxel_World(pos + oSet5); if (v.weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist += weight5.x *	   weight5.y *(distanceBase5 - weight5.z)*v.sdf; colorFloat += weight5.x *	   weight5.y *(distanceBase5 - weight5.z)*vColor;
	//	v = hash.getVoxel_World(pos + oSet6); if (v.weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist += (distanceBase6 - weight6.x)*	   weight6.y *	   weight6.z *v.sdf; colorFloat += (distanceBase6 - weight6.x)*	   weight6.y *	   weight6.z *vColor;
	//	v = hash.getVoxel_World(pos + oSet7); if (v.weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist += weight7.x *(distanceBase7 - weight7.y)*	   weight7.z *v.sdf; colorFloat += weight7.x *(distanceBase7 - weight7.y)*	   weight7.z *vColor;
	//	v = hash.getVoxel_World(pos + oSet8); if (v.weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist += weight8.x *	   weight8.y *	   weight8.z *v.sdf; colorFloat += weight8.x *	   weight8.y *	   weight8.z *vColor;



	//	color = make_uchar4(colorFloat.x, colorFloat.y, colorFloat.z, maskIdx);//v.color;

	//	//if (v.color.w != -1)
	//	//{
	//	//	color = make_uchar3(255, 255, 255);//v.color;
	//	//}
	//	//else {
	//	//	
	//	//	color = make_uchar3(0, 0, 0);
	//	//}
	//	return true;
	//}
	__device__
		bool trilinearInterpolationSimpleFastFast(const HashDataStruct& hash, const float3& pos, float& dist, uchar3& color) const {
		float oSet = c_hashParams.m_virtualVoxelSize;  //0.005
		float3 oSet1 = make_float3(-oSet / 2, -oSet / 2, -oSet / 2);   // 0.0025, 0.005
		float3 oSet2 = make_float3(oSet / 2, -oSet / 2, -oSet / 2);
		float3 oSet3 = make_float3(-oSet / 2, oSet / 2, -oSet / 2);
		float3 oSet4 = make_float3(-oSet / 2, -oSet / 2, oSet / 2);
		float3 oSet5 = make_float3(oSet / 2, oSet / 2, -oSet / 2);
		float3 oSet6 = make_float3(-oSet / 2, oSet / 2, oSet / 2);
		float3 oSet7 = make_float3(oSet / 2, -oSet / 2, oSet / 2);
		float3 oSet8 = make_float3(oSet / 2, oSet / 2, oSet / 2);

		float3 weight1, weight2, weight3, weight4, weight5, weight6, weight7, weight8;
		float distanceBase1, distanceBase2, distanceBase3, distanceBase4, distanceBase5, distanceBase6, distanceBase7, distanceBase8;
		if (hash.isBlockRetouched(hash.worldToVirtualWorldPosFloat(pos)))
		{
			oSet = oSet / 2;
			distanceBase1 = distanceBase2 = distanceBase3 = distanceBase4 = distanceBase5 = distanceBase6 = distanceBase7 = distanceBase8 = 0.5;
			weight1 = weight2 = weight3 = weight4 = weight5 = weight6 = weight7 = weight8 = frac(hash.worldToVirtualWorldPosFloat(pos), true);
			oSet1 = make_float3(-oSet / 2, -oSet / 2, -oSet / 2);   // 0.0025, 0.005
			oSet2 = make_float3(oSet / 2, -oSet / 2, -oSet / 2);
			oSet3 = make_float3(-oSet / 2, oSet / 2, -oSet / 2);
			oSet4 = make_float3(-oSet / 2, -oSet / 2, oSet / 2);
			oSet5 = make_float3(oSet / 2, oSet / 2, -oSet / 2);
			oSet6 = make_float3(-oSet / 2, oSet / 2, oSet / 2);
			oSet7 = make_float3(oSet / 2, -oSet / 2, oSet / 2);
			oSet8 = make_float3(oSet / 2, oSet / 2, oSet / 2);
		}
		else {
			weight1 = weight2 = weight3 = weight4 = weight5 = weight6 = weight7 = weight8 = frac(hash.worldToVirtualWorldPosFloat(pos), false);
			distanceBase1 = distanceBase2 = distanceBase3 = distanceBase4 = distanceBase5 = distanceBase6 = distanceBase7 = distanceBase8 = 1.0;
			if (hash.isBlockRetouched(hash.worldToVirtualWorldPosFloat(pos + oSet1)))
			{
				weight1 = frac(hash.worldToVirtualWorldPosFloat(pos), true);
				distanceBase1 = distanceBase1 / 2;
				oSet1 = make_float3(-oSet / 4, -oSet / 4, -oSet / 4);
			}
			if (hash.isBlockRetouched(hash.worldToVirtualWorldPosFloat(pos + oSet2)))
			{
				weight2 = frac(hash.worldToVirtualWorldPosFloat(pos), true);
				distanceBase2 = distanceBase2 / 2;
				oSet2 = make_float3(oSet / 4, -oSet / 4, -oSet / 4);
			}
			if (hash.isBlockRetouched(hash.worldToVirtualWorldPosFloat(pos + oSet3)))
			{
				weight3 = frac(hash.worldToVirtualWorldPosFloat(pos), true);
				distanceBase3 = distanceBase3 / 2;
				oSet3 = make_float3(-oSet / 4, oSet / 4, -oSet / 4);
			}
			if (hash.isBlockRetouched(hash.worldToVirtualWorldPosFloat(pos + oSet4)))
			{
				weight4 = frac(hash.worldToVirtualWorldPosFloat(pos), true);
				distanceBase4 = distanceBase4 / 2;
				oSet4 = make_float3(-oSet / 4, -oSet / 4, oSet / 4);
			}
			if (hash.isBlockRetouched(hash.worldToVirtualWorldPosFloat(pos + oSet5)))
			{
				weight5 = frac(hash.worldToVirtualWorldPosFloat(pos), true);
				distanceBase5 = distanceBase5 / 2;
				oSet5 = make_float3(oSet / 4, oSet / 4, -oSet / 4);
			}
			if (hash.isBlockRetouched(hash.worldToVirtualWorldPosFloat(pos + oSet6)))
			{
				weight6 = frac(hash.worldToVirtualWorldPosFloat(pos), true);
				distanceBase6 = distanceBase6 / 2;
				oSet6 = make_float3(-oSet / 4, oSet / 4, oSet / 4);
			}
			if (hash.isBlockRetouched(hash.worldToVirtualWorldPosFloat(pos + oSet7)))
			{
				weight7 = frac(hash.worldToVirtualWorldPosFloat(pos), true);
				distanceBase7 = distanceBase7 / 2;
				oSet7 = make_float3(oSet / 4, -oSet / 4, oSet / 4);
			}
			if (hash.isBlockRetouched(hash.worldToVirtualWorldPosFloat(pos + oSet8)))
			{
				weight8 = frac(hash.worldToVirtualWorldPosFloat(pos), true);
				distanceBase8 = distanceBase8 / 2;
				oSet8 = make_float3(oSet / 4, oSet / 4, oSet / 4);
			}
		}

		dist = 0.0f;
		float3 colorFloat = make_float3(0.0f, 0.0f, 0.0f);

		Voxel v = hash.getVoxel_World(pos + oSet1); if (v.weight == 0) return false; float3 vColor = make_float3(v.color.x, v.color.y, v.color.z); dist += (distanceBase1 - weight1.x)*(distanceBase1 - weight1.y)*(distanceBase1 - weight1.z)*v.sdf; colorFloat += (distanceBase1 - weight1.x)*(distanceBase1 - weight1.y)*(distanceBase1 - weight1.z)*vColor;
		v = hash.getVoxel_World(pos + oSet2); if (v.weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist += weight2.x *(distanceBase2 - weight2.y)*(distanceBase2 - weight2.z)*v.sdf; colorFloat += weight2.x *(distanceBase2 - weight2.y)*(distanceBase2 - weight2.z)*vColor;
		v = hash.getVoxel_World(pos + oSet3); if (v.weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist += (distanceBase3 - weight3.x)*	   weight3.y *(distanceBase3 - weight3.z)*v.sdf; colorFloat += (distanceBase3 - weight3.x)*	   weight3.y *(distanceBase3 - weight3.z)*vColor;
		v = hash.getVoxel_World(pos + oSet4); if (v.weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist += (distanceBase4 - weight4.x)*(distanceBase4 - weight4.y)*	   weight4.z *v.sdf; colorFloat += (distanceBase4 - weight4.x)*(distanceBase4 - weight4.y)*	   weight4.z *vColor;
		v = hash.getVoxel_World(pos + oSet5); if (v.weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist += weight5.x *	   weight5.y *(distanceBase5 - weight5.z)*v.sdf; colorFloat += weight5.x *	   weight5.y *(distanceBase5 - weight5.z)*vColor;
		v = hash.getVoxel_World(pos + oSet6); if (v.weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist += (distanceBase6 - weight6.x)*	   weight6.y *	   weight6.z *v.sdf; colorFloat += (distanceBase6 - weight6.x)*	   weight6.y *	   weight6.z *vColor;
		v = hash.getVoxel_World(pos + oSet7); if (v.weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist += weight7.x *(distanceBase7 - weight7.y)*	   weight7.z *v.sdf; colorFloat += weight7.x *(distanceBase7 - weight7.y)*	   weight7.z *vColor;
		v = hash.getVoxel_World(pos + oSet8); if (v.weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist += weight8.x *	   weight8.y *	   weight8.z *v.sdf; colorFloat += weight8.x *	   weight8.y *	   weight8.z *vColor;

		color = make_uchar3(colorFloat.x, colorFloat.y, colorFloat.z);//v.color;

		return true;
	}
	//__device__
	//	bool trilinearInterpolationSimpleFastFast(const HashDataStruct& hash, const float3& pos, float& dist, uchar3& color) const {
	//	float oSet = c_hashParams.m_virtualVoxelSize;
	//	//printf("oSet:%d", oSet);
	//	
	//	bool isRetouched = hash.isBlockRetouched(hash.worldToVirtualWorldPosFloat(pos));
	//	if (isRetouched)
	//	{
	//		oSet = oSet / 2;
	//	}
	//	const float3 posDual = pos - make_float3(oSet / 2.0f, oSet / 2.0f, oSet / 2.0f);
	//	float3 weight = frac(hash.worldToVirtualWorldPosFloat(pos), isRetouched);

	//	dist = 0.0f;
	//	float3 colorFloat = make_float3(0.0f, 0.0f, 0.0f);
	//	float distanceBase = 1.0;
	//	if (isRetouched)
	//	{
	//		distanceBase = distanceBase / 2;
	//	}
	//	Voxel v = hash.getVoxel_World(posDual + make_float3(0.0f, 0.0f, 0.0f)); if (v.weight == 0) return false;    float3 vColor = make_float3(v.color.x, v.color.y, v.color.z); dist += (distanceBase - weight.x)*(distanceBase - weight.y)*(distanceBase - weight.z)*v.sdf; colorFloat += (distanceBase - weight.x)*(distanceBase - weight.y)*(distanceBase - weight.z)*vColor;
	//	      v = hash.getVoxel_World(posDual + make_float3(oSet, 0.0f, 0.0f)); if (v.weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist += weight.x *(distanceBase - weight.y)*(distanceBase - weight.z)*v.sdf; colorFloat += weight.x *(distanceBase - weight.y)*(distanceBase - weight.z)*vColor;
	//		  v = hash.getVoxel_World(posDual + make_float3(0.0f, oSet, 0.0f)); if (v.weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist += (distanceBase - weight.x)*	   weight.y *(distanceBase - weight.z)*v.sdf; colorFloat += (distanceBase - weight.x)*	   weight.y *(distanceBase - weight.z)*vColor;
	//	      v = hash.getVoxel_World(posDual + make_float3(0.0f, 0.0f, oSet)); if (v.weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist += (distanceBase - weight.x)*(distanceBase - weight.y)*	   weight.z *v.sdf; colorFloat += (distanceBase - weight.x)*(distanceBase - weight.y)*	   weight.z *vColor;
	//	      v = hash.getVoxel_World(posDual + make_float3(oSet, oSet, 0.0f)); if (v.weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist += weight.x *	   weight.y *(distanceBase - weight.z)*v.sdf; colorFloat += weight.x *	   weight.y *(distanceBase - weight.z)*vColor;
	//	      v = hash.getVoxel_World(posDual + make_float3(0.0f, oSet, oSet)); if (v.weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist += (distanceBase - weight.x)*	   weight.y *	   weight.z *v.sdf; colorFloat += (distanceBase - weight.x)*	   weight.y *	   weight.z *vColor;
	//	      v = hash.getVoxel_World(posDual + make_float3(oSet, 0.0f, oSet)); if (v.weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist += weight.x *(distanceBase - weight.y)*	   weight.z *v.sdf; colorFloat += weight.x *(distanceBase - weight.y)*	   weight.z *vColor;
	//	      v = hash.getVoxel_World(posDual + make_float3(oSet, oSet, oSet)); if (v.weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist += weight.x *	   weight.y *	   weight.z *v.sdf; colorFloat += weight.x *	   weight.y *	   weight.z *vColor;



	//	color = make_uchar3(colorFloat.x, colorFloat.y, colorFloat.z);//v.color;

	//																		   //if (v.color.w != -1)
	//																		   //{
	//																		   //	color = make_uchar3(255, 255, 255);//v.color;
	//																		   //}
	//																		   //else {
	//																		   //	
	//																		   //	color = make_uchar3(0, 0, 0);
	//																		   //}
	//	return true;
	//}
	//__device__
	//bool trilinearInterpolationSimpleFastFast(const HashData& hash, const float3& pos, float& dist, uchar3& color) const {
	//	const float oSet = c_hashParams.m_virtualVoxelSize;
	//	const float3 posDual = pos-make_float3(oSet/2.0f, oSet/2.0f, oSet/2.0f);
	//	float3 weight = frac(hash.worldToVirtualVoxelPosFloat(pos));

	//	dist = 0.0f;
	//	Voxel v = hash.getVoxel(posDual+make_float3(0.0f, 0.0f, 0.0f)); if(v.weight == 0) return false; dist+= (1.0f-weight.x)*(1.0f-weight.y)*(1.0f-weight.z)*v.sdf;
	//	v = hash.getVoxel(posDual+make_float3(oSet, 0.0f, 0.0f)); if(v.weight == 0) return false;		dist+=	   weight.x *(1.0f-weight.y)*(1.0f-weight.z)*v.sdf;
	//	v = hash.getVoxel(posDual+make_float3(0.0f, oSet, 0.0f)); if(v.weight == 0) return false;		dist+= (1.0f-weight.x)*	   weight.y *(1.0f-weight.z)*v.sdf;
	//	v = hash.getVoxel(posDual+make_float3(0.0f, 0.0f, oSet)); if(v.weight == 0) return false;		dist+= (1.0f-weight.x)*(1.0f-weight.y)*	   weight.z *v.sdf;
	//	v = hash.getVoxel(posDual+make_float3(oSet, oSet, 0.0f)); if(v.weight == 0) return false;		dist+=	   weight.x *	   weight.y *(1.0f-weight.z)*v.sdf;
	//	v = hash.getVoxel(posDual+make_float3(0.0f, oSet, oSet)); if(v.weight == 0) return false;		dist+= (1.0f-weight.x)*	   weight.y *	   weight.z *v.sdf;
	//	v = hash.getVoxel(posDual+make_float3(oSet, 0.0f, oSet)); if(v.weight == 0) return false;		dist+=	   weight.x *(1.0f-weight.y)*	   weight.z *v.sdf;
	//	v = hash.getVoxel(posDual+make_float3(oSet, oSet, oSet)); if(v.weight == 0) return false;		dist+=	   weight.x *	   weight.y *	   weight.z *v.sdf;

	//	color = v.color;

	//	return true;
	//}


	__device__
	float findIntersectionLinear(float tNear, float tFar, float dNear, float dFar) const
	{
		return tNear+(dNear/(dNear-dFar))*(tFar-tNear);
	}
	
	static const unsigned int nIterationsBisection = 3;
	
	// d0 near, d1 far
	__device__
		bool findIntersectionBisection(const HashDataStruct& hash, const float3& worldCamPos, const float3& worldDir, float d0, float r0, float d1, float r1, float& alpha, uchar4& color) const
	{
		float a = r0; float aDist = d0;
		float b = r1; float bDist = d1;
		float c = 0.0f;

#pragma unroll 1
		for(uint i = 0; i<nIterationsBisection; i++)
		{
			c = findIntersectionLinear(a, b, aDist, bDist);

			float cDist;
			if(!trilinearInterpolationSimpleFastFast(hash, worldCamPos+c*worldDir, cDist, color)) return false;

			if(aDist*cDist > 0.0) { a = c; aDist = cDist; }
			else { b = c; bDist = cDist; }
		}

		alpha = c;

		return true;
	}
	
	
	__device__
	float3 gradientForPoint(const HashDataStruct& hash, const float3& pos) const
	{
		const float voxelSize = c_hashParams.m_virtualVoxelSize;
		float3 offset = make_float3(voxelSize, voxelSize, voxelSize);
		/*bool isRetouched = hash.isBlockRetouched(hash.worldToVirtualWorldPosFloat(pos));
		if (isRetouched)
		{
			offset = make_float3(voxelSize / 2.0, voxelSize / 2.0, voxelSize / 2.0);
		}*/
		float distp00; uchar3 colorp00; trilinearInterpolationSimpleFastFast(hash, pos-make_float3(0.5f*offset.x, 0.0f, 0.0f), distp00, colorp00);
		float dist0p0; uchar3 color0p0; trilinearInterpolationSimpleFastFast(hash, pos-make_float3(0.0f, 0.5f*offset.y, 0.0f), dist0p0, color0p0);
		float dist00p; uchar3 color00p; trilinearInterpolationSimpleFastFast(hash, pos-make_float3(0.0f, 0.0f, 0.5f*offset.z), dist00p, color00p);

		float dist100; uchar3 color100; trilinearInterpolationSimpleFastFast(hash, pos+make_float3(0.5f*offset.x, 0.0f, 0.0f), dist100, color100);
		float dist010; uchar3 color010; trilinearInterpolationSimpleFastFast(hash, pos+make_float3(0.0f, 0.5f*offset.y, 0.0f), dist010, color010);
		float dist001; uchar3 color001; trilinearInterpolationSimpleFastFast(hash, pos+make_float3(0.0f, 0.0f, 0.5f*offset.z), dist001, color001);

		float3 grad = make_float3((distp00-dist100)/offset.x, (dist0p0-dist010)/offset.y, (dist00p-dist001)/offset.z);

		float l = length(grad);
		if(l == 0.0f) {
			return make_float3(0.0f, 0.0f, 0.0f);
		}

		return -grad/l;
	}

	static __inline__ __device__
	float depthProjToCameraZ(float z)	{
		return z * (c_rayCastParams.m_maxDepth - c_rayCastParams.m_minDepth) + c_rayCastParams.m_minDepth;
	}
	static __inline__ __device__
	float3 depthToCamera(unsigned int ux, unsigned int uy, float depth) 
	{
		const float x = ((float)ux-c_rayCastParams.mx) / c_rayCastParams.fx;
		const float y = ((float)uy-c_rayCastParams.my) / c_rayCastParams.fy;
		return make_float3(depth*x, depth*y, depth);
	}
	static __inline__ __device__
	float3 cameraToDepthProj(const float3& pos)	{
		float2 proj = make_float2(
			pos.x*c_rayCastParams.fx/pos.z + c_rayCastParams.mx,			
			pos.y*c_rayCastParams.fy/pos.z + c_rayCastParams.my);

		float3 pImage = make_float3(proj.x, proj.y, pos.z);

		pImage.x = (2.0f*pImage.x - (c_rayCastParams.m_width- 1.0f))/(c_rayCastParams.m_width- 1.0f);
		//pImage.y = (2.0f*pImage.y - (c_rayCastParams.m_height-1.0f))/(c_rayCastParams.m_height-1.0f);
		pImage.y = ((c_rayCastParams.m_height-1.0f) - 2.0f*pImage.y)/(c_rayCastParams.m_height-1.0f);
		pImage.z = (pImage.z - c_rayCastParams.m_minDepth)/(c_rayCastParams.m_maxDepth - c_rayCastParams.m_minDepth);

		return pImage;
	}

	__device__
	void traverseCoarseGridSimpleSampleAll(const HashDataStruct& hash, const float3& worldCamPos, const float3& worldDir, const float3& camDir, const int3& dTid, float minInterval, float maxInterval) const
	{
		//std::cout << "123" << std::endl;
		const RayCastParams& rayCastParams = c_rayCastParams;

		// Last Sample
		RayCastSample lastSample; lastSample.sdf = 0.0f; lastSample.alpha = 0.0f; lastSample.weight = 0; // lastSample.color = int3(0, 0, 0);
		const float depthToRayLength = 1.0f/camDir.z; // scale factor to convert from depth to ray length
		
		float rayCurrent = depthToRayLength * max(rayCastParams.m_minDepth, minInterval);	// Convert depth to raylength
		float rayEnd = depthToRayLength * min(rayCastParams.m_maxDepth, maxInterval);		// Convert depth to raylength
		//float rayCurrent = depthToRayLength * rayCastParams.m_minDepth;	// Convert depth to raylength
		//float rayEnd = depthToRayLength * rayCastParams.m_maxDepth;		// Convert depth to raylength

#pragma unroll 1
		while(rayCurrent < rayEnd)
		{
			float3 currentPosWorld = worldCamPos+rayCurrent*worldDir;
			float dist;	uchar3 color;
			//bool isRetouched = hash.isBlockRetouched(hash.worldToVirtualWorldPosFloat(currentPosWorld));
			if(trilinearInterpolationSimpleFastFast(hash, currentPosWorld, dist, color))
			{
				
				if(lastSample.weight > 0 && lastSample.sdf > 0.0f && dist < 0.0f)// current sample is always valid here 
				//if(lastSample.weight > 0 && ((lastSample.sdf > 0.0f && dist < 0.0f) || (lastSample.sdf < 0.0f && dist > 0.0f))) //hack for top down video
				{
					
					

					float alpha; // = findIntersectionLinear(lastSample.alpha, rayCurrent, lastSample.sdf, dist);
					uchar4 color2;
					bool b = findIntersectionBisection(hash, worldCamPos, worldDir, lastSample.sdf, lastSample.alpha, dist, rayCurrent, alpha, color2);
					
					float3 currentIso = worldCamPos+alpha*worldDir;
					if(b && abs(lastSample.sdf - dist) < rayCastParams.m_thresSampleDist)
					{
						if(abs(dist) < rayCastParams.m_thresDist)
						{
							float depth = alpha / depthToRayLength; // Convert ray length to depth depthToRayLength

							d_depth[dTid.y*rayCastParams.m_width+dTid.x] = depth;
							d_depth4[dTid.y*rayCastParams.m_width+dTid.x] = make_float4(depthToCamera(dTid.x, dTid.y, depth), 1.0f);

							d_colors[dTid.y*rayCastParams.m_width+dTid.x] = make_float4(color2.x/255.f, color2.y/255.f, color2.z/255.f, color2.w);
							//int maskIdx = color2.w;
							//printf("x:%d, y:%d, maskid:%d\n", dTid.x, dTid.y, color2.w);
							if(rayCastParams.m_useGradients)
							{
								float3 normal = -gradientForPoint(hash, currentIso);
								float4 n = rayCastParams.m_viewMatrix * make_float4(normal, 0.0f);
								d_normals[dTid.y*rayCastParams.m_width+dTid.x] = make_float4(n.x, n.y, n.z, 1.0f);
							}

							return;
						}
					}
				}

				lastSample.sdf = dist;
				lastSample.alpha = rayCurrent;
				// lastSample.color = color;
				lastSample.weight = 1;
				rayCurrent += rayCastParams.m_rayIncrement;
				/*if (isRetouched)
				{
					rayCurrent += rayCastParams.m_rayIncrement/2;
				}
				else {
					rayCurrent += rayCastParams.m_rayIncrement;
				}*/
				
			} else {
				lastSample.weight = 0;
				/*if (isRetouched)
				{
					rayCurrent += rayCastParams.m_rayIncrement / 2;
				}
				else {
					rayCurrent += rayCastParams.m_rayIncrement;
				}*/
				rayCurrent += rayCastParams.m_rayIncrement;
			}

			
		}
		
	}

#endif // __CUDACC__

	float*  d_depth;
	float4* d_depth4;
	float4* d_normals;
	float4* d_colors;

	float4* d_vertexBuffer; // ray interval splatting triangles, mapped from directx (memory lives there)

	cudaArray* d_rayIntervalSplatMinArray;
	cudaArray* d_rayIntervalSplatMaxArray;
};
