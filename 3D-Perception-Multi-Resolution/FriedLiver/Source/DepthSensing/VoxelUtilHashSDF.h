#pragma once

#ifndef sint
typedef signed int sint;
#endif

#ifndef uint
typedef unsigned int uint;
#endif 

#ifndef slong 
typedef signed long slong;
#endif

#ifndef ulong
typedef unsigned long ulong;
#endif

#ifndef uchar
typedef unsigned char uchar;
#endif

#ifndef schar
typedef signed char schar;
#endif




#include <cutil_inline.h>
#include <cutil_math.h>
#include <device_functions.h>

#include "cuda_SimpleMatrixUtil.h"
#include "CUDAHashParams.h"

#include "DepthCameraUtil.h"

#define HANDLE_COLLISIONS
#define SDF_BLOCK_SIZE 8
#define RETOUCH_MULTI 2
#define HASH_BUCKET_SIZE 4

#ifndef MINF
#define MINF __int_as_float(0xff800000)
#endif

#ifndef PINF
#define PINF __int_as_float(0x7f800000)
#endif

//status flags for hash entries
static const int LOCK_ENTRY = -1;
static const int FREE_ENTRY = -2;
static const int NO_OFFSET = 0;
 
__align__(16)
struct HashEntry 
{
	int3	pos;		//hash position (lower left corner of SDFBlock))
	int		ptr;		//pointer into heap to SDFBlock
	uint	offset;		//offset for collisions

	
	__device__ void operator=(const struct HashEntry& e) {
		//((int*)this)[0] = ((const int*)&e)[0];
		//((int*)this)[1] = ((const int*)&e)[1];
		//((int*)this)[2] = ((const int*)&e)[2];
		//((int*)this)[3] = ((const int*)&e)[3];
		//((int*)this)[4] = ((const int*)&e)[4];
		((long long*)this)[0] = ((const long long*)&e)[0];
		((long long*)this)[1] = ((const long long*)&e)[1];
		((int*)this)[4] = ((const int*)&e)[4];
	}
};

//__align__(8)
struct Voxel {
	float	sdf;		//signed distance function
	float	weight;		//accumulated sdf weight
	uchar4	color;		//color r g b mask
	//uchar   maskIdx;

	//unsigned short sdf;
	//unsigned short weight;
	//uchar4	color;

	__device__ void operator=(const struct Voxel& v) {
		((int*)this)[0] = ((const int*)&v)[0];
		((int*)this)[1] = ((const int*)&v)[1];
		((int*)this)[2] = ((const int*)&v)[2];
		
		//((int*)this)[3] = ((const int*)&v)[3];
		//((int*)this)[3] = ((const int*)&v)[3];
		//((long long*)this)[0] = ((const long long*)&v)[0];	//8 bytes

		//this needs align, which unfortunately is problematic as __align__(16) would require more memory...
		//((long long*)this)[0] = ((const long long*)&v)[0];	//8 bytes
		//((int*)this)[2] = ((const int*)&v)[2];				//4 bytes
	}

};
struct SDFBlock {
	Voxel voxels[SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE];//
	
	bool m_bIsRetouched = false; //用来标记是否需要倍细化
	uint m_bRetouchIndex = -1; //记录对应的细化后的编号 \ 用来控制是否已经完成细化
};
struct SDFBlock_Retouched {
	//16*16*16
	Voxel voxels[(SDF_BLOCK_SIZE * RETOUCH_MULTI)*(SDF_BLOCK_SIZE * RETOUCH_MULTI)*(SDF_BLOCK_SIZE * RETOUCH_MULTI)];//

};

extern  __constant__ HashParams c_hashParams;
extern "C" void updateConstantHashParams(const HashParams& hashParams);
 
struct HashDataStruct {

	///////////////
	// Host part //
	///////////////

	__device__ __host__
	HashDataStruct() {
		d_heap = NULL;
		d_heapCounter = NULL;
		d_hash = NULL;
		d_hashDecision = NULL;
		d_hashDecisionPrefix = NULL;
		d_hashCompactified = NULL;
		d_hashCompactifiedCounter = NULL;
		d_SDFBlocks = NULL;

		d_retouchCounter = NULL;
		d_retouchDecision = NULL;
		d_SDFBlocks_Retouched = NULL;
		d_hashBucketMutex = NULL;
		m_bIsOnGPU = false;
	}

	__host__
	void allocate(const HashParams& params, bool dataOnGPU = true) {
		m_bIsOnGPU = dataOnGPU;
		if (m_bIsOnGPU) {
			cutilSafeCall(cudaMalloc(&d_heap, sizeof(unsigned int) * params.m_numSDFBlocks));
			cutilSafeCall(cudaMalloc(&d_heapCounter, sizeof(unsigned int)));
			cutilSafeCall(cudaMalloc(&d_hash, sizeof(HashEntry)* params.m_hashNumBuckets * params.m_hashBucketSize));
			cutilSafeCall(cudaMalloc(&d_hashDecision, sizeof(int)* params.m_hashNumBuckets * params.m_hashBucketSize));
			cutilSafeCall(cudaMalloc(&d_hashDecisionPrefix, sizeof(int)* params.m_hashNumBuckets * params.m_hashBucketSize));
			cutilSafeCall(cudaMalloc(&d_hashCompactified, sizeof(HashEntry)* params.m_hashNumBuckets * params.m_hashBucketSize));
			cutilSafeCall(cudaMalloc(&d_hashCompactifiedCounter, sizeof(int)));
			cutilSafeCall(cudaMalloc(&d_SDFBlocks, sizeof(SDFBlock) * params.m_numSDFBlocks)); // *params.m_SDFBlockSize*params.m_SDFBlockSize*params.m_SDFBlockSize
		
			cutilSafeCall(cudaMalloc(&d_SDFBlocks_Retouched, sizeof(SDFBlock_Retouched) * params.m_numSDFBlocks / 5));//params.m_numSDFBlocks
			cutilSafeCall(cudaMalloc(&d_retouchCounter, sizeof(unsigned int)));
			cutilSafeCall(cudaMalloc(&d_retouchDecision, sizeof(int) * params.m_numSDFBlocks / 5));//params.m_numSDFBlocks
			
			cutilSafeCall(cudaMalloc(&d_hashBucketMutex, sizeof(int)* params.m_hashNumBuckets));
			
		} 
		else 
		{
			d_heap = new unsigned int[params.m_numSDFBlocks];
			d_heapCounter = new unsigned int[1];
			d_hash = new HashEntry[params.m_hashNumBuckets * params.m_hashBucketSize];
			d_hashDecision = new int[params.m_hashNumBuckets * params.m_hashBucketSize];
			d_hashDecisionPrefix = new int[params.m_hashNumBuckets * params.m_hashBucketSize];
			d_hashCompactifiedCounter = new int[1];
			d_hashCompactified = new HashEntry[params.m_hashNumBuckets * params.m_hashBucketSize];
			d_SDFBlocks = new SDFBlock[params.m_numSDFBlocks]; //* params.m_SDFBlockSize*params.m_SDFBlockSize*params.m_SDFBlockSize
			d_SDFBlocks_Retouched = new SDFBlock_Retouched[params.m_numSDFBlocks / 5];//params.m_numSDFBlocks
			d_retouchCounter = new unsigned int[1];
			d_retouchDecision = new int[params.m_numSDFBlocks / 5];//params.m_numSDFBlocks

			d_hashBucketMutex = new int[params.m_hashNumBuckets];
		}

		updateParams(params);
	}

	__host__
	void updateParams(const HashParams& params) {
		if (m_bIsOnGPU) {
			updateConstantHashParams(params);
		} 
	}

	__host__
	void free() {
		if (m_bIsOnGPU) {
			cutilSafeCall(cudaFree(d_heap));
			cutilSafeCall(cudaFree(d_heapCounter));
			cutilSafeCall(cudaFree(d_hash));
			cutilSafeCall(cudaFree(d_hashDecision));
			cutilSafeCall(cudaFree(d_hashDecisionPrefix));
			cutilSafeCall(cudaFree(d_hashCompactified));
			cutilSafeCall(cudaFree(d_hashCompactifiedCounter));
			cutilSafeCall(cudaFree(d_SDFBlocks));
			cutilSafeCall(cudaFree(d_SDFBlocks_Retouched));
			cutilSafeCall(cudaFree(d_retouchCounter));
			cutilSafeCall(cudaFree(d_retouchDecision));
			
			cutilSafeCall(cudaFree(d_hashBucketMutex));
		} else {
			if (d_heap) delete[] d_heap;
			if (d_heapCounter) delete[] d_heapCounter;
			if (d_hash) delete[] d_hash;
			if (d_hashDecision) delete[] d_hashDecision;
			if (d_hashDecisionPrefix) delete[] d_hashDecisionPrefix;
			if (d_hashCompactified) delete[] d_hashCompactified;
			if (d_hashCompactifiedCounter) delete[] d_hashCompactifiedCounter;
			if (d_SDFBlocks) delete[] d_SDFBlocks;
			if (d_SDFBlocks_Retouched) delete[] d_SDFBlocks_Retouched;
			if (d_retouchCounter) delete[] d_retouchCounter;
			if (d_retouchDecision) delete[] d_retouchDecision;
			
			if (d_hashBucketMutex) delete[] d_hashBucketMutex;
		}

		d_hash = NULL;
		d_heap = NULL;
		d_heapCounter = NULL;
		d_hashDecision = NULL;
		d_hashDecisionPrefix = NULL;
		d_hashCompactified = NULL;
		d_hashCompactifiedCounter = NULL;
		d_SDFBlocks = NULL;
		d_SDFBlocks_Retouched = NULL;
		d_retouchCounter = NULL;
		d_retouchDecision = NULL;

		d_hashBucketMutex = NULL;
	}

	__host__
	HashDataStruct copyToCPU() const {
		HashParams params;
		
		HashDataStruct hashData;
		hashData.allocate(params, false);	//allocate the data on the CPU
		cutilSafeCall(cudaMemcpy(hashData.d_heap, d_heap, sizeof(unsigned int) * params.m_numSDFBlocks, cudaMemcpyDeviceToHost));
		cutilSafeCall(cudaMemcpy(hashData.d_heapCounter, d_heapCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost));
		cutilSafeCall(cudaMemcpy(hashData.d_hash, d_hash, sizeof(HashEntry)* params.m_hashNumBuckets * params.m_hashBucketSize, cudaMemcpyDeviceToHost));
		cutilSafeCall(cudaMemcpy(hashData.d_hashDecision, d_hashDecision, sizeof(int)*params.m_hashNumBuckets * params.m_hashBucketSize, cudaMemcpyDeviceToHost));
		cutilSafeCall(cudaMemcpy(hashData.d_hashDecisionPrefix, d_hashDecisionPrefix, sizeof(int)*params.m_hashNumBuckets * params.m_hashBucketSize, cudaMemcpyDeviceToHost));
		cutilSafeCall(cudaMemcpy(hashData.d_hashCompactified, d_hashCompactified, sizeof(HashEntry)* params.m_hashNumBuckets * params.m_hashBucketSize, cudaMemcpyDeviceToHost));
		cutilSafeCall(cudaMemcpy(hashData.d_SDFBlocks, d_SDFBlocks, sizeof(SDFBlock) * params.m_numSDFBlocks, cudaMemcpyDeviceToHost));// * params.m_SDFBlockSize*params.m_SDFBlockSize*params.m_SDFBlockSize
		cutilSafeCall(cudaMemcpy(hashData.d_SDFBlocks_Retouched, d_SDFBlocks_Retouched, sizeof(SDFBlock_Retouched) * params.m_numSDFBlocks / 5, cudaMemcpyDeviceToHost));
		cutilSafeCall(cudaMemcpy(hashData.d_retouchCounter, d_retouchCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost));
		cutilSafeCall(cudaMemcpy(hashData.d_retouchDecision, d_retouchDecision, sizeof(int) * params.m_numSDFBlocks / 5, cudaMemcpyDeviceToHost));
		
		cutilSafeCall(cudaMemcpy(hashData.d_hashBucketMutex, d_hashBucketMutex, sizeof(int)* params.m_hashNumBuckets, cudaMemcpyDeviceToHost));
		
		return hashData;	//TODO MATTHIAS look at this (i.e,. when does memory get destroyed ; if it's in the destructer it would kill everything here 
	}



	/////////////////
	// Device part //
	/////////////////
//#define __CUDACC__
#ifdef __CUDACC__

	__device__
	const HashParams& params() const {
		return c_hashParams;
	}

	//! see teschner et al. (but with correct prime values)
	__device__ 
	uint computeHashPos(const int3& virtualBlockPos) const { 
		const int p0 = 73856093;
		const int p1 = 19349669;
		const int p2 = 83492791;

		int res = ((virtualBlockPos.x * p0) ^ (virtualBlockPos.y * p1) ^ (virtualBlockPos.z * p2)) % c_hashParams.m_hashNumBuckets;
		if (res < 0) res += c_hashParams.m_hashNumBuckets;
		return (uint)res;
	}

	//! returns the truncation of the SDF for a given distance value
	__device__ 
	float getTruncation(float z) const {
		return c_hashParams.m_truncation + c_hashParams.m_truncScale * z;
	}
	__device__
		float3 worldToVirtualWorldPosFloat(const float3& pos) const {
		return pos / c_hashParams.m_virtualVoxelSize;
	}
	//细化后不再提供整型的virtual voxel pos
	//这里只能支持到一位小数！
	__device__ 
	float3 worldToVirtualWorldPos(const float3& pos) const	{
		const float3 p = pos / c_hashParams.m_virtualVoxelSize;
		int3 P1 = make_int3(p + make_float3(sign(p))*0.5f);//

		return make_float3(P1);
	}

	//通过除以更小的c_hashParams.m_virtualVoxelSize，原本四舍五入，改为趋于0.5的四舍五入
	__device__
		float3 worldToVirtualWorldPos_Retouched(const float3& pos) const {

		const float3 p = pos / (c_hashParams.m_virtualVoxelSize / 2);
		int3 P1 = make_int3(p + make_float3(sign(p))*0.5f); //
		
		return make_float3(P1)/2.0f;
	}
	//__device__ 
	//int3 worldToVirtualVoxelPos(const float3& pos) const {
	//	//const float3 p = pos*g_VirtualVoxelResolutionScalar;
	//	const float3 p = pos / c_hashParams.m_virtualVoxelSize;
	//	return make_int3(p+make_float3(sign(p))*0.5f);
	//}

	//sdf block的编号依然是整型
	__device__ 
	int3 virtualWorldPosToSDFBlock(float3 virtualWorldPos) const {
		if (virtualWorldPos.x < 0) virtualWorldPos.x -= SDF_BLOCK_SIZE-0.5;//负数的四舍五入这里要补值
		if (virtualWorldPos.y < 0) virtualWorldPos.y -= SDF_BLOCK_SIZE-0.5;
		if (virtualWorldPos.z < 0) virtualWorldPos.z -= SDF_BLOCK_SIZE-0.5;

		return make_int3(
			virtualWorldPos.x/SDF_BLOCK_SIZE,
			virtualWorldPos.y/SDF_BLOCK_SIZE,
			virtualWorldPos.z/SDF_BLOCK_SIZE);
	}

	// Computes virtual voxel position of corner sample position
	__device__ 
	float3 SDFBlockToVirtualWorldPos(const int3& sdfBlock) const	{
		return make_float3(sdfBlock*SDF_BLOCK_SIZE);
	}

	__device__ 
	float3 virtualWorldPosToWorld(const float3& pos) const	{
		return pos*c_hashParams.m_virtualVoxelSize;
	}

	__device__ 
	float3 SDFBlockToWorld(const int3& sdfBlock) const	{
		return virtualWorldPosToWorld(SDFBlockToVirtualWorldPos(sdfBlock));
	}

	//__device__
	//	int3 testInRetouch_OneDim(int3 blockRetouch, int3 block) const{
	//	HashEntry entry = getHashEntryForSDFBlockPos(make_int3(blockRetouch.x, blockRetouch.y, blockRetouch.z));
	//	if (entry.ptr != FREE_ENTRY && d_SDFBlocks[entry.ptr].m_bRetouchIndex != -1)
	//	{
	//		return blockRetouch;
	//	}
	//	else {
	//		return block;
	//	}
	//}
	//__device__
	//	int3 testInRetouch_TwoDim(int3 blockRetouch, int3 block) const {
	//	HashEntry entry = getHashEntryForSDFBlockPos(make_int3(blockRetouch.x, blockRetouch.y, blockRetouch.z));
	//	if (entry.ptr != FREE_ENTRY && d_SDFBlocks[entry.ptr].m_bRetouchIndex != -1)
	//	{
	//		return blockRetouch;
	//	}
	//	else {
	//		int3 offset = block - blockRetouch;
	//		if (offset.x == 0)
	//		{
	//			int3 result1 = testInRetouch_OneDim(make_int3(blockRetouch.x, blockRetouch.y + 1, blockRetouch.z), block);
	//			if (result1.x == block.x && result1.y == block.y && result1.z == block.z)
	//			{
	//				int3 result2 = testInRetouch_OneDim(make_int3(blockRetouch.x, blockRetouch.y, blockRetouch.z + 1), block);
	//				if (result2.x == block.x && result2.y == block.y && result2.z == block.z)
	//				{
	//					return block;
	//				}
	//				else {
	//					return result2;
	//				}
	//			}
	//			else {
	//				return result1;
	//			}
	//		}
	//		if (offset.y == 0)
	//		{
	//			int3 result1 = testInRetouch_OneDim(make_int3(blockRetouch.x + 1, blockRetouch.y, blockRetouch.z), block);
	//			if (result1.x == block.x && result1.y == block.y && result1.z == block.z)
	//			{
	//				int3 result2 = testInRetouch_OneDim(make_int3(blockRetouch.x, blockRetouch.y, blockRetouch.z + 1), block);
	//				if (result2.x == block.x && result2.y == block.y && result2.z == block.z)
	//				{
	//					return block;
	//				}
	//				else {
	//					return result2;
	//				}
	//			}
	//			else {
	//				return result1;
	//			}
	//		}
	//		if (offset.z == 0)
	//		{
	//			int3 result1 = testInRetouch_OneDim(make_int3(blockRetouch.x, blockRetouch.y + 1, blockRetouch.z), block);
	//			if (result1.x == block.x && result1.y == block.y && result1.z == block.z)
	//			{
	//				int3 result2 = testInRetouch_OneDim(make_int3(blockRetouch.x + 1, blockRetouch.y, blockRetouch.z), block);
	//				if (result2.x == block.x && result2.y == block.y && result2.z == block.z)
	//				{
	//					return block;
	//				}
	//				else {
	//					return result2;
	//				}
	//			}
	//			else {
	//				return result1;
	//			}
	//		}
	//		return block;
	//	}
	//}
	//__device__
	//	int3 testInRetouch_ThreeDim(int3 blockRetouch, int3 block) const {
	//	HashEntry entry = getHashEntryForSDFBlockPos(make_int3(blockRetouch.x, blockRetouch.y, blockRetouch.z));
	//	if (entry.ptr != FREE_ENTRY && d_SDFBlocks[entry.ptr].m_bRetouchIndex != -1)
	//	{
	//		return blockRetouch;
	//	}
	//	else {
	//		int3 result1 = testInRetouch_TwoDim(make_int3(blockRetouch.x + 1, blockRetouch.y, blockRetouch.z), block);
	//		if (result1.x == block.x && result1.y == block.y && result1.z == block.z)
	//		{
	//			int3 result2 = testInRetouch_TwoDim(make_int3(blockRetouch.x, blockRetouch.y + 1, blockRetouch.z), block);
	//			if (result2.x == block.x && result2.y == block.y && result2.z == block.z)
	//			{
	//				int3 result3 = testInRetouch_TwoDim(make_int3(blockRetouch.x, blockRetouch.y, blockRetouch.z + 1), block);
	//				if (result3.x == block.x && result3.y == block.y && result3.z == block.z)
	//				{
	//					//到此说明依然没有找到被细化的，下面开始找一维的
	//					int3 result4 = testInRetouch_OneDim(make_int3(blockRetouch.x + 1, blockRetouch.y + 1, blockRetouch.z), block);
	//					if (result4.x == block.x && result4.y == block.y && result4.z == block.z)
	//					{
	//						int3 result5 = testInRetouch_OneDim(make_int3(blockRetouch.x + 1, blockRetouch.y, blockRetouch.z + 1), block);
	//						if (result5.x == block.x && result5.y == block.y && result5.z == block.z)
	//						{
	//							int3 result6 = testInRetouch_OneDim(make_int3(blockRetouch.x, blockRetouch.y + 1, blockRetouch.z + 1), block);
	//							if (result6.x == block.x && result6.y == block.y && result6.z == block.z)
	//							{
	//								return block;
	//							}
	//							else {
	//								return result6;
	//							}
	//						}
	//						else {
	//							return result5;
	//						}
	//					}
	//					else {
	//						return result4;
	//					}
	//				}
	//				else {
	//					return result3;
	//				}
	//			}
	//			else {
	//				return result2;
	//			}
	//		}
	//		else {
	//			return result1;
	//		}
	//	}
	//}

    //ok
	__device__ 
	int3 worldToSDFBlock(const float3& worldPos) const	{
		//扩大100倍后做四舍五入。这样的分界就是7.5
		float3 virtualWorld = worldToVirtualWorldPos(worldPos);
		int3 block = virtualWorldPosToSDFBlock(virtualWorld);

		HashEntry entry = getHashEntryForSDFBlockPos(make_int3(block.x, block.y, block.z));
		//return block;

		if (entry.ptr == FREE_ENTRY || d_SDFBlocks[entry.ptr].m_bRetouchIndex == -1)
		{
			return block;
		}
		else {
			float3 virtualWorld_Retouch = worldToVirtualWorldPos_Retouched(worldPos);
			int3 blockRetouch = virtualWorldPosToSDFBlock(virtualWorld_Retouch);
			return blockRetouch;
		}
		
			
		

		//int3 offset = block - blockRetouch; 
		//int difference = (offset.x + offset.y + offset.z);
		/////////////要重写
		//if (difference == 0)
		//{
		//	return block;
		//}
		//else if (difference == 1)
		//{
		//	return testInRetouch_OneDim(blockRetouch, block);
		//}
		//else if (difference == 2)
		//{
		//	return testInRetouch_TwoDim(blockRetouch, block);
		//}
		//else
		//{//difference == 3
		//	return testInRetouch_ThreeDim(blockRetouch, block);
		//}
		
		
	}

	__device__
	bool isSDFBlockInCameraFrustumApprox(const int3& sdfBlock) {
		
		bool isRetouched = isBlockRetouched(SDFBlockToVirtualWorldPos(sdfBlock));
		float voxelSize;
		int blockSize;
		if (isRetouched)
		{
			voxelSize = c_hashParams.m_virtualVoxelSize * 0.5;
			blockSize = SDF_BLOCK_SIZE * 2;
		}
		else {
			voxelSize = c_hashParams.m_virtualVoxelSize;
			blockSize = SDF_BLOCK_SIZE;
		}
		float3 posWorld = virtualWorldPosToWorld(SDFBlockToVirtualWorldPos(sdfBlock)) + voxelSize * 0.5f * (blockSize - 1.0f);
		return DepthCameraData::isInCameraFrustumApprox(c_hashParams.m_rigidTransformInverse, posWorld);
	}

	//! computes the (local) virtual voxel pos of an index; idx in [0;511]
	__device__ 
	int3 delinearizeVoxelIndex(uint idx) const	{
		uint x = idx % SDF_BLOCK_SIZE;
		uint y = (idx % (SDF_BLOCK_SIZE * SDF_BLOCK_SIZE)) / SDF_BLOCK_SIZE;
		uint z = idx / (SDF_BLOCK_SIZE * SDF_BLOCK_SIZE);	
		return make_int3(x,y,z);
	}

	//! computes the linearized index of a local virtual voxel pos; pos in [0;7]^3
	__device__ 
	uint linearizeLocalVoxelPos_Unretouched(const int3& localVoxelPos)	const {
		return  
			localVoxelPos.z * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE +
			localVoxelPos.y * SDF_BLOCK_SIZE +
			localVoxelPos.x;
	}
	__device__
		uint linearizeLocalVoxelPos_Retouched(const int3& localVoxelPos)	const {

		int3 base = make_int3(localVoxelPos.x / 2, localVoxelPos.y / 2, localVoxelPos.z / 2);
		int3 line = make_int3(localVoxelPos.x % 2, localVoxelPos.y % 2, localVoxelPos.z % 2);

		
		return

			base.z * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * 8  +
			base.y * SDF_BLOCK_SIZE * 8 +
			base.x * 8 +

			line.z * 2 * 2 +
			line.y * 2 +
			line.x;
	}
	__device__ 
	int virtualWorldPosToLocalSDFBlockIndex_Unretouched(const int3& virtualWorldPos) const	{
		
		int3 localVoxelPos = make_int3(
			virtualWorldPos.x % SDF_BLOCK_SIZE,
			virtualWorldPos.y % SDF_BLOCK_SIZE,
			virtualWorldPos.z % SDF_BLOCK_SIZE);

		if (localVoxelPos.x < 0) localVoxelPos.x += SDF_BLOCK_SIZE;
		if (localVoxelPos.y < 0) localVoxelPos.y += SDF_BLOCK_SIZE;
		if (localVoxelPos.z < 0) localVoxelPos.z += SDF_BLOCK_SIZE;

		return linearizeLocalVoxelPos_Unretouched(localVoxelPos);
	}
	__device__
		int virtualWorldPosToLocalSDFBlockIndex_Retouched(const float3& virtualWorldPos) const {

		//float3 virtualWorldPos2 = virtualWorldPos;
		//这一步先扩大双倍
		int3 virtualWorldPos_int_Double = make_int3(virtualWorldPos * 2);

		int3 localVoxelPos_int_Double = make_int3(
			virtualWorldPos_int_Double.x % (SDF_BLOCK_SIZE * 2),
			virtualWorldPos_int_Double.y % (SDF_BLOCK_SIZE * 2),
			virtualWorldPos_int_Double.z % (SDF_BLOCK_SIZE * 2));

		if (localVoxelPos_int_Double.x < 0) localVoxelPos_int_Double.x += (SDF_BLOCK_SIZE * 2);
		if (localVoxelPos_int_Double.y < 0) localVoxelPos_int_Double.y += (SDF_BLOCK_SIZE * 2);
		if (localVoxelPos_int_Double.z < 0) localVoxelPos_int_Double.z += (SDF_BLOCK_SIZE * 2);
		return linearizeLocalVoxelPos_Retouched(localVoxelPos_int_Double);
	}
	/*__device__ 
	int worldToLocalSDFBlockIndex(const float3& world) const	{
		float3 virtualVoxelPos = worldToVirtualVoxelPos(world);
		return virtualVoxelPosToLocalSDFBlockIndex(virtualVoxelPos);
	}*/


		//! returns the hash entry for a given worldPos; if there was no hash entry the returned entry will have a ptr with FREE_ENTRY set
	__device__ 
	HashEntry getHashEntry(const float3& worldPos) const	{
		//int3 blockID = worldToSDFVirtualVoxelPos(worldPos)/SDF_BLOCK_SIZE;	//position of sdf block
		int3 blockID = worldToSDFBlock(worldPos);
		return getHashEntryForSDFBlockPos(blockID);
	}


	__device__ 
		void deleteHashEntry(uint id) {
			deleteHashEntry(d_hash[id]);
	}

	__device__ 
		void deleteHashEntry(HashEntry& hashEntry) {
			hashEntry.pos = make_int3(0);
			hashEntry.offset = 0;
			hashEntry.ptr = FREE_ENTRY;
	}

	__device__ 
		bool voxelExists(const float3& worldPos) const	{
			HashEntry hashEntry = getHashEntry(worldPos);
			return (hashEntry.ptr != FREE_ENTRY);
	}

	/*__device__ 
		void deleteVoxel(uint blockId, uint voxelLocalId) {
		if (d_SDFBlocks[blockId].m_bRetouchIndex != -1)
		{
			deleteVoxel(d_SDFBlocks[blockId].voxels_retouched[voxelLocalId]);
		}
		else {
			deleteVoxel(d_SDFBlocks[blockId].voxels[voxelLocalId]);
		}
			
	}*/
	__device__
		void deleteVoxel(Voxel& v) const {
		v.color = make_uchar4(0, 0, 0, 0);
		v.weight = 0.0f;
		v.sdf = 0.0f;
	}

	__device__
		void deleteBlock(uint blockId) {

		for (int i = 0; i < SDF_BLOCK_SIZE*SDF_BLOCK_SIZE*SDF_BLOCK_SIZE; i++)
		{
			deleteVoxel(d_SDFBlocks[blockId].voxels[i]);

		}
		if (d_SDFBlocks[blockId].m_bRetouchIndex != -1)
		{
			//uint oldCounter = atomicAdd(&d_retouchCounter[0], 1);
			/*for (int i = 0; i < SDF_BLOCK_SIZE*SDF_BLOCK_SIZE*SDF_BLOCK_SIZE*8; i++)
			{
				deleteVoxel(d_SDFBlocks_Retouched[d_SDFBlocks[blockId].m_bRetouchIndex].voxels[i]);
			}*/

			//d_retouchDecision[oldCounter+1] = d_SDFBlocks[blockId].m_bRetouchIndex;
			
			d_SDFBlocks[blockId].m_bRetouchIndex = -1;
			d_SDFBlocks[blockId].m_bIsRetouched = false;
		}

	}
	__device__
		bool isBlockRetouched(const float3& virtualWorldPos) const {
		//可能不对
		bool isRetouched;

		float3 worldPos = virtualWorldPosToWorld(virtualWorldPos);
		HashEntry hashEntry = getHashEntry(worldPos);
		
		if (hashEntry.ptr == FREE_ENTRY) {
			isRetouched = false;
		}
		else {
			SDFBlock b = d_SDFBlocks[hashEntry.ptr];
			if (b.m_bRetouchIndex == -1)
			{
				isRetouched = false;
			}
			else {
				isRetouched = true;
			}
			//isRetouched = b.m_bIsRetouchedDown;
		}
		
		return isRetouched;
	}

	__device__ 
	Voxel getVoxel_World(const float3& worldPos) const	{
		//printf("getVoxel_World: x:%f, y:%f, z:%f\n", worldPos.x, worldPos.y, worldPos.z);
		HashEntry hashEntry = getHashEntry(worldPos);//ok
		Voxel v;
		if (hashEntry.ptr == FREE_ENTRY) {
			deleteVoxel(v);			
		} else {
			//这里要根据block是否细化来区分，
			SDFBlock b = d_SDFBlocks[hashEntry.ptr];
			if (b.m_bIsRetouched && b.m_bRetouchIndex!=-1)
			{//被细化，不可以被0.01整除
				
				//float3 virtualWorldPos = worldToVirtualWorldPos_Retouched(worldPos);//这里算到的virtualWorldPos是以0.5为最小单位的
				
				//int index = virtualWorldPosToLocalSDFBlockIndex_Retouched(virtualWorldPos);
				//printf("index: %d\n", index);
				float3 virtualWorldPos = worldToVirtualWorldPos_Retouched(worldPos);
				//printf("worldPos: x:%f,y:%f,z:%f, blockidx: %d \nvirtualWorldPos: x:%f,y:%f,z:%f\n", worldPos.x, worldPos.y, worldPos.z, hashEntry.ptr, virtualWorldPos.x, virtualWorldPos.y, virtualWorldPos.z);
				v = d_SDFBlocks_Retouched[b.m_bRetouchIndex].voxels[virtualWorldPosToLocalSDFBlockIndex_Retouched(virtualWorldPos)];
				//v = b.voxels_retouched[virtualWorldPosToLocalSDFBlockIndex_Retouched(virtualWorldPos)];
			}
			else {//未被细化,可以被0.01整除
				int3 virtualWorldPos = make_int3(worldToVirtualWorldPos(worldPos));
				//printf("worldPos: x:%f,y:%f,z:%f, blockidx: %d \nvirtualWorldPos: x:%d,y:%d,z:%d\n", worldPos.x, worldPos.y, worldPos.z, hashEntry.ptr, virtualWorldPos.x, virtualWorldPos.y, virtualWorldPos.z);
				v = b.voxels[virtualWorldPosToLocalSDFBlockIndex_Unretouched(virtualWorldPos)];
			}
			
			//v = d_SDFBlocks[hashEntry.ptr + virtualVoxelPosToLocalSDFBlockIndex(virtualWorldPos)];
		}
		return v;
	}

	//__device__ 
	//Voxel getVoxel_VirtualWorld(const float3& virtualWorldPos) const	{
	//	
	//	HashEntry hashEntry = getHashEntryForSDFBlockPos(virtualWorldPosToSDFBlock(virtualWorldPos));
	//	Voxel v;
	//	if (hashEntry.ptr == FREE_ENTRY) {
	//		deleteVoxel(v);			
	//	} else {
	//		SDFBlock b = d_SDFBlocks[hashEntry.ptr];
	//		if (b.m_bIsRetouched)
	//		{//被细化，不可以被0.01整除
	//			v = b.voxels[virtualWorldPosToLocalSDFBlockIndex_Retouched(virtualWorldPos)];
	//		}
	//		else {//未被细化,可以被0.01整除
	//			v = b.voxels[virtualWorldPosToLocalSDFBlockIndex_Unretouched(make_int3(virtualWorldPos))];
	//		}
	//		//v = d_SDFBlocks[hashEntry.ptr + virtualWorldPosToLocalSDFBlockIndex(virtualWorldPos)];
	//	}
	//	return v;
	//}
	
	//__device__ 
	//void setVoxel(const float3& virtualWorldPos, Voxel& voxelInput) const {
	//	HashEntry hashEntry = getHashEntryForSDFBlockPos(virtualWorldPosToSDFBlock(virtualWorldPos));
	//	if (hashEntry.ptr != FREE_ENTRY) {
	//		SDFBlock b = d_SDFBlocks[hashEntry.ptr];
	//		if (b.m_bIsRetouchedDown)
	//		{//被细化，不可以被0.01整除
	//			b.voxels_retouched[virtualWorldPosToLocalSDFBlockIndex_Retouched(virtualWorldPos)] = voxelInput;
	//		}
	//		else {//未被细化,可以被0.01整除
	//			b.voxels[virtualWorldPosToLocalSDFBlockIndex_Unretouched(make_int3(virtualWorldPos))] = voxelInput;
	//		}
	//		//d_SDFBlocks[hashEntry.ptr + virtualVoxelPosToLocalSDFBlockIndex(virtualVoxelPos)] = voxelInput;
	//	}
	//}

	//! returns the hash entry for a given sdf block id; if there was no hash entry the returned entry will have a ptr with FREE_ENTRY set
	__device__ 
	HashEntry getHashEntryForSDFBlockPos(const int3& sdfBlock) const
	{
		uint h = computeHashPos(sdfBlock);			//hash bucket
		uint hp = h * HASH_BUCKET_SIZE;	//hash position

		HashEntry entry;
		entry.pos = sdfBlock;
		entry.offset = 0;
		entry.ptr = FREE_ENTRY;

		for (uint j = 0; j < HASH_BUCKET_SIZE; j++) {
			uint i = j + hp;
			HashEntry curr = d_hash[i];
			if (curr.pos.x == entry.pos.x && curr.pos.y == entry.pos.y && curr.pos.z == entry.pos.z && curr.ptr != FREE_ENTRY) {
				return curr;
			}
		}

#ifdef HANDLE_COLLISIONS
		const uint idxLastEntryInBucket = (h+1)*HASH_BUCKET_SIZE - 1;
		int i = idxLastEntryInBucket;	//start with the last entry of the current bucket
		HashEntry curr;
		//traverse list until end: memorize idx at list end and memorize offset from last element of bucket to list end

		unsigned int maxIter = 0;
		uint g_MaxLoopIterCount = c_hashParams.m_hashMaxCollisionLinkedListSize;
		#pragma unroll 1 
		while (maxIter < g_MaxLoopIterCount) {
			curr = d_hash[i];

			if (curr.pos.x == entry.pos.x && curr.pos.y == entry.pos.y && curr.pos.z == entry.pos.z && curr.ptr != FREE_ENTRY) {
				return curr;
			}

			if (curr.offset == 0) {	//we have found the end of the list
				break;
			}
			i = idxLastEntryInBucket + curr.offset;						//go to next element in the list
			i %= (HASH_BUCKET_SIZE * c_hashParams.m_hashNumBuckets);	//check for overflow

			maxIter++;
		}
#endif
		return entry;
	}

	//for histogram (no collision traversal)
	__device__ 
	unsigned int getNumHashEntriesPerBucket(unsigned int bucketID) {
		unsigned int h = 0;
		for (uint i = 0; i < HASH_BUCKET_SIZE; i++) {
			if (d_hash[bucketID*HASH_BUCKET_SIZE+i].ptr != FREE_ENTRY) {
				h++;
			}
		} 
		return h;
	}

	//for histogram (collisions traversal only)
	__device__ 
	unsigned int getNumHashLinkedList(unsigned int bucketID) {
		unsigned int listLen = 0;

#ifdef HANDLE_COLLISIONS
		const uint idxLastEntryInBucket = (bucketID+1)*HASH_BUCKET_SIZE - 1;
		unsigned int i = idxLastEntryInBucket;	//start with the last entry of the current bucket
		//int offset = 0;
		HashEntry curr;	curr.offset = 0;
		//traverse list until end: memorize idx at list end and memorize offset from last element of bucket to list end

		unsigned int maxIter = 0;
		uint g_MaxLoopIterCount = c_hashParams.m_hashMaxCollisionLinkedListSize;
		#pragma unroll 1 
		while (maxIter < g_MaxLoopIterCount) {
			//offset = curr.offset;
			//curr = getHashEntry(g_Hash, i);
			curr = d_hash[i];

			if (curr.offset == 0) {	//we have found the end of the list
				break;
			}
			i = idxLastEntryInBucket + curr.offset;		//go to next element in the list
			i %= (HASH_BUCKET_SIZE * c_hashParams.m_hashNumBuckets);	//check for overflow
			listLen++;

			maxIter++;
		}
#endif
		
		return listLen;
	}



	__device__
	uint consumeHeap() {
		uint addr = atomicSub(&d_heapCounter[0], 1);
		//TODO MATTHIAS check some error handling?
		return d_heap[addr];
	}
	__device__
	void appendHeap(uint ptr) {
		uint addr = atomicAdd(&d_heapCounter[0], 1);
		//TODO MATTHIAS check some error handling?
		d_heap[addr+1] = ptr;
	}
	//pos in SDF block coordinates
	__device__
	void allocBlock(const int3& pos) {


		uint h = computeHashPos(pos);				//hash bucket
		uint hp = h * HASH_BUCKET_SIZE;	//hash position

		int firstEmpty = -1;
		for (uint j = 0; j < HASH_BUCKET_SIZE; j++) {
			uint i = j + hp;		
			const HashEntry& curr = d_hash[i];

			//in that case the SDF-block is already allocated and corresponds to the current position -> exit thread
			if (curr.pos.x == pos.x && curr.pos.y == pos.y && curr.pos.z == pos.z && curr.ptr != FREE_ENTRY) {
				return;
			}

			//store the first FREE_ENTRY hash entry
			if (firstEmpty == -1 && curr.ptr == FREE_ENTRY) {
				firstEmpty = i;
			}
		}


#ifdef HANDLE_COLLISIONS
		//updated variables as after the loop
		const uint idxLastEntryInBucket = (h+1)*HASH_BUCKET_SIZE - 1;	//get last index of bucket
		uint i = idxLastEntryInBucket;											//start with the last entry of the current bucket
		//int offset = 0;
		HashEntry curr;	curr.offset = 0;
		//traverse list until end: memorize idx at list end and memorize offset from last element of bucket to list end
		//int k = 0;

		unsigned int maxIter = 0;
		uint g_MaxLoopIterCount = c_hashParams.m_hashMaxCollisionLinkedListSize;
		#pragma  unroll 1 
		while (maxIter < g_MaxLoopIterCount) {
			//offset = curr.offset;
			curr = d_hash[i];	//TODO MATTHIAS do by reference
			if (curr.pos.x == pos.x && curr.pos.y == pos.y && curr.pos.z == pos.z && curr.ptr != FREE_ENTRY) {
				return;
			}
			if (curr.offset == 0) {	//we have found the end of the list
				break;
			}
			i = idxLastEntryInBucket + curr.offset;		//go to next element in the list
			i %= (HASH_BUCKET_SIZE * c_hashParams.m_hashNumBuckets);	//check for overflow

			maxIter++;
		}
#endif

		if (firstEmpty != -1) {	//if there is an empty entry and we haven't allocated the current entry before
			//int prevValue = 0;
			//InterlockedExchange(d_hashBucketMutex[h], LOCK_ENTRY, prevValue);	//lock the hash bucket
			int prevValue = atomicExch(&d_hashBucketMutex[h], LOCK_ENTRY);
			if (prevValue != LOCK_ENTRY) {	//only proceed if the bucket has been locked
				HashEntry& entry = d_hash[firstEmpty];
				entry.pos = pos;
				entry.offset = NO_OFFSET;		
				entry.ptr = consumeHeap() ;	//memory alloc//* SDF_BLOCK_SIZE*SDF_BLOCK_SIZE*SDF_BLOCK_SIZE
			}
			return;
		}

#ifdef HANDLE_COLLISIONS
		//if (i != idxLastEntryInBucket) return;
		int offset = 0;
		//linear search for free entry

		maxIter = 0;
		#pragma  unroll 1 
		while (maxIter < g_MaxLoopIterCount) {
			offset++;
			i = (idxLastEntryInBucket + offset) % (HASH_BUCKET_SIZE * c_hashParams.m_hashNumBuckets);	//go to next hash element
			if ((offset % HASH_BUCKET_SIZE) == 0) continue;			//cannot insert into a last bucket element (would conflict with other linked lists)
			curr = d_hash[i];
			//if (curr.pos.x == pos.x && curr.pos.y == pos.y && curr.pos.z == pos.z && curr.ptr != FREE_ENTRY) {
			//	return;
			//} 
			if (curr.ptr == FREE_ENTRY) {	//this is the first free entry
				//int prevValue = 0;
				//InterlockedExchange(g_HashBucketMutex[h], LOCK_ENTRY, prevValue);	//lock the original hash bucket
				int prevValue = atomicExch(&d_hashBucketMutex[h], LOCK_ENTRY);
				if (prevValue != LOCK_ENTRY) {
					HashEntry lastEntryInBucket = d_hash[idxLastEntryInBucket];
					h = i / HASH_BUCKET_SIZE;
					//InterlockedExchange(g_HashBucketMutex[h], LOCK_ENTRY, prevValue);	//lock the hash bucket where we have found a free entry
					prevValue = atomicExch(&d_hashBucketMutex[h], LOCK_ENTRY);
					if (prevValue != LOCK_ENTRY) {	//only proceed if the bucket has been locked
						HashEntry& entry = d_hash[i];
						entry.pos = pos;
						entry.offset = lastEntryInBucket.offset;		
						entry.ptr = consumeHeap();	//memory alloc// * SDF_BLOCK_SIZE*SDF_BLOCK_SIZE*SDF_BLOCK_SIZE

						lastEntryInBucket.offset = offset;
						d_hash[idxLastEntryInBucket] = lastEntryInBucket;
						//setHashEntry(g_Hash, idxLastEntryInBucket, lastEntryInBucket);
					}
				} 
				return;	//bucket was already locked
			}

			maxIter++;
		} 
#endif
	}

	
	//!inserts a hash entry without allocating any memory: used by streaming: TODO MATTHIAS check the atomics in this function
	__device__
	bool insertHashEntry(HashEntry entry)
	{
		uint h = computeHashPos(entry.pos);
		uint hp = h * HASH_BUCKET_SIZE;

		for (uint j = 0; j < HASH_BUCKET_SIZE; j++) {
			uint i = j + hp;		
			//const HashEntry& curr = d_hash[i];
			int prevWeight = 0;
			//InterlockedCompareExchange(hash[3*i+2], FREE_ENTRY, LOCK_ENTRY, prevWeight);
			prevWeight = atomicCAS(&d_hash[i].ptr, FREE_ENTRY, LOCK_ENTRY);
			if (prevWeight == FREE_ENTRY) {
				d_hash[i] = entry;
				//setHashEntry(hash, i, entry);
				return true;
			}
		}

#ifdef HANDLE_COLLISIONS
		//updated variables as after the loop
		const uint idxLastEntryInBucket = (h+1)*HASH_BUCKET_SIZE - 1;	//get last index of bucket

		uint i = idxLastEntryInBucket;											//start with the last entry of the current bucket
		HashEntry curr;

		unsigned int maxIter = 0;
		//[allow_uav_condition]
		uint g_MaxLoopIterCount = c_hashParams.m_hashMaxCollisionLinkedListSize;
		#pragma  unroll 1 
		while (maxIter < g_MaxLoopIterCount) {									//traverse list until end // why find the end? we you are inserting at the start !!!
			//curr = getHashEntry(hash, i);
			curr = d_hash[i];	//TODO MATTHIAS do by reference
			if (curr.offset == 0) break;									//we have found the end of the list
			i = idxLastEntryInBucket + curr.offset;							//go to next element in the list
			i %= (HASH_BUCKET_SIZE * c_hashParams.m_hashNumBuckets);	//check for overflow

			maxIter++;
		}

		maxIter = 0;
		int offset = 0;
		#pragma  unroll 1 
		while (maxIter < g_MaxLoopIterCount) {													//linear search for free entry
			offset++;
			uint i = (idxLastEntryInBucket + offset) % (HASH_BUCKET_SIZE * c_hashParams.m_hashNumBuckets);	//go to next hash element
			if ((offset % HASH_BUCKET_SIZE) == 0) continue;										//cannot insert into a last bucket element (would conflict with other linked lists)

			int prevWeight = 0;
			//InterlockedCompareExchange(hash[3*i+2], FREE_ENTRY, LOCK_ENTRY, prevWeight);		//check for a free entry
			uint* d_hashUI = (uint*)d_hash;
			prevWeight = prevWeight = atomicCAS(&d_hashUI[3*idxLastEntryInBucket+1], (uint)FREE_ENTRY, (uint)LOCK_ENTRY);
			if (prevWeight == FREE_ENTRY) {														//if free entry found set prev->next = curr & curr->next = prev->next
				//[allow_uav_condition]
				//while(hash[3*idxLastEntryInBucket+2] == LOCK_ENTRY); // expects setHashEntry to set the ptr last, required because pos.z is packed into the same value -> prev->next = curr -> might corrput pos.z

				HashEntry lastEntryInBucket = d_hash[idxLastEntryInBucket];			//get prev (= lastEntry in Bucket)

				int newOffsetPrev = (offset << 16) | (lastEntryInBucket.pos.z & 0x0000ffff);	//prev->next = curr (maintain old z-pos)
				int oldOffsetPrev = 0;
				//InterlockedExchange(hash[3*idxLastEntryInBucket+1], newOffsetPrev, oldOffsetPrev);	//set prev offset atomically
				uint* d_hashUI = (uint*)d_hash;
				oldOffsetPrev = prevWeight = atomicExch(&d_hashUI[3*idxLastEntryInBucket+1], newOffsetPrev);
				entry.offset = oldOffsetPrev >> 16;													//remove prev z-pos from old offset

				//setHashEntry(hash, i, entry);														//sets the current hashEntry with: curr->next = prev->next
				d_hash[i] = entry;
				return true;
			}

			maxIter++;
		} 
#endif

		return false;
	}



	//! deletes a hash entry position for a given sdfBlock index (returns true uppon successful deletion; otherwise returns false)
	__device__
	bool deleteHashEntryElement(const int3& sdfBlock) {
		uint h = computeHashPos(sdfBlock);	//hash bucket
		uint hp = h * HASH_BUCKET_SIZE;		//hash position

		for (uint j = 0; j < HASH_BUCKET_SIZE; j++) {
			uint i = j + hp;
			const HashEntry& curr = d_hash[i];
			if (curr.pos.x == sdfBlock.x && curr.pos.y == sdfBlock.y && curr.pos.z == sdfBlock.z && curr.ptr != FREE_ENTRY) {
#ifndef HANDLE_COLLISIONS
				const uint linBlockSize = 1;// SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;
				appendHeap(curr.ptr / linBlockSize);
				//heapAppend.Append(curr.ptr / linBlockSize);
				deleteHashEntry(i);
				return true;
#endif
#ifdef HANDLE_COLLISIONS
				if (curr.offset != 0) {	//if there was a pointer set it to the next list element
					//int prevValue = 0;
					//InterlockedExchange(bucketMutex[h], LOCK_ENTRY, prevValue);	//lock the hash bucket
					int prevValue = atomicExch(&d_hashBucketMutex[h], LOCK_ENTRY);
					if (prevValue == LOCK_ENTRY)	return false;
					if (prevValue != LOCK_ENTRY) {
						const uint linBlockSize = 1;// SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;
						appendHeap(curr.ptr / linBlockSize);
						//heapAppend.Append(curr.ptr / linBlockSize);
						int nextIdx = (i + curr.offset) % (HASH_BUCKET_SIZE*c_hashParams.m_hashNumBuckets);
						//setHashEntry(hash, i, getHashEntry(hash, nextIdx));
						d_hash[i] = d_hash[nextIdx];
						deleteHashEntry(nextIdx);
						return true;
					}
				} else {
					const uint linBlockSize = 1;// SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;
					appendHeap(curr.ptr / linBlockSize);
					//heapAppend.Append(curr.ptr / linBlockSize);
					deleteHashEntry(i);
					return true;
				}
#endif	//HANDLE_COLLSISION
			}
		}	
#ifdef HANDLE_COLLISIONS
		const uint idxLastEntryInBucket = (h+1)*HASH_BUCKET_SIZE - 1;
		int i = idxLastEntryInBucket;
		HashEntry curr;
		curr = d_hash[i];
		int prevIdx = i;
		i = idxLastEntryInBucket + curr.offset;							//go to next element in the list
		i %= (HASH_BUCKET_SIZE * c_hashParams.m_hashNumBuckets);	//check for overflow

		unsigned int maxIter = 0;
		uint g_MaxLoopIterCount = c_hashParams.m_hashMaxCollisionLinkedListSize;

		#pragma  unroll 1 
		while (maxIter < g_MaxLoopIterCount) {
			curr = d_hash[i];
			//found that dude that we need/want to delete
			if (curr.pos.x == sdfBlock.x && curr.pos.y == sdfBlock.y && curr.pos.z == sdfBlock.z && curr.ptr != FREE_ENTRY) {
				//int prevValue = 0;
				//InterlockedExchange(bucketMutex[h], LOCK_ENTRY, prevValue);	//lock the hash bucket
				int prevValue = atomicExch(&d_hashBucketMutex[h], LOCK_ENTRY);
				if (prevValue == LOCK_ENTRY)	return false;
				if (prevValue != LOCK_ENTRY) {
					const uint linBlockSize = 1;// SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;
					appendHeap(curr.ptr / linBlockSize);
					//heapAppend.Append(curr.ptr / linBlockSize);
					deleteHashEntry(i);
					HashEntry prev = d_hash[prevIdx];				
					prev.offset = curr.offset;
					//setHashEntry(hash, prevIdx, prev);
					d_hash[prevIdx] = prev;
					return true;
				}
			}

			if (curr.offset == 0) {	//we have found the end of the list
				return false;	//should actually never happen because we need to find that guy before
			}
			prevIdx = i;
			i = idxLastEntryInBucket + curr.offset;		//go to next element in the list
			i %= (HASH_BUCKET_SIZE * c_hashParams.m_hashNumBuckets);	//check for overflow

			maxIter++;
		}
#endif	// HANDLE_COLLSISION
		return false;
	}

#endif	//CUDACC

	uint*		d_heap;						//heap that manages free memory
	uint*		d_heapCounter;				//single element; used as an atomic counter (points to the next free block)
	int*		d_hashDecision;				//
	int*		d_hashDecisionPrefix;		//
	HashEntry*	d_hash;						//hash that stores pointers to sdf blocks
	HashEntry*	d_hashCompactified;			//same as before except that only valid pointers are there
	int*		d_hashCompactifiedCounter;	//atomic counter to add compactified entries atomically 
	SDFBlock*	d_SDFBlocks;				//sub-blocks that contain 8x8x8 voxels (linearized); are allocated by heap
	
	SDFBlock_Retouched* d_SDFBlocks_Retouched;
	uint*       d_retouchCounter;           //计数
	int*        d_retouchDecision;          //描述


	int*		d_hashBucketMutex;			//binary flag per hash bucket; used for allocation to atomically lock a bucket
	bool		m_bIsOnGPU;					//the class be be used on both cpu and gpu
};