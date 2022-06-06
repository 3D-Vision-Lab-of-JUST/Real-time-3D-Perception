
#include <cutil_inline.h>
#include <cutil_math.h>
#include "VoxelUtilHashSDF.h"
#include "RayCastSDFUtil.h"
#include "MarchingCubesSDFUtil.h"

using namespace std;


__global__ void resetMarchingCubesKernel(MarchingCubesData data) 
{
	*data.d_numTriangles = 0;
}

__global__ void extractIsoBlocksKernel(const HashDataStruct& hashData, const HashParams& hashParams, vector<float4>* mask_voxels)
{
	/*const HashEntry& entry = hashData.d_hashCompactified[blockIdx.x];
	uint blockIdx = entry.ptr;
	uint x = threadIdx.x;
	uint y = threadIdx.y;
	uint z = threadIdx.z;

	if (entry.ptr == FREE_ENTRY)
	{
		return;
	}
	else {
		float3 pi_base = hashData.SDFBlockToVirtualWorldPos(entry.pos);
		float3 pi = pi_base + make_float3(x, y, z);

		int retouchedNum = 1;
		if (hashData.d_SDFBlocks[blockIdx].m_bRetouchIndex != -1)
		{
			retouchedNum = 8;
		}
		for (int i = 0; i < retouchedNum; i++)
		{
			float3 newpi;
			if (i == 0)
			{
				newpi = pi;
			}
			else if (i == 1)
			{
				newpi = pi + make_float3(0.5, 0, 0);
			}
			else if (i == 2)
			{
				newpi = pi + make_float3(0, 0.5, 0);
			}
			else if (i == 3)
			{
				newpi = pi + make_float3(0.5, 0.5, 0);
			}
			else if (i == 4)
			{
				newpi = pi + make_float3(0, 0, 0.5);
			}
			else if (i == 5)
			{
				newpi = pi + make_float3(0.5, 0, 0.5);
			}
			else if (i == 6)
			{
				newpi = pi + make_float3(0, 0.5, 0.5);
			}
			else if (i == 7)
			{
				newpi = pi + make_float3(0.5, 0.5, 0.5);
			}
			float3 worldPos = newpi * 0.01;
			Voxel v = hashData.getVoxel_World(worldPos);
			if (v.weight > 0)
			{
				float4 voxel = make_float4(worldPos.x, worldPos.y, worldPos.z, 0);
				(*mask_voxels).push_back(voxel);
			}
		}
	}*/

	
}

__global__ void extractIsoSurfaceKernel(HashDataStruct hashData, RayCastData rayCastData, MarchingCubesData data)
{
	uint idx = blockIdx.x;

	const HashEntry& entry = hashData.d_hash[idx];
	if (entry.ptr != FREE_ENTRY) {
		float3 pi_base = hashData.SDFBlockToVirtualWorldPos(entry.pos);
		float3 pi = pi_base + make_float3(threadIdx);
		//uint voxelIndex = hashData.linearizeLocalVoxelPos_Unretouched(make_int3((int)threadIdx.x, (int)threadIdx.y, (int)threadIdx.z));
		
		if (hashData.d_SDFBlocks[entry.ptr].m_bRetouchIndex!=-1)
		{
			float3 pi1 = pi + make_float3(0, 0, 0);
			float3 pi2 = pi + make_float3(0.5, 0, 0);
			float3 pi3 = pi + make_float3(0, 0.5, 0);
			float3 pi4 = pi + make_float3(0, 0, 0.5);
			float3 pi5 = pi + make_float3(0.5, 0.5, 0);
			float3 pi6 = pi + make_float3(0.5, 0, 0.5);
			float3 pi7 = pi + make_float3(0, 0.5, 0.5);
			float3 pi8 = pi + make_float3(0.5, 0.5, 0.5);

			float3 worldPos1 = hashData.virtualWorldPosToWorld(pi1);
			float3 worldPos2 = hashData.virtualWorldPosToWorld(pi2);
			float3 worldPos3 = hashData.virtualWorldPosToWorld(pi3);
			float3 worldPos4 = hashData.virtualWorldPosToWorld(pi4);
			float3 worldPos5 = hashData.virtualWorldPosToWorld(pi5);
			float3 worldPos6 = hashData.virtualWorldPosToWorld(pi6);
			float3 worldPos7 = hashData.virtualWorldPosToWorld(pi7);
			float3 worldPos8 = hashData.virtualWorldPosToWorld(pi8);

			data.extractIsoSurfaceAtPosition(worldPos1, hashData, rayCastData);
			data.extractIsoSurfaceAtPosition(worldPos2, hashData, rayCastData);
			data.extractIsoSurfaceAtPosition(worldPos3, hashData, rayCastData);
			data.extractIsoSurfaceAtPosition(worldPos4, hashData, rayCastData);
			data.extractIsoSurfaceAtPosition(worldPos5, hashData, rayCastData);
			data.extractIsoSurfaceAtPosition(worldPos6, hashData, rayCastData);
			data.extractIsoSurfaceAtPosition(worldPos7, hashData, rayCastData);
			data.extractIsoSurfaceAtPosition(worldPos8, hashData, rayCastData);
		}
		else {
			float3 worldPos = hashData.virtualWorldPosToWorld(pi);
			data.extractIsoSurfaceAtPosition(worldPos, hashData, rayCastData);
		}
		

	}
}

extern "C" void resetMarchingCubesCUDA(MarchingCubesData& data)
{
	const dim3 blockSize(1, 1, 1);
	const dim3 gridSize(1, 1, 1);

	resetMarchingCubesKernel<<<gridSize, blockSize>>>(data);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}
extern "C" void extractIsoBlocksCUDA(const HashDataStruct& hashData, vector<float4>* mask_voxels, const HashParams& hashParams)
{
	const dim3 gridSize(hashParams.m_numSDFBlocks, 1);
	const dim3 blockSize(hashParams.m_SDFBlockSize, hashParams.m_SDFBlockSize, hashParams.m_SDFBlockSize);
	//extractIsoBboxKernel << <gridSize, blockSize >> >(hashData, mask_voxels);
	extractIsoBlocksKernel <<<gridSize, blockSize >>>(hashData, hashParams, mask_voxels);
}
uint linearizeVoxelPos(const int3& virtualVoxelPos)
{
	return
		virtualVoxelPos.z * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE +
		virtualVoxelPos.y * SDF_BLOCK_SIZE +
		virtualVoxelPos.x;
}
int virtualVoxelPosToLocalSDFBlockIndex(const int3& virtualVoxelPos)
{
	int3 localVoxelPos = make_int3(
		virtualVoxelPos.x % SDF_BLOCK_SIZE,
		virtualVoxelPos.y % SDF_BLOCK_SIZE,
		virtualVoxelPos.z % SDF_BLOCK_SIZE);

	if (localVoxelPos.x < 0) localVoxelPos.x += SDF_BLOCK_SIZE;
	if (localVoxelPos.y < 0) localVoxelPos.y += SDF_BLOCK_SIZE;
	if (localVoxelPos.z < 0) localVoxelPos.z += SDF_BLOCK_SIZE;

	return linearizeVoxelPos(localVoxelPos);
}
extern "C" void extractIsoSurfaceCUDA(const HashDataStruct& hashData, const RayCastData& rayCastData, const MarchingCubesParams& params, MarchingCubesData& data, vector<float4>* mask_voxels, const HashParams& hashParams)
{
	unsigned int numHashEntries = hashParams.m_hashBucketSize *hashParams.m_hashNumBuckets;
	HashEntry* hashCPU = new HashEntry[numHashEntries];
	cudaMemcpy(hashCPU, hashData.d_hash, sizeof(HashEntry)*numHashEntries, cudaMemcpyDeviceToHost);
	SDFBlock* sdfBlocksCPU = new SDFBlock[hashParams.m_numSDFBlocks];
	cudaMemcpy(sdfBlocksCPU, hashData.d_SDFBlocks, sizeof(SDFBlock)*hashParams.m_numSDFBlocks, cudaMemcpyDeviceToHost);

	for (int i = 0; i < params.m_hashNumBuckets*params.m_hashBucketSize; i++)
	{
		for (int x = 0; x < params.m_sdfBlockSize; x++)
		{
			for (int y = 0; y < params.m_sdfBlockSize; y++)
			{
				for (int z = 0; z < params.m_sdfBlockSize; z++)
				{

					HashEntry& entry_cpu = hashCPU[i];

					if (entry_cpu.ptr != FREE_ENTRY) {
						int3 pi_base = entry_cpu.pos * 8;
						int3 pi = pi_base + make_int3(x, y, z);
						float3 worldPos = make_float3(pi) * hashParams.m_virtualVoxelSize;

						float3 p = worldPos / hashParams.m_virtualVoxelSize;
						int3 virtualVoxelPos = make_int3(p + make_float3(sign(p))*0.5f);

						Voxel v = sdfBlocksCPU[entry_cpu.ptr].voxels[virtualVoxelPosToLocalSDFBlockIndex(virtualVoxelPos)];

						//这里根据v的信息来搞boundingbox
						if (v.weight> 0 && v.color.w>0)
						{
							//这个在显存中的
							float4 voxel = make_float4(worldPos.x, worldPos.y, worldPos.z, v.color.w);

							(*mask_voxels).push_back(voxel);

						}
					}
				}
			}

		}
	}

	const dim3 gridSize(params.m_hashNumBuckets*params.m_hashBucketSize, 1, 1);
	const dim3 blockSize(params.m_sdfBlockSize, params.m_sdfBlockSize, params.m_sdfBlockSize);
	//extractIsoBboxKernel << <gridSize, blockSize >> >(hashData, mask_voxels);
	extractIsoSurfaceKernel <<<gridSize, blockSize >>>(hashData, rayCastData, data);

	
	
	
	
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}