#include "stdafx.h"

#include "VoxelUtilHashSDF.h"
#include "RayCastSDFUtil.h"
#include "CUDAMarchingCubesHashSDF.h"

#include <iostream>
#include <fstream>
extern "C" void resetMarchingCubesCUDA(MarchingCubesData& data);
extern "C" void extractIsoSurfaceCUDA(const HashDataStruct& hashData,
										 const RayCastData& rayCastData,
										 const MarchingCubesParams& params,
										 MarchingCubesData& data,
                                         vector<float4>* mask_voxels, const HashParams& hashParams);
extern "C" void extractIsoBlocksCUDA(const HashDataStruct& hashData, vector<float4>* mask_voxels, const HashParams& hashParams);

void CUDAMarchingCubesHashSDF::create(const MarchingCubesParams& params)
{
	m_params = params;
	m_data.allocate(m_params);

	resetMarchingCubesCUDA(m_data);
}

void CUDAMarchingCubesHashSDF::destroy(void)
{
	m_data.free();
}

void CUDAMarchingCubesHashSDF::copyTrianglesToCPU() {

	MarchingCubesData cpuData = m_data.copyToCPU();

	unsigned int nTriangles = *cpuData.d_numTriangles;

	//std::cout << "Marching Cubes: #triangles = " << nTriangles << std::endl;

	if (nTriangles != 0) {
		unsigned int baseIdx = (unsigned int)m_meshData.m_Vertices.size();
		m_meshData.m_Vertices.resize(baseIdx + 3 * nTriangles);
		m_meshData.m_Colors.resize(baseIdx + 3 * nTriangles);

		vec3f* vc = (vec3f*)cpuData.d_triangles;
		for (unsigned int i = 0; i < 3 * nTriangles; i++) {
			m_meshData.m_Vertices[baseIdx + i] = vc[2 * i + 0];
			m_meshData.m_Colors[baseIdx + i] = vec4f(vc[2 * i + 1]);
		}
	}
	cpuData.free();
}

void CUDAMarchingCubesHashSDF::saveMesh(const std::string& filename, const mat4f *transform /*= NULL*/, bool overwriteExistingFile /*= false*/)
{
	std::string folder = util::directoryFromPath(filename);
	if (!util::directoryExists(folder)) {
		util::makeDirectory(folder);
	}

	std::string actualFilename = filename;
	if (!overwriteExistingFile) {
		while (util::fileExists(actualFilename)) {
			std::string path = util::directoryFromPath(actualFilename);
			std::string curr = util::fileNameFromPath(actualFilename);
			std::string ext = util::getFileExtension(curr);
			curr = util::removeExtensions(curr);
			std::string base = util::getBaseBeforeNumericSuffix(curr);
			unsigned int num = util::getNumericSuffix(curr);
			if (num == (unsigned int)-1) {
				num = 0;
			}
			actualFilename = path + base + std::to_string(num + 1) + "." + ext;
		}
	}

	//create index buffer (required for merging the triangle soup)
	m_meshData.m_FaceIndicesVertices.resize(m_meshData.m_Vertices.size());
	for (unsigned int i = 0; i < (unsigned int)m_meshData.m_Vertices.size()/3; i++) {
		m_meshData.m_FaceIndicesVertices[i][0] = 3*i+0;
		m_meshData.m_FaceIndicesVertices[i][1] = 3*i+1;
		m_meshData.m_FaceIndicesVertices[i][2] = 3*i+2;
	}
	std::cout << "size before:\t" << m_meshData.m_Vertices.size() << std::endl;

	//std::cout << "saving initial mesh...";
	//MeshIOf::saveToFile("./Scans/scan_initial.ply", m_meshData);
	//std::cout << "done!" << std::endl;
	
	//m_meshData.removeDuplicateVertices();
	//m_meshData.mergeCloseVertices(0.00001f);
	std::cout << "merging close vertices... ";
	m_meshData.mergeCloseVertices(0.00001f, true);
	std::cout << "done!" << std::endl;
	std::cout << "removing duplicate faces... ";
	m_meshData.removeDuplicateFaces();
	std::cout << "done!" << std::endl;

	std::cout << "size after:\t" << m_meshData.m_Vertices.size() << std::endl;

	if (transform) {
		m_meshData.applyTransform(*transform);
	}

	std::cout << "saving mesh (" << actualFilename << ") ...";
	MeshIOf::saveToFile(actualFilename, m_meshData);
	std::cout << "done!" << std::endl;

	clearMeshBuffer();
	
}

struct myPoint3d
{
	float x;
	float y;
	float z;
};
struct blockItem3d
{
	myPoint3d minPoint;
	myPoint3d maxPoint;
	vector<myPoint3d> pointIndex;
	int maskIdx;
};
void GenBoundingBox3d(vector<float4> src, vector<myPoint3d>& bbox_min, vector<myPoint3d>& bbox_max)
{
	vector<blockItem3d> blockItemVec;

	for (int i = 0; i < src.size(); i++)
	{
		if (src[i].w > 0)
		{
			bool isInOneBlock = false;
			for (int n = 0; n < blockItemVec.size(); n++)
			{
				int mask = blockItemVec[n].maskIdx;
				if (mask != src[i].w)
				{
					continue;
				}
				isInOneBlock = true;
				float minx = blockItemVec[n].minPoint.x;
				float miny = blockItemVec[n].minPoint.y;
				float minz = blockItemVec[n].minPoint.z;

				float maxx = blockItemVec[n].maxPoint.x;
				float maxy = blockItemVec[n].maxPoint.y;
				float maxz = blockItemVec[n].maxPoint.z;

				if (src[i].x >= minx - 0.011 && src[i].x <= maxx + 0.011 && src[i].y >= miny - 0.011 && src[i].y <= maxy + 0.011 && src[i].z >= minz - 0.011 && src[i].z<=maxz + 0.011)
				{
					if (src[i].x < minx)
					{
						blockItemVec[n].minPoint.x = src[i].x;
					}
					else if (src[i].x > maxx)
					{
						blockItemVec[n].maxPoint.x = src[i].x;
					}
					if (src[i].y < miny)
					{
						blockItemVec[n].minPoint.y = src[i].y;
					}
					else if (src[i].y > maxy)
					{
						blockItemVec[n].maxPoint.y = src[i].y;
					}
					if (src[i].z < minz)
					{
						blockItemVec[n].minPoint.z = src[i].z;
					}
					else if (src[i].z > maxz)
					{
						blockItemVec[n].maxPoint.z = src[i].z;
					}
					myPoint3d pp = { src[i].x,src[i].y,src[i].z };
					blockItemVec[n].pointIndex.push_back(pp);
					//这里再加上一个block之间的融合
					for (int m = 0; m < blockItemVec.size(); m++)
					{
						int mask2 = blockItemVec[m].maskIdx;
						if (mask2 != blockItemVec[n].maskIdx || m == n)
						{
							continue;
						}
						if (blockItemVec[n].minPoint.x >= blockItemVec[m].maxPoint.x && blockItemVec[n].minPoint.y >= blockItemVec[m].maxPoint.y && blockItemVec[n].minPoint.z >= blockItemVec[m].maxPoint.z)
						{
							blockItemVec[n].minPoint = blockItemVec[m].minPoint;
							blockItemVec.erase(blockItemVec.begin() + m);
							break;
						}
						if (blockItemVec[n].maxPoint.x <= blockItemVec[m].minPoint.x && blockItemVec[n].maxPoint.y <= blockItemVec[m].minPoint.y && blockItemVec[n].maxPoint.z <= blockItemVec[m].minPoint.z)
						{
							blockItemVec[n].maxPoint = blockItemVec[m].maxPoint;
							blockItemVec.erase(blockItemVec.begin() + m);
							break;
						}
					}
					continue;
				}
			}
			if (!isInOneBlock)
			{
				//新建一个
				myPoint3d minPoint = { src[i].x,src[i].y,src[i].z };
				myPoint3d maxPoint = { src[i].x,src[i].y,src[i].z };
				vector<myPoint3d> pointIndex;
				myPoint3d pp = { src[i].x,src[i].y,src[i].z };
				pointIndex.push_back(pp);
				blockItem3d item = { minPoint , maxPoint , pointIndex , src[i].w };
				blockItemVec.push_back(item);
			}

		}
		
	}
	for (int i = 0; i < blockItemVec.size(); i++)
	{

		bbox_min.push_back(blockItemVec[i].minPoint);
		bbox_max.push_back(blockItemVec[i].maxPoint);
	}

}
void CUDAMarchingCubesHashSDF::extractIsoSurface(const HashDataStruct& hashData, const HashParams& hashParams, const RayCastData& rayCastData, const vec3f& minCorner, const vec3f& maxCorner, bool boxEnabled)
{
	resetMarchingCubesCUDA(m_data);

	m_params.m_maxCorner = MatrixConversion::toCUDA(maxCorner);
	m_params.m_minCorner = MatrixConversion::toCUDA(minCorner);
	m_params.m_boxEnabled = boxEnabled;
	m_data.updateParams(m_params);
	//这个地方 开辟的内存太大了。。。
	vector<float4> mask_voxels;
	//cudaMalloc(&mask_voxels, sizeof(float4) * m_params.m_hashNumBuckets * m_params.m_hashBucketSize * m_params.m_sdfBlockSize * m_params.m_sdfBlockSize * m_params.m_sdfBlockSize);
	//extractIsoBlocksCUDA(hashData, &mask_voxels, hashParams);
	extractIsoSurfaceCUDA(hashData, rayCastData, m_params, m_data, &mask_voxels, hashParams);
	//float4* mask_voxels_cpu;
	//cudaMemcpy(mask_voxels_cpu, mask_voxels, sizeof(float4)*(m_params.m_hashNumBuckets*m_params.m_hashBucketSize*m_params.m_sdfBlockSize*m_params.m_sdfBlockSize*m_params.m_sdfBlockSize), cudaMemcpyDeviceToHost);
	
	vector<myPoint3d> minPoints;
	vector<myPoint3d> maxPoints;
	//GenBoundingBox3d(mask_voxels, minPoints, maxPoints);
	//std::ofstream outfile("boundingbox.txt", std::ios::trunc);

	std::string end = ".txt";
	
	for (int i = 0; i < mask_voxels.size(); i++)
	{
		std::string name = "pointcloud_" + std::to_string(mask_voxels[i].w) + end;
		
		std::ofstream outfile3;
		outfile3.open(name, std::ios::app);
		outfile3 << mask_voxels[i].x << "," << mask_voxels[i].y << ","
			<< mask_voxels[i].z << std::endl;

		outfile3.close();
	}
	

	std::ofstream outfile2("pointcloud.txt", std::ios::trunc);
	int3 RGB[11];
	RGB[0] = make_int3(0, 255, 0);
	RGB[1] = make_int3(0, 0, 255);
	RGB[2] = make_int3(255, 0, 0);
	RGB[3] = make_int3(0, 255, 255);
	RGB[4] = make_int3(255, 255, 0);
	RGB[5] = make_int3(255, 0, 255);
	RGB[6] = make_int3(80, 70, 180);
	RGB[7] = make_int3(250, 80, 190);
	RGB[8] = make_int3(245, 145, 50);
	RGB[9] = make_int3(70, 150, 250);
	RGB[10] = make_int3(50, 190, 190);
	for (int i = 0; i < mask_voxels.size(); i++)
	{
		int index = ((int)mask_voxels[i].w) % 11;
		outfile2 << mask_voxels[i].x << "," << mask_voxels[i].y << ","
			<< mask_voxels[i].z << "," << RGB[index].x << ","
			<< RGB[index].y << "," << RGB[index].z << std::endl;
	}

	/*for (int i = 0; i < minPoints.size(); i++)
	{
		outfile << minPoints[i].x << "," << minPoints[i].y << ","
			<< minPoints[i].z << ";" << maxPoints[i].x << ","
			<< maxPoints[i].y << "," << maxPoints[i].z << std::endl;
	}*/
	//outfile.close();
	outfile2.close();
	copyTrianglesToCPU();
}

//11-28取消，
//void CUDAMarchingCubesHashSDF::extractIsoSurface( CUDASceneRepChunkGrid& chunkGrid, const RayCastData& rayCastData, const vec3f& camPos, float radius)
//{
//
//	chunkGrid.stopMultiThreading();
//
//	const vec3i& minGridPos = chunkGrid.getMinGridPos();
//	const vec3i& maxGridPos = chunkGrid.getMaxGridPos();
//
//	clearMeshBuffer();
//
//	chunkGrid.streamOutToCPUAll();
//
//	for (int x = minGridPos.x; x < maxGridPos.x; x++)	{
//		for (int y = minGridPos.y; y < maxGridPos.y; y++) {
//			for (int z = minGridPos.z; z < maxGridPos.z; z++) {
//
//				vec3i chunk(x, y, z);
//				if (chunkGrid.containsSDFBlocksChunk(chunk)) {
//					std::cout << "Marching Cubes on chunk (" << x << ", " << y << ", " << z << ") " << std::endl;
//
//					chunkGrid.streamInToGPUChunkNeighborhood(chunk, 1);
//
//					const vec3f& chunkCenter = chunkGrid.getWorldPosChunk(chunk);
//					const vec3f& voxelExtends = chunkGrid.getVoxelExtends();
//					float virtualVoxelSize = chunkGrid.getHashParams().m_virtualVoxelSize;
//
//					vec3f minCorner = chunkCenter-voxelExtends/2.0f-vec3f(virtualVoxelSize, virtualVoxelSize, virtualVoxelSize)*(float)chunkGrid.getHashParams().m_SDFBlockSize;
//					vec3f maxCorner = chunkCenter+voxelExtends/2.0f+vec3f(virtualVoxelSize, virtualVoxelSize, virtualVoxelSize)*(float)chunkGrid.getHashParams().m_SDFBlockSize;
//
//					
//					extractIsoSurface(chunkGrid.getHashData(), chunkGrid.getHashParams(), rayCastData, minCorner, maxCorner, true);
//
//					chunkGrid.streamOutToCPUAll();
//				}
//			}
//		}
//	}
//
//	unsigned int nStreamedBlocks;
//	chunkGrid.streamInToGPUAll(camPos, radius, true, nStreamedBlocks);
//
//	chunkGrid.startMultiThreading();
//}


/*
void CUDAMarchingCubesHashSDF::extractIsoSurfaceCPU(const HashData& hashData, const HashParams& hashParams, const RayCastData& rayCastData)
{
	reset();
	m_params.m_numOccupiedSDFBlocks = hashParams.m_numOccupiedBlocks;
	m_data.updateParams(m_params);

	MarchingCubesData cpuData = m_data.copyToCPU();
	HashData		  cpuHashData = hashData.copyToCPU();

	for (unsigned int sdfBlockId = 0; sdfBlockId < m_params.m_numOccupiedSDFBlocks; sdfBlockId++) {
		for (int x = 0; x < hashParams.m_SDFBlockSize; x++) {
			for (int y = 0; y < hashParams.m_SDFBlockSize; y++) {
				for (int z = 0; z < hashParams.m_SDFBlockSize; z++) {

					const HashEntry& entry = cpuHashData.d_hashCompactified[sdfBlockId];
					if (entry.ptr != FREE_ENTRY) {
						int3 pi_base = cpuHashData.SDFBlockToVirtualVoxelPos(entry.pos);
						int3 pi = pi_base + make_int3(x,y,z);
						float3 worldPos = cpuHashData.virtualVoxelPosToWorld(pi);

						cpuData.extractIsoSurfaceAtPosition(worldPos, cpuHashData, rayCastData);
					}

				} // z
			} // y
		} // x
	} // sdf block id

	// save mesh
	{
		std::cout << "saving mesh..." << std::endl;
		std::string filename = "Scans/scan.ply";
		unsigned int nTriangles = *cpuData.d_numTriangles;

		std::cout << "marching cubes: #triangles = " << nTriangles << std::endl;

		if (nTriangles == 0) return;

		unsigned int baseIdx = (unsigned int)m_meshData.m_Vertices.size();
		m_meshData.m_Vertices.resize(baseIdx + 3*nTriangles);
		m_meshData.m_Colors.resize(baseIdx + 3*nTriangles);

		vec3f* vc = (vec3f*)cpuData.d_triangles;
		for (unsigned int i = 0; i < 3*nTriangles; i++) {
			m_meshData.m_Vertices[baseIdx + i] = vc[2*i+0];
			m_meshData.m_Colors[baseIdx + i] = vc[2*i+1];
		}

		//create index buffer (required for merging the triangle soup)
		m_meshData.m_FaceIndicesVertices.resize(nTriangles);
		for (unsigned int i = 0; i < nTriangles; i++) {
			m_meshData.m_FaceIndicesVertices[i][0] = 3*i+0;
			m_meshData.m_FaceIndicesVertices[i][1] = 3*i+1;
			m_meshData.m_FaceIndicesVertices[i][2] = 3*i+2;
		}

		//m_meshData.removeDuplicateVertices();
		//m_meshData.mergeCloseVertices(0.00001f);
		std::cout << "merging close vertices... ";
		m_meshData.mergeCloseVertices(0.00001f, true);
		std::cout << "done!" << std::endl;
		std::cout << "removing duplicate faces... ";
		m_meshData.removeDuplicateFaces();
		std::cout << "done!" << std::endl;

		std::cout << "saving mesh (" << filename << ") ...";
		MeshIOf::saveToFile(filename, m_meshData);
		std::cout << "done!" << std::endl;

		clearMeshBuffer();
	}

	cpuData.free();
}
*/
