#pragma once

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "DXUT.h"
#include "DX11Utils.h"

#include "cudaUtil.h"
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h> 
#include "mLib.h"


struct CB_RGBDRenderer
{
	D3DXMATRIX		m_mIntrinsicInverse;
	D3DXMATRIX		m_mExtrinsic;			//model-view
	D3DXMATRIX		m_mIntrinsicNew;		//for 'real-world' depth range
	D3DXMATRIX		m_mProjection;			//for graphics rendering

	unsigned int	m_uScreenWidth;
	unsigned int	m_uScreenHeight;
	unsigned int	m_uDepthImageWidth;
	unsigned int	m_uDepthImageHeight;

	float			m_fDepthThreshOffset;
	float			m_fDepthThreshLin;
	float2			m_vDummy;
};

class DX11RGBDRenderer
{
public:

	DX11RGBDRenderer()
	{
		m_EmptyVS = NULL;
		m_RGBDRendererGS = NULL;
		m_RGBDRendererRawDepthPS = NULL;
		m_cbRGBDRenderer = NULL;
		m_PointSampler = NULL;
		m_LinearSampler = NULL;

		m_pTextureFloat = NULL;
		m_pTextureFloatSRV = NULL;
		m_dCudaFloat = NULL;

		m_pTextureFloat4 = NULL;
		m_pTextureFloat4SRV = NULL;
		m_dCudaFloat4 = NULL;

		m_width = 0;
		m_height = 0;
	}

	~DX11RGBDRenderer()
	{
		OnD3D11DestroyDevice();
	}

	HRESULT OnD3D11CreateDevice(ID3D11Device* pd3dDevice, unsigned int width, unsigned int height)
	{
		HRESULT hr = S_OK;

		D3D11_BUFFER_DESC desc;
		ZeroMemory(&desc, sizeof(D3D11_BUFFER_DESC));
		desc.Usage = D3D11_USAGE_DYNAMIC;
		desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
		desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
		desc.MiscFlags = 0;
		desc.ByteWidth = sizeof( CB_RGBDRenderer );
		V_RETURN( pd3dDevice->CreateBuffer( &desc, NULL, &m_cbRGBDRenderer ) );

		ID3DBlob* pBlob = NULL;
		V_RETURN(CompileShaderFromFile(L"Shaders/RGBDRenderer.hlsl", "EmptyVS", "vs_5_0", &pBlob));
		V_RETURN(pd3dDevice->CreateVertexShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &m_EmptyVS));

		V_RETURN(CompileShaderFromFile(L"Shaders/RGBDRenderer.hlsl", "RGBDRendererGS", "gs_5_0", &pBlob));
		V_RETURN(pd3dDevice->CreateGeometryShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &m_RGBDRendererGS));

		V_RETURN(CompileShaderFromFile(L"Shaders/RGBDRenderer.hlsl", "RGBDRendererRawDepthPS", "ps_5_0", &pBlob));
		V_RETURN(pd3dDevice->CreatePixelShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &m_RGBDRendererRawDepthPS));

		SAFE_RELEASE(pBlob);
		
		D3D11_SAMPLER_DESC sdesc;
		ZeroMemory(&sdesc, sizeof(sdesc));
		sdesc.AddressU =  D3D11_TEXTURE_ADDRESS_CLAMP;
		sdesc.AddressV =  D3D11_TEXTURE_ADDRESS_CLAMP;
		sdesc.AddressW =  D3D11_TEXTURE_ADDRESS_CLAMP;
		sdesc.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
		V_RETURN(pd3dDevice->CreateSamplerState(&sdesc, &m_PointSampler));
		sdesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
		V_RETURN(pd3dDevice->CreateSamplerState(&sdesc, &m_LinearSampler));

		V_RETURN(OnResize(pd3dDevice, width, height));

		return hr;
	}

	HRESULT OnResize(ID3D11Device* pd3dDevice, unsigned int width, unsigned int height)
	{
		HRESULT hr = S_OK;
		if (width == 0 || height == 0)	return S_FALSE;
		if (m_width == width && m_height == height)	return hr;

		m_width = width;
		m_height = height;

		{
			SAFE_RELEASE(m_pTextureFloat);
			SAFE_RELEASE(m_pTextureFloatSRV);
			if (m_dCudaFloat) {
				cutilSafeCall(cudaGraphicsUnregisterResource(m_dCudaFloat));
				m_dCudaFloat = NULL;
			}

			D3D11_TEXTURE2D_DESC descTex;
			ZeroMemory(&descTex, sizeof(D3D11_TEXTURE2D_DESC));
			descTex.Usage = D3D11_USAGE_DEFAULT;
			descTex.BindFlags = D3D11_BIND_SHADER_RESOURCE;
			descTex.CPUAccessFlags = 0;
			descTex.MiscFlags = 0;
			descTex.SampleDesc.Count = 1;
			descTex.SampleDesc.Quality = 0;
			descTex.ArraySize = 1;
			descTex.MipLevels = 1;
			descTex.Format = DXGI_FORMAT_R32_FLOAT;
			descTex.Width = m_width;
			descTex.Height = m_height;

			V_RETURN(pd3dDevice->CreateTexture2D(&descTex, NULL, &m_pTextureFloat));
			V_RETURN(pd3dDevice->CreateShaderResourceView(m_pTextureFloat, NULL, &m_pTextureFloatSRV));

			cutilSafeCall(cudaGraphicsD3D11RegisterResource(&m_dCudaFloat, m_pTextureFloat, cudaGraphicsRegisterFlagsNone));
			cutilSafeCall(cudaGraphicsResourceSetMapFlags(m_dCudaFloat, cudaGraphicsMapFlagsWriteDiscard));
		}

		{
			SAFE_RELEASE(m_pTextureFloat4);
			SAFE_RELEASE(m_pTextureFloat4SRV);
			if (m_dCudaFloat4) {
				cutilSafeCall(cudaGraphicsUnregisterResource(m_dCudaFloat4));
				m_dCudaFloat4 = NULL;
			}

			D3D11_TEXTURE2D_DESC descTex;
			ZeroMemory(&descTex, sizeof(D3D11_TEXTURE2D_DESC));
			descTex.Usage = D3D11_USAGE_DEFAULT;
			descTex.BindFlags = D3D11_BIND_SHADER_RESOURCE;
			descTex.CPUAccessFlags = 0;
			descTex.MiscFlags = 0;
			descTex.SampleDesc.Count = 1;
			descTex.SampleDesc.Quality = 0;
			descTex.ArraySize = 1;
			descTex.MipLevels = 1;
			descTex.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
			descTex.Width = m_width;
			descTex.Height = m_height;

			V_RETURN(pd3dDevice->CreateTexture2D(&descTex, NULL, &m_pTextureFloat4));
			V_RETURN(pd3dDevice->CreateShaderResourceView(m_pTextureFloat4, NULL, &m_pTextureFloat4SRV));

			cutilSafeCall(cudaGraphicsD3D11RegisterResource(&m_dCudaFloat4, m_pTextureFloat4, cudaGraphicsRegisterFlagsNone));
			cutilSafeCall(cudaGraphicsResourceSetMapFlags(m_dCudaFloat4, cudaGraphicsMapFlagsWriteDiscard));
		}
		
		return hr;
	}

	void OnD3D11DestroyDevice()
	{
		SAFE_RELEASE(m_EmptyVS);
		SAFE_RELEASE(m_RGBDRendererGS);
		SAFE_RELEASE(m_RGBDRendererRawDepthPS);
		SAFE_RELEASE(m_cbRGBDRenderer);
		SAFE_RELEASE(m_PointSampler);
		SAFE_RELEASE(m_LinearSampler);

		SAFE_RELEASE(m_pTextureFloat);
		SAFE_RELEASE(m_pTextureFloatSRV);
		if (m_dCudaFloat)
		{
			cutilSafeCall(cudaGraphicsUnregisterResource(m_dCudaFloat));
			m_dCudaFloat = NULL;
		}

		SAFE_RELEASE(m_pTextureFloat4);
		SAFE_RELEASE(m_pTextureFloat4SRV);
		if (m_dCudaFloat4)
		{
			cutilSafeCall(cudaGraphicsUnregisterResource(m_dCudaFloat4));
			m_dCudaFloat4 = NULL;
		}

		m_width = 0;
		m_height = 0;
	}

	//渲染mask
	//HRESULT RenderMaskBox(
	//	ID3D11DeviceContext* pd3dDeviceContext,
	//	float* d_depthMap,
	//	float4* d_colorMap,
	//	unsigned int width,
	//	unsigned int height,
	//	const mat4f& intrinsicDepthToWorld,
	//	const mat4f& modelview,
	//	const mat4f& intrinsicWorldToDepth,
	//	unsigned int screenWidth,
	//	unsigned int screenheight,
	//	float depthThreshOffset,
	//	float depthThreshLin
	//)
	//{
	//	HRESULT hr = S_OK;
	//	V_RETURN(OnResize(DXUTGetD3D11Device(), width, height));	//returns if width/height did not change

	//	float4* d_colorMap_cpu = (float4*)malloc(sizeof(float4)*width*height);
	//	cutilSafeCall(cudaMemcpy(d_colorMap_cpu, d_colorMap, sizeof(float4)*width*height, cudaMemcpyDeviceToHost));

	//	float* d_maskMap = (float*)malloc(sizeof(float)*width*height);
	//	for (int i = 0; i < width*height; i++)
	//	{
	//		if (d_colorMap_cpu[i].w > 0)
	//		{

	//		}
	//	}
	//	cudaArray* in_array;
	//	cutilSafeCall(cudaGraphicsMapResources(1, &m_dCudaFloat, 0));	// Map DX texture to Cuda
	//	cutilSafeCall(cudaGraphicsSubResourceGetMappedArray(&in_array, m_dCudaFloat, 0, 0));
	//	cutilSafeCall(cudaMemcpyToArray(in_array, 0, 0, d_depthMap, sizeof(float)*width*height, cudaMemcpyHostToDevice));
	//	cutilSafeCall(cudaGraphicsUnmapResources(1, &m_dCudaFloat, 0));	// Unmap DX texture
	//	//这里要根据d_colorMap中的标签信息来生成boundingbox

	//	
	//	
	//	
	//	//考虑是否加入cv
	//	//BoundingBox2d()
	//	/*D3DXFrameCalculateBoundingSphere()
	//	D3DXComputeBoundingBox(d_maskMap, );*/
	//	ID3DXLine* dxLine;                 // 此COM接口可以用来画线
	//	ID3D11Device* device = DXUTGetD3D11Device();
	//	

	//	pd3dDeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_POINTLIST);
	//	pd3dDeviceContext->IASetInputLayout(NULL);
	//	unsigned int stride = 0;
	//	unsigned int offset = 0;
	//	pd3dDeviceContext->IASetVertexBuffers(0, 0, NULL, &stride, &offset);

	//	ID3D11SamplerState* ss[] = { m_PointSampler, m_LinearSampler };
	//	pd3dDeviceContext->GSSetSamplers(0, 2, ss);
	//	pd3dDeviceContext->PSSetSamplers(0, 2, ss);
	//	pd3dDeviceContext->VSSetShader(m_EmptyVS, NULL, 0);
	//	pd3dDeviceContext->GSSetShader(m_RGBDRendererGS, NULL, 0);
	//	pd3dDeviceContext->PSSetShader(m_RGBDRendererRawDepthPS, NULL, 0);

	//	//mapping the constant buffer
	//	{
	//		D3D11_MAPPED_SUBRESOURCE MappedResource;
	//		V(pd3dDeviceContext->Map(m_cbRGBDRenderer, 0, D3D11_MAP_WRITE_DISCARD, 0, &MappedResource));
	//		CB_RGBDRenderer* pCB = (CB_RGBDRenderer*)MappedResource.pData;
	//		memcpy(&pCB->m_mIntrinsicInverse, &intrinsicDepthToWorld, sizeof(float) * 16);
	//		memcpy(&pCB->m_mIntrinsicNew, &intrinsicWorldToDepth, sizeof(float) * 16);
	//		memcpy(&pCB->m_mExtrinsic, &modelview, sizeof(float) * 16);
	//		pCB->m_uScreenHeight = screenheight;
	//		pCB->m_uScreenWidth = screenWidth;
	//		pCB->m_uDepthImageWidth = m_width;
	//		pCB->m_uDepthImageHeight = m_height;
	//		pCB->m_fDepthThreshOffset = depthThreshOffset;
	//		pCB->m_fDepthThreshLin = depthThreshLin;
	//		pd3dDeviceContext->Unmap(m_cbRGBDRenderer, 0);

	//		pd3dDeviceContext->GSSetConstantBuffers(0, 1, &m_cbRGBDRenderer);
	//		pd3dDeviceContext->PSSetConstantBuffers(0, 1, &m_cbRGBDRenderer);
	//	}
	//	pd3dDeviceContext->GSSetShaderResources(0, 1, &m_pTextureFloatSRV);
	//	pd3dDeviceContext->PSSetShaderResources(0, 1, &m_pTextureFloatSRV);

	//	pd3dDeviceContext->GSSetShaderResources(1, 1, &m_pTextureFloat4SRV);
	//	pd3dDeviceContext->PSSetShaderResources(1, 1, &m_pTextureFloat4SRV);


	//	unsigned int numQuads = ;

	//	pd3dDeviceContext->Draw(numQuads, 0);

	//	//! reset the state
	//	pd3dDeviceContext->VSSetShader(NULL, NULL, 0);
	//	pd3dDeviceContext->GSSetShader(NULL, NULL, 0);
	//	pd3dDeviceContext->PSSetShader(NULL, NULL, 0);

	//	ID3D11ShaderResourceView* srvNULL[] = { NULL, NULL };
	//	pd3dDeviceContext->GSSetShaderResources(0, 2, srvNULL);
	//	pd3dDeviceContext->PSSetShaderResources(0, 2, srvNULL);
	//	ID3D11Buffer* buffNULL[] = { NULL };
	//	pd3dDeviceContext->GSSetConstantBuffers(0, 1, buffNULL);
	//	pd3dDeviceContext->PSSetConstantBuffers(0, 1, buffNULL);

	//	return hr;
	//}
struct myPoint
{
	int x;
	int y;
};
struct blockItem
{
	myPoint minPoint;
	myPoint maxPoint;
	vector<myPoint>	pointIndex;
	int maskIdx;
};
	void GenBoundingBox(float* src, int width, int height, vector<myPoint>& bbox_min, vector<myPoint>& bbox_max)
	{
		vector<blockItem> blockItemVec;
		
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (src[i*width + j] > 0)
				{
					bool isInOneBlock = false;
					for (int n = 0; n < blockItemVec.size(); n++)
					{
						int mask = blockItemVec[n].maskIdx;
						if (mask != src[i*width + j])
						{
							continue;
						}
						isInOneBlock = true;
						int left = blockItemVec[n].minPoint.x;
						int right = blockItemVec[n].maxPoint.x;
						int top = blockItemVec[n].minPoint.y;
						int bottom = blockItemVec[n].maxPoint.y;

						if (i >= top - 2 && i <= bottom+2 && j >= left-2 && j <= right+2)
						{
							if (i < top)
							{
								blockItemVec[n].minPoint.y = i;
							}
							else if (i > bottom)
							{
								blockItemVec[n].maxPoint.y = i;
							}
							if (j < left)
							{
								blockItemVec[n].minPoint.x = j;
							}
							else if (j > right)
							{
								blockItemVec[n].maxPoint.x = j;
							}
							myPoint pp = { j,i };
							blockItemVec[n].pointIndex.push_back(pp);
							break;
						}
					}
					if (!isInOneBlock)
					{
						//新建一个
						myPoint minPoint = { j,i };
						myPoint maxPoint = { j,i };
						vector<myPoint>	pointIndex;
						myPoint pp = { j,i };
						pointIndex.push_back(pp);
						blockItem item = { minPoint , maxPoint , pointIndex , src[i*width + j] };
						blockItemVec.push_back(item);
					}

				}
			}
		}
		for (int i = 0; i < blockItemVec.size(); i++)
		{
			
			bbox_min.push_back(blockItemVec[i].minPoint);
			bbox_max.push_back(blockItemVec[i].maxPoint);
		}
	
	}


	HRESULT RenderDepthMap(
		ID3D11DeviceContext* pd3dDeviceContext, 
		float* d_depthMap,
		float4* d_colorMap,
		unsigned int width, 
		unsigned int height, 
		const mat4f& intrinsicDepthToWorld, 
		const mat4f& modelview, 
		const mat4f& intrinsicWorldToDepth, 
		unsigned int screenWidth,
		unsigned int screenheight,
		float depthThreshOffset,
		float depthThreshLin,
		vector<float>& xs,
		vector<float>& ys,
		vector<float>& zs)
	{
		HRESULT hr = S_OK;
		V_RETURN(OnResize(DXUTGetD3D11Device(), width, height));	//returns if width/height did not change
		float4* d_colorMap_cpu = (float4*)malloc(sizeof(float4)*width*height);
		cutilSafeCall(cudaMemcpy(d_colorMap_cpu, d_colorMap, sizeof(float4)*width*height, cudaMemcpyDeviceToHost));
		float* d_depthMap_cpu = (float*)malloc(sizeof(float)*width*height);
		cutilSafeCall(cudaMemcpy(d_depthMap_cpu, d_depthMap, sizeof(float)*width*height, cudaMemcpyDeviceToHost));
		
		
		float* d_maskMap = (float*)malloc(sizeof(float)*width*height);
		for (int i = 0; i < width*height; i++)
		{
			d_maskMap[i] = d_colorMap_cpu[i].w;
		}
		/*cv::Mat M = cv::Mat(height, width, CV_16F, d_maskMap);
		
		cv::imwrite("maskMap.jpg", M);*/
		vector<myPoint> bbox_min;
		vector<myPoint> bbox_max;
		GenBoundingBox(d_maskMap, width, height, bbox_min, bbox_max);

		double camera_px = 321.084;
		double camera_py = 238.077;
		double camera_fx = 609.441;
		double camera_fy = 607.96;

		for (int i = 0; i < bbox_min.size(); i++)
		{
			int r = (bbox_min[i].y + bbox_max[i].y) / 2;
			int c = (bbox_min[i].x + bbox_max[i].x) / 2;

			float d = d_depthMap_cpu[r * width + c];
			float x = d * (c - camera_px/2) / camera_fx;
			float y = d * (r - camera_py/2) / camera_fy;
			float z = d;

			xs.push_back(x);
			ys.push_back(y);
			zs.push_back(z);
			//float dist = sqrt(x * x + y * y + z * z);

			
		}
		//Mat objectMask(outMasks.size[2], outMasks.size[3], CV_32F, outMasks.ptr<float>(i, classId));
		
		
		for (int i = 0; i < bbox_min.size(); i++)
		{
			for (int r = bbox_min[i].y; r < bbox_max[i].y; r++)
			{
				int c = bbox_min[i].x;

				d_depthMap_cpu[r * width + c] = 0;
				d_colorMap_cpu[r * width + c] = make_float4(1.0, 1.0, 1.0, 1.0);
			}
			for (int r = bbox_min[i].y; r < bbox_max[i].y; r++)
			{
				int c = bbox_max[i].x;

				d_depthMap_cpu[r * width + c] = 0;
				d_colorMap_cpu[r * width + c] = make_float4(1.0, 1.0, 1.0, 1.0);
			}
			for (int c = bbox_min[i].x; c < bbox_max[i].x; c++)
			{
				int r = bbox_min[i].y;
				d_depthMap_cpu[r * width + c] = 0;
				d_colorMap_cpu[r * width + c] = make_float4(1.0, 1.0, 1.0, 1.0);
			}
			for (int c = bbox_min[i].x; c < bbox_max[i].x; c++)
			{
				int r = bbox_max[i].y;
				d_depthMap_cpu[r * width + c] = 0;
				d_colorMap_cpu[r * width + c] = make_float4(1.0, 1.0, 1.0, 1.0);
			}
		}
		cudaArray* in_array;
		cutilSafeCall(cudaGraphicsMapResources(1, &m_dCudaFloat, 0));	// Map DX texture to Cuda
		cutilSafeCall(cudaGraphicsSubResourceGetMappedArray(&in_array, m_dCudaFloat, 0, 0));
		cutilSafeCall(cudaMemcpyToArray(in_array, 0, 0, d_depthMap_cpu, sizeof(float)*width*height, cudaMemcpyHostToDevice));
		cutilSafeCall(cudaGraphicsUnmapResources(1, &m_dCudaFloat, 0));	// Unmap DX texture

		cutilSafeCall(cudaGraphicsMapResources(1, &m_dCudaFloat4, 0));	// Map DX texture to Cuda
		cutilSafeCall(cudaGraphicsSubResourceGetMappedArray(&in_array, m_dCudaFloat4, 0, 0));
		cutilSafeCall(cudaMemcpyToArray(in_array, 0, 0, d_colorMap_cpu, 4*sizeof(float)*width*height, cudaMemcpyHostToDevice));
		cutilSafeCall(cudaGraphicsUnmapResources(1, &m_dCudaFloat4, 0));	// Unmap DX texture



		//这里要根据d_colorMap中的标签信息来生成boundingbox

		/*float* d_depthMap_cpu = (float*)malloc(sizeof(float)*width*height);
		cutilSafeCall(cudaMemcpy(d_depthMap_cpu, d_depthMap, sizeof(float)*width*height, cudaMemcpyDeviceToHost));
		for (int i = 0; i < width*height; i++)
		{
			printf("i:%d, d:%f\n", i, d_depthMap_cpu[i]);
		}*/
		//
		////float4* d_maskMap = (float4*)malloc(sizeof(float4)*width*height);

		//for (int i = 0; i < width*height; i++)
		//{
		//	if (d_colorMap_cpu[i].w > 0)
		//	{
		//		d_colorMap_cpu[i] = make_float4(1.0, 1.0, 1.0, 1.0);
		//	}
		//}
		//cutilSafeCall(cudaGraphicsMapResources(1, &m_dCudaFloat4, 0));	// Map DX texture to Cuda
		//cutilSafeCall(cudaGraphicsSubResourceGetMappedArray(&in_array, m_dCudaFloat4, 0, 0));
		//cutilSafeCall(cudaMemcpyToArray(in_array, 0, 0, d_colorMap_cpu, 4 * sizeof(float)*width*height, cudaMemcpyHostToDevice));
		//cutilSafeCall(cudaGraphicsUnmapResources(1, &m_dCudaFloat4, 0));	// Unmap DX texture


		pd3dDeviceContext->IASetPrimitiveTopology( D3D11_PRIMITIVE_TOPOLOGY_POINTLIST );
		pd3dDeviceContext->IASetInputLayout( NULL );
		unsigned int stride = 0;
		unsigned int offset = 0;
		pd3dDeviceContext->IASetVertexBuffers(0, 0, NULL, &stride, &offset);

		ID3D11SamplerState* ss[] = {m_PointSampler, m_LinearSampler};
		pd3dDeviceContext->GSSetSamplers(0, 2, ss);
		pd3dDeviceContext->PSSetSamplers(0, 2, ss);
		pd3dDeviceContext->VSSetShader(m_EmptyVS, NULL, 0);
		pd3dDeviceContext->GSSetShader(m_RGBDRendererGS, NULL, 0);
		pd3dDeviceContext->PSSetShader(m_RGBDRendererRawDepthPS, NULL, 0);

		//mapping the constant buffer
		{
			D3D11_MAPPED_SUBRESOURCE MappedResource;
			V(pd3dDeviceContext->Map( m_cbRGBDRenderer, 0, D3D11_MAP_WRITE_DISCARD, 0, &MappedResource ));
			CB_RGBDRenderer* pCB = ( CB_RGBDRenderer* )MappedResource.pData;
			memcpy(&pCB->m_mIntrinsicInverse, &intrinsicDepthToWorld, sizeof(float)*16);
			memcpy(&pCB->m_mIntrinsicNew, &intrinsicWorldToDepth, sizeof(float)*16);
			memcpy(&pCB->m_mExtrinsic, &modelview, sizeof(float)*16);
			pCB->m_uScreenHeight = screenheight;
			pCB->m_uScreenWidth = screenWidth;
			pCB->m_uDepthImageWidth = m_width;
			pCB->m_uDepthImageHeight = m_height;
			pCB->m_fDepthThreshOffset = depthThreshOffset;
			pCB->m_fDepthThreshLin = depthThreshLin;
			pd3dDeviceContext->Unmap( m_cbRGBDRenderer, 0 );

			pd3dDeviceContext->GSSetConstantBuffers(0, 1, &m_cbRGBDRenderer);
			pd3dDeviceContext->PSSetConstantBuffers(0, 1, &m_cbRGBDRenderer);
		}
		pd3dDeviceContext->GSSetShaderResources(0, 1, &m_pTextureFloatSRV);
		pd3dDeviceContext->PSSetShaderResources(0, 1, &m_pTextureFloatSRV);

		pd3dDeviceContext->GSSetShaderResources(1, 1, &m_pTextureFloat4SRV);
		pd3dDeviceContext->PSSetShaderResources(1, 1, &m_pTextureFloat4SRV);


		unsigned int numQuads = width*height;
		pd3dDeviceContext->Draw(numQuads, 0);

		//! reset the state
		pd3dDeviceContext->VSSetShader(NULL, NULL, 0);
		pd3dDeviceContext->GSSetShader(NULL, NULL, 0);
		pd3dDeviceContext->PSSetShader(NULL, NULL, 0);

		ID3D11ShaderResourceView* srvNULL[] = {NULL, NULL};
		pd3dDeviceContext->GSSetShaderResources(0, 2, srvNULL);
		pd3dDeviceContext->PSSetShaderResources(0, 2, srvNULL);	
		ID3D11Buffer* buffNULL[] = {NULL};
		pd3dDeviceContext->GSSetConstantBuffers(0, 1, buffNULL);
		pd3dDeviceContext->PSSetConstantBuffers(0, 1, buffNULL);

		return hr;
	}

private:

	unsigned int m_width;
	unsigned int m_height;

	ID3D11VertexShader*		m_EmptyVS;
	ID3D11GeometryShader*	m_RGBDRendererGS;
	ID3D11PixelShader*		m_RGBDRendererRawDepthPS;

	ID3D11Buffer*			m_cbRGBDRenderer;
	ID3D11SamplerState*		m_PointSampler;
	ID3D11SamplerState*		m_LinearSampler;

	ID3D11Texture2D*			m_pTextureFloat;
	ID3D11ShaderResourceView*	m_pTextureFloatSRV;
	cudaGraphicsResource*		m_dCudaFloat;

	ID3D11Texture2D*			m_pTextureFloat4;
	ID3D11ShaderResourceView*	m_pTextureFloat4SRV;
	cudaGraphicsResource*		m_dCudaFloat4;

};
