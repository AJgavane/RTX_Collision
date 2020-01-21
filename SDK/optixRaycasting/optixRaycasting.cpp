/*
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

 //-----------------------------------------------------------------------------
 //
 //  This sample uses OptiX as a replacement for Prime, to compute hits only.  Compare to primeSimplePP.
 //  Shading and ray generation are done with separate CUDA kernels and interop.
 //  Also supports an optional mask texture for geometry transparency (the hole in the default fish model).
 //
 //-----------------------------------------------------------------------------

#include <cuda_runtime.h>

#include "Common.h"
#include "OptiXRaycastingContext.h"
#include "optixRaycastingKernels.h"

#include <optixu/optixu_math_namespace.h>
#include <sutil.h>
#include <Mesh.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <sstream>


void printUsageAndExit(const char* argv0)
{
	std::cerr
		<< "Usage  : " << argv0 << " [options]\n"
		<< "App options:\n"
		<< "  -h  | --help                               Print this usage message\n"
		<< "  -m  | --mesh <mesh_file>                   Model to be rendered\n"
		<< "        --mask <ppm_file>                    Mask texture (optional)\n"
		<< "  -w  | --width <number>                     Output image width\n"
		<< std::endl;

	exit(1);
}


void writePPM(const char* filename, const float* image, int width, int height)
{
	std::ofstream out(filename, std::ios::out | std::ios::binary);
	if (!out)
	{
		std::cerr << "Cannot open file " << filename << "'" << std::endl;
		return;
	}

	out << "P6\n" << width << " " << height << "\n255" << std::endl;
	for (int y = height - 1; y >= 0; --y) // flip vertically
	{
		for (int x = 0; x < width * 3; ++x)
		{
			float val = image[y*width * 3 + x];
			unsigned char cval = val < 0.0f ? 0u : val > 1.0f ? 255u : static_cast<unsigned char>(val*255.0f);
			out.put(cval);
		}
	}

	std::cout << "Wrote file " << filename << std::endl;
}


void PrintRays(Ray* rays_device, int sizeOfInput);

void PrintHits(Hit* hit, Ray* rays_device, int sizeOfInput);

int main(int argc, char** argv)
{
	std::string objFilename;
	std::string maskFilename;
	int width = 640;

	// parse arguments
	for (int i = 1; i < argc; ++i)
	{
		std::string arg(argv[i]);
		if (arg == "-h" || arg == "--help")
		{
			printUsageAndExit(argv[0]);
		}
		else if ((arg == "-m" || arg == "--mesh") && i + 1 < argc)
		{
			objFilename = argv[++i];
		}
		else if (arg == "--mask" && i + 1 < argc)
		{
			maskFilename = argv[++i];
		}
		else if ((arg == "-w" || arg == "--width") && i + 1 < argc)
		{
			width = atoi(argv[++i]);
		}
		else
		{
			std::cerr << "Bad option: '" << arg << "'" << std::endl;
			printUsageAndExit(argv[0]);
		}
	}

	// Set default scene with mask if user did not specify scene
	if (objFilename.empty()) {
		objFilename = std::string(sutil::samplesDir()) + "/data/fish.obj";
		if (maskFilename.empty()) {
			maskFilename = std::string(sutil::samplesDir()) + "/data/fish_mask.ppm";
		}
	}

	try {

		//
		// Create Context
		//
		OptiXRaycastingContext context;

		//
		// Create model on host
		//
		objFilename = std::string(sutil::samplesDir()) + "/data/WashingtonDC.obj";
		std::cout << objFilename << std::endl;
		std::cerr << "Loading model: " << objFilename << std::endl;
		HostMesh model(objFilename);
		context.setTriangles(model.num_triangles, model.tri_indices, model.num_vertices, model.positions,
			model.has_texcoords ? model.texcoords : NULL);

		//
		// Create CUDA buffers for rays and hits
		//
		const optix::float3& bbox_min = *reinterpret_cast<const optix::float3*>(model.bbox_min);
		const optix::float3& bbox_max = *reinterpret_cast<const optix::float3*>(model.bbox_max);
		const optix::float3 bbox_span = bbox_max - bbox_min;
		int height = static_cast<int>(width * bbox_span.y / bbox_span.x);

		cudaError_t err = cudaSetDevice(context.getCudaDeviceOrdinal());
		if (err != cudaSuccess)
		{
			printf("cudaSetDevice failed (%s): %s\n", cudaGetErrorName(err), cudaGetErrorString(err));
			exit(1);
		}

		// Populate rays using CUDA kernel
		// casting one ray

		// Read rays from inputfile and copy to cuda.
		std::string filename = std::string(sutil::samplesDir()) + "/data/input.csv";
		std::fstream fin;
		fin.open(filename, std::ios::in);
		std::vector <std::string> row;
		std::string line, word, temp;
		std::vector <Ray> rays_host;
		getline(fin, line);
		while (getline(fin, line))
		{
			row.clear();
			std::stringstream s(line);
			while( getline(s, word, ',') )
			{
				row.push_back(word);
			}
			optix::float3 origin = optix::make_float3(stof(row[0]), stof(row[1]), stof(row[2]));
			optix::float3 dest = optix::make_float3(stof(row[3]), stof(row[4]), stof(row[5]));
			Ray  r;
			r.origin = origin;
			r.dir = dest - origin;
			r.tmin = 0.0f; r.tmax = 1.0f;
			rays_host.push_back(r);
		}
		fin.close();
		// Copy to cuda
		Ray* rays_d = NULL;
		int sizeOfInput = rays_host.size();	
		err = cudaMalloc((void**)&rays_d, sizeof(Ray)*sizeOfInput);
		if (err != cudaSuccess)
		{
			printf("cudaMalloc failed (%s): %s\n", cudaGetErrorName(err), cudaGetErrorString(err));
			exit(1);
		}
		
		cudaMemcpy(rays_d, &rays_host[0], sizeof(Ray)*sizeOfInput, cudaMemcpyHostToDevice);
		
		//PrintRays(rays_d, sizeOfInput);
		context.setRaysDevicePointer(rays_d, size_t(sizeOfInput));

		//////////////////Init hits/////////////////////////
		Hit h;
		for(int i = 0; i < NUM_OF_HITS; i++)
		{
			h.t[i] = 0.0; h.triId[i] = 0;
		}
		h.nhits = 1;
		
		Hit* hits_d = NULL;
		std::cout << "h size: " << sizeof(h) << std::endl;
		err = cudaMalloc(&hits_d, sizeof(Hit)*sizeOfInput);
		std::cout << "hits_d size: " << sizeof(Hit)*sizeOfInput << std::endl;

		if (err != cudaSuccess)
		{
			printf("cudaMalloc failed (%s): %s\n", cudaGetErrorName(err), cudaGetErrorString(err));
			exit(1);
		}
		context.setHitsDevicePointer(hits_d, size_t(sizeOfInput));
		context.execute();

	//	PrintRays(rays_d, sizeOfInput);
		PrintHits(hits_d, rays_d, sizeOfInput);

		rays_host.clear();
		cudaFree(rays_d);
		cudaFree(hits_d);
	}
	catch (std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		exit(1);
	}

	return 0;
}

void PrintRays(Ray* rays_device, int sizeOfInput)
{
	Ray* rays_h = (Ray*)malloc(sizeof(Ray)*sizeOfInput);
	cudaMemcpy(rays_h, rays_device, sizeof(Ray)*sizeOfInput, cudaMemcpyDeviceToHost);
	std::cout << "Rays: " << std::endl;
	for (int i = 0; i < sizeOfInput; i++) {
		std::cout << "Origin: [" << rays_h[i].origin.x << ", " << rays_h[i].origin.y << ", " << rays_h[i].origin.z << "]\t";
		std::cout << "Dir: [" << rays_h[i].dir.x << ", " << rays_h[i].dir.y << ", " << rays_h[i].dir.z << "]" << std::endl;
	}
	std::cout << std::endl;
	delete rays_h;
}

void PrintHits(Hit* hits_device, Ray* rays_device, int sizeOfInput)
{
	Ray* rays_h = (Ray*)malloc(sizeof(Ray)*sizeOfInput);
	cudaMemcpy(rays_h, rays_device, sizeof(Ray)*sizeOfInput, cudaMemcpyDeviceToHost);
	Hit* hits_h = (Hit*)malloc(sizeof(Hit)*sizeOfInput);
	cudaMemcpy(hits_h, hits_device, sizeof(Ray)*sizeOfInput, cudaMemcpyDeviceToHost);
	for (int i = 0; i < sizeOfInput; i++) {
		std::cout << "Origin: [" << rays_h[i].origin.x << ", " << rays_h[i].origin.y << ", " << rays_h[i].origin.z << "]\t";
		std::cout << "Dir: [" << rays_h[i].dir.x << ", " << rays_h[i].dir.y << ", " << rays_h[i].dir.z << "]\t" << hits_h[i].nhits << std::endl;
		for (int j = 0; j < NUM_OF_HITS; j++) {
			if(hits_h[i].t[j] <= 0)
				continue;
			std::cout << "t: " << hits_h[i].t[j] << "    triId: " << hits_h[i].triId[j];
			optix::float3 poi = rays_h[i].origin + hits_h[i].t[j] * rays_h[i].dir;
			std::cout << "    POI: " << poi.x << ", " << poi.y << ", " << poi.z << std::endl;
		}
		std::cout << std::endl;
	}
	
	delete hits_h;
	delete rays_h;
}