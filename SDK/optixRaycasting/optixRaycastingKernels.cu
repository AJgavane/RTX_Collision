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

#include <cuda_runtime.h>

#include "Common.h"
#include "optixRaycastingKernels.h"
#include <cstdio>


inline int idivCeil( int x, int y )
{
  return (x + y-1)/y;
}



__global__ void CreateRaysKernel(Ray* rays, int width, int height, int depth)
{
	const int rayx = threadIdx.x + blockIdx.x*blockDim.x;
	const int rayy = threadIdx.y + blockIdx.y*blockDim.y;
	const int rayz = threadIdx.z + blockIdx.z*blockDim.z;
	if( rayx >= width || rayy >= height || rayz >= depth)
		return;

	printf("rayx: %d\t rayy: %d\t rayz: %d\n", rayx, rayy, rayz);
  
	  const int idx = rayz*width*height + rayy * width + rayx;
	printf("index: %d \n", idx);
	  rays[idx].origin = make_float3( 6.452f ,1.461f, 0.06142f );
	  rays[idx].tmin = 0.0f;
	  rays[idx].dir =   make_float3( -35.05f, 10.74f, 15.93f) - rays[idx].origin;
	  rays[idx].tmax = 1.0;
}

void CreateRaysDevice(Ray* rays_device, int width, int height, int depth)
{
	dim3 blockSize( 32, 16, 2 );
	dim3 gridSize( idivCeil( width, blockSize.x ), idivCeil( height, blockSize.y ), idivCeil(depth, blockSize.z));
	CreateRaysKernel<<<gridSize,blockSize>>>( rays_device, width, height, depth);
}

