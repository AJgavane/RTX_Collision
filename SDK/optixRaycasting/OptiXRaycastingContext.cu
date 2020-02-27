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

#include "Common.h"

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_aabb_namespace.h>

//
// OptiX programs for raycasting context
//


rtBuffer<float3> vertex_buffer;     
rtBuffer<int3>   index_buffer;
rtBuffer<float2> texcoord_buffer;  // per vertex, indexed with index_buffer

rtDeclareVariable( Hit, hit_attr, attribute hit_attr, ); 

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );


RT_PROGRAM void intersect( int primIdx )
{
    const int3 v_idx = index_buffer[ primIdx ];

    const float3 p0 = vertex_buffer[ v_idx.x ];
    const float3 p1 = vertex_buffer[ v_idx.y ];
    const float3 p2 = vertex_buffer[ v_idx.z ];

    // Intersect ray with triangle
    float3 normal;
    float  t, beta, gamma;
    if( intersect_triangle( ray, p0, p1, p2, normal, t, beta, gamma ) )
    {
        if(  rtPotentialIntersection( t ) )
        {
            Hit h;
            h.t[0] = t;
            h.triId[0] = primIdx;
        	h.nhits = 1;
        	hit_attr = h;
            rtReportIntersection( /*material index*/ 0 );
        }
    }
}


//------------------------------------------------------------------------------
//
// Bounds program
//
//------------------------------------------------------------------------------

RT_PROGRAM void bounds( int primIdx, float result[6] )
{
    const int3 v_idx = index_buffer[ primIdx ];

    const float3 p0 = vertex_buffer[ v_idx.x ];
    const float3 p1 = vertex_buffer[ v_idx.y ];
    const float3 p2 = vertex_buffer[ v_idx.z ];

    optix::Aabb* aabb = (optix::Aabb*)result;
    aabb->m_min = fminf( fminf( p0, p1), p2 );
    aabb->m_max = fmaxf( fmaxf( p0, p1), p2 );
}


//------------------------------------------------------------------------------
//
// Hit program copies hit attribute into hit PRD 
//
//------------------------------------------------------------------------------

rtDeclareVariable( Hit, hit_prd, rtPayload, );

RT_PROGRAM void closest_hit()
{
    hit_prd = hit_attr;
}


//------------------------------------------------------------------------------
//
// Any-hit program masks geometry with a texture
//
//------------------------------------------------------------------------------

//rtTextureSampler<uchar4, 2, cudaReadModeNormalizedFloat> mask_sampler;

RT_PROGRAM void any_hit()
{
	
	bool flag = true;
	int i = 0;
	for(i = 0; flag == true && i < NUM_OF_HITS; ++i)
	{
		if(hit_prd.t[i] == -1){
			hit_prd.nhits += 1;
			hit_prd.t[i] = hit_attr.t[0];
			hit_prd.triId[i] = hit_attr.triId[0];
			flag = false;
		}		
	}//*/
	rtIgnoreIntersection();	
}



//------------------------------------------------------------------------------
//
// Ray generation
//
//------------------------------------------------------------------------------

rtDeclareVariable(unsigned int, launch_index, rtLaunchIndex, );

rtDeclareVariable(rtObject, top_object, , );

rtBuffer<Hit, 1>  hits;
rtBuffer<Ray, 1>  rays;


RT_PROGRAM void ray_gen()
{
    Hit hit_prd;
	for(int i = 0; i < NUM_OF_HITS; i++)
	{
		hit_prd.t[i]           = -1.0f;
		hit_prd.triId[i]       = -1;
	}    
    hit_prd.nhits = 0;
    Ray ray = rays[launch_index];
    rtTrace( top_object,
             optix::make_Ray( ray.origin, ray.dir, 1, ray.tmin, ray.tmax ),
             hit_prd );
    hits[ launch_index ] = hit_prd;
}

//------------------------------------------------------------------------------
//
// Exception program for debugging only
//
//------------------------------------------------------------------------------


RT_PROGRAM void exception()
{
  const unsigned int code = rtGetExceptionCode();
  rtPrintf( "Caught exception 0x%X at launch index (%d)\n", code, launch_index );
  Hit hit_prd;
	for(int i = 0; i < NUM_OF_HITS; i++)
	{
		hit_prd.t[i]           = -100.0f;
		hit_prd.triId[i]       = -100;
	}
	hit_prd.nhits = -1;
   hits[ launch_index ] = hit_prd;
}

