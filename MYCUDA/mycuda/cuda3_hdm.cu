// ***********************************************************************
//
// Demo program for education in subject
// Computer Architectures and Paralel Systems.
// Petr Olivka, dep. of Computer Science, FEI, VSB-TU Ostrava
// email:petr.olivka@vsb.cz
//
// Example of CUDA Technology Usage without unified memory.
//
// Manipulation with prepared image.
//
// ***********************************************************************

#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#include "pic_type.h"

__global__ void kernel_animation( CudaPic t_cuda_pic )
{
	// X,Y coordinates 
	int l_y = blockDim.y * blockIdx.y + threadIdx.y;
	int l_x = blockDim.x * blockIdx.x + threadIdx.x;
	if ( l_x >= t_cuda_pic.m_size.x ) return;
	if ( l_y >= t_cuda_pic.m_size.y ) return;

	// Point [l_x,l_y] selection from image
	uchar3 l_bgr, l_tmp = t_cuda_pic.m_p_uchar3[ l_y * t_cuda_pic.m_size.x + l_x ];

	// color rotation
    l_bgr.x = l_tmp.x;
    l_bgr.y = l_tmp.y;
    l_bgr.z = l_tmp.z;

	// Store point [l_x,l_y] back to image
	t_cuda_pic.m_p_uchar3[ l_y * t_cuda_pic.m_size.x + l_x ] = l_bgr;
}

void cu_run_animation( CudaPic t_pic, uint2 t_block_size )
{
	cudaError_t l_cerr;

	// Grid creation with computed organization
	dim3 l_grid( ( t_pic.m_size.x + t_block_size.x - 1 ) / t_block_size.x,
			     ( t_pic.m_size.y + t_block_size.y - 1 ) / t_block_size.y );
	kernel_animation<<< l_grid, dim3( t_block_size.x, t_block_size.y ) >>>( t_pic );

	if ( ( l_cerr = cudaGetLastError() ) != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( l_cerr ) );

	cudaDeviceSynchronize();
}
