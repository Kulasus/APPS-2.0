// ***********************************************************************
//
// Demo program for education in subject
// Computer Architectures and Parallel Systems.
// Petr Olivka, dep. of Computer Science, FEI, VSB-TU Ostrava
// email:petr.olivka@vsb.cz
//
// Example of CUDA Technology Usage without unified memory.
//
// Simple animation.
//
// ***********************************************************************

#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#include "pic_type.h"
#include "animation.h"

// Demo kernel to create chess board
__global__ void kernel_creategradient( CudaPic t_color_pic )
{
	// X,Y coordinates and check imaa CUDA 0b0100ge dimensions
	int l_y = blockDim.y * blockIdx.y + threadIdx.y;
	int l_x = blockDim.x * blockIdx.x + threadIdx.x;
	if ( l_y >= t_color_pic.m_size.y ) return;
	if ( l_x >= t_color_pic.m_size.x ) return;
	
	int l_dy = l_x * t_color_pic.m_size.y / t_color_pic.m_size.x + l_y - t_color_pic.m_size.y;
	unsigned char l_color = 255 * abs( l_dy ) / t_color_pic.m_size.y;

	uchar3 l_bgr = ( l_dy < 0 ) ? ( uchar3 ) { l_color, 255 - l_color, 0 } : ( uchar3 ) { 0, 255 - l_color, l_color };

	// Store point into image
	t_color_pic.m_p_uchar3[ l_y * t_color_pic.m_size.x + l_x ] = l_bgr;
}

// -----------------------------------------------------------------------------------------------
// ---------------------------------ROLETA1--------------------------------------
// -----------------------------------------------------------------------------------------------


// Demo kernel to create picture with alpha channel gradient
__global__ void kernel_insertimage( CudaPic t_big_pic, CudaPic t_small_pic, int2 t_position )
{
	// X,Y coordinates and check image dimensions
	int l_y = blockDim.y * blockIdx.y + threadIdx.y;
	int l_x = blockDim.x * blockIdx.x + threadIdx.x;
	if ( l_y >= t_small_pic.m_size.y ) return;
	if ( l_x >= t_small_pic.m_size.x ) return;
	int l_by = l_y + t_position.y;
	int l_bx = l_x + t_position.x;
	if ( l_by >= t_big_pic.m_size.y || l_by < 0 ) return;
	if ( l_bx >= t_big_pic.m_size.x || l_bx < 0 ) return;

	t_big_pic.data_of_picture3(l_by, l_bx) = t_small_pic.data_of_picture3(l_y, l_x);
}

void cu_insertimage( CudaPic t_big_pic, CudaPic t_small_pic, int2 t_position )
{
	cudaError_t l_cerr;

	// Grid creation, size of grid must be equal or greater than images
	int l_block_size = 32;
	dim3 l_blocks( ( t_small_pic.m_size.x + l_block_size - 1 ) / l_block_size,
			       ( t_small_pic.m_size.y + l_block_size - 1 ) / l_block_size );
	dim3 l_threads( l_block_size, l_block_size );
	kernel_insertimage<<< l_blocks, l_threads >>>( t_big_pic, t_small_pic, t_position );

	if ( ( l_cerr = cudaGetLastError() ) != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( l_cerr ) );

	cudaDeviceSynchronize();
}

// -----------------------------------------------------------------------------------------------
// ---------------------------------ROLETA1--------------------------------------
// -----------------------------------------------------------------------------------------------


void Animation::start( CudaPic t_bg_pic, CudaPic t_ins_pic )
{
	if ( m_initialized ) return;
	cudaError_t l_cerr;

	m_cuda_bg_pic = t_bg_pic;
	m_cuda_res_pic = t_bg_pic;
	m_cuda_ins_pic = t_ins_pic;

	// Memory allocation in GPU device
	// Memory for background
	l_cerr = cudaMalloc( &m_cuda_bg_pic.m_p_void, m_cuda_bg_pic.m_size.x * m_cuda_bg_pic.m_size.y * sizeof( uchar3 ) );
	if ( l_cerr != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( l_cerr ) );

	// Creation of background gradient
	int l_block_size = 32;
	dim3 l_blocks( ( m_cuda_bg_pic.m_size.x + l_block_size - 1 ) / l_block_size,
			       ( m_cuda_bg_pic.m_size.y + l_block_size - 1 ) / l_block_size );
	dim3 l_threads( l_block_size, l_block_size );
	kernel_creategradient<<< l_blocks, l_threads >>>( m_cuda_bg_pic );

	m_initialized = 1;
}
//roleta1
void Animation::next( CudaPic t_res_pic, int2 t_position )
{
	if ( !m_initialized ) return;

	cudaError_t cerr;

	// Copy data internally GPU from background into result
	cerr = cudaMemcpy( m_cuda_res_pic.m_p_void, m_cuda_bg_pic.m_p_void, m_cuda_bg_pic.m_size.x * m_cuda_bg_pic.m_size.y * sizeof( uchar3 ), cudaMemcpyDeviceToDevice );
	if ( cerr != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cerr ) );

	// insert picture
	int l_block_size = 32;
	dim3 l_blocks( ( m_cuda_ins_pic.m_size.x + l_block_size - 1 ) / l_block_size,
			       ( m_cuda_ins_pic.m_size.y + l_block_size - 1 ) / l_block_size );
	dim3 l_threads( l_block_size, l_block_size );
	kernel_insertimage<<< l_blocks, l_threads >>>( m_cuda_res_pic, m_cuda_ins_pic, t_position );

	// Copy data to GPU device
	cerr = cudaMemcpy( t_res_pic.m_p_void, m_cuda_res_pic.m_p_void, m_cuda_res_pic.m_size.x * m_cuda_res_pic.m_size.y * sizeof( uchar3 ), cudaMemcpyDeviceToHost );
	if ( cerr != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cerr ) );

}

void Animation::stop()
{
	if ( !m_initialized ) return;

	cudaFree( m_cuda_bg_pic.m_p_void );
	cudaFree( m_cuda_res_pic.m_p_void );
	cudaFree( m_cuda_ins_pic.m_p_void );

	m_initialized = 0;
}







