// ***********************************************************************
//
// Demo program for education in subject
// Computer Architectures and Paralel Systems.
// Petr Olivka, dep. of Computer Science, FEI, VSB-TU Ostrava
// email:petr.olivka@vsb.cz
//
// Example of CUDA Technology Usage with unified memory.
//
// Manipulation with prepared image.
//
// ***********************************************************************

#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "pic_type.h"

// Every threads identifies its position in grid and in block and modify image
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
    l_bgr.x = l_tmp.y;
    l_bgr.y = l_tmp.z;
    l_bgr.z = l_tmp.x;

	// Store point [l_x,l_y] back to image
	t_cuda_pic.m_p_uchar3[ l_y * t_cuda_pic.m_size.x + l_x ] = l_bgr;
}
__global__ void kernel_mask( CudaPic t_cuda_pic, uchar3 rgb_umask )
{
	// X,Y coordinates 
	int l_y = blockDim.y * blockIdx.y + threadIdx.y;
	int l_x = blockDim.x * blockIdx.x + threadIdx.x;
	if ( l_x >= t_cuda_pic.m_size.x ) return;
	if ( l_y >= t_cuda_pic.m_size.y ) return;

	uchar3 l_bgr = t_cuda_pic.at3(l_y,l_x);

	// MASKA - funguje ale naopak, misto aby barvu ubrala tak ji přida
	l_bgr.x = (l_bgr.x & rgb_umask.x);
	l_bgr.y = (l_bgr.y & rgb_umask.y);
	l_bgr.z = (l_bgr.z & rgb_umask.z);

	t_cuda_pic.at3(l_y, l_x) = l_bgr;
}

// Demo kernel to create chess board
__global__ void kernel_chessboard( CudaPic t_pic, uchar3 color1, uchar3 color2 )
{
	int l_y = blockDim.y * blockIdx.y + threadIdx.y;
	int l_x = blockDim.x * blockIdx.x + threadIdx.x;
	if ( l_y >= t_pic.m_size.y ) return;
	if ( l_x >= t_pic.m_size.x ) return;

	if(( blockIdx.x + blockIdx.y ) & 1){
		t_pic.m_p_uchar3[ l_y * t_pic.m_size.x + l_x ] = { color1.x, color1.y, color1.z };
	}
	else{
		t_pic.m_p_uchar3[ l_y * t_pic.m_size.x + l_x ] = { color2.x, color2.y, color2.z };
	}	
}

__global__ void kernel_split( CudaPic t_pic_in, CudaPic t_pic_out_r, CudaPic t_pic_out_g, CudaPic t_pic_out_b )
{
	int l_y = blockDim.y * blockIdx.y + threadIdx.y;
	int l_x = blockDim.x * blockIdx.x + threadIdx.x;
	if ( l_y >= t_pic_in.m_size.y ) return;
	if ( l_x >= t_pic_in.m_size.x ) return;

	t_pic_out_r.m_p_uchar3[l_y * t_pic_out_r.m_size.x + l_x] = {t_pic_in.at3(l_y, l_x).x, 0 , 0};
	t_pic_out_g.m_p_uchar3[l_y * t_pic_out_g.m_size.x + l_x] = {0, t_pic_in.at3(l_y, l_x).y , 0};
	t_pic_out_b.m_p_uchar3[l_y * t_pic_out_b.m_size.x + l_x] = {0, 0 , t_pic_in.at3(l_y, l_x).z};
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

void cu_run_mask( CudaPic t_pic, uint2 t_block_size, uchar3 rgb_umask)
{
	cudaError_t l_cerr;

	// Grid creation with computed organization
	dim3 l_grid( ( t_pic.m_size.x + t_block_size.x - 1 ) / t_block_size.x,
			     ( t_pic.m_size.y + t_block_size.y - 1 ) / t_block_size.y );
	kernel_mask<<< l_grid, dim3( t_block_size.x, t_block_size.y ) >>>( t_pic, rgb_umask );

	if ( ( l_cerr = cudaGetLastError() ) != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( l_cerr ) );

	cudaDeviceSynchronize();

}

void cu_create_chessboard( CudaPic t_pic, int t_square_size, uchar3 color1, uchar3 color2 )
{
	cudaError_t l_cerr;

	dim3 l_blocks( ( t_pic.m_size.x + t_square_size - 1 ) / t_square_size,
			       ( t_pic.m_size.y + t_square_size - 1 ) / t_square_size );
	dim3 l_threads( t_square_size, t_square_size );
	kernel_chessboard<<< l_blocks, l_threads >>>( t_pic, color1, color2 );

	if ( ( l_cerr = cudaGetLastError() ) != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( l_cerr ) );

	cudaDeviceSynchronize();
}

void cu_run_rgb_split( CudaPic t_pic_in, CudaPic t_pic_out_r, CudaPic t_pic_out_g, CudaPic t_pic_out_b )
{
	cudaError_t l_cerr;

	int l_block_size = 32; // MUST BE BIGGER OR AT LEAST EQUAL TO IMAGE 
	dim3 l_blocks( ( t_pic_in.m_size.x + l_block_size - 1 ) / l_block_size,
			       ( t_pic_in.m_size.y + l_block_size - 1 ) / l_block_size );
	dim3 l_threads( l_block_size, l_block_size );
	kernel_split<<< l_blocks, l_threads >>>( t_pic_in, t_pic_out_r, t_pic_out_g, t_pic_out_b );

	if ( ( l_cerr = cudaGetLastError() ) != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( l_cerr ) );

	cudaDeviceSynchronize();
}
