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

// Every threads identifies its position in grid and in block and modify image
__global__ void kernel_animation( CudaPic t_cuda_pic )
{
	
	// X,Y coordinates 
	int l_y = blockDim.y * blockIdx.y + threadIdx.y;
	int l_x = blockDim.x * blockIdx.x + threadIdx.x;
	if ( l_x >= t_cuda_pic.m_size.x ) return;
	if ( l_y >= t_cuda_pic.m_size.y ) return;
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

//************************************************************    	
//**********************  	REVERSE		**********************  	
//************************************************************   

__global__ void reverse_animation( CudaPic t_cuda_pic )
{
	// X,Y coordinates 
	int l_y = blockDim.y * blockIdx.y + threadIdx.y;
	int l_x = blockDim.x * blockIdx.x + threadIdx.x;
	if ( l_x >= t_cuda_pic.m_size.x ) return;
	if ( l_y >= t_cuda_pic.m_size.y ) return;

	uchar3 l_bgr, l_tmp = t_cuda_pic.data_of_picture3(l_y, l_x);
	
	l_bgr.x = l_tmp.y;

    l_bgr.y = l_tmp.z;

    l_bgr.z = l_tmp.x;

	t_cuda_pic.data_of_picture3(l_y, l_x) = l_bgr;
}
void cu_run_reverse(CudaPic t_pic, uint2 t_block_size)
{
	cudaError_t l_cerr;

	// Grid creation with computed organization
	dim3 l_grid( ( t_pic.m_size.x + t_block_size.x - 1 ) / t_block_size.x,
			     ( t_pic.m_size.y + t_block_size.y - 1 ) / t_block_size.y );
	reverse_animation<<< l_grid, dim3( t_block_size.x, t_block_size.y ) >>>( t_pic );

	if ( ( l_cerr = cudaGetLastError() ) != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( l_cerr ) );

	cudaDeviceSynchronize();

	
}

//************************************************************    	
//**********************  	MASKING		**********************  	
//************************************************************  

__global__ void masking_animation( CudaPic t_cuda_pic, uchar3 mask)
{
	// X,Y coordinates 
	int l_y = blockDim.y * blockIdx.y + threadIdx.y;
	int l_x = blockDim.x * blockIdx.x + threadIdx.x;
	if ( l_x >= t_cuda_pic.m_size.x ) return;
	if ( l_y >= t_cuda_pic.m_size.y ) return;
	uchar3 l_mask = mask;
	uchar3 l_tmp = t_cuda_pic.data_of_picture3(l_y, l_x);
	
	if(l_tmp.x == 0xFF && l_tmp.z == 0xFF && l_tmp.y == 0xFF)
	{
		t_cuda_pic.data_of_picture3(l_y, l_x) = l_mask;
	}	
}

void cu_run_masking(CudaPic t_pic, uint2 t_block_size, uchar3 mask)
{
	cudaError_t l_cerr;

	// Grid creation with computed organization
	dim3 l_grid( ( t_pic.m_size.x + t_block_size.x - 1 ) / t_block_size.x,
			     ( t_pic.m_size.y + t_block_size.y - 1 ) / t_block_size.y );
	masking_animation<<< l_grid, dim3( t_block_size.x, t_block_size.y ) >>>( t_pic,mask );

	if ( ( l_cerr = cudaGetLastError() ) != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( l_cerr ) );

	cudaDeviceSynchronize();
}

//************************************************************    	
//**********************	CHESSBOARD	**********************  	
//************************************************************  

__global__ void kernel_chessboard( CudaPic t_cuda_pic, uchar3 t_suda_barva, uchar3 t_licha_barva )
{
	int l_y = blockDim.y * blockIdx.y + threadIdx.y;
	int l_x = blockDim.x * blockIdx.x + threadIdx.x;
	if ( l_y >= t_cuda_pic.m_size.y ) return;
	if ( l_x >= t_cuda_pic.m_size.x ) return;

	if((( blockIdx.x + blockIdx.y ) & 1) == 1){
		t_cuda_pic.data_of_picture3(l_y,l_x) = t_suda_barva;
	}
	if((( blockIdx.x + blockIdx.y ) & 1) == 0){
		t_cuda_pic.data_of_picture3(l_y,l_x) = t_licha_barva;
	}	
}

void cu_run_chessboard( CudaPic t_pic, int t_velikost_policka, uchar3 t_suda_barva, uchar3 t_licha_barva )
{
	cudaError_t l_cerr;

	dim3 l_blocks( ( t_pic.m_size.x + t_velikost_policka - 1 ) / t_velikost_policka,
			       ( t_pic.m_size.y + t_velikost_policka - 1 ) / t_velikost_policka );
	dim3 l_threads( t_velikost_policka, t_velikost_policka );
	kernel_chessboard<<< l_blocks, l_threads >>>( t_pic, t_suda_barva, t_licha_barva );

	if ( ( l_cerr = cudaGetLastError() ) != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( l_cerr ) );

	cudaDeviceSynchronize();
}

//************************************************************    	
//**********************	ROTATION	**********************  	
//************************************************************  

__global__ void kernel_rotate(CudaPic t_pic_in, CudaPic t_pic_out, int t_smer)
{
	// X,Y coordinates and check image dimensions
	int l_y = blockDim.y * blockIdx.y + threadIdx.y;
	int l_x = blockDim.x * blockIdx.x + threadIdx.x;
	if ( l_y >= t_pic_out.m_size.y ) return;
	if ( l_x >= t_pic_out.m_size.x ) return;

	int y_smer, x_smer;

	if(t_smer)
	{
		y_smer = l_x;
		x_smer = t_pic_out.m_size.y - l_y - 1;
	}
	else
	{
		x_smer = l_y;
		y_smer = t_pic_out.m_size.x - l_x - 1;
		
	}
	t_pic_out.data_of_picture3(l_y, l_x) = t_pic_in.data_of_picture3(y_smer, x_smer);
}

void cu_run_rotate(CudaPic t_pic_in, CudaPic t_pic_out, int t_smer)
{
	cudaError_t l_cerr;

	int l_block_size = 32;
	dim3 l_blocks( ( t_pic_out.m_size.x + l_block_size - 1 ) / l_block_size,
				   ( t_pic_out.m_size.y + l_block_size - 1 ) / l_block_size );
	dim3 l_threads( l_block_size, l_block_size );
	kernel_rotate<<< l_blocks, l_threads >>>(t_pic_in, t_pic_out, t_smer);

	if ( ( l_cerr = cudaGetLastError() ) != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( l_cerr ) );

	cudaDeviceSynchronize();
}

//************************************************************    	
//**********************	   RGB		**********************  	
//************************************************************  

__global__ void kernel_rgb_rozklad( CudaPic t_pic_in, CudaPic t_pic_out_r, CudaPic t_pic_out_g, CudaPic t_pic_out_b )
{
	int l_y = blockDim.y * blockIdx.y + threadIdx.y;
	int l_x = blockDim.x * blockIdx.x + threadIdx.x;
	if ( l_y >= t_pic_in.m_size.y ) return;
	if ( l_x >= t_pic_in.m_size.x ) return;

	t_pic_out_r.data_of_picture3(l_y, l_x) = {t_pic_in.data_of_picture3(l_y, l_x).x, 0 , 0};
	t_pic_out_g.data_of_picture3(l_y, l_x) = {0, t_pic_in.data_of_picture3(l_y, l_x).y , 0};
	t_pic_out_b.data_of_picture3(l_y, l_x) = {0, 0 , t_pic_in.data_of_picture3(l_y, l_x).z};
}

void cu_run_rgb_rozklad( CudaPic t_pic_in, CudaPic t_pic_out_r, CudaPic t_pic_out_g, CudaPic t_pic_out_b )
{
	cudaError_t l_cerr;

	int l_block_size = 32; 
	dim3 l_blocks( ( t_pic_in.m_size.x + l_block_size - 1 ) / l_block_size,
			       ( t_pic_in.m_size.y + l_block_size - 1 ) / l_block_size );
	dim3 l_threads( l_block_size, l_block_size );
	kernel_rgb_rozklad<<< l_blocks, l_threads >>>( t_pic_in, t_pic_out_r, t_pic_out_g, t_pic_out_b );

	if ( ( l_cerr = cudaGetLastError() ) != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( l_cerr ) );

	cudaDeviceSynchronize();
}

//************************************************************    	
//**********************	  SMALL		**********************  	
//************************************************************  

__global__ void kernel_zmensit_obrazek(CudaPic t_cuda_pic, CudaPic t_cuda_pic_small)
{
	// X,Y coordinates and check image dimensions
	int l_y = blockDim.y * blockIdx.y + threadIdx.y;
	int l_x = blockDim.x * blockIdx.x + threadIdx.x;
	if (l_x >= t_cuda_pic_small.m_size.y) return;
	if (l_x >= t_cuda_pic_small.m_size.x) return;

	int pic_X = l_x * 2;
	int pic_Y = l_y * 2;

	uchar pic_b_px = (t_cuda_pic.data_of_picture3(pic_X, pic_Y).x 
	+ t_cuda_pic.data_of_picture3(pic_X + 1, pic_Y).x 
	+ t_cuda_pic.data_of_picture3(pic_X, pic_Y + 1).x 
	+ t_cuda_pic.data_of_picture3(pic_X + 1, pic_Y + 1).x)/4;

	uchar pic_g_px = (t_cuda_pic.data_of_picture3(pic_X, pic_Y).y 
	+ t_cuda_pic.data_of_picture3(pic_X + 1, pic_Y).y 
	+ t_cuda_pic.data_of_picture3(pic_X, pic_Y + 1).y 
	+ t_cuda_pic.data_of_picture3(pic_X + 1, pic_Y + 1).y)/4;

	uchar pic_r_px = (t_cuda_pic.data_of_picture3(pic_X, pic_Y).z 
	+ t_cuda_pic.data_of_picture3(pic_X + 1, pic_Y).z 
	+ t_cuda_pic.data_of_picture3(pic_X, pic_Y + 1).z 
	+ t_cuda_pic.data_of_picture3(pic_X + 1, pic_Y + 1).z)/4;
	
	
	
	t_cuda_pic_small.data_of_picture3(l_x, l_y) = {pic_b_px, pic_g_px, pic_r_px};
}

void cu_run_zmensit_obrazek(CudaPic t_pic, CudaPic t_cuda_pic_small)
{
	cudaError_t l_cerr;

	int l_block_size = 32;
	dim3 l_blocks( ( t_cuda_pic_small.m_size.x + l_block_size - 1 ) / l_block_size,
				   ( t_cuda_pic_small.m_size.y + l_block_size - 1 ) / l_block_size );
	dim3 l_threads( l_block_size, l_block_size );
	kernel_zmensit_obrazek<<< l_blocks, l_threads >>>(t_pic, t_cuda_pic_small);

	if ( ( l_cerr = cudaGetLastError() ) != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( l_cerr ) );

	cudaDeviceSynchronize();
}