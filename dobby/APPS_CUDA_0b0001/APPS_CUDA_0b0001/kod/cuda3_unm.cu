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
#include <string.h>
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
//**********************	  INSERT	**********************  	
//************************************************************  

__global__ void kernel_insert_image( CudaPic t_big_pic, CudaPic t_small_pic, int2 t_position )
{

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

void cu_insert_image( CudaPic t_big_pic, CudaPic t_small_pic, int2 t_position )
{
	cudaError_t l_cerr;

	// Grid creation, size of grid must be equal or greater than images
	int l_block_size = 32;
	dim3 l_blocks( ( t_small_pic.m_size.x + l_block_size - 1 ) / l_block_size,
			       ( t_small_pic.m_size.y + l_block_size - 1 ) / l_block_size );
	dim3 l_threads( l_block_size, l_block_size );
	kernel_insert_image<<< l_blocks, l_threads >>>( t_big_pic, t_small_pic, t_position );

	if ( ( l_cerr = cudaGetLastError() ) != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( l_cerr ) );

	cudaDeviceSynchronize();
}

//************************************************************    	
//******************	 CHESSBOARD MIX		******************
//************************************************************ 

__global__ void kernel_chessboard_mix( CudaPic t_cuda_pic1, CudaPic t_cuda_pic2)
{
	int l_y = blockDim.y * blockIdx.y + threadIdx.y;
	int l_x = blockDim.x * blockIdx.x + threadIdx.x;
	if ( l_y >= t_cuda_pic1.m_size.y ) return;
	if ( l_x >= t_cuda_pic1.m_size.x ) return;

	
	if((( blockIdx.x + blockIdx.y ) & 1) == 0){
		t_cuda_pic1.data_of_picture3(l_y,l_x) = t_cuda_pic2.data_of_picture3(l_y,l_x);
	}	
}

void cu_run_chessboard_mix( CudaPic t_pic1,CudaPic t_pic2, int t_velikost_policka)
{
	cudaError_t l_cerr;

	dim3 l_blocks( ( t_pic1.m_size.x + t_velikost_policka - 1 ) / t_velikost_policka,
			       ( t_pic1.m_size.y + t_velikost_policka - 1 ) / t_velikost_policka );
	dim3 l_threads( t_velikost_policka, t_velikost_policka );

	kernel_chessboard_mix<<< l_blocks, l_threads >>>( t_pic1, t_pic2 );

	if ( ( l_cerr = cudaGetLastError() ) != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( l_cerr ) );

	cudaDeviceSynchronize();
}

//************************************************************    	
//******************	 	SELECT			******************
//************************************************************ 

__global__ void kernel_select_image( CudaPic t_big_pic, CudaPic t_small_pic, int2 t_position )
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

	t_small_pic.data_of_picture3(l_y, l_x) = t_big_pic.data_of_picture3(l_by, l_bx);

}

void cu_select_image( CudaPic t_big_pic, CudaPic t_small_pic, int2 t_position )
{
	cudaError_t l_cerr;

	// Grid creation, size of grid must be equal or greater than images
	int l_block_size = 32;
	dim3 l_blocks( ( t_small_pic.m_size.x + l_block_size - 1 ) / l_block_size,
			       ( t_small_pic.m_size.y + l_block_size - 1 ) / l_block_size );
	dim3 l_threads( l_block_size, l_block_size );
	kernel_select_image<<< l_blocks, l_threads >>>( t_big_pic, t_small_pic, t_position );

	if ( ( l_cerr = cudaGetLastError() ) != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( l_cerr ) );

	cudaDeviceSynchronize();
}

//************************************************************    	
//******************	 	INSERT ALPHA	******************
//************************************************************ 

__global__ void kernel_insert_image_alpha( CudaPic t_big_pic, CudaPic t_small_pic, int2 t_position )
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

	
	uchar4 l_fg_bgra = t_small_pic.data_of_picture4(l_y, l_x);
	uchar3 l_bg_bgr = t_big_pic.data_of_picture3(l_by, l_bx);
	uchar3 l_bgr = { 0, 0, 0 };

	// compose point from small and big image according alpha channel
	l_bgr.x = l_fg_bgra.x * l_fg_bgra.w / 255 + l_bg_bgr.x * ( 255 - l_fg_bgra.w ) / 255;
	l_bgr.y = l_fg_bgra.y * l_fg_bgra.w / 255 + l_bg_bgr.y * ( 255 - l_fg_bgra.w ) / 255;
	l_bgr.z = l_fg_bgra.z * l_fg_bgra.w / 255 + l_bg_bgr.z * ( 255 - l_fg_bgra.w ) / 255;

	// Store point into image
	t_big_pic.data_of_picture3(l_by, l_bx) = l_bgr;
}

void cu_insert_image_alpha( CudaPic t_big_pic, CudaPic t_small_pic, int2 t_position )
{
	cudaError_t l_cerr;

	// Grid creation, size of grid must be equal or greater than images
	int l_block_size = 32;
	dim3 l_blocks( ( t_small_pic.m_size.x + l_block_size - 1 ) / l_block_size,
			       ( t_small_pic.m_size.y + l_block_size - 1 ) / l_block_size );
	dim3 l_threads( l_block_size, l_block_size );
	kernel_insert_image_alpha<<< l_blocks, l_threads >>>( t_big_pic, t_small_pic, t_position );

	if ( ( l_cerr = cudaGetLastError() ) != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( l_cerr ) );

	cudaDeviceSynchronize();
}

//************************************************************    	
//******************	 	DRAW TEXT		******************
//************************************************************ 

__global__ void kernel_draw_text(  CudaPic t_color_pic, int2 t_pos, const char *t_text, char *t_font, uchar2 t_fsize, double t_alpha, uchar3 t_color)
{
	// X,Y coordinates and check image dimensions
	int l_y = blockDim.y * blockIdx.y + threadIdx.y;
	int l_x = blockDim.x * blockIdx.x + threadIdx.x;
	if ( l_y >= t_color_pic.m_size.y ) return;
	if ( l_x >= t_color_pic.m_size.x ) return;

	char l_znak = t_text[blockIdx.x];
	char l_bity = t_font[l_znak * t_fsize.y + threadIdx.y];

	// Store point into image
	
	
	uchar3 l_bg_bgr = t_color_pic.data_of_picture3(l_y, l_x);
	uchar3 l_bgr = { 0, 0, 0 };

	// compose point from small and big image according alpha channel
	l_bgr.x = t_color.x * (t_alpha * 255) / 255 + l_bg_bgr.x * ( 255 - (t_alpha * 255) ) / 255;
	l_bgr.y = t_color.y * (t_alpha * 255) / 255 + l_bg_bgr.y * ( 255 - (t_alpha * 255) ) / 255;
	l_bgr.z = t_color.z * (t_alpha * 255) / 255 + l_bg_bgr.z * ( 255 - (t_alpha * 255) ) / 255;

	// Store point into image
	if(l_bity & (1 << threadIdx.x))
	t_color_pic.data_of_picture3(l_y, l_x) = l_bgr;
	
}

void cu_draw_text( CudaPic t_color_pic, int2 t_pos, const char *t_text, char *t_font, uchar2 t_fsize, double t_alpha, uchar3 t_color )
{
	cudaError_t l_cerr;

	// Grid creation, size of grid must be equal or greater than images
	int l_block_size_x = t_fsize.x;
	int l_block_size_y = t_fsize.y;
	dim3 l_blocks( strlen( t_text ), 1 );
	dim3 l_threads( l_block_size_x, l_block_size_y);
	char *l_text;
	cudaMallocManaged( &l_text, strlen(t_text));
	strcpy(l_text, t_text);

	kernel_draw_text<<< l_blocks, l_threads >>>( t_color_pic, t_pos, l_text, t_font, t_fsize, t_alpha, t_color );

	if ( ( l_cerr = cudaGetLastError() ) != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( l_cerr ) );

	cudaDeviceSynchronize();
}