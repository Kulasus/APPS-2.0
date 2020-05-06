#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "uni_mem_allocator.h"
#include "pic_type.h"

// Prototype of function in .cu file
void cu_run_animation( CudaPic t_pic, uint2 t_block_size );
void cu_run_mask( CudaPic t_pic, uint2 t_block_size, uchar3 rgb_umask);
void cu_create_chessboard( CudaPic t_pic, int t_square_size, uchar3 color1, uchar3 color2 );
void cu_run_rgb_split( CudaPic t_pic_in, CudaPic t_pic_out_r, CudaPic t_pic_out_g, CudaPic t_pic_out_b );

// Image size
#define SIZEX 432 // Width of image
#define	SIZEY 321 // Height of image
// Block size for threads
#define BLOCKX 40 // block width
#define BLOCKY 25 // block height

int main()
{
	// Uniform Memory allocator for Mat
	UniformAllocator allocator;
	cv::Mat::setDefaultAllocator( &allocator );

	// Creation of empty image.
	// Image is stored line by line.
	cv::Mat l_cv_img( SIZEY, SIZEX, CV_8UC3 );

	// Image filling by color gradient blue-green-red
	for ( int y = 0; y < l_cv_img.rows; y++ )
		for ( int x  = 0; x < l_cv_img.cols; x++ )
		{
			int l_dx = x - l_cv_img.cols / 2;

			unsigned char l_grad = 255 * abs( l_dx ) / ( l_cv_img.cols / 2 );
			unsigned char l_inv_grad = 255 - l_grad;

			// left or right half of gradient
			uchar3 l_bgr = ( l_dx < 0 ) ? ( uchar3 ) { l_grad, l_inv_grad, 0 } : ( uchar3 ) { 0, l_inv_grad, l_grad };

			// put pixel into image
			cv::Vec3b l_v3bgr( l_bgr.x, l_bgr.y, l_bgr.z );
			l_cv_img.at<cv::Vec3b>( y, x ) = l_v3bgr;
			// also possible: cv_img.at<uchar3>( y, x ) = bgr;
		}

	// 1 DEMO ANIMATION
	/*
	CudaPic l_pic_img(&l_cv_img);
	cv::imshow( "B-G-R Gradient", l_cv_img );
	uint2 l_block_size = { BLOCKX, BLOCKY };
	cu_run_animation( l_pic_img, l_block_size );
	cv::imshow( "B-G-R Gradient & Color Rotation", l_cv_img );
	*/

	// 2 RGB MASK
	/*
	cv::Mat l_shrek = cv::imread("shrek2.jpg", CV_LOAD_IMAGE_UNCHANGED);
	cv::imshow("SHREK",l_shrek);
	uchar3 rgb_mask = {150,255,0};
	CudaPic l_pic_shrek(&l_shrek);
	cu_run_mask(l_pic_shrek,l_block_size,rgb_mask);
	cv::imshow("SHREK MODIFIED",l_shrek);
	*/

	//3 CHESSBOARD
	/*
	cv::Mat l_cv_chessboard( 600, 800, CV_8UC3 );
	CudaPic l_pic_chessboard(&l_cv_chessboard);
	uchar3 rgb1 = {255,128,0};
	uchar3 rgb2 = {0,128,255};
	cu_create_chessboard( l_pic_chessboard, 20, rgb1, rgb2 );
	cv::imshow( "Chess Board", l_cv_chessboard );
	*/

	//4 RGB SPLIT
	cv::Mat l_shrek = cv::imread("rgb.jpg", CV_LOAD_IMAGE_UNCHANGED);
	cv::imshow("SHREK",l_shrek);
	CudaPic l_pic_shrek(&l_shrek);
	cv::Mat l_pic_shrek_r(l_shrek.rows, l_shrek.cols, CV_8UC3);
	cv::Mat l_pic_shrek_g(l_shrek.rows, l_shrek.cols, CV_8UC3);
	cv::Mat l_pic_shrek_b(l_shrek.rows, l_shrek.cols, CV_8UC3);
	CudaPic l_pic_shrek_R_cp(&l_pic_shrek_r);
	CudaPic l_pic_shrek_G_cp(&l_pic_shrek_g);
	CudaPic l_pic_shrek_B_cp(&l_pic_shrek_b);
	cu_run_rgb_split(l_pic_shrek, l_pic_shrek_R_cp, l_pic_shrek_G_cp, l_pic_shrek_B_cp);
	cv::imshow("SHREK BLUE", l_pic_shrek_r);
	cv::imshow("SHREK GREEN", l_pic_shrek_g);
	cv::imshow("SHREK RED", l_pic_shrek_b);
	//END
	cv::waitKey( 0 );
}

