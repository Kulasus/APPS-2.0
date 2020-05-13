#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "pic_type.h"

// Prototype of function in .cu file
void cu_run_animation( CudaPic t_pic, uint2 t_block_size );

// Image size
#define SIZEX 432 // Width of image
#define	SIZEY 321 // Height of image
// Block size for threads
#define BLOCKX 40 // block width
#define BLOCKY 25 // block height

int main()
{
	cv::Mat l_cv_image = cv::imread( "slunicko.jpg" ,CV_LOAD_IMAGE_COLOR );
	cv::imshow("slunicko", l_cv_image);
	
	cv::Mat l_cv_img( SIZEY, SIZEX, CV_8UC3 );

	/*// Image filling by color gradient blue-green-red
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

	CudaPic l_pic_img(&l_cv_img);
	//l_pic_img.m_size.x = l_cv_img.size().width; // equivalent to cv_img.cols
	//l_pic_img.m_size.y = l_cv_img.size().height; // equivalent to cv_img.rows
	//l_pic_img.m_p_uchar3 = ( uchar3* ) l_cv_img.data;

	// Show image before modification
	cv::imshow( "B-G-R Gradient", l_cv_img );

	// Function calling from .cu file
	uint2 l_block_size = { BLOCKX, BLOCKY };
	cu_run_animation( l_pic_img, l_block_size );

	// Show modified image
	cv::imshow( "B-G-R Gradient & Color Rotation", l_cv_img );*/
	
	cv::waitKey( 0 );
}