// ***********************************************************************
//
// Demo program for education in subject
// Computer Architectures and Parallel Systems.
// Petr Olivka, dep. of Computer Science, FEI, VSB-TU Ostrava
// email:petr.olivka@vsb.cz
//
// Example of CUDA Technology Usage without unified memory.
//
// Image creation and its modification using CUDA.
// Image manipulation is performed by OpenCV library. 
//
// ***********************************************************************

#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "pic_type.h"

// Prototype of function in .cu file
void cu_run_animation( CudaPic t_pic, uint2 t_block_size );
void cu_run_reverse(CudaPic t_pic, uint2 t_block_size);
void cu_run_masking(CudaPic t_pic, uint2 t_block_size, uchar3 mask);
void cu_run_chessboard( CudaPic t_color_pic, int t_velikost_policka, uchar3 t_suda_barva, uchar3 t_licha_barva);
void cu_run_rotate(CudaPic t_pic, CudaPic t_rotated, int t_smer);
void cu_run_rgb_rozklad( CudaPic t_pic_in, CudaPic t_pic_out_r, CudaPic t_pic_out_g, CudaPic t_pic_out_b );
void cu_run_zmensit_obrazek(CudaPic t_pic, CudaPic t_cuda_pic_small);

// Image size
#define SIZEX 432 // Width of image
#define	SIZEY 321 // Height of image
// Block size for threads
#define BLOCKX 40 // block width
#define BLOCKY 25 // block height

int main()
{
	UniformAllocator allocator;
	cv::Mat::setDefaultAllocator( &allocator );
	
	cv::Mat l_cv_slunicko = cv::imread("slunicko.jpeg", CV_LOAD_IMAGE_COLOR );
	cv::imshow("Slunicko", l_cv_slunicko);
	CudaPic l_cu_slunicko(&l_cv_slunicko);
	uint2 l_block_size = { BLOCKX, BLOCKY };

	//************************************************************    	
	//**********************  	REVERSE		**********************  	
	//************************************************************ 
	
	/*cu_run_reverse(l_cu_slunicko,l_block_size);
	cv::imshow("Slunicko zamena barev",l_cv_slunicko);*/

	//************************************************************    	
	//**********************  	MASKING		**********************  	
	//************************************************************  

	/*uchar3 mask = {0,0,0};
	cu_run_masking(l_cu_slunicko,l_block_size,mask);
	cv::imshow("Slunicko zamena barev a maska",l_cv_slunicko);*/

	//************************************************************    	
	//**********************	CHESSBOARD	**********************  	
	//************************************************************   

	/*int l_velikost_policka = 12;
	cv::Mat l_cv_chessboard( l_velikost_policka * l_velikost_policka, l_velikost_policka * l_velikost_policka, CV_8UC3 );
	CudaPic l_cu_chessboard(&l_cv_chessboard);
	uchar3 l_suda_barva = {255,0,0};
	uchar3 l_licha_barva = {0,0,255};
	cu_run_chessboard( l_cu_chessboard, l_velikost_policka, l_suda_barva, l_licha_barva);
	cv::imshow( "Chess Board", l_cv_chessboard );*/

	//************************************************************  
	//**********************	ROTATION	**********************  	
	//************************************************************    
	
	/*cv::Mat l_cv_slunicko_rotated(l_cv_slunicko.size().width, l_cv_slunicko.size().height, CV_8UC3);
	CudaPic l_cu_slunicko0(&l_cv_slunicko);
	CudaPic l_cu_slunicko1(&l_cv_slunicko_rotated);
	int t_smer = 0;
	cu_run_rotate(l_cu_slunicko0, l_cu_slunicko1, t_smer);
	cv::imshow("Slunicko pred rotaci", l_cv_slunicko);
	cv::imshow("Slunicko po rotaci", l_cv_slunicko_rotated);*/
	

	//************************************************************    	
	//**********************	   RGB		**********************  	
	//************************************************************  

	/*cv::Mat l_cv_rgb_picture = cv::imread("rgb.png", CV_LOAD_IMAGE_UNCHANGED);
	cv::Mat l_cv_rgb_picture_red(l_cv_rgb_picture.rows, l_cv_rgb_picture.cols, CV_8UC3);
	cv::Mat l_cv_rgb_picture_green(l_cv_rgb_picture.rows, l_cv_rgb_picture.cols, CV_8UC3);
	cv::Mat l_cv_rgb_picture_blue(l_cv_rgb_picture.rows, l_cv_rgb_picture.cols, CV_8UC3);

	cv::imshow("RGB picture",l_cv_rgb_picture);

	CudaPic l_cu_rgb_picture(&l_cv_rgb_picture);
	CudaPic l_cu_rgb_picture_red(&l_cv_rgb_picture_red);
	CudaPic l_cu_rgb_picture_green(&l_cv_rgb_picture_green);
	CudaPic l_cu_rgb_picture_blue(&l_cv_rgb_picture_blue);

	cu_run_rgb_rozklad(l_cu_rgb_picture, l_cu_rgb_picture_red, l_cu_rgb_picture_green, l_cu_rgb_picture_blue);

	cv::imshow("RGB picture red", l_cv_rgb_picture_red);
	cv::imshow("RGB picture green", l_cv_rgb_picture_green);
	cv::imshow("RGB picture blue", l_cv_rgb_picture_blue);*/

	//************************************************************    	
	//**********************	   SMALL		**********************  	
	//************************************************************ 

	cv::Mat l_cv_slunicko_small(l_cv_slunicko.size().height/2, l_cv_slunicko.size().width/2, CV_8UC3);
	CudaPic l_cu_slunicko_small(&l_cv_slunicko_small);
	cu_run_zmensit_obrazek(l_cu_slunicko, l_cu_slunicko_small);
	cv::imshow("Smaller", l_cv_slunicko_small);

	
	cv::waitKey( 0 );
}