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
// Image manipulation is performed by OpenCV library. 
//
// ***********************************************************************

#include <stdio.h>
#include <sys/time.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "uni_mem_allocator.h"
#include "pic_type.h"
#include "animation.h"

// Function prototype from .cu file

int main( int t_numarg, char **t_arg )
{
	// Uniform Memory allocator for Mat
	UniformAllocator allocator;
	cv::Mat::setDefaultAllocator( &allocator );
	
	// -----------------------------------------------------------------------------------------------
	// ---------------------------------ROLETA1--------------------------------------
	// -----------------------------------------------------------------------------------------------

	Animation l_animation;
	cv::Mat l_cv_brouk= cv::imread("brouk.png", CV_LOAD_IMAGE_COLOR );
	CudaPic l_cu_brouk(&l_cv_brouk);
	cv::Mat l_cv_koupelna( l_cv_brouk.rows, l_cv_brouk.cols, CV_8UC3 );
	//cv::Mat l_cv_koupelna = cv::imread("koupelna.png", CV_LOAD_IMAGE_COLOR );
	CudaPic l_cu_koupelna(&l_cv_koupelna);
	
	l_animation.start( l_cu_koupelna, l_cu_brouk );
	
	timeval l_start_time, l_cur_time, l_old_time, l_delta_time;
	gettimeofday( &l_old_time, NULL );
	l_start_time = l_old_time;
	int l_iterations = 0;
	int2 l_positions = {0,0};
	
	int position_y = l_cv_brouk.rows;
	int l_fps = 1000;
	cv::VideoWriter l_out("roleta.mkv", cv::VideoWriter::fourcc('M', 'P', 'E', 'G'), l_fps, l_cv_koupelna.size());
	printf("Zacatek");
	while ( l_iterations < 5)
	{
		cv::waitKey( 1 );

		gettimeofday( &l_cur_time, NULL );
		timersub( &l_cur_time, &l_old_time, &l_delta_time );
		if ( l_delta_time.tv_usec < 1000 ) continue; // too short time
		l_old_time = l_cur_time;
		float l_delta_sec = ( float ) l_delta_time.tv_usec / 1E6; // time in seconds
		l_delta_sec = 1.0 / l_fps;
		
		
		if(l_positions.y == -position_y || l_positions.y == 1)
		{
			l_iterations++;
		}

		if(l_iterations % 2 == 0)
		{
			l_positions.y -= 1;
		}
		else
		{
			l_positions.y += 1;
		}
		

		l_animation.next(l_cu_koupelna, l_positions);


		l_out.write(l_cv_koupelna);

	}
	printf("Konec");
	l_animation.stop();
	
	
}

