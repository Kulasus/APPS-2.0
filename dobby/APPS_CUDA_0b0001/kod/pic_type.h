// ***********************************************************************
//
// Demo program for education in subject
// Computer Architectures and Parallel Systems.
// Petr Olivka, dep. of Computer Science, FEI, VSB-TU Ostrava
// email:petr.olivka@vsb.cz
//
// Example of CUDA Technology Usage.
//
// Image interface for CUDA
//
// ***********************************************************************

#pragma once

#include "uni_mem_allocator.h"


// Structure definition for exchanging data between Host and Device
struct CudaPic
{
	uint3 m_size;				// size of picture
	union {
		  void   *m_p_void;		// data of picture
		  uchar1 *m_p_uchar1;	// data of picture
		  uchar3 *m_p_uchar3;	// data of picture
		  uchar4 *m_p_uchar4;	// data of picture
	};

	__host__ CudaPic() {	
		m_size.x = 0;
		m_size.y = 0;
		m_size.z = 0;
		m_p_void = nullptr;
	}
	
	__host__ CudaPic(cv::Mat *img)
	{
		m_size.x = img->cols;
		m_size.y = img->rows;
	    m_size.z = img->channels();
		m_p_void = img->data;
    }

	__device__ __host__  uchar1 &data_of_picture1(uint32_t y,uint32_t x)
	{
		return m_p_uchar1[y * m_size.x + x];
	}

	__device__ __host__ uchar3 &data_of_picture3(uint32_t y,uint32_t x)
	{
		return m_p_uchar3[y * m_size.x + x];
	}

	__device__ __host__  uchar4 &data_of_picture4(uint32_t y,uint32_t x)
	{
		return m_p_uchar4[y * m_size.x + x];
	}


};