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

	__host__ CudaPic(cv::Mat *image)
	{
		m_size.x = image->cols;
		m_size.y = image->rows;
		m_p_void = image->data;
        m_size.z = image->channels();
	}

	__device__ __host__  uchar1 &at1(uint32_t y,uint32_t x)
	{
		return m_p_uchar1[y*m_size.x + x];
	}

	__device__ __host__ uchar3 &at3(uint32_t y,uint32_t x)
	{
		return m_p_uchar3[y*m_size.x + x];
	}

	__device__ __host__  uchar4 &at4(uint32_t y,uint32_t x)
	{
		return m_p_uchar4[y*m_size.x + x];
	}
};
