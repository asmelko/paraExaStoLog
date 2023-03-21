#include <iostream>

#include "cu_context.h"

void cuda_check(cudaError_t e, const char* file, int line)
{
	if (e != cudaSuccess)
	{
		std::printf("CUDA API failed at %s:%d with error: %s (%d)\n", file, line, cudaGetErrorString(e), e);
		std::exit(EXIT_FAILURE);
	}
}

void cusparse_check(cusparseStatus_t e, const char* file, int line)
{
	if (e != CUSPARSE_STATUS_SUCCESS)
	{
		std::printf("CUSPARSE API failed at %s:%d with error: %s (%d)\n", file, line, cusparseGetErrorString(e), e);
		std::exit(EXIT_FAILURE);
	}
}

void cusolver_check(cusolverStatus_t e, const char* file, int line)
{
	if (e != CUSOLVER_STATUS_SUCCESS)
	{
		std::printf("CUSOLVER API failed at %s:%d with error: %d\n", file, line, e);
		std::exit(EXIT_FAILURE);
	}
}

cu_context::cu_context()
{
	static bool heap_init = false;

	if (!heap_init)
	{
		size_t size;
		CHECK_CUDA(cudaDeviceGetLimit(&size, cudaLimitMallocHeapSize));
		std::cout << "cudaLimitMallocHeapSize " << size << std::endl;
		size = 1UL << 30;
		CHECK_CUDA(cudaDeviceSetLimit(cudaLimitMallocHeapSize, size));
		heap_init = true;
	}
	CHECK_CUSPARSE(cusparseCreate(&cusparse_handle));
	CHECK_CUSOLVER(cusolverSpCreate(&cusolver_handle));
}

cu_context::~cu_context()
{
	CHECK_CUSPARSE(cusparseDestroy(cusparse_handle));
	CHECK_CUSOLVER(cusolverSpDestroy(cusolver_handle));
}
