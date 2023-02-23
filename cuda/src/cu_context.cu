#include "cu_context.cuh"

#include <iostream>

void cuda_check(cudaError_t e)
{
	if (e != cudaSuccess)
	{
		std::printf("CUDA API failed at line %d with error: %s (%d)\n", __LINE__, cudaGetErrorString(e), e);
		std::exit(EXIT_FAILURE);
	}
}

void cusparse_check(cusparseStatus_t e)
{
	if (e != CUSPARSE_STATUS_SUCCESS)
	{
		std::printf("CUSPARSE API failed at line %d with error: %s (%d)\n", __LINE__, cusparseGetErrorString(e), e);
		std::exit(EXIT_FAILURE);
	}
}

cu_context::cu_context() { CHECK_CUSPARSE(cusparseCreate(&cusparse_handle)); }

cu_context::~cu_context() { CHECK_CUSPARSE(cusparseDestroy(cusparse_handle)); }
