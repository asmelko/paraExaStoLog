#include <cusparse.h>

void cuda_check(cudaError_t e);
void cusparse_check(cusparseStatus_t e);

#define CHECK_CUDA(func) cuda_check(func)
#define CHECK_CUSPARSE(func) cusparse_check(func)

struct cu_context
{
	cusparseHandle_t cusparse_handle = nullptr;

	cu_context();
	~cu_context();
};
