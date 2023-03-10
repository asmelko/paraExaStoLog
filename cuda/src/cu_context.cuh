#include <cusparse.h>

void cuda_check(cudaError_t e, const char* file, int line);
void cusparse_check(cusparseStatus_t e, const char* file, int line);

#define CHECK_CUDA(func) cuda_check(func, __FILE__, __LINE__)
#define CHECK_CUSPARSE(func) cusparse_check(func, __FILE__, __LINE__)

struct cu_context
{
	cusparseHandle_t cusparse_handle = nullptr;

	cu_context();
	~cu_context();
};
