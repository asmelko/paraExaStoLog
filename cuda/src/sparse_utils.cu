#include "sparse_utils.h"


void coo2csc(cusparseHandle_t handle, index_t n, d_idxvec& rows, d_idxvec& cols, d_idxvec& indptr)
{
	size_t buffersize;
	CHECK_CUSPARSE(cusparseXcscsort_bufferSizeExt(handle, n, n, (int)cols.size(), rows.data().get(), cols.data().get(),
												  &buffersize));

	thrust::device_vector<char> buffer(buffersize);

	d_idxvec P(cols.size());
	CHECK_CUSPARSE(cusparseCreateIdentityPermutation(handle, P.size(), P.data().get()));

	CHECK_CUSPARSE(cusparseXcoosortByColumn(handle, n, n, (int)cols.size(), rows.data().get(), cols.data().get(),
											P.data().get(), buffer.data().get()));

	indptr.resize(n + 1);

	CHECK_CUSPARSE(cusparseXcoo2csr(handle, cols.data().get(), (int)rows.size(), n, indptr.data().get(),
									CUSPARSE_INDEX_BASE_ZERO));
}

// coded just for symbolic non rectangular matrices
sparse_csc_matrix coo2csc(cusparseHandle_t handle, sparse_coo_matrix& coo)
{
	d_idxvec indptr;
	coo2csc(handle, coo.h, coo.rows, coo.cols, indptr);

	return sparse_csc_matrix(std::move(indptr), std::move(coo.rows), std::move(coo.data));
}
