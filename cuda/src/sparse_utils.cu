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


sparse_csr_matrix matmul(cusparseHandle_t handle, index_t* lhs_indptr, index_t* lhs_indices, float* lhs_data,
						 index_t lhs_rows, index_t lhs_cols, index_t lhs_nnz, index_t* rhs_indptr, index_t* rhs_indices,
						 float* rhs_data, index_t rhs_rows, index_t rhs_cols, index_t rhs_nnz)
{
	sparse_csr_matrix out;

	cusparseSpGEMMDescr_t spgemmDesc;
	CHECK_CUSPARSE(cusparseSpGEMM_createDescr(&spgemmDesc));

	cusparseSpMatDescr_t lhs_descr, rhs_descr, out_descr;
	CHECK_CUSPARSE(cusparseCreateCsr(&lhs_descr, lhs_rows, lhs_cols, lhs_nnz, lhs_indptr, lhs_indices, lhs_data,
									 CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
	CHECK_CUSPARSE(cusparseCreateCsr(&rhs_descr, rhs_rows, rhs_cols, rhs_nnz, rhs_indptr, rhs_indices, rhs_data,
									 CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
	CHECK_CUSPARSE(cusparseCreateCsr(&out_descr, lhs_rows, rhs_cols, 0, nullptr, nullptr, nullptr, CUSPARSE_INDEX_32I,
									 CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

	float alpha = 1.f;
	float beta = 0.f;

	size_t bufferSize1;
	// ask bufferSize1 bytes for external memory
	CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(
		handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, lhs_descr, rhs_descr, &beta,
		out_descr, CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize1, NULL));

	thrust::device_vector<char> buffer1(bufferSize1);

	// inspect the matrices A and B to understand the memory requirement for
	// the next step
	CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(
		handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, lhs_descr, rhs_descr, &beta,
		out_descr, CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize1, buffer1.data().get()));

	size_t bufferSize2;
	// ask bufferSize2 bytes for external memory
	CHECK_CUSPARSE(cusparseSpGEMM_compute(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
										  &alpha, lhs_descr, rhs_descr, &beta, out_descr, CUDA_R_32F,
										  CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize2, NULL));

	thrust::device_vector<char> buffer2(bufferSize2);

	// compute the intermediate product of A * B
	CHECK_CUSPARSE(cusparseSpGEMM_compute(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
										  &alpha, lhs_descr, rhs_descr, &beta, out_descr, CUDA_R_32F,
										  CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize2, buffer2.data().get()));

	// get matrix C non-zero entries C_nnz1
	int64_t out_rows, out_cols, out_nnz;
	CHECK_CUSPARSE(cusparseSpMatGetSize(out_descr, &out_rows, &out_cols, &out_nnz));
	// allocate matrix C

	out.indptr.resize(out_rows + 1);
	out.indices.resize(out_nnz);
	out.data.resize(out_nnz);

	// NOTE: if 'beta' != 0, the values of C must be update after the allocation
	//       of dC_values, and before the call of cusparseSpGEMM_copy

	// update matC with the new pointers
	CHECK_CUSPARSE(
		cusparseCsrSetPointers(out_descr, out.indptr.data().get(), out.indices.data().get(), out.data.data().get()));

	// if beta != 0, cusparseSpGEMM_copy reuses/updates the values of dC_values

	// copy the final products to the matrix C
	CHECK_CUSPARSE(cusparseSpGEMM_copy(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
									   &alpha, lhs_descr, rhs_descr, &beta, out_descr, CUDA_R_32F,
									   CUSPARSE_SPGEMM_DEFAULT, spgemmDesc));

	// destroy matrix/vector descriptors
	CHECK_CUSPARSE(cusparseSpGEMM_destroyDescr(spgemmDesc));
	CHECK_CUSPARSE(cusparseDestroySpMat(lhs_descr));
	CHECK_CUSPARSE(cusparseDestroySpMat(rhs_descr));
	CHECK_CUSPARSE(cusparseDestroySpMat(out_descr));

	return out;
}

sparse_csr_matrix matmul(cusparseHandle_t handle, sparse_csr_matrix& lhs, sparse_csr_matrix& rhs)
{
	return matmul(handle, lhs.indptr.data().get(), lhs.indices.data().get(), lhs.data.data().get(), lhs.h, lhs.w,
				  lhs.nnz(), rhs.indptr.data().get(), rhs.indices.data().get(), rhs.data.data().get(), rhs.h, rhs.w,
				  rhs.nnz());
}

void transpose_sparse_matrix(cusparseHandle_t handle, const index_t* in_indptr, const index_t* in_indices,
							 const float* in_data, index_t in_n, index_t out_n, index_t nnz, d_idxvec& out_indptr,
							 d_idxvec& out_indices, thrust::device_vector<float>& out_data)
{
	out_indptr.resize(out_n + 1);
	out_indices.resize(nnz);
	out_data.resize(nnz);

	size_t buffersize;
	CHECK_CUSPARSE(cusparseCsr2cscEx2_bufferSize(handle, in_n, out_n, nnz, in_data, in_indptr, in_indices,
												 out_data.data().get(), out_indptr.data().get(),
												 out_indices.data().get(), CUDA_R_32F, CUSPARSE_ACTION_NUMERIC,
												 CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, &buffersize));

	thrust::device_vector<char> buffer(buffersize);
	CHECK_CUSPARSE(cusparseCsr2cscEx2(handle, in_n, out_n, nnz, in_data, in_indptr, in_indices, out_data.data().get(),
									  out_indptr.data().get(), out_indices.data().get(), CUDA_R_32F,
									  CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1,
									  buffer.data().get()));
}

sparse_csc_matrix csr2csc(cusparseHandle_t handle, const sparse_csr_matrix& in, index_t in_n, index_t out_n,
						  index_t nnz)
{
	sparse_csc_matrix out;
	transpose_sparse_matrix(handle, in.indptr.data().get(), in.indices.data().get(), in.data.data().get(), in_n, out_n,
							nnz, out.indptr, out.indices, out.data);
	return out;
}

sparse_csr_matrix csc2csr(cusparseHandle_t handle, const sparse_csc_matrix& in, index_t in_n, index_t out_n,
						  index_t nnz)
{
	sparse_csr_matrix out;
	transpose_sparse_matrix(handle, in.indptr.data().get(), in.indices.data().get(), in.data.data().get(), in_n, out_n,
							nnz, out.indptr, out.indices, out.data);
	return out;
}


d_datvec mvmul(cusparseHandle_t handle, const d_idxvec& indptr, const d_idxvec& indices, const d_datvec& data,
			   cs_kind k, index_t rows, index_t cols, const d_datvec& x)
{
	d_datvec y(rows);
	float alpha = 1.0f;
	float beta = 0.0f;

	cusparseConstSpMatDescr_t matA;
	cusparseConstDnVecDescr_t vecX;
	cusparseDnVecDescr_t vecY;
	size_t bufferSize = 0;

	if (k == cs_kind::CSR)
		CHECK_CUSPARSE(cusparseCreateConstCsr(&matA, rows, cols, data.size(), indptr.data().get(), indices.data().get(),
											  data.data().get(), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
											  CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
	else
		CHECK_CUSPARSE(cusparseCreateConstCsc(&matA, rows, cols, data.size(), indptr.data().get(), indices.data().get(),
											  data.data().get(), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
											  CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

	// Create dense vector X
	CHECK_CUSPARSE(cusparseCreateConstDnVec(&vecX, cols, x.data().get(), CUDA_R_32F));
	// Create dense vector y
	CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, rows, y.data().get(), CUDA_R_32F));
	// allocate an external buffer if needed
	CHECK_CUSPARSE(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY,
										   CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
	thrust::device_vector<char> buffer(bufferSize);

	// execute SpMV
	CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
								CUSPARSE_SPMV_ALG_DEFAULT, buffer.data().get()));

	// destroy matrix/vector descriptors
	CHECK_CUSPARSE(cusparseDestroySpMat(matA));
	CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
	CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));

	return y;
}
