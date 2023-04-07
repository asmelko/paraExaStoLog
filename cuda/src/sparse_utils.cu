#include <cusolverSp_LOWLEVEL_PREVIEW.h>

#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/partition.h>
#include <thrust/tuple.h>

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

sparse_csr_matrix matmul(cusparseHandle_t handle, index_t* lhs_indptr, index_t* lhs_indices, real_t* lhs_data,
						 index_t lhs_rows, index_t lhs_cols, index_t lhs_nnz, index_t* rhs_indptr, index_t* rhs_indices,
						 real_t* rhs_data, index_t rhs_rows, index_t rhs_cols, index_t rhs_nnz)
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

	real_t alpha = 1.f;
	real_t beta = 0.f;

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
							 const real_t* in_data, index_t in_n, index_t out_n, index_t nnz, d_idxvec& out_indptr,
							 d_idxvec& out_indices, d_datvec& out_data)
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

d_datvec mvmul(cusparseHandle_t handle, d_idxvec& indptr, d_idxvec& indices, d_datvec& data, cs_kind k, index_t rows,
			   index_t cols, d_datvec& x)
{
	d_datvec y(rows);
	real_t alpha = 1.0f;
	real_t beta = 0.0f;

	cusparseSpMatDescr_t matA;
	cusparseDnVecDescr_t vecX;
	cusparseDnVecDescr_t vecY;
	size_t bufferSize = 0;

	if (k == cs_kind::CSR)
		CHECK_CUSPARSE(cusparseCreateCsr(&matA, rows, cols, data.size(), indptr.data().get(), indices.data().get(),
										 data.data().get(), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
										 CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
	else
		CHECK_CUSPARSE(cusparseCreateCsc(&matA, rows, cols, data.size(), indptr.data().get(), indices.data().get(),
										 data.data().get(), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
										 CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

	// Create dense vector X
	CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, cols, x.data().get(), CUDA_R_32F));
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

d_datvec sparse2dense(cusparseHandle_t handle, sparse_csr_matrix& M)
{
	auto rows = M.h;
	auto cols = M.w;

	cusparseSpMatDescr_t matM;
	cusparseDnMatDescr_t matDn;
	CHECK_CUSPARSE(cusparseCreateCsr(&matM, rows, cols, M.nnz(), M.indptr.data().get(), M.indices.data().get(),
									 M.data.data().get(), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
									 CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

	d_datvec mat_dn(rows * cols);
	CHECK_CUSPARSE(cusparseCreateDnMat(&matDn, rows, cols, rows, mat_dn.data().get(), CUDA_R_32F, CUSPARSE_ORDER_COL));

	size_t buffer_size;
	CHECK_CUSPARSE(
		cusparseSparseToDense_bufferSize(handle, matM, matDn, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, &buffer_size));

	thrust::device_vector<char> buffer(buffer_size);

	CHECK_CUSPARSE(cusparseSparseToDense(handle, matM, matDn, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, buffer.data().get()));

	return mat_dn;
}

sparse_csr_matrix dense2sparse(cusparseHandle_t handle, d_datvec mat_dn, index_t rows, index_t cols)
{
	sparse_csr_matrix M;

	cusparseSpMatDescr_t matM;
	cusparseDnMatDescr_t matDn;
	CHECK_CUSPARSE(cusparseCreateCsr(&matM, 0, 0, 0, nullptr, nullptr, nullptr, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
									 CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

	CHECK_CUSPARSE(cusparseCreateDnMat(&matDn, rows, cols, rows, mat_dn.data().get(), CUDA_R_32F, CUSPARSE_ORDER_COL));

	size_t buffer_size;
	CHECK_CUSPARSE(
		cusparseDenseToSparse_bufferSize(handle, matDn, matM, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, &buffer_size));

	thrust::device_vector<char> buffer(buffer_size);

	CHECK_CUSPARSE(
		cusparseDenseToSparse_analysis(handle, matDn, matM, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, buffer.data().get()));

	int64_t rows_, cols_, nnz;

	CHECK_CUSPARSE(cusparseSpMatGetSize(matM, &rows_, &cols_, &nnz));

	M.indptr.resize(rows + 1);
	M.indices.resize(nnz);
	M.data.resize(nnz);

	cusparseCsrSetPointers(matM, M.indptr.data().get(), M.indices.data().get(), M.data.data().get());

	CHECK_CUSPARSE(
		cusparseDenseToSparse_convert(handle, matDn, matM, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, buffer.data().get()));

	return M;
}

void dense_lu(cusolverDnHandle_t handle, d_datvec& A, index_t rows, index_t cols)
{
	size_t d_lwork = 0; /* size of workspace */
	size_t h_lwork = 0; /* size of workspace */

	cusolverDnParams_t params;
	CHECK_CUSOLVER(cusolverDnCreateParams(&params));
	CHECK_CUSOLVER(cusolverDnSetAdvOptions(params, CUSOLVERDN_GETRF, CUSOLVER_ALG_0));

	CHECK_CUSOLVER(cusolverDnXgetrf_bufferSize(handle, params, rows, cols, CUDA_R_32F, A.data().get(), rows, CUDA_R_32F,
											   &d_lwork, &h_lwork));

	thrust::device_vector<char> d_buffer(d_lwork);
	thrust::host_vector<char> h_buffer(h_lwork);
	thrust::device_vector<int> info(1);

	CHECK_CUSOLVER(cusolverDnXgetrf(handle, params, rows, cols, CUDA_R_32F, A.data().get(), rows, nullptr, CUDA_R_32F,
									d_buffer.data().get(), d_lwork, h_buffer.data(), h_lwork, info.data().get()));

	if (info[0] != 0)
		std::cout << "unexpected error at dense LU" << std::endl;

	CHECK_CUSOLVER(cusolverDnDestroyParams(params));
}

void host_lu(cusolverSpHandle_t handle, const host_sparse_csr_matrix& h, host_sparse_csr_matrix& l,
			 host_sparse_csr_matrix& u)
{
	index_t n = h.indptr.size() - 1;
	index_t nnz = h.indices.size();

	csrluInfoHost_t info;
	CHECK_CUSOLVER(cusolverSpCreateCsrluInfoHost(&info));

	cusparseMatDescr_t descr, descr_L, descr_U;
	CHECK_CUSPARSE(cusparseCreateMatDescr(&descr));
	CHECK_CUSPARSE(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));
	CHECK_CUSPARSE(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
	CHECK_CUSPARSE(cusparseSetMatDiagType(descr, CUSPARSE_DIAG_TYPE_NON_UNIT));

	CHECK_CUSPARSE(cusparseCreateMatDescr(&descr_L));
	CHECK_CUSPARSE(cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ZERO));
	CHECK_CUSPARSE(cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL));
	CHECK_CUSPARSE(cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER));
	CHECK_CUSPARSE(cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_UNIT));

	CHECK_CUSPARSE(cusparseCreateMatDescr(&descr_U));
	CHECK_CUSPARSE(cusparseSetMatIndexBase(descr_U, CUSPARSE_INDEX_BASE_ZERO));
	CHECK_CUSPARSE(cusparseSetMatType(descr_U, CUSPARSE_MATRIX_TYPE_GENERAL));
	CHECK_CUSPARSE(cusparseSetMatFillMode(descr_U, CUSPARSE_FILL_MODE_UPPER));
	CHECK_CUSPARSE(cusparseSetMatDiagType(descr_U, CUSPARSE_DIAG_TYPE_NON_UNIT));

	CHECK_CUSOLVER(cusolverSpXcsrluAnalysisHost(handle, n, nnz, descr, h.indptr.data(), h.indices.data(), info));

	size_t internal_data, workspace;
	CHECK_CUSOLVER(cusolverSpScsrluBufferInfoHost(handle, n, nnz, descr, h.data.data(), h.indptr.data(),
												  h.indices.data(), info, &internal_data, &workspace));

	std::vector<char> buffer(workspace);

	CHECK_CUSOLVER(cusolverSpScsrluFactorHost(handle, n, nnz, descr, h.data.data(), h.indptr.data(), h.indices.data(),
											  info, 0.1f, buffer.data()));

	int nnz_l, nnz_u;
	CHECK_CUSOLVER(cusolverSpXcsrluNnzHost(handle, &nnz_l, &nnz_u, info));

	std::vector<index_t> P(n), Q(n);

	l.indptr.resize(n + 1);
	u.indptr.resize(n + 1);

	l.indices.resize(nnz_l);
	l.data.resize(nnz_l);
	u.indices.resize(nnz_u);
	u.data.resize(nnz_u);

	CHECK_CUSOLVER(cusolverSpScsrluExtractHost(handle, P.data(), Q.data(), descr_L, l.data.data(), l.indptr.data(),
											   l.indices.data(), descr_U, u.data.data(), u.indptr.data(),
											   u.indices.data(), info, buffer.data()));

	CHECK_CUSPARSE(cusparseDestroyMatDescr(descr));
	CHECK_CUSPARSE(cusparseDestroyMatDescr(descr_L));
	CHECK_CUSPARSE(cusparseDestroyMatDescr(descr_U));
	CHECK_CUSOLVER(cusolverSpDestroyCsrluInfoHost(info));
}

double host_det(cusolverSpHandle_t handle, const host_sparse_csr_matrix& h)
{
	index_t n = h.indptr.size() - 1;

	host_sparse_csr_matrix l, u;

	host_lu(handle, h, l, u);

	thrust::host_vector<double> diag;
	diag.resize(n);

	thrust::for_each(thrust::host, thrust::make_counting_iterator<index_t>(0),
					 thrust::make_counting_iterator<index_t>(n), [&](index_t i) {
						 auto begin = u.indptr[i];
						 auto end = u.indptr[i + 1];

						 for (auto col_idx = begin; col_idx != end; col_idx++)
						 {
							 if (u.indices[col_idx] == i)
							 {
								 diag[i] = u.data[col_idx];
								 break;
							 }
						 }
					 });

	return thrust::reduce(diag.begin(), diag.end(), 1., thrust::multiplies<double>());
}

void create_minor(cusparseHandle_t handle, d_idxvec& indptr, d_idxvec& indices, d_datvec& data,
				  const index_t remove_vertex)
{
	const auto nnz = indices.size();
	const auto n = indptr.size() - 1;

	d_idxvec indptr_decompressed(nnz);

	// this decompresses indptr into cols
	CHECK_CUSPARSE(cusparseXcsr2coo(handle, indptr.data().get(), nnz, n, indptr_decompressed.data().get(),
									CUSPARSE_INDEX_BASE_ZERO));

	{
		auto part_point = thrust::stable_partition(
			thrust::make_zip_iterator(indices.begin(), indptr_decompressed.begin(), data.begin()),
			thrust::make_zip_iterator(indices.end(), indptr_decompressed.end(), data.end()),
			[remove_vertex] __device__(thrust::tuple<index_t, index_t, real_t> x) {
				return thrust::get<0>(x) != remove_vertex && thrust::get<1>(x) != remove_vertex;
			});

		auto removed_n = thrust::get<0>(part_point.get_iterator_tuple()) - indices.begin();

		indices.resize(removed_n);
		indptr_decompressed.resize(removed_n);
		data.resize(removed_n);

		thrust::transform_if(
			indices.begin(), indices.end(), indices.begin(), [] __device__(index_t x) { return x - 1; },
			[remove_vertex] __device__(index_t x) { return x > remove_vertex; });

		thrust::transform_if(
			indptr_decompressed.begin(), indptr_decompressed.end(), indptr_decompressed.begin(),
			[] __device__(index_t x) { return x - 1; },
			[remove_vertex] __device__(index_t x) { return x > remove_vertex; });
	}

	indptr.resize(n);

	// this compresses back into indptr
	CHECK_CUSPARSE(cusparseXcoo2csr(handle, indptr_decompressed.data().get(), indptr_decompressed.size(), n - 1,
									indptr.data().get(), CUSPARSE_INDEX_BASE_ZERO));
}

void sort_sparse_matrix(cusparseHandle_t handle, sparse_csr_matrix& N)
{
	index_t n = N.n();
	index_t nnz = N.nnz();

	cusparseMatDescr_t descr_N = 0;
	size_t pBufferSizeInBytes;

	CHECK_CUSPARSE(cusparseCreateMatDescr(&descr_N));
	CHECK_CUSPARSE(cusparseSetMatIndexBase(descr_N, CUSPARSE_INDEX_BASE_ZERO));
	CHECK_CUSPARSE(cusparseSetMatType(descr_N, CUSPARSE_MATRIX_TYPE_GENERAL));

	// step 1: allocate buffer
	CHECK_CUSPARSE(cusparseXcsrsort_bufferSizeExt(handle, n, n, nnz, N.indptr.data().get(), N.indices.data().get(),
												  &pBufferSizeInBytes));

	// step 2: setup permutation vector P to identity
	d_idxvec P(nnz);
	CHECK_CUSPARSE(cusparseCreateIdentityPermutation(handle, nnz, P.data().get()));

	{
		// step 3: sort CSR format
		thrust::device_vector<char> buffer(pBufferSizeInBytes);
		CHECK_CUSPARSE(cusparseXcsrsort(handle, n, n, nnz, descr_N, N.indptr.data().get(), N.indices.data().get(),
										P.data().get(), buffer.data().get()));
	}

	// step 4: gather sorted csrVal
	d_datvec sorted_data(nnz);
	thrust::gather(P.begin(), P.end(), N.data.begin(), sorted_data.begin());
	N.data = std::move(sorted_data);

	CHECK_CUSPARSE(cusparseDestroyMatDescr(descr_N));
}
