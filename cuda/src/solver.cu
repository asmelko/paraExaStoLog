#include <cusolverSp_LOWLEVEL_PREVIEW.h>
#include <cusparse_v2.h>
#include <device_launch_parameters.h>

#include <thrust/adjacent_difference.h>
#include <thrust/execution_policy.h>
#include <thrust/partition.h>

#include "solver.h"
#include "utils.h"

struct equals_ftor : public thrust::unary_function<index_t, bool>
{
	index_t value;

	equals_ftor(index_t value) : value(value) {}

	__host__ __device__ bool operator()(index_t x) const { return x == value; }
};

solver::solver(cu_context& context, const transition_table& t, transition_graph g, initial_state s)
	: context_(context),
	  initial_state_(std::move(s.state)),
	  labels_(std::move(g.labels)),
	  terminals_(std::move(g.terminals)),
	  rows_(t.rows),
	  cols_(t.cols),
	  indptr_(t.indptr),
	  submatrix_vertex_mapping_(labels_.size())
{}

__global__ void scatter_rows_data(const index_t* __restrict__ dst_indptr, index_t* __restrict__ dst_rows,
								  float* __restrict__ dst_data, const index_t* __restrict__ src_rows,
								  const index_t* __restrict__ src_indptr, const index_t* __restrict__ src_perm,
								  int perm_size)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx >= perm_size)
		return;

	index_t src_begin = src_indptr[src_perm[idx]];
	index_t src_end = src_indptr[src_perm[idx] + 1];

	index_t dst_begin = dst_indptr[idx];

	int i = 0;
	for (; i < src_end - src_begin; i++)
	{
		dst_rows[dst_begin + i] = src_rows[src_begin + i];
	}

	dst_rows[dst_begin + i] = src_perm[idx];
	dst_data[dst_begin + i] = -(float)i;
}

float solver::determinant(const d_idxvec& indptr, const d_idxvec& rows, const thrust::device_vector<float>& data, int n,
						  int nnz)
{
	thrust::host_vector<index_t> h_indptr = indptr;
	thrust::host_vector<index_t> h_rows = rows;
	thrust::host_vector<float> h_data = data;

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

	CHECK_CUSOLVER(
		cusolverSpXcsrluAnalysisHost(context_.cusolver_handle, n, nnz, descr, h_indptr.data(), h_rows.data(), info));

	size_t internal_data, workspace;
	CHECK_CUSOLVER(cusolverSpScsrluBufferInfoHost(context_.cusolver_handle, n, nnz, descr, h_data.data(),
												  h_indptr.data(), h_rows.data(), info, &internal_data, &workspace));

	std::vector<char> buffer(workspace);

	CHECK_CUSOLVER(cusolverSpScsrluFactorHost(context_.cusolver_handle, n, nnz, descr, h_data.data(), h_indptr.data(),
											  h_rows.data(), info, 0.1f, buffer.data()));

	int nnz_l, nnz_u;
	CHECK_CUSOLVER(cusolverSpXcsrluNnzHost(context_.cusolver_handle, &nnz_l, &nnz_u, info));

	std::vector<index_t> P(n), Q(n), L_indptr(n + 1), U_indptr(n + 1), L_cols(nnz_l), U_cols(nnz_u);
	std::vector<float> L_data(nnz_l), U_data(nnz_u);

	CHECK_CUSOLVER(cusolverSpScsrluExtractHost(context_.cusolver_handle, P.data(), Q.data(), descr_L, L_data.data(),
											   L_indptr.data(), L_cols.data(), descr_U, U_data.data(), U_indptr.data(),
											   U_cols.data(), info, buffer.data()));

	std::vector<float> diag(n);

	thrust::for_each(thrust::host, thrust::make_counting_iterator<index_t>(0),
					 thrust::make_counting_iterator<index_t>(n), [&](index_t i) {
						 auto begin = U_indptr[i];
						 auto end = U_indptr[i + 1];

						 for (auto col_idx = begin; col_idx != end; col_idx++)
						 {
							 if (U_cols[col_idx] == i)
								 diag[i] = U_data[col_idx];
						 }
					 });

	float determinant = thrust::reduce(thrust::host, diag.begin(), diag.end(), 1, thrust::multiplies<float>());

	CHECK_CUSPARSE(cusparseDestroyMatDescr(descr));
	CHECK_CUSPARSE(cusparseDestroyMatDescr(descr_L));
	CHECK_CUSPARSE(cusparseDestroyMatDescr(descr_U));
	CHECK_CUSOLVER(cusolverSpDestroyCsrluInfoHost(info));

	return determinant;
}

void solver::reorganize_terminal_sccs()
{
	// vector of vertex indices
	sccs_ =
		d_idxvec(thrust::make_counting_iterator<index_t>(0), thrust::make_counting_iterator<index_t>(labels_.size()));

	// vector of terminal scc begins and ends
	sccs_offsets_.reserve(terminals_.size() + 1);
	sccs_offsets_.push_back(0);

	// we partition labels_ and sccs multiple times so the ordering is T1, ..., Tn, NT1, ..., NTn
	auto partition_point = thrust::make_zip_iterator(sccs_.begin(), labels_.begin());
	for (auto it = terminals_.begin(); it != terminals_.end(); it++)
	{
		partition_point =
			thrust::stable_partition(partition_point, thrust::make_zip_iterator(sccs_.end(), labels_.end()),
									 [terminal_idx = *it] __device__(thrust::tuple<index_t, index_t> x) {
										 return thrust::get<1>(x) == terminal_idx;
									 });

		print("partitioned sccs ", sccs_);
		print("partitioned labs ", labels_);

		sccs_offsets_.push_back(thrust::get<0>(partition_point.get_iterator_tuple()) - sccs_.begin());
	}
}

index_t solver::take_submatrix(index_t n, d_idxvec::const_iterator vertices_subset_begin, d_idxvec& submatrix_indptr,
							   d_idxvec& submatrix_rows, thrust::device_vector<float>& submatrix_data,
							   bool mapping_prefilled)
{
	submatrix_indptr.resize(n + 1);
	submatrix_indptr[0] = 0;

	print("vertices mapping ", submatrix_vertex_mapping_);

	// this creates indptr of scc in CSC
	{
		auto scc_begins_b = thrust::make_permutation_iterator(indptr_.begin(), vertices_subset_begin);
		auto scc_begins_e = thrust::make_permutation_iterator(indptr_.begin(), vertices_subset_begin + n);

		auto scc_ends_b = thrust::make_permutation_iterator(indptr_.begin() + 1, vertices_subset_begin);
		auto scc_ends_e = thrust::make_permutation_iterator(indptr_.begin() + 1, vertices_subset_begin + n);

		// first get sizes of each col - also add 1 for diagonal part
		thrust::transform(
			thrust::make_zip_iterator(scc_begins_b, scc_ends_b), thrust::make_zip_iterator(scc_begins_e, scc_ends_e),
			submatrix_indptr.begin() + 1,
			[] __device__(thrust::tuple<index_t, index_t> x) { return 1 + thrust::get<1>(x) - thrust::get<0>(x); });

		print("submatrix_indptr sizes before ", submatrix_indptr);

		thrust::inclusive_scan(submatrix_indptr.begin(), submatrix_indptr.end(), submatrix_indptr.begin());

		print("submatrix_indptr sizes after  ", submatrix_indptr);
	}

	index_t nnz = submatrix_indptr.back();
	submatrix_rows.resize(nnz);
	submatrix_data = thrust::device_vector<float>(nnz, 1.f);

	// this creates rows and data of scc
	{
		int blocksize = 512;
		int gridsize = (n + blocksize - 1) / blocksize;
		scatter_rows_data<<<gridsize, blocksize>>>(submatrix_indptr.data().get(), submatrix_rows.data().get(),
												   submatrix_data.data().get(), rows_.data().get(),
												   indptr_.data().get(), (&*vertices_subset_begin).get(), n);

		CHECK_CUDA(cudaDeviceSynchronize());
	}

	// finally we transform rows so they start from 0
	{
		print("submatrix_rows before  ", submatrix_rows);

		if (!mapping_prefilled)
		{
			// create map for scc vertices so they start from 0
			thrust::copy(thrust::make_counting_iterator<intptr_t>(0), thrust::make_counting_iterator<intptr_t>(n),
						 thrust::make_permutation_iterator(submatrix_vertex_mapping_.begin(), vertices_subset_begin));
		}

		thrust::transform(submatrix_rows.begin(), submatrix_rows.end(), submatrix_rows.begin(),
						  [map = submatrix_vertex_mapping_.data().get()] __device__(index_t x) { return map[x]; });

		print("submatrix_rows after   ", submatrix_rows);
	}

	return nnz;
}

void solver::solve_terminal_part()
{
	d_idxvec reverse_labels(labels_.size());

	// BEWARE this expects that scc_offsets_ contains just terminal scc indices
	term_indptr = sccs_offsets_;
	term_rows.resize(sccs_offsets_.back());
	term_data.resize(sccs_offsets_.back());

	thrust::copy(sccs_.begin(), sccs_.begin() + sccs_offsets_.back(), term_rows.begin());

	for (size_t terminal_scc_idx = 1; terminal_scc_idx < sccs_offsets_.size(); terminal_scc_idx++)
	{
		size_t scc_size = sccs_offsets_[terminal_scc_idx] - sccs_offsets_[terminal_scc_idx - 1];

		if (scc_size == 1)
		{
			term_data[sccs_offsets_[terminal_scc_idx - 1]] = 1;
			continue;
		}

		d_idxvec scc_indptr, scc_rows;
		thrust::device_vector<float> scc_data;

		auto nnz = take_submatrix(scc_size, sccs_.begin() + sccs_offsets_[terminal_scc_idx - 1], scc_indptr, scc_rows,
								  scc_data);

		d_idxvec scc_cols(nnz);

		// this decompresses indptr into cols
		CHECK_CUSPARSE(cusparseXcsr2coo(context_.cusparse_handle, scc_indptr.data().get(), nnz, scc_size,
										scc_cols.data().get(), CUSPARSE_INDEX_BASE_ZERO));

		print("scc_cols ", scc_cols);

		std::cout << "Row to remove" << scc_size - 1 << std::endl;

		// this removes last row
		{
			auto part_point = thrust::stable_partition(
				thrust::make_zip_iterator(scc_rows.begin(), scc_cols.begin(), scc_data.begin()),
				thrust::make_zip_iterator(scc_rows.end(), scc_cols.end(), scc_data.end()),
				[remove_row = scc_size - 1] __device__(thrust::tuple<index_t, index_t, float> x) {
					return thrust::get<0>(x) != remove_row;
				});

			auto removed_n = thrust::get<0>(part_point.get_iterator_tuple()) - scc_rows.begin();

			scc_rows.resize(removed_n);
			scc_cols.resize(removed_n);
			scc_data.resize(removed_n);
		}

		// this compresses rows back into indptr
		CHECK_CUSPARSE(cusparseXcoo2csr(context_.cusparse_handle, scc_cols.data().get(), scc_cols.size(), scc_size,
										scc_indptr.data().get(), CUSPARSE_INDEX_BASE_ZERO));

		print("scc_indptr -1r: ", scc_indptr);
		print("scc_rows   -1r: ", scc_rows);
		print("scc_data   -1r: ", scc_data);
		print("scc_cols   -1r: ", scc_cols);

		// now we do minors
		d_idxvec minor_indptr(scc_indptr.size()), minor_rows(scc_rows.size());
		thrust::device_vector<float> minor_data(scc_rows.size());

		thrust::host_vector<index_t> h_scc_indptr = scc_indptr;

		thrust::host_vector<float> h_minors(scc_size);
		for (size_t minor_i = 0; minor_i < scc_size; minor_i++)
		{
			// copy indptr
			thrust::copy(scc_indptr.begin(), scc_indptr.begin() + minor_i + 1, minor_indptr.begin());
			auto offset = h_scc_indptr[minor_i + 1] - h_scc_indptr[minor_i];
			thrust::transform(scc_indptr.begin() + minor_i + 2, scc_indptr.end(), minor_indptr.begin() + minor_i + 1,
							  [offset] __device__(index_t x) { return x - offset; });

			// copy rows
			thrust::copy(scc_rows.begin(), scc_rows.begin() + h_scc_indptr[minor_i], minor_rows.begin());
			thrust::copy(scc_rows.begin() + h_scc_indptr[minor_i + 1], scc_rows.end(),
						 minor_rows.begin() + h_scc_indptr[minor_i]);
			// copy data
			thrust::copy(scc_data.begin(), scc_data.begin() + h_scc_indptr[minor_i], minor_data.begin());
			thrust::copy(scc_data.begin() + h_scc_indptr[minor_i + 1], scc_data.end(),
						 minor_data.begin() + h_scc_indptr[minor_i]);

			print("indptr -1c: ", minor_indptr);
			print("rows   -1c: ", minor_rows);
			print("data   -1c: ", minor_data);

			h_minors[minor_i] = std::abs(
				determinant(minor_indptr, minor_rows, minor_data, scc_indptr.size() - 2, scc_data.size() - offset));
		}

		thrust::device_vector<float> minors = h_minors;
		auto sum = thrust::reduce(minors.begin(), minors.end(), 0.f, thrust::plus<float>());

		thrust::transform(minors.begin(), minors.end(), term_data.begin() + sccs_offsets_[terminal_scc_idx - 1],
						  [sum] __device__(float x) { return x / sum; });
	}
}

void solver::csr_csc_switch(const index_t* in_indptr, const index_t* in_indices, const float* in_data, index_t in_n,
							index_t out_n, index_t nnz, d_idxvec& out_indptr, d_idxvec& out_indices,
							thrust::device_vector<float>& out_data)
{
	out_indptr.resize(out_n + 1);
	out_indices.resize(nnz);
	out_data.resize(nnz);

	size_t buffersize;
	cusparseCsr2cscEx2_bufferSize(context_.cusparse_handle, in_n, out_n, nnz, in_data, in_indptr, in_indices,
								  out_data.data().get(), out_indptr.data().get(), out_indices.data().get(), CUDA_R_32F,
								  CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG_DEFAULT,
								  &buffersize);

	thrust::device_vector<char> buffer(buffersize);
	cusparseCsr2cscEx2(context_.cusparse_handle, in_n, out_n, nnz, in_data, in_indptr, in_indices,
					   out_data.data().get(), out_indptr.data().get(), out_indices.data().get(), CUDA_R_32F,
					   CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG_DEFAULT,
					   buffer.data().get());
}

void solver::matmul(index_t* lhs_indptr, index_t* lhs_indices, float* lhs_data, index_t lhs_rows, index_t lhs_cols,
					index_t lhs_nnz, index_t* rhs_indptr, index_t* rhs_indices, float* rhs_data, index_t rhs_rows,
					index_t rhs_cols, index_t rhs_nnz, d_idxvec& out_indptr, d_idxvec& out_indices,
					thrust::device_vector<float>& out_data)
{
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
		context_.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, lhs_descr,
		rhs_descr, &beta, out_descr, CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize1, NULL));

	thrust::device_vector<char> buffer1(bufferSize1);

	// inspect the matrices A and B to understand the memory requirement for
	// the next step
	CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(context_.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
												 CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, lhs_descr, rhs_descr, &beta,
												 out_descr, CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc,
												 &bufferSize1, buffer1.data().get()));

	size_t bufferSize2;
	// ask bufferSize2 bytes for external memory
	CHECK_CUSPARSE(cusparseSpGEMM_compute(
		context_.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, lhs_descr,
		rhs_descr, &beta, out_descr, CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize2, NULL));

	thrust::device_vector<char> buffer2(bufferSize2);

	// compute the intermediate product of A * B
	CHECK_CUSPARSE(cusparseSpGEMM_compute(context_.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
										  CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, lhs_descr, rhs_descr, &beta,
										  out_descr, CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize2,
										  buffer2.data().get()));

	// get matrix C non-zero entries C_nnz1
	int64_t out_rows, out_cols, out_nnz;
	CHECK_CUSPARSE(cusparseSpMatGetSize(out_descr, &out_rows, &out_cols, &out_nnz));
	// allocate matrix C

	out_indptr.resize(out_rows + 1);
	out_indices.resize(out_nnz);
	out_data.resize(out_nnz);

	// NOTE: if 'beta' != 0, the values of C must be update after the allocation
	//       of dC_values, and before the call of cusparseSpGEMM_copy

	// update matC with the new pointers
	CHECK_CUSPARSE(
		cusparseCsrSetPointers(out_descr, out_indptr.data().get(), out_indices.data().get(), out_data.data().get()));

	// if beta != 0, cusparseSpGEMM_copy reuses/updates the values of dC_values

	// copy the final products to the matrix C
	CHECK_CUSPARSE(cusparseSpGEMM_copy(context_.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
									   CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, lhs_descr, rhs_descr, &beta, out_descr,
									   CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc));

	// destroy matrix/vector descriptors
	CHECK_CUSPARSE(cusparseSpGEMM_destroyDescr(spgemmDesc));
	CHECK_CUSPARSE(cusparseDestroySpMat(lhs_descr));
	CHECK_CUSPARSE(cusparseDestroySpMat(rhs_descr));
	CHECK_CUSPARSE(cusparseDestroySpMat(out_descr));
}

void solver::solve_tri_system()
{
	// Suppose that A is m x m sparse matrix represented by CSR format,
	// Assumption:
	// - handle is already created by cusparseCreate(),
	// - (d_csrRowPtr, d_csrColInd, d_csrVal) is CSR of A on device memory,
	// - d_x is right hand side vector on device memory,
	// - d_y is solution vector on device memory.
	// - d_z is intermediate result on device memory.

	cusparseMatDescr_t descr_M = 0;
	cusparseMatDescr_t descr_L = 0;
	cusparseMatDescr_t descr_U = 0;
	csrilu02Info_t info_M = 0;
	csrsv2Info_t info_L = 0;
	csrsv2Info_t info_U = 0;
	int pBufferSize_M;
	int pBufferSize_L;
	int pBufferSize_U;
	int pBufferSize;
	void* pBuffer = 0;
	int structural_zero;
	int numerical_zero;
	const double alpha = 1.;
	const cusparseSolvePolicy_t policy_M = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
	const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
	const cusparseSolvePolicy_t policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
	const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE;
	const cusparseOperation_t trans_U = CUSPARSE_OPERATION_NON_TRANSPOSE;

	// step 1: create a descriptor which contains
	// - matrix M is base-1
	// - matrix L is base-1
	// - matrix L is lower triangular
	// - matrix L has unit diagonal
	// - matrix U is base-1
	// - matrix U is upper triangular
	// - matrix U has non-unit diagonal
	cusparseCreateMatDescr(&descr_M);
	cusparseSetMatIndexBase(descr_M, CUSPARSE_INDEX_BASE_ONE);
	cusparseSetMatType(descr_M, CUSPARSE_MATRIX_TYPE_GENERAL);

	cusparseCreateMatDescr(&descr_L);
	cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ONE);
	cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER);
	cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_UNIT);

	cusparseCreateMatDescr(&descr_U);
	cusparseSetMatIndexBase(descr_U, CUSPARSE_INDEX_BASE_ONE);
	cusparseSetMatType(descr_U, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatFillMode(descr_U, CUSPARSE_FILL_MODE_UPPER);
	cusparseSetMatDiagType(descr_U, CUSPARSE_DIAG_TYPE_NON_UNIT);

	// step 2: create a empty info structure
	// we need one info for csrilu02 and two info's for csrsv2
	cusparseCreateCsrilu02Info(&info_M);
	cusparseCreateCsrsv2Info(&info_L);
	cusparseCreateCsrsv2Info(&info_U);

	cusparseSbsrsm2_bufferSize()

	// step 3: query how much memory used in csrilu02 and csrsv2, and allocate the buffer
	cusparseDcsrilu02_bufferSize(handle, m, nnz, descr_M, d_csrVal, d_csrRowPtr, d_csrColInd, info_M, &pBufferSize_M);
	cusparseDcsrsv2_bufferSize(handle, trans_L, m, nnz, descr_L, d_csrVal, d_csrRowPtr, d_csrColInd, info_L,
							   &pBufferSize_L);
	cusparseDcsrsv2_bufferSize(handle, trans_U, m, nnz, descr_U, d_csrVal, d_csrRowPtr, d_csrColInd, info_U,
							   &pBufferSize_U);

	pBufferSize = max(pBufferSize_M, max(pBufferSize_L, pBufferSize_U));

	// pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
	cudaMalloc((void**)&pBuffer, pBufferSize);

	// step 4: perform analysis of incomplete Cholesky on M
	//         perform analysis of triangular solve on L
	//         perform analysis of triangular solve on U
	// The lower(upper) triangular part of M has the same sparsity pattern as L(U),
	// we can do analysis of csrilu0 and csrsv2 simultaneously.

	cusparseDcsrilu02_analysis(handle, m, nnz, descr_M, d_csrVal, d_csrRowPtr, d_csrColInd, info_M, policy_M, pBuffer);
	status = cusparseXcsrilu02_zeroPivot(handle, info_M, &structural_zero);
	if (CUSPARSE_STATUS_ZERO_PIVOT == status)
	{
		printf("A(%d,%d) is missing\n", structural_zero, structural_zero);
	}

	cusparseDcsrsv2_analysis(handle, trans_L, m, nnz, descr_L, d_csrVal, d_csrRowPtr, d_csrColInd, info_L, policy_L,
							 pBuffer);

	cusparseDcsrsv2_analysis(handle, trans_U, m, nnz, descr_U, d_csrVal, d_csrRowPtr, d_csrColInd, info_U, policy_U,
							 pBuffer);

	// step 5: M = L * U
	cusparseDcsrilu02(handle, m, nnz, descr_M, d_csrVal, d_csrRowPtr, d_csrColInd, info_M, policy_M, pBuffer);
	status = cusparseXcsrilu02_zeroPivot(handle, info_M, &numerical_zero);
	if (CUSPARSE_STATUS_ZERO_PIVOT == status)
	{
		printf("U(%d,%d) is zero\n", numerical_zero, numerical_zero);
	}

	// step 6: solve L*z = x
	cusparseDcsrsv2_solve(handle, trans_L, m, nnz, &alpha, descr_L, d_csrVal, d_csrRowPtr, d_csrColInd, info_L, d_x,
						  d_z, policy_L, pBuffer);

	// step 7: solve U*y = z
	cusparseDcsrsv2_solve(handle, trans_U, m, nnz, &alpha, descr_U, d_csrVal, d_csrRowPtr, d_csrColInd, info_U, d_z,
						  d_y, policy_U, pBuffer);

	// step 6: free resources
	cudaFree(pBuffer);
	cusparseDestroyMatDescr(descr_M);
	cusparseDestroyMatDescr(descr_L);
	cusparseDestroyMatDescr(descr_U);
	cusparseDestroyCsrilu02Info(info_M);
	cusparseDestroyCsrsv2Info(info_L);
	cusparseDestroyCsrsv2Info(info_U);
	cusparseDestroy(handle);
}

void solver::solve_nonterminal_part()
{
	index_t n = labels_.size();
	index_t terminal_vertices_n = sccs_offsets_.back();
	index_t nonterminal_vertices_n = n - terminal_vertices_n;

	// -U
	d_idxvec& U_indptr_csr = term_indptr;
	d_idxvec U_cols(term_rows.size());

	thrust::transform(term_rows.begin(), term_rows.end(), U_cols.begin(),
					  [offset = terminal_vertices_n] __device__(index_t x) { return x - offset; });

	thrust::device_vector<float> U_data(term_rows.size(), -1.f);

	// NB

	d_idxvec nb_indptr_csc, nb_rows;
	thrust::device_vector<float> nb_data_csc;

	thrust::copy(
		thrust::make_counting_iterator<intptr_t>(0), thrust::make_counting_iterator<intptr_t>(n),
		thrust::make_permutation_iterator(submatrix_vertex_mapping_.begin(), sccs_.begin() + sccs_offsets_.back()));

	thrust::copy(thrust::make_counting_iterator<intptr_t>(n), thrust::make_counting_iterator<intptr_t>(labels_.size()),
				 thrust::make_permutation_iterator(submatrix_vertex_mapping_.begin(), sccs_.begin()));

	auto nnz = take_submatrix(n, sccs_.begin() + sccs_offsets_.back(), nb_indptr_csc, nb_rows, nb_data_csc, true);

	d_idxvec nb_indptr_csr, nb_cols;
	thrust::device_vector<float> nb_data_csr;

	csr_csc_switch(nb_indptr_csc.data().get(), nb_rows.data().get(), nb_data_csc.data().get(), nonterminal_vertices_n,
				   nonterminal_vertices_n + terminal_vertices_n, nnz, nb_indptr_csr, nb_cols, nb_data_csr);

	index_t n_nnz = nb_indptr_csr[nonterminal_vertices_n];
	index_t b_nnz = nnz - n_nnz;

	thrust::transform(nb_indptr_csr.begin() + nonterminal_vertices_n, nb_indptr_csr.end(),
					  nb_indptr_csr.begin() + nonterminal_vertices_n,
					  [n_nnz] __device__(index_t x) { return x - n_nnz; });

	d_idxvec A_indptr, A_indices;
	thrust::device_vector<float> A_data;

	matmul(U_indptr_csr.data().get(), U_cols.data().get(), U_data.data().get(), sccs_offsets_.size() - 1,
		   terminal_vertices_n, U_cols.size(), nb_indptr_csr.data().get() + nonterminal_vertices_n,
		   nb_cols.data().get() + nonterminal_vertices_n, nb_data_csr.data().get() + nonterminal_vertices_n,
		   terminal_vertices_n, nonterminal_vertices_n, b_nnz, A_indptr, A_indices, A_data);

	nb_indptr_csr[nonterminal_vertices_n] = n_nnz;

	cusparseScscsm
}

void solver::solve()
{
	reorganize_terminal_sccs();
	solve_terminal_part();
}
