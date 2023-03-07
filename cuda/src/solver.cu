#include <cusolverSp_LOWLEVEL_PREVIEW.h>
#include <cusparse_v2.h>
#include <device_launch_parameters.h>

#include <thrust/adjacent_difference.h>
#include <thrust/execution_policy.h>
#include <thrust/partition.h>

constexpr float zero_threshold = 1e-10f;

__device__ __host__ bool is_zero(float x) { return (x > 0 ? x : -x) <= zero_threshold; }

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

__global__ void hstack(const index_t* __restrict__ out_indptr, index_t* __restrict__ out_indices,
					   float* __restrict__ out_data, const index_t* __restrict__ lhs_indptr,
					   const index_t* __restrict__ rhs_indptr, const index_t* __restrict__ lhs_indices,
					   const index_t* __restrict__ rhs_indices, const float* __restrict__ lhs_data,
					   const float* __restrict__ rhs_data, int n)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx >= 2 * n)
		return;

	const index_t* __restrict__ my_indptr = (idx >= n) ? rhs_indptr : lhs_indptr;
	const index_t* __restrict__ my_indices = (idx >= n) ? rhs_indices : lhs_indices;
	const float* __restrict__ my_data = (idx >= n) ? rhs_data : lhs_data;
	const int my_offset = (idx >= n) ? lhs_indptr[idx - n + 1] - lhs_indptr[idx - n] : 0;
	idx -= (idx >= n) ? n : 0;

	auto out_begin = out_indptr[idx] + my_offset;
	auto in_begin = my_indptr[idx];

	auto count = my_indptr[idx + 1] - in_begin;

	for (int i = 0; i < count; i++)
	{
		out_indices[out_begin + i] = my_indices[in_begin + i];
		out_data[out_begin + i] = my_data[in_begin + i];
	}
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

		sccs_offsets_.push_back(thrust::get<0>(partition_point.get_iterator_tuple()) - sccs_.begin());
	}
}

index_t solver::take_submatrix(index_t n, d_idxvec::const_iterator vertices_subset_begin, d_idxvec& submatrix_indptr,
							   d_idxvec& submatrix_rows, thrust::device_vector<float>& submatrix_data,
							   bool mapping_prefilled)
{
	submatrix_indptr.resize(n + 1);
	submatrix_indptr[0] = 0;

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

		thrust::inclusive_scan(submatrix_indptr.begin(), submatrix_indptr.end(), submatrix_indptr.begin());
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
		if (!mapping_prefilled)
		{
			// create map for scc vertices so they start from 0
			thrust::copy(thrust::make_counting_iterator<intptr_t>(0), thrust::make_counting_iterator<intptr_t>(n),
						 thrust::make_permutation_iterator(submatrix_vertex_mapping_.begin(), vertices_subset_begin));
		}

		thrust::transform(submatrix_rows.begin(), submatrix_rows.end(), submatrix_rows.begin(),
						  [map = submatrix_vertex_mapping_.data().get()] __device__(index_t x) { return map[x]; });
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
	CHECK_CUSPARSE(cusparseCsr2cscEx2_bufferSize(context_.cusparse_handle, in_n, out_n, nnz, in_data, in_indptr,
												 in_indices, out_data.data().get(), out_indptr.data().get(),
												 out_indices.data().get(), CUDA_R_32F, CUSPARSE_ACTION_NUMERIC,
												 CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG_DEFAULT, &buffersize));

	thrust::device_vector<char> buffer(buffersize);
	CHECK_CUSPARSE(cusparseCsr2cscEx2(context_.cusparse_handle, in_n, out_n, nnz, in_data, in_indptr, in_indices,
									  out_data.data().get(), out_indptr.data().get(), out_indices.data().get(),
									  CUDA_R_32F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
									  CUSPARSE_CSR2CSC_ALG_DEFAULT, buffer.data().get()));
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

void solver::solve_tri_system(const d_idxvec& indptr, const d_idxvec& rows, const thrust::device_vector<float>& data,
							  int n, int cols, int nnz, const d_idxvec& b_indptr, const d_idxvec& b_indices,
							  const thrust::device_vector<float>& b_data, d_idxvec& x_indptr, d_idxvec& x_indices,
							  thrust::device_vector<float>& x_data)
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


	thrust::host_vector<index_t> hb_indptr = b_indptr;
	thrust::host_vector<index_t> hb_indices = b_indices;


	thrust::host_vector<float> x_vec(cols);

	thrust::host_vector<index_t> hx_indptr(b_indptr.size());
	hx_indptr[0] = 0;

	for (int b_idx = 0; b_idx < b_indptr.size() - 1; b_idx++)
	{
		thrust::host_vector<float> b_vec(n, 0.f);
		auto start = hb_indptr[b_idx];
		auto end = hb_indptr[b_idx + 1];
		thrust::copy(b_data.begin() + start, b_data.begin() + end,
					 thrust::make_permutation_iterator(b_vec.begin(), hb_indices.begin() + start));

		CHECK_CUSOLVER(
			cusolverSpScsrluSolveHost(context_.cusolver_handle, n, b_vec.data(), x_vec.data(), info, buffer.data()));

		thrust::device_vector<float> dx_vec = x_vec;

		auto x_nnz = thrust::count_if(dx_vec.begin(), dx_vec.end(), [] __device__(float x) { return !is_zero(x); });

		auto size_before = x_indices.size();
		x_indices.resize(x_indices.size() + x_nnz);
		x_data.resize(x_data.size() + x_nnz);

		hx_indptr[b_idx + 1] = hx_indptr[b_idx] + x_nnz;

		thrust::copy_if(thrust::make_zip_iterator(dx_vec.begin(), thrust::make_counting_iterator<index_t>(0)),
						thrust::make_zip_iterator(dx_vec.end(), thrust::make_counting_iterator<index_t>(dx_vec.size())),
						thrust::make_zip_iterator(x_data.begin() + size_before, x_indices.begin() + size_before),
						[] __device__(thrust::tuple<float, index_t> x) { return !is_zero(thrust::get<0>(x)); });
	}

	x_indptr = hx_indptr;

	CHECK_CUSPARSE(cusparseDestroyMatDescr(descr));
	CHECK_CUSPARSE(cusparseDestroyMatDescr(descr_L));
	CHECK_CUSPARSE(cusparseDestroyMatDescr(descr_U));
	CHECK_CUSOLVER(cusolverSpDestroyCsrluInfoHost(info));
}

void solver::solve_nonterminal_part()
{
	index_t n = labels_.size();
	index_t terminal_vertices_n = sccs_offsets_.back();
	index_t nonterminal_vertices_n = n - terminal_vertices_n;

	std::cout << "terminal vertices " << terminal_vertices_n << std::endl;
	std::cout << "nonterminal vertices " << nonterminal_vertices_n << std::endl;

	if (nonterminal_vertices_n == 0)
	{
		nonterm_indptr = term_indptr;
		nonterm_cols = term_rows;
		nonterm_data = thrust::device_vector<float>(term_rows.size(), 1.f);

		return;
	}

	// -U
	d_idxvec& U_indptr_csr = term_indptr;
	d_idxvec U_cols(term_rows.size());

	thrust::copy(thrust::make_counting_iterator<intptr_t>(0),
				 thrust::make_counting_iterator<intptr_t>(terminal_vertices_n),
				 thrust::make_permutation_iterator(submatrix_vertex_mapping_.begin(), sccs_.begin()));

	thrust::transform(sccs_.begin(), sccs_.begin() + sccs_offsets_.back(), U_cols.begin(),
					  [map = submatrix_vertex_mapping_.data().get()] __device__(index_t x) { return map[x]; });

	thrust::device_vector<float> U_data(term_rows.size(), -1.f);

	// NB

	d_idxvec nb_indptr_csc, nb_rows;
	thrust::device_vector<float> nb_data_csc;

	// custom mapping
	{
		thrust::copy(
			thrust::make_counting_iterator<intptr_t>(0),
			thrust::make_counting_iterator<intptr_t>(nonterminal_vertices_n),
			thrust::make_permutation_iterator(submatrix_vertex_mapping_.begin(), sccs_.begin() + sccs_offsets_.back()));

		thrust::copy(thrust::make_counting_iterator<intptr_t>(nonterminal_vertices_n),
					 thrust::make_counting_iterator<intptr_t>(labels_.size()),
					 thrust::make_permutation_iterator(submatrix_vertex_mapping_.begin(), sccs_.begin()));
	}

	// create vstack(N, B) matrix in csc
	auto nnz = take_submatrix(nonterminal_vertices_n, sccs_.begin() + sccs_offsets_.back(), nb_indptr_csc, nb_rows,
							  nb_data_csc, true);

	d_idxvec nb_indptr_csr, nb_cols;
	thrust::device_vector<float> nb_data_csr;

	// vstack(N, B) to csr
	csr_csc_switch(nb_indptr_csc.data().get(), nb_rows.data().get(), nb_data_csc.data().get(), nonterminal_vertices_n,
				   nonterminal_vertices_n + terminal_vertices_n, nnz, nb_indptr_csr, nb_cols, nb_data_csr);


	index_t n_nnz = nb_indptr_csr[nonterminal_vertices_n];
	index_t b_nnz = nnz - n_nnz;

	std::cout << "N nnz " << n_nnz << std::endl;
	std::cout << "B nnz " << b_nnz << std::endl;

	// offset B part of indptr
	thrust::transform(nb_indptr_csr.begin() + nonterminal_vertices_n, nb_indptr_csr.end(),
					  nb_indptr_csr.begin() + nonterminal_vertices_n,
					  [n_nnz] __device__(index_t x) { return x - n_nnz; });

	d_idxvec A_indptr, A_indices;
	thrust::device_vector<float> A_data;

	matmul(U_indptr_csr.data().get(), U_cols.data().get(), U_data.data().get(), sccs_offsets_.size() - 1,
		   terminal_vertices_n, U_cols.size(), nb_indptr_csr.data().get() + nonterminal_vertices_n,
		   nb_cols.data().get() + n_nnz, nb_data_csr.data().get() + n_nnz, terminal_vertices_n, nonterminal_vertices_n,
		   b_nnz, A_indptr, A_indices, A_data);

	nb_indptr_csr[nonterminal_vertices_n] = n_nnz;

	csr_csc_switch(nb_indptr_csr.data().get(), nb_cols.data().get(), nb_data_csr.data().get(), nonterminal_vertices_n,
				   nonterminal_vertices_n, n_nnz, nb_indptr_csc, nb_rows, nb_data_csc);

	nb_indptr_csc.resize(nonterminal_vertices_n + 1);
	nb_rows.resize(n_nnz);
	nb_data_csc.resize(n_nnz);

	d_idxvec X_indptr, X_indices;
	thrust::device_vector<float> X_data;

	solve_tri_system(nb_indptr_csc, nb_rows, nb_data_csc, nonterminal_vertices_n, nonterminal_vertices_n, n_nnz,
					 A_indptr, A_indices, A_data, X_indptr, X_indices, X_data);


	nonterm_indptr.resize(U_indptr_csr.size());
	index_t nonterm_nnz = U_indptr_csr.back() + X_indptr.back();
	nonterm_cols.resize(nonterm_nnz);
	nonterm_data.resize(nonterm_nnz);

	thrust::transform(
		thrust::make_zip_iterator(U_indptr_csr.begin(), X_indptr.begin()),
		thrust::make_zip_iterator(U_indptr_csr.end(), X_indptr.end()), nonterm_indptr.begin(),
		[] __device__(thrust::tuple<index_t, index_t> x) { return thrust::get<0>(x) + thrust::get<1>(x); });


	// -U back to U
	thrust::transform(U_data.begin(), U_data.end(), U_data.begin(), thrust::negate<float>());

	// nonterminal vertices from 0, ..., n_nt to actual indices
	{
		thrust::copy(sccs_.begin() + sccs_offsets_.back(), sccs_.end(), submatrix_vertex_mapping_.begin());

		thrust::transform(X_indices.begin(), X_indices.end(), X_indices.begin(),
						  [map = submatrix_vertex_mapping_.data().get()] __device__(index_t x) { return map[x]; });
	}


	// hstack(U,X)
	{
		int blocksize = 512;
		int gridsize = (2 * (nonterm_indptr.size() - 1) + blocksize - 1) / blocksize;

		std::cout << "blockxgrid size " << blocksize << "x" << gridsize << std::endl;

		hstack<<<gridsize, blocksize>>>(nonterm_indptr.data().get(), nonterm_cols.data().get(),
										nonterm_data.data().get(), U_indptr_csr.data().get(), X_indptr.data().get(),
										term_rows.data().get(), X_indices.data().get(), U_data.data().get(),
										X_data.data().get(), nonterm_indptr.size() - 1);

		CHECK_CUDA(cudaDeviceSynchronize());
	}
}

void solver::compute_final_states()
{
	thrust::device_vector<float> y(terminals_.size());
	{
		float alpha = 1.0f;
		float beta = 0.0f;

		cusparseSpMatDescr_t matA;
		cusparseDnVecDescr_t vecX, vecY;
		size_t bufferSize = 0;

		// Create sparse matrix A in CSR format
		CHECK_CUSPARSE(cusparseCreateCsr(&matA, terminals_.size(), labels_.size(), nonterm_data.size(),
										 nonterm_indptr.data().get(), nonterm_cols.data().get(),
										 nonterm_data.data().get(), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
										 CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
		// Create dense vector X
		CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, labels_.size(), initial_state_.data().get(), CUDA_R_32F));
		// Create dense vector y
		CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, terminals_.size(), y.data().get(), CUDA_R_32F));
		// allocate an external buffer if needed
		CHECK_CUSPARSE(cusparseSpMV_bufferSize(context_.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA,
											   vecX, &beta, vecY, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
		thrust::device_vector<char> buffer(bufferSize);

		// execute SpMV
		CHECK_CUSPARSE(cusparseSpMV(context_.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX,
									&beta, vecY, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, buffer.data().get()));

		// destroy matrix/vector descriptors
		CHECK_CUSPARSE(cusparseDestroySpMat(matA));
		CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
		CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
	}

	final_state.resize(labels_.size());
	{
		float alpha = 1.0f;
		float beta = 0.0f;

		cusparseSpMatDescr_t matA;
		cusparseDnVecDescr_t vecX, vecY;
		size_t bufferSize = 0;

		// Create sparse matrix A in CSC format
		CHECK_CUSPARSE(cusparseCreateCsc(&matA, labels_.size(), terminals_.size(), term_data.size(),
										 term_indptr.data().get(), term_rows.data().get(), term_data.data().get(),
										 CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
		// Create dense vector y
		CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, terminals_.size(), y.data().get(), CUDA_R_32F));
		// Create dense vector final
		CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, labels_.size(), final_state.data().get(), CUDA_R_32F));
		// allocate an external buffer if needed
		CHECK_CUSPARSE(cusparseSpMV_bufferSize(context_.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA,
											   vecX, &beta, vecY, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
		thrust::device_vector<char> buffer(bufferSize);

		// execute SpMV
		CHECK_CUSPARSE(cusparseSpMV(context_.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX,
									&beta, vecY, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, buffer.data().get()));

		// destroy matrix/vector descriptors
		CHECK_CUSPARSE(cusparseDestroySpMat(matA));
		CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
		CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
	}
}

void solver::solve()
{
	reorganize_terminal_sccs();
	solve_terminal_part();
	solve_nonterminal_part();

	compute_final_states();
}
