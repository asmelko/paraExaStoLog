#include <cusolverSp_LOWLEVEL_PREVIEW.h>
#include <cusparse_v2.h>
#include <device_launch_parameters.h>

#include <thrust/adjacent_difference.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/partition.h>
#include <thrust/tabulate.h>

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
	  rows_(t.rows),
	  cols_(t.cols),
	  indptr_(t.indptr),
	  ordered_vertices_(std::move(g.reordered_vertices)),
	  submatrix_vertex_mapping_(ordered_vertices_.size())
{
	terminals_offsets_ =
		thrust::host_vector<index_t>(g.sccs_offsets.begin(), g.sccs_offsets.begin() + g.terminals_count + 1);

	nonterminals_offsets_ =
		thrust::host_vector<index_t>(g.sccs_offsets.begin() + g.terminals_count, g.sccs_offsets.end());
}

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


__global__ void partial_factorize(index_t n, index_t scc_size, const index_t* __restrict__ nonempty_rows,
								  const index_t* __restrict__ U_indptr, const index_t* __restrict__ U_indices,
								  const real_t* __restrict__ U_data, const index_t* __restrict__ indptr,
								  const index_t* __restrict__ indices, real_t* __restrict__ data,
								  real_t* __restrict__ out, real_t* __restrict__ scratch_data)
{
	auto idx = blockDim.x * blockIdx.x + threadIdx.x;

	scratch_data = scratch_data + idx * scc_size;

	for (int i = 0; i < scc_size; i++)
		scratch_data[i] = 0.f;

	if (idx >= n)
		return;

	auto row_id = nonempty_rows[idx];
	out = out + row_id * scc_size;

	auto row_begin = indptr[row_id];
	auto row_size = indptr[row_id + 1] - row_begin;

	auto min_col_idx = scc_size;
	// fill scratchpad with sorted data
	for (int i = row_begin; i < row_begin + row_size; i++)
	{
		auto col_idx = indices[i];

		min_col_idx = (min_col_idx > col_idx) ? col_idx : min_col_idx;

		scratch_data[col_idx] = data[i];
	}

	for (int col_idx = min_col_idx; col_idx < scc_size; col_idx++)
	{
		float data = scratch_data[col_idx];
		if (data == 0.f)
			continue;

		float pivot;

		// find pivot
		auto U_begin = U_indptr[col_idx];
		auto U_end = U_indptr[col_idx + 1];
		for (int U_i = U_begin; U_i < U_end; U_i++)
		{
			if (U_indices[U_i] == col_idx)
			{
				pivot = U_data[U_i];
				continue;
			}
		}

		float factor = -data / pivot;

		out[col_idx] = factor;

		for (int U_i = U_begin; U_i < U_end; U_i++)
		{
			scratch_data[U_indices[U_i]] += factor * U_data[U_i];
		}
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

void solver::LU_factorization(const index_t* indptr, const index_t* indices, const float* data, int n, int nnz,
							  d_idxvec& L_indptr, d_idxvec& L_indices, d_datvec& L_data, d_idxvec& U_indptr,
							  d_idxvec& U_indices, d_datvec& U_data)
{
	thrust::host_vector<index_t> h_indptr(indptr, indptr + n + 1);
	thrust::host_vector<index_t> h_rows(indices, indices + nnz);
	thrust::host_vector<float> h_data(data, data + nnz);

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

	std::vector<index_t> hP(n), hQ(n), hL_indptr(n + 1), hU_indptr(n + 1), hL_cols(nnz_l), hU_cols(nnz_u);
	std::vector<float> hL_data(nnz_l), hU_data(nnz_u);

	CHECK_CUSOLVER(cusolverSpScsrluExtractHost(context_.cusolver_handle, hP.data(), hQ.data(), descr_L, hL_data.data(),
											   hL_indptr.data(), hL_cols.data(), descr_U, hU_data.data(),
											   hU_indptr.data(), hU_cols.data(), info, buffer.data()));

	L_indptr = hL_indptr;
	L_indices = hL_cols;
	L_data = hL_data;

	U_indptr = hU_indptr;
	U_indices = hU_cols;
	U_data = hU_data;

	CHECK_CUSPARSE(cusparseDestroyMatDescr(descr));
	CHECK_CUSPARSE(cusparseDestroyMatDescr(descr_L));
	CHECK_CUSPARSE(cusparseDestroyMatDescr(descr_U));
	CHECK_CUSOLVER(cusolverSpDestroyCsrluInfoHost(info));
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
	// BEWARE this expects that scc_offsets_ contains just terminal scc indices

	std::cout << "to back" << terminals_offsets_.back() << std::endl;
	term_indptr = terminals_offsets_;
	term_rows.resize(terminals_offsets_.back());
	term_data.resize(terminals_offsets_.back());

	std::cout << "strat solving terminal part" << std::endl;

	thrust::copy(ordered_vertices_.begin(), ordered_vertices_.begin() + terminals_offsets_.back(), term_rows.begin());

	for (size_t terminal_scc_idx = 1; terminal_scc_idx < terminals_offsets_.size(); terminal_scc_idx++)
	{
		size_t scc_size = terminals_offsets_[terminal_scc_idx] - terminals_offsets_[terminal_scc_idx - 1];

		if (scc_size == 1)
		{
			term_data[terminals_offsets_[terminal_scc_idx - 1]] = 1;
			continue;
		}

		d_idxvec scc_indptr, scc_rows;
		thrust::device_vector<float> scc_data;

		auto nnz = take_submatrix(scc_size, ordered_vertices_.begin() + terminals_offsets_[terminal_scc_idx - 1],
								  scc_indptr, scc_rows, scc_data);

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

		thrust::transform(minors.begin(), minors.end(), term_data.begin() + terminals_offsets_[terminal_scc_idx - 1],
						  [sum] __device__(float x) { return x / sum; });
	}

	print("terminal indptr  ", term_indptr);
	print("terminal indices ", term_rows);
	print("terminal data    ", term_data);
}


void solver::transpose_sparse_matrix(cusparseHandle_t handle, const index_t* in_indptr, const index_t* in_indices,
									 const float* in_data, index_t in_n, index_t out_n, index_t nnz,
									 d_idxvec& out_indptr, d_idxvec& out_indices,
									 thrust::device_vector<float>& out_data)
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

void solver::csr_csc_switch(const index_t* in_indptr, const index_t* in_indices, const float* in_data, index_t in_n,
							index_t out_n, index_t nnz, d_idxvec& out_indptr, d_idxvec& out_indices,
							thrust::device_vector<float>& out_data)
{
	transpose_sparse_matrix(context_.cusparse_handle, in_indptr, in_indices, in_data, in_n, out_n, nnz, out_indptr,
							out_indices, out_data);
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

// input: block lower diagonal CSC matrix
void solver::solve_single_nonterm(index_t nonterm_idx, const d_idxvec& indptr, const d_idxvec& indices,
								  const thrust::device_vector<float>& data, d_idxvec& L_indptr, d_idxvec& L_indices,
								  d_datvec& L_data, d_idxvec& U_indptr, d_idxvec& U_indices, d_datvec& U_data)
{
	index_t scc_size = nonterminals_offsets_[nonterm_idx + 1] - nonterminals_offsets_[nonterm_idx];
	if (scc_size == 1)
		return;

	index_t mat_offset = nonterminals_offsets_[nonterm_idx] - nonterminals_offsets_.front();
	index_t included_terminals = indptr.size() - 1 - mat_offset;

	index_t scc_indices_begin = indptr[mat_offset];
	index_t scc_indices_end = indptr[mat_offset + scc_size + 1];

	d_idxvec scc_indptr_csc(scc_size + 1);
	d_idxvec scc_indices_csc(scc_indices_end - scc_indices_begin);
	thrust::device_vector<float> scc_data_csc(scc_indices_csc.size());

	thrust::transform(indptr.begin() + mat_offset, indptr.begin() + mat_offset + scc_size + 1, scc_indptr_csc.begin(),
					  [=] __device__(index_t x) { return x - scc_indices_begin; });
	thrust::copy(indices.begin() + scc_indices_begin, indices.begin() + scc_indices_end, scc_indices_csc.begin());
	thrust::copy(data.begin() + scc_indices_begin, data.begin() + scc_indices_end, scc_data_csc.begin());

	// make indices start from 0
	{
		thrust::copy(thrust::make_counting_iterator<intptr_t>(0),
					 thrust::make_counting_iterator<intptr_t>(included_terminals),
					 thrust::make_permutation_iterator(submatrix_vertex_mapping_.begin(),
													   ordered_vertices_.begin() + nonterminals_offsets_[nonterm_idx]));

		thrust::transform(scc_indices_csc.begin(), scc_indices_csc.end(), scc_indices_csc.begin(),
						  [map = submatrix_vertex_mapping_.data().get()] __device__(index_t x) { return map[x]; });
	}

	d_idxvec scc_indptr_csr, scc_indices_csr;
	thrust::device_vector<float> scc_data_csr;

	csr_csc_switch(scc_indptr_csc.data().get(), scc_indices_csc.data().get(), scc_data_csc.data().get(), scc_size,
				   included_terminals, scc_indices_csc.size(), scc_indptr_csr, scc_indices_csr, scc_data_csr);

	index_t scc_nnz = scc_indptr_csr[scc_size];
	index_t transitions_nnz = scc_data_csr.size() - scc_nnz;

	std::cout << "SCC nnz " << scc_nnz << std::endl;
	std::cout << "TRA nnz " << transitions_nnz << std::endl;

	LU_factorization(scc_indptr_csr.data().get(), scc_indices_csr.data().get(), scc_data_csr.data().get(), scc_size,
					 scc_nnz, L_indptr, L_indices, L_data, U_indptr, U_indices, U_data);

	// now offset scc_*_csr so we have only transitions

	d_idxvec nonempty_rows(included_terminals - scc_size);

	auto end = thrust::copy_if(
		thrust::make_zip_iterator(scc_indptr_csr.begin() + scc_size, scc_indptr_csr.begin() + scc_size + 1,
								  thrust::make_counting_iterator<index_t>(0)),
		thrust::make_zip_iterator(scc_indptr_csr.end() - 1, scc_indptr_csr.end(),
								  thrust::make_counting_iterator<index_t>(included_terminals - scc_size)),
		thrust::make_zip_iterator(thrust::make_constant_iterator<index_t>(0),
								  thrust::make_constant_iterator<index_t>(0), nonempty_rows.begin()),
		[] __device__(thrust::tuple<index_t, index_t, index_t> x) {
			return thrust::get<1>(x) - thrust::get<0>(x) != 0;
		});

	nonempty_rows.resize(thrust::get<2>(end.get_iterator_tuple()) - nonempty_rows.begin());

	auto L_data_trans_offset = L_data.size();
	L_data.resize(L_data.size() + nonempty_rows.size() * scc_size);
	thrust::fill(L_data.begin() + L_data_trans_offset, L_data.end(), 0);

	d_datvec scratch(nonempty_rows.size() * scc_size);

	{
		int blocksize = 512;
		int gridsize = (nonempty_rows.size() + blocksize - 1) / blocksize;

		partial_factorize<<<gridsize, blocksize>>>(
			nonempty_rows.size(), scc_size, nonempty_rows.data().get(), U_indptr.data().get(), U_indices.data().get(),
			U_data.data().get(), scc_indptr_csr.data().get() + scc_size, scc_indices_csr.data().get() + scc_nnz,
			scc_data_csr.data().get() + scc_nnz, L_data.data().get() + L_data_trans_offset, scratch.data().get());

		CHECK_CUDA(cudaDeviceSynchronize());
	}

	L_indices.resize(L_data.size());

	thrust::tabulate(L_indices.begin() + L_data_trans_offset, L_indices.end(),
					 [scc_size] __device__(index_t idx) { return idx % scc_size; });

	L_indptr.resize(included_terminals + 1);
	thrust::fill(L_indptr.begin() + scc_size + 1, L_indptr.end(), 0);

	thrust::for_each(
		nonempty_rows.begin(), nonempty_rows.end(),
		[ind = L_indptr.data().get() + scc_size, scc_size] __device__(index_t x) { ind[x + 1] = scc_size; });

	thrust::inclusive_scan(L_indptr.begin() + scc_size, L_indptr.end(), L_indptr.begin() + scc_size);
}

template <bool L>
__global__ void nway_hstack_indptr(index_t n, const index_t* __restrict__ N_indptr, const index_t* __restrict__ offsets,
								   const index_t* const* __restrict__ in_indptr, index_t* __restrict__ out_indptr)
{
	index_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= n)
		return;

	if (idx == 0)
		out_indptr[0] = 0;

	auto begin = offsets[idx];
	auto end = offsets[idx + 1];

	index_t one[2] = { 0, 1 };

	if constexpr (L)
		if (end - begin == 1)
		{
			one[1] = N_indptr[begin + 1] - N_indptr[begin];
			// printf("idx %i one %i \n", idx, one);
		}

	const index_t* my_indptr = (end - begin == 1) ? one : (in_indptr[idx]);

	for (index_t i = begin; i < end; i++)
	{
		out_indptr[i + 1] = my_indptr[begin - i + 1] - my_indptr[begin - i];
	}
}

// This is for all sccs of U and for non trivial sccs of L
template <bool L>
__global__ void nway_hstack_indices_and_data(index_t n, const index_t* __restrict__ N_indptr,
											 const index_t* __restrict__ N_indices, const float* __restrict__ N_data,
											 const index_t* __restrict__ offsets,
											 const index_t* const* __restrict__ in_indices,
											 const float* const* __restrict__ in_data,
											 const index_t* __restrict__ out_indptr, index_t* __restrict__ out_indices,
											 float* __restrict__ out_data)
{
	index_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= 2 * n)
		return;

	bool data = idx >= n;
	idx = idx % n;

	const auto indptr_begin = offsets[idx];
	const auto indptr_end = offsets[idx + 1];

	if constexpr (L)
		if (indptr_end - indptr_begin == 1)
			return;

	auto my_begin = out_indptr[indptr_begin];
	auto scc_nnz_size = out_indptr[indptr_end] - my_begin;

	if (!data)
	{
		index_t zero = 0;
		const index_t* my_indices = (scc_nnz_size == 1) ? &zero : in_indices[idx];

		for (index_t i = 0; i < scc_nnz_size; i++)
		{
			out_indices[my_begin + i] = my_indices[i] + indptr_begin;
		}
	}
	else
	{
		// find pivot (data with negative value)
		// optimization: make negative value always the first value
		float pivot;
		if (scc_nnz_size == 1)
		{
			auto N_begin = N_indptr[indptr_begin];
			auto N_end = N_indptr[indptr_begin + 1];
			for (auto i = N_begin; i < N_end; i++)
			{
				pivot = N_data[i];
				if (pivot < 0)
				{
					break;
				}
			}
		}
		const float* my_data = (scc_nnz_size == 1) ? &pivot : in_data[idx];

		for (index_t i = 0; i < scc_nnz_size; i++)
		{
			out_data[my_begin + i] = my_data[i];
		}
	}
}


__global__ void nway_hstack_indices_and_data_trivial_L(
	index_t n, const index_t* __restrict__ N_indptr, const index_t* __restrict__ N_indices,
	const float* __restrict__ N_data, const index_t* __restrict__ offsets,
	const index_t* const* __restrict__ in_indices, const float* const* __restrict__ in_data,
	const index_t* __restrict__ out_indptr, index_t* __restrict__ out_indices, float* __restrict__ out_data)
{
	index_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= n)
		return;

	const auto indptr_begin = offsets[idx];
	const auto indptr_end = offsets[idx + 1];

	if (indptr_end - indptr_begin != 1)
		return;

	auto my_begin = out_indptr[indptr_begin];
	// auto scc_nnz_size = out_indptr[indptr_end] - my_begin;

	// find pivot and its index
	float pivot;
	auto N_begin = N_indptr[indptr_begin];
	auto N_end = N_indptr[indptr_begin + 1];
	for (auto i = N_begin; i < N_end; i++)
	{
		pivot = N_data[i];
		if (pivot < 0.f)
		{
			break;
		}
	}

	// printf("idx %i pivot %f p_idx %i\n", idx, pivot, pivot_index);

	for (auto i = N_begin; i < N_end; i++)
	{
		out_indices[my_begin + i - N_begin] = N_indices[i];
		out_data[my_begin + i - N_begin] = N_data[i] / pivot;
	}
}

struct LU_part_t
{
	d_idxvec L_indptr, L_indices, U_indptr, U_indices;
	d_datvec L_data, U_data;
};

void solver::solve_system(const d_idxvec& indptr, const d_idxvec& rows, const thrust::device_vector<float>& data, int n,
						  int cols, int nnz, const d_idxvec& b_indptr, const d_idxvec& b_indices,
						  const thrust::device_vector<float>& b_data, d_idxvec& x_indptr, d_idxvec& x_indices,
						  thrust::device_vector<float>& x_data)
{
	index_t nt_n = nonterminals_offsets_.size() - 1;

	// print("N indptr  ", indptr);
	// print("N indices ", rows);
	// print("N data    ", data);

	d_idxvec L_indptr(nt_n + 1), L_indices, U_indptr(nt_n + 1), U_indices;
	d_datvec L_data, U_data;

	{
		thrust::device_vector<LU_part_t> lu_parts(nt_n);
		thrust::device_vector<index_t*> L_indptr_vec(nt_n), L_indices_vec(nt_n), U_indptr_vec(nt_n),
			U_indices_vec(nt_n);
		thrust::device_vector<real_t*> L_data_vec(nt_n), U_data_vec(nt_n);

		std::cout << "foreach" << std::endl;

		thrust::for_each(thrust::host, thrust::make_counting_iterator(0), thrust::make_counting_iterator(nt_n),
						 [&](index_t nonterm_idx) {
							 LU_part_t p;

							 solve_single_nonterm(nonterm_idx, indptr, rows, data, p.L_indptr, p.L_indices, p.L_data,
												  p.U_indptr, p.U_indices, p.U_data);

							 L_indptr_vec[nonterm_idx] = p.L_indptr.data().get();
							 L_indices_vec[nonterm_idx] = p.L_indices.data().get();
							 L_data_vec[nonterm_idx] = p.L_data.data().get();

							 U_indptr_vec[nonterm_idx] = p.U_indptr.data().get();
							 U_indices_vec[nonterm_idx] = p.U_indices.data().get();
							 U_data_vec[nonterm_idx] = p.U_data.data().get();
						 });

		d_idxvec offsets = nonterminals_offsets_;
		thrust::transform(offsets.begin(), offsets.end(), offsets.begin(),
						  [off = nonterminals_offsets_.front()] __device__(index_t x) { return x - off; });

		auto blocksize = 256;
		auto gridsize = (nt_n + blocksize - 1) / blocksize;

		std::cout << "hstack" << std::endl;

		nway_hstack_indptr<false><<<gridsize, blocksize>>>(nt_n, indptr.data().get(), offsets.data().get(),
														   U_indptr_vec.data().get(), U_indptr.data().get());
		nway_hstack_indptr<true><<<gridsize, blocksize>>>(nt_n, indptr.data().get(), offsets.data().get(),
														  L_indptr_vec.data().get(), L_indptr.data().get());

		CHECK_CUDA(cudaDeviceSynchronize());

		thrust::inclusive_scan(U_indptr.begin(), U_indptr.end(), U_indptr.begin());
		thrust::inclusive_scan(L_indptr.begin(), L_indptr.end(), L_indptr.begin());

		index_t U_nnz = U_indptr.back();
		index_t L_nnz = L_indptr.back();

		std::cout << "U_nnz " << U_nnz << std::endl;
		std::cout << "L_nnz " << L_nnz << std::endl;



		// print("U_indptr ", U_indptr);
		// print("L_indptr ", L_indptr);

		U_indices.resize(U_nnz);
		U_data.resize(U_nnz);

		L_indices.resize(L_nnz);
		L_data.resize(L_nnz);

		std::cout << "hstack2" << std::endl;

		nway_hstack_indices_and_data<false>
			<<<gridsize * 2, blocksize>>>(nt_n, indptr.data().get(), rows.data().get(), data.data().get(),
									  offsets.data().get(), U_indices_vec.data().get(), U_data_vec.data().get(),
									  U_indptr.data().get(), U_indices.data().get(), U_data.data().get());

		nway_hstack_indices_and_data<true>
			<<<gridsize * 2, blocksize>>>(nt_n, indptr.data().get(), rows.data().get(), data.data().get(),
									  offsets.data().get(), L_indices_vec.data().get(), L_data_vec.data().get(),
									  L_indptr.data().get(), L_indices.data().get(), L_data.data().get());

		nway_hstack_indices_and_data_trivial_L<<<gridsize, blocksize>>>(
			nt_n, indptr.data().get(), rows.data().get(), data.data().get(), offsets.data().get(),
			L_indices_vec.data().get(), L_data_vec.data().get(), L_indptr.data().get(), L_indices.data().get(),
			L_data.data().get());

		CHECK_CUDA(cudaDeviceSynchronize());


		// print("U_indices ", U_indices);
		// print("U_data    ", U_data);
		// print("L_indices ", L_indices);
		// print("L_data    ", L_data);


		for (size_t i = 0; i < nt_n; i++)
		{
			if (U_indptr[i] != i)
				std::cout << "bad at indptr " << i << " what: " << U_indptr[i] << std::endl;
		}


		for (size_t i = 0; i < U_nnz; i++)
		{
			if (U_indices[i] != i)
				std::cout << "bad at indices " << i << " what: " << U_indices[i] << std::endl;

			float pivot;
			index_t N_begin = indptr[i];
			index_t N_end = indptr[i + 1];
			for (auto j = N_begin; j < N_end; j++)
			{
				pivot = data[j];
				if (pivot < 0.f)
				{
					break;
				}
			}

			if (U_data[i] != pivot)
				std::cout << "bad at data " << i << " what: " << U_data[i] << std::endl;
		}

		for (size_t i = 0; i < nt_n; i++)
		{
			if (L_indptr[i] != indptr[i])
				std::cout << "bad at Lindptr " << i << " what: " << L_indptr[i] << std::endl;
		}


		for (size_t i = 0; i < nt_n; i++)
		{


			float pivot;
			index_t begin = indptr[i];
			index_t end = indptr[i + 1];
			for (auto j = begin; j < end; j++)
			{
				if (L_indices[j] != rows[j])
					std::cout << "bad at Lindices " << j << " what: " << L_indices[j] << std::endl;

				pivot = data[j];
				if (pivot < 0.f)
				{
					break;
				}
			}

			for (auto j = begin; j < end; j++)
			{
				if (L_data[j] != data[j] / pivot)
					std::cout << "bad at Ldata " << j << " what: " << L_data[j] << std::endl;
			}
		}

		// for (size_t i = 0; i < nt_n; i++)
		// {
		// 	if (L_indptr[i] != indptr[i])
		// 		std::cout << "bad at " << i << " what: " << L_indptr[i] << std::endl;
		// }


		// for (size_t i = 0; i < L_nnz; i++)
		// {
		// 	if (L_indices[i] != rows[i])
		// 		std::cout << "bad at " << i << " what: " << L_indices[i] << std::endl;

		// 	float pivot;
		// 	auto N_begin = indptr[i];
		// 	auto N_end = indptr[i + 1];
		// 	for (auto i = N_begin; i < N_end; i++)
		// 	{
		// 		pivot = data[i];
		// 		if (pivot < 0.f)
		// 		{
		// 			break;
		// 		}
		// 	}

		// 	if (L_data[i] != data[i] / pivot)
		// 		std::cout << "bad at " << i << " what: " << L_data[i] << std::endl;
		// }
	}


	cusparseMatDescr_t descr_L = 0;
	cusparseMatDescr_t descr_U = 0;
	bsrsv2Info_t info_L = 0;
	bsrsv2Info_t info_U = 0;
	int pBufferSize_L;
	int pBufferSize_U;
	void* pBufferL = 0;
	void* pBufferU = 0;
	const float alpha = 1.;
	const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
	const cusparseSolvePolicy_t policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
	const cusparseOperation_t trans_L = CUSPARSE_OPERATION_TRANSPOSE;
	const cusparseOperation_t trans_U = CUSPARSE_OPERATION_NON_TRANSPOSE;

	CHECK_CUSPARSE(cusparseCreateMatDescr(&descr_L));
	CHECK_CUSPARSE(cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ZERO));
	CHECK_CUSPARSE(cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL));
	CHECK_CUSPARSE(cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_UPPER));
	CHECK_CUSPARSE(cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_NON_UNIT));

	CHECK_CUSPARSE(cusparseCreateMatDescr(&descr_U));
	CHECK_CUSPARSE(cusparseSetMatIndexBase(descr_U, CUSPARSE_INDEX_BASE_ZERO));
	CHECK_CUSPARSE(cusparseSetMatType(descr_U, CUSPARSE_MATRIX_TYPE_GENERAL));
	CHECK_CUSPARSE(cusparseSetMatFillMode(descr_U, CUSPARSE_FILL_MODE_UPPER));
	CHECK_CUSPARSE(cusparseSetMatDiagType(descr_U, CUSPARSE_DIAG_TYPE_NON_UNIT));

	// step 2: create a empty info structure
	CHECK_CUSPARSE(cusparseCreateBsrsv2Info(&info_L));
	CHECK_CUSPARSE(cusparseCreateBsrsv2Info(&info_U));

	// step 3: query how much memory used in csrilu02 and csrsv2, and allocate the buffer
	CHECK_CUSPARSE(cusparseSbsrsv2_bufferSize(context_.cusparse_handle, CUSPARSE_DIRECTION_ROW, trans_L, n,
											  L_data.size(), descr_L, L_data.data().get(), L_indptr.data().get(),
											  L_indices.data().get(), 1, info_L, &pBufferSize_L));
	CHECK_CUSPARSE(cusparseSbsrsv2_bufferSize(context_.cusparse_handle, CUSPARSE_DIRECTION_ROW, trans_U, n,
											  U_data.size(), descr_U, U_data.data().get(), U_indptr.data().get(),
											  U_indices.data().get(), 1, info_U, &pBufferSize_U));

	cudaMalloc((void**)&pBufferL, pBufferSize_L);
	cudaMalloc((void**)&pBufferU, pBufferSize_U);

	CHECK_CUSPARSE(cusparseSbsrsv2_analysis(context_.cusparse_handle, CUSPARSE_DIRECTION_ROW, trans_L, n, L_data.size(),
											descr_L, L_data.data().get(), L_indptr.data().get(), L_indices.data().get(),
											1, info_L, policy_L, pBufferL));

	CHECK_CUSPARSE(cusparseSbsrsv2_analysis(context_.cusparse_handle, CUSPARSE_DIRECTION_ROW, trans_U, n, U_data.size(),
											descr_U, U_data.data().get(), U_indptr.data().get(), U_indices.data().get(),
											1, info_U, policy_U, pBufferU));

	thrust::host_vector<index_t> hb_indptr = b_indptr;
	thrust::host_vector<index_t> hb_indices = b_indices;


	thrust::device_vector<float> x_vec(cols);
	thrust::device_vector<float> z_vec(cols);

	thrust::host_vector<index_t> hx_indptr(b_indptr.size());
	hx_indptr[0] = 0;

	std::cout << "solve begin" << std::endl;


	for (int b_idx = 0; b_idx < b_indptr.size() - 1; b_idx++)
	{
		thrust::device_vector<float> b_vec(n, 0.f);
		auto start = hb_indptr[b_idx];
		auto end = hb_indptr[b_idx + 1];
		thrust::copy(b_data.begin() + start, b_data.begin() + end,
					 thrust::make_permutation_iterator(b_vec.begin(), b_indices.begin() + start));

		// print("b ", b_vec);

		// step 6: solve L*z = x
		CHECK_CUSPARSE(cusparseSbsrsv2_solve(context_.cusparse_handle, CUSPARSE_DIRECTION_ROW, trans_L, n,
											 L_data.size(), &alpha, descr_L, L_data.data().get(), L_indptr.data().get(),
											 L_indices.data().get(), 1, info_L, b_vec.data().get(), z_vec.data().get(),
											 policy_L, pBufferL));

		// print("z ", z_vec);

		CHECK_CUSPARSE(cusparseSbsrsv2_solve(context_.cusparse_handle, CUSPARSE_DIRECTION_ROW, trans_U, n,
											 U_data.size(), &alpha, descr_U, U_data.data().get(), U_indptr.data().get(),
											 U_indices.data().get(), 1, info_U, z_vec.data().get(), x_vec.data().get(),
											 policy_U, pBufferU));


		auto x_nnz = thrust::count_if(x_vec.begin(), x_vec.end(), [] __device__(float x) { return !is_zero(x); });

		// print("x ", x_vec);

		auto size_before = x_indices.size();
		x_indices.resize(x_indices.size() + x_nnz);
		x_data.resize(x_data.size() + x_nnz);

		hx_indptr[b_idx + 1] = hx_indptr[b_idx] + x_nnz;

		thrust::copy_if(thrust::make_zip_iterator(x_vec.begin(), thrust::make_counting_iterator<index_t>(0)),
						thrust::make_zip_iterator(x_vec.end(), thrust::make_counting_iterator<index_t>(x_vec.size())),
						thrust::make_zip_iterator(x_data.begin() + size_before, x_indices.begin() + size_before),
						[] __device__(thrust::tuple<float, index_t> x) { return !is_zero(thrust::get<0>(x)); });
	}
	std::cout << "solve end" << std::endl;


	x_indptr = hx_indptr;

	// step 6: free resources
	CHECK_CUSPARSE(cusparseDestroyMatDescr(descr_L));
	CHECK_CUSPARSE(cusparseDestroyMatDescr(descr_U));
	CHECK_CUSPARSE(cusparseDestroyBsrsv2Info(info_L));
	CHECK_CUSPARSE(cusparseDestroyBsrsv2Info(info_U));
}

void solver::solve_nonterminal_part()
{
	index_t n = ordered_vertices_.size();
	index_t terminal_vertices_n = terminals_offsets_.back();
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
				 thrust::make_permutation_iterator(submatrix_vertex_mapping_.begin(), ordered_vertices_.begin()));

	thrust::transform(ordered_vertices_.begin(), ordered_vertices_.begin() + terminals_offsets_.back(), U_cols.begin(),
					  [map = submatrix_vertex_mapping_.data().get()] __device__(index_t x) { return map[x]; });

	thrust::device_vector<float> U_data(term_rows.size(), -1.f);

	// NB

	d_idxvec nb_indptr_csc, nb_rows;
	thrust::device_vector<float> nb_data_csc;

	// custom mapping
	{
		thrust::copy(thrust::make_counting_iterator<intptr_t>(0),
					 thrust::make_counting_iterator<intptr_t>(nonterminal_vertices_n),
					 thrust::make_permutation_iterator(submatrix_vertex_mapping_.begin(),
													   ordered_vertices_.begin() + terminals_offsets_.back()));

		thrust::copy(thrust::make_counting_iterator<intptr_t>(nonterminal_vertices_n),
					 thrust::make_counting_iterator<intptr_t>(n),
					 thrust::make_permutation_iterator(submatrix_vertex_mapping_.begin(), ordered_vertices_.begin()));
	}

	// create vstack(N, B) matrix in csc
	auto nnz = take_submatrix(nonterminal_vertices_n, ordered_vertices_.begin() + terminals_offsets_.back(),
							  nb_indptr_csc, nb_rows, nb_data_csc, true);

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

	std::cout << "matmul begin" << std::endl;

	matmul(U_indptr_csr.data().get(), U_cols.data().get(), U_data.data().get(), terminals_offsets_.size() - 1,
		   terminal_vertices_n, U_cols.size(), nb_indptr_csr.data().get() + nonterminal_vertices_n,
		   nb_cols.data().get() + n_nnz, nb_data_csr.data().get() + n_nnz, terminal_vertices_n, nonterminal_vertices_n,
		   b_nnz, A_indptr, A_indices, A_data);

	nb_indptr_csr[nonterminal_vertices_n] = n_nnz;

	std::cout << "NB switch begin" << std::endl;

	csr_csc_switch(nb_indptr_csr.data().get(), nb_cols.data().get(), nb_data_csr.data().get(), nonterminal_vertices_n,
				   nonterminal_vertices_n, n_nnz, nb_indptr_csc, nb_rows, nb_data_csc);

	nb_indptr_csc.resize(nonterminal_vertices_n + 1);
	nb_rows.resize(n_nnz);
	nb_data_csc.resize(n_nnz);

	d_idxvec N_indptr_csr, N_indices_csr;
	thrust::device_vector<float> N_data_csr;

	csr_csc_switch(nb_indptr_csc.data().get(), nb_rows.data().get(), nb_data_csc.data().get(), nonterminal_vertices_n,
				   nonterminal_vertices_n, n_nnz, N_indptr_csr, N_indices_csr, N_data_csr);

	d_idxvec X_indptr, X_indices;
	thrust::device_vector<float> X_data;

	std::cout << "Trisystem begin" << std::endl;

	solve_system(N_indptr_csr, N_indices_csr, N_data_csr, nonterminal_vertices_n, nonterminal_vertices_n, n_nnz,
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
		thrust::copy(ordered_vertices_.begin() + terminals_offsets_.back(), ordered_vertices_.end(),
					 submatrix_vertex_mapping_.begin());

		thrust::transform(X_indices.begin(), X_indices.end(), X_indices.begin(),
						  [map = submatrix_vertex_mapping_.data().get()] __device__(index_t x) { return map[x]; });
	}

	std::cout << "hstack begin" << std::endl;

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
	std::cout << "Rx begin" << std::endl;
	thrust::device_vector<float> y(terminals_offsets_.size() - 1);
	{
		float alpha = 1.0f;
		float beta = 0.0f;

		cusparseSpMatDescr_t matA;
		cusparseDnVecDescr_t vecX, vecY;
		size_t bufferSize = 0;

		// Create sparse matrix A in CSR format
		CHECK_CUSPARSE(cusparseCreateCsr(&matA, terminals_offsets_.size() - 1, ordered_vertices_.size(),
										 nonterm_data.size(), nonterm_indptr.data().get(), nonterm_cols.data().get(),
										 nonterm_data.data().get(), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
										 CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
		// Create dense vector X
		CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, ordered_vertices_.size(), initial_state_.data().get(), CUDA_R_32F));
		// Create dense vector y
		CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, terminals_offsets_.size() - 1, y.data().get(), CUDA_R_32F));
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

	std::cout << "Ly begin" << std::endl;

	final_state.resize(ordered_vertices_.size());
	{
		float alpha = 1.0f;
		float beta = 0.0f;

		cusparseSpMatDescr_t matA;
		cusparseDnVecDescr_t vecX, vecY;
		size_t bufferSize = 0;

		// Create sparse matrix A in CSC format
		CHECK_CUSPARSE(cusparseCreateCsc(&matA, ordered_vertices_.size(), terminals_offsets_.size() - 1,
										 term_data.size(), term_indptr.data().get(), term_rows.data().get(),
										 term_data.data().get(), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
										 CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
		// Create dense vector y
		CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, terminals_offsets_.size() - 1, y.data().get(), CUDA_R_32F));
		// Create dense vector final
		CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, ordered_vertices_.size(), final_state.data().get(), CUDA_R_32F));
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
	solve_terminal_part();
	solve_nonterminal_part();

	compute_final_states();
}
