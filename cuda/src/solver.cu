#include <cusolverSp_LOWLEVEL_PREVIEW.h>
#include <cusparse_v2.h>
#include <device_launch_parameters.h>

#include <thrust/adjacent_difference.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/partition.h>

#include "numml/splu.h"
#include "solver.h"
#include "utils.h"

constexpr float zero_threshold = 1e-10f;

__device__ __host__ bool is_zero(float x) { return (x > 0 ? x : -x) <= zero_threshold; }


struct equals_ftor : public thrust::unary_function<index_t, bool>
{
	index_t value;

	equals_ftor(index_t value) : value(value) {}

	__host__ __device__ bool operator()(index_t x) const { return x == value; }
};

__global__ void scatter_rows_data(const index_t* __restrict__ dst_indptr, index_t* __restrict__ dst_rows,
								  float* __restrict__ dst_data, const index_t* __restrict__ src_rows,
								  const index_t* __restrict__ src_indptr, const index_t* __restrict__ src_perm,
								  int perm_size)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx >= perm_size)
		return;

	index_t diag = src_perm[idx];
	index_t src_begin = src_indptr[diag];
	index_t size = src_indptr[diag + 1] - src_begin;

	index_t dst_begin = dst_indptr[idx];

	bool diag_inserted = false;

	for (int i = 0; i < size; i++)
	{
		index_t r = src_rows[src_begin + i];

		if (!diag_inserted && r > diag)
		{
			dst_rows[dst_begin + i] = diag;
			dst_data[dst_begin + i] = -(float)size;
			diag_inserted = true;
			dst_begin++;
		}

		dst_rows[dst_begin + i] = r;
	}

	if (!diag_inserted)
	{
		dst_rows[dst_begin + size] = diag;
		dst_data[dst_begin + size] = -(float)size;
	}
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

	nonterminals_offsets_.assign(g.sccs_offsets.begin() + g.terminals_count, g.sccs_offsets.end());
}

float solver::determinant(const d_idxvec& indptr, const d_idxvec& rows, const thrust::device_vector<float>& data, int n,
						  int nnz)
{
	d_idxvec scc(2);
	scc[0] = 0;
	scc[1] = n;

	d_idxvec M_indptr, M_indices;
	d_datvec M_data;

	splu(context_, scc, indptr, rows, data, M_indptr, M_indices, M_data);

	d_idxvec diag(n);

	thrust::for_each(thrust::device, thrust::make_counting_iterator<index_t>(0),
					 thrust::make_counting_iterator<index_t>(n),
					 [M_indptr_v = M_indptr.data().get(), M_indices_v = M_indices.data().get(),
					  M_data_v = M_data.data().get(), diag_v = diag.data().get()] __device__(index_t i) {
						 auto begin = M_indptr_v[i];
						 auto end = M_indptr_v[i + 1];

						 for (auto col_idx = begin; col_idx != end; col_idx++)
						 {
							 if (M_indices_v[col_idx] == i)
								 diag_v[i] = M_data_v[col_idx];
						 }
					 });

	return thrust::reduce(diag.begin(), diag.end(), 1, thrust::multiplies<float>());
}

void solver::take_submatrix(index_t n, d_idxvec::const_iterator vertices_subset_begin, sparse_csc_matrix& m,
							bool mapping_prefilled)
{
	m.indptr.resize(n + 1);
	m.indptr[0] = 0;

	// this creates indptr of scc in CSC
	{
		auto scc_begins_b = thrust::make_permutation_iterator(indptr_.begin(), vertices_subset_begin);
		auto scc_begins_e = thrust::make_permutation_iterator(indptr_.begin(), vertices_subset_begin + n);

		auto scc_ends_b = thrust::make_permutation_iterator(indptr_.begin() + 1, vertices_subset_begin);
		auto scc_ends_e = thrust::make_permutation_iterator(indptr_.begin() + 1, vertices_subset_begin + n);

		// first get sizes of each col - also add 1 for diagonal part
		thrust::transform(
			thrust::make_zip_iterator(scc_begins_b, scc_ends_b), thrust::make_zip_iterator(scc_begins_e, scc_ends_e),
			m.indptr.begin() + 1,
			[] __device__(thrust::tuple<index_t, index_t> x) { return 1 + thrust::get<1>(x) - thrust::get<0>(x); });

		thrust::inclusive_scan(m.indptr.begin(), m.indptr.end(), m.indptr.begin());
	}

	index_t nnz = m.indptr.back();
	m.indices.resize(nnz);
	m.data.assign(nnz, 1.f);

	// this creates rows and data of scc
	{
		int blocksize = 512;
		int gridsize = (n + blocksize - 1) / blocksize;
		scatter_rows_data<<<gridsize, blocksize>>>(m.indptr.data().get(), m.indices.data().get(), m.data.data().get(),
												   rows_.data().get(), indptr_.data().get(),
												   (&*vertices_subset_begin).get(), n);

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

		thrust::transform(m.indices.begin(), m.indices.end(), m.indices.begin(),
						  [map = submatrix_vertex_mapping_.data().get()] __device__(index_t x) { return map[x]; });
	}
}

void solver::create_minor(d_idxvec& indptr, d_idxvec& rows, d_datvec& data, const index_t remove_vertex)
{
	const auto nnz = rows.size();
	const auto n = indptr.size() - 1;

	d_idxvec cols(nnz);

	// this decompresses indptr into cols
	CHECK_CUSPARSE(cusparseXcsr2coo(context_.cusparse_handle, indptr.data().get(), nnz, n, cols.data().get(),
									CUSPARSE_INDEX_BASE_ZERO));

	{
		auto part_point = thrust::stable_partition(
			thrust::make_zip_iterator(rows.begin(), cols.begin(), data.begin()),
			thrust::make_zip_iterator(rows.end(), cols.end(), data.end()),
			[remove_vertex] __device__(thrust::tuple<index_t, index_t, float> x) {
				return thrust::get<0>(x) != remove_vertex && thrust::get<1>(x) != remove_vertex;
			});

		auto removed_n = thrust::get<0>(part_point.get_iterator_tuple()) - rows.begin();

		rows.resize(removed_n);
		cols.resize(removed_n);
		data.resize(removed_n);

		thrust::transform_if(
			rows.begin(), rows.end(), rows.begin(), [] __device__(index_t x) { return x - 1; },
			[remove_vertex] __device__(index_t x) { return x > remove_vertex; });

		thrust::transform_if(
			cols.begin(), cols.end(), cols.begin(), [] __device__(index_t x) { return x - 1; },
			[remove_vertex] __device__(index_t x) { return x > remove_vertex; });
	}

	indptr.resize(n);

	// this compresses rows back into indptr
	CHECK_CUSPARSE(cusparseXcoo2csr(context_.cusparse_handle, cols.data().get(), cols.size(), n - 1,
									indptr.data().get(), CUSPARSE_INDEX_BASE_ZERO));
}

void solver::solve_terminal_part()
{
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

		sparse_csc_matrix scc;

		take_submatrix(scc_size, ordered_vertices_.begin() + terminals_offsets_[terminal_scc_idx - 1], scc);

		thrust::host_vector<float> h_minors(scc_size);
		for (size_t minor_i = 0; minor_i < scc_size; minor_i++)
		{
			// copy indptr
			d_idxvec minor_indptr = scc.indptr;
			d_idxvec minor_rows = scc.indices;
			d_datvec minor_data = scc.data;

			create_minor(minor_indptr, minor_rows, minor_data, minor_i);

			h_minors[minor_i] =
				std::abs(determinant(minor_indptr, minor_rows, minor_data, minor_indptr.size() - 1, minor_data.size()));
		}

		thrust::device_vector<float> minors = h_minors;
		auto sum = thrust::reduce(minors.begin(), minors.end(), 0.f, thrust::plus<float>());

		thrust::transform(minors.begin(), minors.end(), term_data.begin() + terminals_offsets_[terminal_scc_idx - 1],
						  [sum] __device__(float x) { return x / sum; });
	}
}

void solver::solve_system(const d_idxvec& indptr, d_idxvec& rows, thrust::device_vector<float>& data, int n, int cols,
						  int nnz, const d_idxvec& b_indptr, const d_idxvec& b_indices,
						  const thrust::device_vector<float>& b_data, d_idxvec& x_indptr, d_idxvec& x_indices,
						  thrust::device_vector<float>& x_data)
{
	// print("A indptr ", indptr);
	// print("A indice ", rows);
	// print("A data   ", data);

	{
		cusparseMatDescr_t descr_N = 0;
		size_t pBufferSizeInBytes;

		CHECK_CUSPARSE(cusparseCreateMatDescr(&descr_N));
		CHECK_CUSPARSE(cusparseSetMatIndexBase(descr_N, CUSPARSE_INDEX_BASE_ZERO));
		CHECK_CUSPARSE(cusparseSetMatType(descr_N, CUSPARSE_MATRIX_TYPE_GENERAL));

		// step 1: allocate buffer
		CHECK_CUSPARSE(cusparseXcsrsort_bufferSizeExt(context_.cusparse_handle, n, n, nnz, indptr.data().get(),
													  rows.data().get(), &pBufferSizeInBytes));

		// step 2: setup permutation vector P to identity
		d_idxvec P(nnz);
		CHECK_CUSPARSE(cusparseCreateIdentityPermutation(context_.cusparse_handle, nnz, P.data().get()));

		{
			// step 3: sort CSR format
			thrust::device_vector<char> buffer(pBufferSizeInBytes);
			CHECK_CUSPARSE(cusparseXcsrsort(context_.cusparse_handle, n, n, nnz, descr_N, indptr.data().get(),
											rows.data().get(), P.data().get(), buffer.data().get()));
		}

		// step 4: gather sorted csrVal
		d_datvec sorted_data(data.size());
		thrust::gather(P.begin(), P.end(), data.begin(), sorted_data.begin());
		data = std::move(sorted_data);

		CHECK_CUSPARSE(cusparseDestroyMatDescr(descr_N));
	}

	d_idxvec M_indptr, M_indices;
	d_datvec M_data;

	splu(context_, nonterminals_offsets_, indptr, rows, data, M_indptr, M_indices, M_data);

	cusparseMatDescr_t descr_L = 0;
	cusparseMatDescr_t descr_U = 0;
	bsrsv2Info_t info_L = 0;
	bsrsv2Info_t info_U = 0;
	int pBufferSize_L;
	int pBufferSize_U;
	int pBufferSize;
	void* pBufferL = 0;
	void* pBufferU = 0;
	int structural_zero;
	int numerical_zero;
	const float alpha = 1.;
	const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
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

	// step 2: create a empty info structure
	// we need one info for csrilu02 and two info's for csrsv2
	CHECK_CUSPARSE(cusparseCreateBsrsv2Info(&info_L));
	CHECK_CUSPARSE(cusparseCreateBsrsv2Info(&info_U));

	// step 3: query how much memory used in csrilu02 and csrsv2, and allocate the buffer
	CHECK_CUSPARSE(cusparseSbsrsv2_bufferSize(context_.cusparse_handle, CUSPARSE_DIRECTION_ROW, trans_L, n,
											  M_data.size(), descr_L, M_data.data().get(), M_indptr.data().get(),
											  M_indices.data().get(), 1, info_L, &pBufferSize_L));
	CHECK_CUSPARSE(cusparseSbsrsv2_bufferSize(context_.cusparse_handle, CUSPARSE_DIRECTION_ROW, trans_U, n,
											  M_data.size(), descr_U, M_data.data().get(), M_indptr.data().get(),
											  M_indices.data().get(), 1, info_U, &pBufferSize_U));

	// pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
	cudaMalloc((void**)&pBufferL, pBufferSize_L);
	cudaMalloc((void**)&pBufferU, pBufferSize_U);

	// step 4: perform analysis of incomplete Cholesky on M
	//         perform analysis of triangular solve on L
	//         perform analysis of triangular solve on U
	// The lower(upper) triangular part of M has the same sparsity pattern as L(U),
	// we can do analysis of csrilu0 and csrsv2 simultaneously.

	CHECK_CUSPARSE(cusparseSbsrsv2_analysis(context_.cusparse_handle, CUSPARSE_DIRECTION_ROW, trans_L, n, M_data.size(),
											descr_L, M_data.data().get(), M_indptr.data().get(), M_indices.data().get(),
											1, info_L, policy_L, pBufferL));

	CHECK_CUSPARSE(cusparseSbsrsv2_analysis(context_.cusparse_handle, CUSPARSE_DIRECTION_ROW, trans_U, n, M_data.size(),
											descr_U, M_data.data().get(), M_indptr.data().get(), M_indices.data().get(),
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

		std::cout << ".";

		// step 6: solve L*z = x
		CHECK_CUSPARSE(cusparseSbsrsv2_solve(context_.cusparse_handle, CUSPARSE_DIRECTION_ROW, trans_L, n,
											 M_data.size(), &alpha, descr_L, M_data.data().get(), M_indptr.data().get(),
											 M_indices.data().get(), 1, info_L, b_vec.data().get(), z_vec.data().get(),
											 policy_L, pBufferL));

		CHECK_CUSPARSE(cusparseSbsrsv2_solve(context_.cusparse_handle, CUSPARSE_DIRECTION_ROW, trans_U, n,
											 M_data.size(), &alpha, descr_U, M_data.data().get(), M_indptr.data().get(),
											 M_indices.data().get(), 1, info_U, z_vec.data().get(), x_vec.data().get(),
											 policy_U, pBufferU));


		auto x_nnz = thrust::count_if(x_vec.begin(), x_vec.end(), [] __device__(float x) { return !is_zero(x); });

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

void solver::break_NB(sparse_csc_matrix&& NB, sparse_csc_matrix& N, sparse_csc_matrix& B)
{
	const auto nnz = NB.data.size();
	const auto n = NB.indptr.size() - 1;
	const auto nonterm_n = ordered_vertices_.size() - terminals_offsets_.back();

	std::cout << "nonterm_n " << nonterm_n << std::endl;

	/*print("NB indptr ", NB.indptr);
	print("NB indice ", NB.indices);
	print("NB data   ", NB.data);*/

	d_idxvec NB_decomp_indptr(nnz);

	// this decompresses indptr into cols
	CHECK_CUSPARSE(cusparseXcsr2coo(context_.cusparse_handle, NB.indptr.data().get(), nnz, n,
									NB_decomp_indptr.data().get(), CUSPARSE_INDEX_BASE_ZERO));

	auto part_point = thrust::stable_partition(
		thrust::make_zip_iterator(NB_decomp_indptr.begin(), NB.indices.begin(), NB.data.begin()),
		thrust::make_zip_iterator(NB_decomp_indptr.end(), NB.indices.end(), NB.data.end()),
		[point = nonterm_n] __device__(thrust::tuple<index_t, index_t, float> x) { return thrust::get<1>(x) < point; });

	auto N_size = thrust::get<0>(part_point.get_iterator_tuple()) - NB_decomp_indptr.begin();

	std::cout << "N_size " << N_size << std::endl;

	d_idxvec B_decomp_indptr(NB_decomp_indptr.begin() + N_size, NB_decomp_indptr.end());
	B.indices.assign(NB.indices.begin() + N_size, NB.indices.end());
	B.data.assign(NB.data.begin() + N_size, NB.data.end());

	NB_decomp_indptr.resize(N_size);
	N.indices = std::move(NB.indices);
	N.indices.resize(N_size);
	N.data = std::move(NB.data);
	N.data.resize(N_size);

	thrust::transform(B.indices.begin(), B.indices.end(), B.indices.begin(),
					  [point = nonterm_n] __device__(index_t x) { return x - point; });

	// this compresses rows back into indptr
	N.indptr.resize(nonterm_n + 1);
	CHECK_CUSPARSE(cusparseXcoo2csr(context_.cusparse_handle, NB_decomp_indptr.data().get(), NB_decomp_indptr.size(),
									nonterm_n, N.indptr.data().get(), CUSPARSE_INDEX_BASE_ZERO));

	B.indptr.resize(nonterm_n + 1);
	CHECK_CUSPARSE(cusparseXcoo2csr(context_.cusparse_handle, B_decomp_indptr.data().get(), B_decomp_indptr.size(),
									nonterm_n, B.indptr.data().get(), CUSPARSE_INDEX_BASE_ZERO));


	// print("N indptr ", N.indptr);
	// print("N indice ", N.indices);
	// print("N data   ", N.data);

	// print("B indptr ", B.indptr);
	// print("B indice ", B.indices);
	// print("B data   ", B.data);
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
	sparse_csr_matrix U;
	{
		U.indptr = term_indptr;
		U.indices.resize(term_rows.size());

		thrust::copy(thrust::make_counting_iterator<intptr_t>(0),
					 thrust::make_counting_iterator<intptr_t>(terminal_vertices_n),
					 thrust::make_permutation_iterator(submatrix_vertex_mapping_.begin(), ordered_vertices_.begin()));

		thrust::transform(ordered_vertices_.begin(), ordered_vertices_.begin() + terminals_offsets_.back(),
						  U.indices.begin(),
						  [map = submatrix_vertex_mapping_.data().get()] __device__(index_t x) { return map[x]; });

		U.data.assign(term_rows.size(), -1.f);
	}

	// NB
	sparse_csc_matrix NB;
	{
		// custom mapping
		{
			thrust::copy(thrust::make_counting_iterator<intptr_t>(0),
						 thrust::make_counting_iterator<intptr_t>(nonterminal_vertices_n),
						 thrust::make_permutation_iterator(submatrix_vertex_mapping_.begin(),
														   ordered_vertices_.begin() + terminals_offsets_.back()));

			thrust::copy(
				thrust::make_counting_iterator<intptr_t>(nonterminal_vertices_n),
				thrust::make_counting_iterator<intptr_t>(n),
				thrust::make_permutation_iterator(submatrix_vertex_mapping_.begin(), ordered_vertices_.begin()));
		}

		// create vstack(N, B) matrix in csc
		take_submatrix(nonterminal_vertices_n, ordered_vertices_.begin() + terminals_offsets_.back(), NB, true);
	}

	// extract B
	sparse_csc_matrix N, B;
	break_NB(std::move(NB), N, B);

	std::cout << "N nnz " << N.nnz() << std::endl;
	std::cout << "B nnz " << B.nnz() << std::endl;

	auto B_t = csc2csr(context_.cusparse_handle, B, nonterminal_vertices_n, terminal_vertices_n, B.nnz());

	std::cout << "matmul begin" << std::endl;

	sparse_csr_matrix A =
		matmul(context_.cusparse_handle, U.indptr.data().get(), U.indices.data().get(), U.data.data().get(),
			   terminals_offsets_.size() - 1, terminal_vertices_n, U.indices.size(), B_t.indptr.data().get(),
			   B_t.indices.data().get(), B_t.data.data().get(), terminal_vertices_n, nonterminal_vertices_n, B.nnz());

	sparse_csr_matrix X;

	solve_system(N.indptr, N.indices, N.data, nonterminal_vertices_n, nonterminal_vertices_n, N.nnz(), A.indptr,
				 A.indices, A.data, X.indptr, X.indices, X.data);


	nonterm_indptr.resize(U.indptr.size());
	index_t nonterm_nnz = U.indptr.back() + X.indptr.back();
	nonterm_cols.resize(nonterm_nnz);
	nonterm_data.resize(nonterm_nnz);

	thrust::transform(
		thrust::make_zip_iterator(U.indptr.begin(), X.indptr.begin()),
		thrust::make_zip_iterator(U.indptr.end(), X.indptr.end()), nonterm_indptr.begin(),
		[] __device__(thrust::tuple<index_t, index_t> x) { return thrust::get<0>(x) + thrust::get<1>(x); });


	// -U back to U
	thrust::transform(U.data.begin(), U.data.end(), U.data.begin(), thrust::negate<float>());

	// nonterminal vertices from 0, ..., n_nt to actual indices
	{
		thrust::copy(ordered_vertices_.begin() + terminals_offsets_.back(), ordered_vertices_.end(),
					 submatrix_vertex_mapping_.begin());

		thrust::transform(X.indices.begin(), X.indices.end(), X.indices.begin(),
						  [map = submatrix_vertex_mapping_.data().get()] __device__(index_t x) { return map[x]; });
	}

	std::cout << "hstack begin" << std::endl;

	// hstack(U,X)
	{
		int blocksize = 512;
		int gridsize = (2 * (nonterm_indptr.size() - 1) + blocksize - 1) / blocksize;

		std::cout << "blockxgrid size " << blocksize << "x" << gridsize << std::endl;

		hstack<<<gridsize, blocksize>>>(nonterm_indptr.data().get(), nonterm_cols.data().get(),
										nonterm_data.data().get(), U.indptr.data().get(), X.indptr.data().get(),
										term_rows.data().get(), X.indices.data().get(), U.data.data().get(),
										X.data.data().get(), nonterm_indptr.size() - 1);

		CHECK_CUDA(cudaDeviceSynchronize());
	}
}

void solver::compute_final_states()
{
	std::cout << "Rx begin" << std::endl;
	auto y = mvmul(context_.cusparse_handle, nonterm_indptr, nonterm_cols, nonterm_data, cs_kind::CSR,
				   terminals_offsets_.size() - 1, ordered_vertices_.size(), initial_state_);

	std::cout << "Ly begin" << std::endl;
	final_state = mvmul(context_.cusparse_handle, term_indptr, term_rows, term_data, cs_kind::CSC,
						ordered_vertices_.size(), terminals_offsets_.size() - 1, y);
}

void solver::solve()
{
	solve_terminal_part();
	solve_nonterminal_part();

	compute_final_states();
}
