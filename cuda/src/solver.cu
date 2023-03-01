#include <cusolverSp.h>

#include <thrust/adjacent_difference.h>
#include <thrust/execution_policy.h>
#include <thrust/partition.h>

#include "solver.h"
#include "utils.cuh"

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
	  indptr_(t.indptr)
{}

__global__ void scatter_rows_data(const __restrict__ index_t* dst_indptr, __restrict__ index_t* dst_rows,
								  __restrict__ float* dst_data, const __restrict__ index_t* src_rows,
								  const __restrict__ index_t* src_indptr, const __restrict__ index_t* src_perm,
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

float solver::determinant(const d_idxvec& indptr, const d_idxvec& rows, const thrust::device_vector<float>& data)
{
	cusolverSpHandle_t handle;
	cusolverSpCreate(&handle);

	thrust::host_vector<index_t> h_indptr = indptr;
	thrust::host_vector<index_t> h_rows = rows;
	thrust::host_vector<float> h_data = data;

	csrluInfoHost_t info;
	cusolverSpCreateCsrluInfoHost(&info);

	cusparseMatDescr_t desc, descr_L, descr_U;
	CHECK_CUSPARSE(cusparseCreateMatDescr(&desc));
	CHECK_CUSPARSE(cusparseSetMatIndexBase(desc, CUSPARSE_INDEX_BASE_ZERO);
	CHECK_CUSPARSE(cusparseSetMatType(desc, CUSPARSE_MATRIX_TYPE_GENERAL);


	CHECK_CUSPARSE(cusparseCreateMatDescr(&descr_L);
	CHECK_CUSPARSE(cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ZERO);
	CHECK_CUSPARSE(cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL);
	CHECK_CUSPARSE(cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER);
	CHECK_CUSPARSE(cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_UNIT);

	CHECK_CUSPARSE(cusparseCreateMatDescr(&descr_U);
	CHECK_CUSPARSE(cusparseSetMatIndexBase(descr_U, CUSPARSE_INDEX_BASE_ZERO);
	CHECK_CUSPARSE(cusparseSetMatType(descr_U, CUSPARSE_MATRIX_TYPE_GENERAL);
	CHECK_CUSPARSE(cusparseSetMatFillMode(descr_U, CUSPARSE_FILL_MODE_UPPER);
	CHECK_CUSPARSE(cusparseSetMatDiagType(descr_U, CUSPARSE_DIAG_TYPE_NON_UNIT);

	cusolverSpXcsrluAnalysisHost(handle, indptr.size() - 1, rows.size(), desc, h_indptr.data(), h_rows.data(), info);

	size_t internal_data, workspace;
	cusolverSpScsrluBufferInfoHost(handle, indptr.size() - 1, rows.size(), desc, h_data.data(), h_indptr.data(),
								   h_rows.data(), info, &internal_data, &workspace);

	std::vector<char> buffer(workspace);

	cusolverSpScsrluFactorHost(handle, indptr.size() - 1, rows.size(), desc, h_data.data(), h_indptr.data(),
							   h_rows.data(), info, 0.f, buffer.data());

	int nnz_l, nnz_u;
	cusolverSpXcsrluNnzHost(handle, &nnz_l, &nnz_u, info);

	std::vector<index_t> P(indptr.size() - 1), Q(indptr.size() - 1), L_indptr(indptr.size()), U_indptr(indptr.size()),
		L_cols(nnz_l), U_cols(nnz_u);
	std::vector<float> L_data(nnz_l), U_data(nnz_u);

	cusolverSpScsrluExtractHost(handle, P.data(), Q.data(), descr_L, L_data.data(), L_indptr.data(), L_cols.data(),
								descr_U, U_data.data(), U_indptr.data(), U_cols.data(), info, buffer.data());

	std::vector<float> diag(indptr.size() - 1);

	thrust::for_each(thrust::host, thrust::make_counting_iterator<index_t>(0),
					 thrust::make_counting_iterator<index_t>(indptr.size() - 1), [&](index_t i) {
						 auto begin = U_indptr[i];
						 auto end = U_indptr[i + 1];

						 for (auto col_idx = begin; col_idx != end; col_idx++)
						 {
							 if (U_cols[col_idx] == i)
								 diag[i] = U_data[col_idx];
						 }
					 });

	float determinant = thrust::reduce(thrust::host, diag.begin(), diag.end(), 0, thrust::multiplies<float>());

	CHECK_CUSPARSE(cusparseDestroyMatDescr(desc);
	CHECK_CUSPARSE(cusparseDestroyMatDescr(descr_L);
	CHECK_CUSPARSE(cusparseDestroyMatDescr(descr_U);
	cusolverSpDestroyCsrluInfoHost(info);
	cusolverSpDestroy(handle);

	return determinant;
}

void solver::solve_terminal_part()
{
	// vector of vertex indices
	d_idxvec sccs(thrust::make_counting_iterator<index_t>(0), thrust::make_counting_iterator<index_t>(labels_.size()));

	// vector of terminal scc begins and ends
	std::vector<size_t> terminals_offsets;
	terminals_offsets.reserve(terminals_.size() + 1);
	terminals_offsets.push_back(0);

	d_idxvec reverse_labels(labels_.size());

	// we partition labels_ and sccs multiple times so the ordering is T1, ..., Tn, NT1, ..., NTn
	auto partition_point = thrust::make_zip_iterator(sccs.begin(), labels_.begin());
	for (auto it = terminals_.begin(); it != terminals_.end(); it++)
	{
		partition_point =
			thrust::stable_partition(partition_point, thrust::make_zip_iterator(sccs.end(), labels_.end()),
									 [terminal_idx = *it] __device__(thrust::tuple<index_t, index_t> x) {
										 return thrust::get<1>(x) == terminal_idx;
									 });

		terminals_offsets.push_back(partition_point - thrust::make_zip_iterator(sccs.begin(), labels_.begin()));
	}

	// this is point that partitions terminals and nonterminals
	auto sccs_terminals_end = partition_point;

	for (size_t i = 1; i < terminals_offsets.size(); i++)
	{
		size_t scc_size = terminals_offsets[i] - terminals_offsets[i - 1];
		d_idxvec scc_indptr(scc_size + 1);
		scc_indptr[0] = 0;

		// create map for scc vertices so they start from 0
		thrust::copy(
			thrust::make_counting_iterator<intptr_t>(0), thrust::make_counting_iterator<intptr_t>(scc_size),
			thrust::make_permutation_iterator(reverse_labels.begin(), sccs.begin() + terminals_offsets[i - 1]));

		// this creates indptr of scc in CSC
		{
			auto scc_begins_b =
				thrust::make_permutation_iterator(indptr_.begin() + 1, sccs.begin() + terminals_offsets[i - 1]);

			auto scc_begins_e =
				thrust::make_permutation_iterator(indptr_.begin() + 1, sccs.begin() + terminals_offsets[i]);

			thrust::adjacent_difference(scc_begins_b, scc_begins_e, scc_indptr.begin() + 1);

			// add 1 to each col for diagonal part
			thrust::transform(scc_indptr.begin() + 1, scc_indptr.end(), scc_indptr.begin() + 1,
							  [] __device__(index_t x) { return x + 1; });

			thrust::inclusive_scan(scc_indptr.begin(), scc_indptr.end(), scc_indptr.begin());
		}

		index_t nnz = scc_indptr.back();
		d_idxvec scc_cols(nnz), scc_rows(nnz);
		thrust::device_vector<float> scc_data(nnz, 1.f);

		// this creates rows and data of scc
		{
			int blocksize = 512;
			int gridsize = (scc_size + blocksize - 1) / blocksize;
			scatter_rows_data<<<gridsize, blocksize>>>(scc_indptr.data().get(), scc_rows.data().get(),
													   scc_data.data().get(), rows_.data().get(), indptr_.data().get(),
													   sccs.data().get() + terminals_offsets[i], scc_size);

			CHECK_CUDA(cudaDeviceSynchronize());

			thrust::transform(scc_rows.begin(), scc_rows.end(), scc_rows.begin(),
							  [map = reverse_labels.data().get()] __device__(index_t x) { return map[x]; });
		}

		// this decompresses indptr into cols
		CHECK_CUSPARSE(cusparseXcsr2coo(context_.cusparse_handle, scc_indptr.data().get(), nnz, scc_size,
										scc_cols.data().get(), CUSPARSE_INDEX_BASE_ZERO));

		index_t row_to_remove = scc_rows.front();

		// this removes one row
		{
			auto part_point = thrust::stable_partition(thrust::make_zip_iterator(scc_rows.begin(), scc_cols.begin()),
													   thrust::make_zip_iterator(scc_rows.end(), scc_cols.end()),
													   [row_to_remove] __device__(thrust::tuple<index_t, index_t> x) {
														   return thrust::get<0>(x) != row_to_remove;
													   });

			scc_rows.resize(part_point - thrust::make_zip_iterator(scc_rows.begin(), scc_cols.begin()));
			scc_cols.resize(part_point - thrust::make_zip_iterator(scc_rows.begin(), scc_cols.begin()));
		}

		// this compresses rows back into indptr
		CHECK_CUSPARSE(cusparseXcoo2csr(context_.cusparse_handle, scc_cols.data().get(), scc_cols.size(), scc_size,
										scc_indptr.data().get(), CUSPARSE_INDEX_BASE_ZERO));

		// now we do minors
		d_idxvec minor_indptr(scc_size - 1), minor_rows(scc_rows.size());
		thrust::host_vector<index_t> h_scc_indptr = scc_indptr;
		for (size_t minor_i = 0; minor_i < scc_size; i++)
		{
			// copy indptr
			thrust::copy(scc_indptr.begin(), scc_indptr.begin() + i + 1, minor_indptr.begin());
			auto offset = h_scc_indptr[i + 1] - h_scc_indptr[i];
			thrust::transform(scc_indptr.begin() + i + 2, scc_indptr.end(), minor_indptr.begin() + i + 1,
							  [offset] __device__(index_t x) { x - offset; });

			// copy rows
			thrust::copy(scc_rows.begin(), scc_rows.begin() + h_scc_indptr[i], minor_rows.begin());
			thrust::copy(scc_rows.begin() + h_scc_indptr[i + 1], scc_rows.end(), minor_rows.begin() + h_scc_indptr[i]);

			// determinant
		}
	}
}

void solver::solve() {}
