#include <cusolverSp_LOWLEVEL_PREVIEW.h>
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
	  indptr_(t.indptr)
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

void solver::solve_terminal_part()
{
	// vector of vertex indices
	d_idxvec sccs(thrust::make_counting_iterator<index_t>(0), thrust::make_counting_iterator<index_t>(labels_.size()));

	// vector of terminal scc begins and ends
	thrust::host_vector<index_t> terminals_offsets;
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

		print("partitioned sccs ", sccs);
		print("partitioned labs ", labels_);

		terminals_offsets.push_back(partition_point - thrust::make_zip_iterator(sccs.begin(), labels_.begin()));
	}

	// this is point that partitions terminals and nonterminals
	auto sccs_terminals_end = partition_point.get_iterator_tuple();

	term_indptr = terminals_offsets;
	term_rows.resize(terminals_offsets.back());
	term_data.resize(terminals_offsets.back());

	thrust::copy(sccs.begin(), sccs.begin() + terminals_offsets.back(), term_rows.begin());

	for (size_t terminal_scc_idx = 1; terminal_scc_idx < terminals_offsets.size(); terminal_scc_idx++)
	{
		size_t scc_size = terminals_offsets[terminal_scc_idx] - terminals_offsets[terminal_scc_idx - 1];

		if (scc_size == 1)
		{
			term_data[terminals_offsets[terminal_scc_idx - 1]] = 1;
			continue;
		}

		d_idxvec scc_indptr(scc_size + 1);
		scc_indptr[0] = 0;

		// create map for scc vertices so they start from 0
		thrust::copy(thrust::make_counting_iterator<intptr_t>(0), thrust::make_counting_iterator<intptr_t>(scc_size),
					 thrust::make_permutation_iterator(reverse_labels.begin(),
													   sccs.begin() + terminals_offsets[terminal_scc_idx - 1]));

		print("labels map ", reverse_labels);

		// this creates indptr of scc in CSC
		{
			auto scc_begins_b = thrust::make_permutation_iterator(
				indptr_.begin(), sccs.begin() + terminals_offsets[terminal_scc_idx - 1]);

			auto scc_begins_e =
				thrust::make_permutation_iterator(indptr_.begin(), sccs.begin() + terminals_offsets[terminal_scc_idx]);

			auto scc_ends_b = thrust::make_permutation_iterator(indptr_.begin() + 1,
																sccs.begin() + terminals_offsets[terminal_scc_idx - 1]);

			auto scc_ends_e = thrust::make_permutation_iterator(indptr_.begin() + 1,
																sccs.begin() + terminals_offsets[terminal_scc_idx]);

			// first get sizes of each col - also add 1 for diagonal part
			thrust::transform(
				thrust::make_zip_iterator(scc_begins_b, scc_ends_b),
				thrust::make_zip_iterator(scc_begins_e, scc_ends_e), scc_indptr.begin() + 1,
				[] __device__(thrust::tuple<index_t, index_t> x) { return 1 + thrust::get<1>(x) - thrust::get<0>(x); });

			print("scc_indptr sizes before ", scc_indptr);

			thrust::inclusive_scan(scc_indptr.begin(), scc_indptr.end(), scc_indptr.begin());

			print("scc_indptr sizes after  ", scc_indptr);
		}

		index_t nnz = scc_indptr.back();
		d_idxvec scc_cols(nnz), scc_rows(nnz);
		thrust::device_vector<float> scc_data(nnz, 1.f);

		// this creates rows and data of scc
		{
			int blocksize = 512;
			int gridsize = (scc_size + blocksize - 1) / blocksize;
			scatter_rows_data<<<gridsize, blocksize>>>(
				scc_indptr.data().get(), scc_rows.data().get(), scc_data.data().get(), rows_.data().get(),
				indptr_.data().get(), sccs.data().get() + terminals_offsets[terminal_scc_idx - 1], scc_size);

			CHECK_CUDA(cudaDeviceSynchronize());

			print("scc_rows before  ", scc_rows);

			thrust::transform(scc_rows.begin(), scc_rows.end(), scc_rows.begin(),
							  [map = reverse_labels.data().get()] __device__(index_t x) { return map[x]; });

			print("scc_rows after   ", scc_rows);
		}

		// this decompresses indptr into cols
		CHECK_CUSPARSE(cusparseXcsr2coo(context_.cusparse_handle, scc_indptr.data().get(), nnz, scc_size,
										scc_cols.data().get(), CUSPARSE_INDEX_BASE_ZERO));

		print("scc_cols ", scc_cols);

		std::cout << "Row to remove" << scc_indptr.size() - 2 << std::endl;

		// this removes last row
		{
			auto part_point = thrust::stable_partition(
				thrust::make_zip_iterator(scc_rows.begin(), scc_cols.begin(), scc_data.begin()),
				thrust::make_zip_iterator(scc_rows.end(), scc_cols.end(), scc_data.end()),
				[remove_row = scc_indptr.size() - 2] __device__(thrust::tuple<index_t, index_t, float> x) {
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

		thrust::transform(minors.begin(), minors.end(), term_data.begin() + terminals_offsets[terminal_scc_idx - 1],
						  [sum] __device__(float x) { return x / sum; });
	}
}

void solver::solve() {}
