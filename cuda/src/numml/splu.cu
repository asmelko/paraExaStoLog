#include <cooperative_groups.h>
#include <cusolverSp_LOWLEVEL_PREVIEW.h>
#include <device_launch_parameters.h>

#include <cooperative_groups/reduce.h>
#include <thrust/adjacent_difference.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/partition.h>
#include <thrust/set_operations.h>
#include <thrust/unique.h>

#include "../solver.h"
#include "../sparse_utils.h"
#include "../utils.h"
#include "splu.h"

namespace cg = cooperative_groups;

constexpr size_t big_scc_threshold = 2;

__device__ index_t merge_size(const index_t this_row, const index_t* __restrict__ this_row_indices,
							  const index_t this_row_size, const index_t merging_row,
							  const index_t* __restrict__ merging_row_indices, const index_t merging_row_size)
{
	index_t this_idx = 0;
	index_t merging_idx = 0;

	index_t count = 0;

	while (merging_idx < merging_row_size && merging_row_indices[merging_idx] <= merging_row)
		merging_idx++;

	while (merging_idx < merging_row_size && this_idx < this_row_size)
	{
		const index_t this_data = this_row_indices[this_idx];
		const index_t merging_data = merging_row_indices[merging_idx];

		if (this_data == merging_data)
		{
			this_idx++;
			merging_idx++;
		}
		else if (this_data < merging_data)
		{
			this_idx++;
		}
		else
		{
			merging_idx++;
		}

		count++;
	}

	return count + this_row_size - this_idx + merging_row_size - merging_idx;
}

__device__ void merge(const index_t this_row, const index_t* __restrict__ this_row_indices,
					  const real_t* __restrict__ this_row_data, const index_t this_row_size, const index_t merging_row,
					  const index_t* __restrict__ merging_row_indices, const real_t* __restrict__ merging_row_data,
					  const index_t merging_row_size, index_t* __restrict__ out_indices, real_t* __restrict__ out_data)
{
	index_t this_idx = 0;
	index_t merging_idx = 0;

	index_t out_idx = 0;

	real_t divisor = this_row_data[0] / merging_row_data[0];
	out_data[0] = divisor;
	out_indices[0] = merging_row;

	out_idx++;
	this_idx++;
	merging_idx++;

	while (merging_idx < merging_row_size && this_idx < this_row_size)
	{
		const index_t this_col = this_row_indices[this_idx];
		const index_t merging_col = merging_row_indices[merging_idx];

		if (this_col == merging_col)
		{
			out_indices[out_idx] = this_col;
			out_data[out_idx] = this_row_data[this_idx] - divisor * merging_row_data[merging_idx];
			this_idx++;
			merging_idx++;
			out_idx++;
		}
		else if (this_col < merging_col)
		{
			out_indices[out_idx] = this_col;
			out_data[out_idx] = this_row_data[this_idx];
			this_idx++;
			out_idx++;
		}
		else
		{
			out_indices[out_idx] = merging_col;
			out_data[out_idx] = -divisor * merging_row_data[merging_idx];
			merging_idx++;
			out_idx++;
		}
	}

	while (merging_idx < merging_row_size)
	{
		out_indices[out_idx] = merging_row_indices[merging_idx];
		out_data[out_idx] = -divisor * merging_row_data[merging_idx];
		merging_idx++;
		out_idx++;
	}
	while (this_idx < this_row_size)
	{
		out_indices[out_idx] = this_row_indices[this_idx];
		out_data[out_idx] = this_row_data[this_idx];
		this_idx++;
		out_idx++;
	}
}



__global__ void cuda_kernel_splu_symbolic_fact_triv_nnz(const index_t sccs_rows, const index_t scc_count,
														const index_t* __restrict__ scc_sizes,
														const index_t* __restrict__ scc_offsets,
														const index_t* __restrict__ A_indices,
														const index_t* __restrict__ A_indptr,
														index_t* __restrict__ As_nnz)
{
	index_t row = blockIdx.x * blockDim.x + threadIdx.x;

	if (row >= sccs_rows)
		return;

	const index_t scc_index = thrust::upper_bound(thrust::seq, scc_sizes, scc_sizes + scc_count, row) - scc_sizes;

	const index_t scc_offset = scc_offsets[scc_index];
	const index_t in_scc_offset = row - (scc_index == 0 ? 0 : scc_sizes[scc_index - 1]);

	const index_t scc_size = scc_index == 0 ? scc_sizes[scc_index] : scc_sizes[scc_index] - scc_sizes[scc_index - 1];

	// printf("row %i idx %i off %i inoff %i size %i\n", row, scc_index, scc_offset, in_scc_offset, scc_size);

	row = scc_offset + in_scc_offset;


	if (scc_size > big_scc_threshold)
	{
		printf("problem\n");
		return;
	}

	if (scc_size == 1 || in_scc_offset == 0)
	{
		const index_t row_indices_begin = A_indptr[row];
		index_t row_size = A_indptr[row + 1] - row_indices_begin;

		As_nnz[row] = row_size;
	}
	else
	{
		const index_t row_indices_begin = A_indptr[row];
		index_t row_size = A_indptr[row + 1] - row_indices_begin;
		const index_t* row_indices = A_indices + row_indices_begin;

		const index_t merging_row = row - 1;
		const index_t merging_row_indices_begin = A_indptr[merging_row];
		index_t merging_row_size = A_indptr[merging_row + 1] - merging_row_indices_begin;
		const index_t* merging_row_indices = A_indices + merging_row_indices_begin;

		const index_t new_row_size =
			merge_size(row, row_indices, row_size, merging_row, merging_row_indices, merging_row_size);

		As_nnz[row] = new_row_size;
	}
}

__global__ void cuda_kernel_splu_symbolic_fact_triv_populate(
	const index_t sccs_rows, const index_t scc_count, const index_t* __restrict__ scc_sizes,
	const index_t* __restrict__ scc_offsets, const index_t* __restrict__ A_indptr,
	const index_t* __restrict__ A_indices, const real_t* __restrict__ A_data, index_t* __restrict__ As_indptr,
	index_t* __restrict__ As_indices, real_t* __restrict__ As_data)
{
	index_t row = blockIdx.x * blockDim.x + threadIdx.x;

	if (row >= sccs_rows)
		return;

	const index_t scc_index = thrust::upper_bound(thrust::seq, scc_sizes, scc_sizes + scc_count, row) - scc_sizes;

	const index_t scc_offset = scc_offsets[scc_index];
	const index_t in_scc_offset = row - (scc_index == 0 ? 0 : scc_sizes[scc_index - 1]);

	const index_t scc_size = scc_index == 0 ? scc_sizes[scc_index] : scc_sizes[scc_index] - scc_sizes[scc_index - 1];

	// printf("row %i idx %i off %i inoff %i size %i\n", row, scc_index, scc_offset, in_scc_offset, scc_size);

	row = scc_offset + in_scc_offset;

	if (scc_size > big_scc_threshold)
	{
		printf("problem\n");
		return;
	}

	if (in_scc_offset >= scc_size)
	{
		printf("problem\n");
		return;
	}

	if (scc_size == 1 || in_scc_offset == 0)
	{
		const index_t row_indices_begin = A_indptr[row];
		index_t row_size = A_indptr[row + 1] - row_indices_begin;

		const index_t out_row_indices_begin = As_indptr[row];

		thrust::copy(thrust::seq, A_indices + row_indices_begin, A_indices + row_indices_begin + row_size,
					 As_indices + out_row_indices_begin);

		thrust::copy(thrust::seq, A_data + row_indices_begin, A_data + row_indices_begin + row_size,
					 As_data + out_row_indices_begin);
	}
	else
	{
		const index_t row_indices_begin = A_indptr[row];
		index_t row_size = A_indptr[row + 1] - row_indices_begin;
		const index_t* row_indices = A_indices + row_indices_begin;
		const real_t* row_data = A_data + row_indices_begin;

		const index_t merging_row = row - 1;
		const index_t merging_row_indices_begin = A_indptr[merging_row];
		index_t merging_row_size = A_indptr[merging_row + 1] - merging_row_indices_begin;
		const index_t* merging_row_indices = A_indices + merging_row_indices_begin;
		const real_t* merging_row_data = A_data + merging_row_indices_begin;

		const index_t out_row_indices_begin = As_indptr[row];
		index_t* out_row_indices = As_indices + out_row_indices_begin;
		real_t* out_row_data = As_data + out_row_indices_begin;

		merge(row, row_indices, row_data, row_size, merging_row, merging_row_indices, merging_row_data,
			  merging_row_size, out_row_indices, out_row_data);
	}
}

void host_lu(cusolverSpHandle_t handle, thrust::host_vector<index_t>& indptr, thrust::host_vector<index_t>& rows,
			 const thrust::host_vector<float>& data, thrust::host_vector<index_t>& M_indptr,
			 thrust::host_vector<index_t>& M_indices, thrust::host_vector<float>& M_data)
{
	// print("scc indptr ", d_idxvec(indptr));
	// print("scc indice ", d_idxvec(rows));
	// print("scc data   ", d_datvec(data));

	auto orig_n = indptr.size() - 1;
	auto nnz = rows.size();

	thrust::host_vector<index_t> big_rows(nnz);
	thrust::host_vector<index_t> map;

	// modify
	{
		auto end =
			thrust::copy_if(rows.begin(), rows.end(), big_rows.begin(), [orig_n](index_t x) { return x >= orig_n; });

		big_rows.resize(end - big_rows.begin());

		if (big_rows.size())
		{
			thrust::sort(big_rows.begin(), big_rows.end());
			end = thrust::unique(big_rows.begin(), big_rows.end());

			big_rows.resize(end - big_rows.begin());

			map.resize(big_rows.back() + 1);

			thrust::for_each_n(thrust::host, thrust::counting_iterator<index_t>(0), big_rows.size(),
							   [&](index_t i) { map[big_rows[i]] = orig_n + i; });

			thrust::transform_if(
				rows.begin(), rows.end(), rows.begin(), [&](index_t x) { return map[x]; },
				[orig_n](index_t x) { return x >= orig_n; });

			indptr.resize(indptr.size() + big_rows.size());
			thrust::for_each_n(thrust::host, thrust::counting_iterator<index_t>(0), big_rows.size(),
							   [&](index_t i) { indptr[orig_n + 1 + i] = nnz; });
		}
	}

	auto n = indptr.size() - 1;

	// std::cout << "lu orig n: " << orig_n << " lu n: " << n << std::endl;

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

	CHECK_CUSOLVER(cusolverSpXcsrluAnalysisHost(handle, n, nnz, descr, indptr.data(), rows.data(), info));

	size_t internal_data, workspace;
	CHECK_CUSOLVER(cusolverSpScsrluBufferInfoHost(handle, n, nnz, descr, data.data(), indptr.data(), rows.data(), info,
												  &internal_data, &workspace));

	std::vector<char> buffer(workspace);

	CHECK_CUSOLVER(cusolverSpScsrluFactorHost(handle, n, nnz, descr, data.data(), indptr.data(), rows.data(), info,
											  0.1f, buffer.data()));

	int nnz_l, nnz_u;
	CHECK_CUSOLVER(cusolverSpXcsrluNnzHost(handle, &nnz_l, &nnz_u, info));

	thrust::host_vector<index_t> P(n), Q(n), L_indptr(n + 1), U_indptr(n + 1), L_cols(nnz_l), U_cols(nnz_u);
	thrust::host_vector<float> L_data(nnz_l), U_data(nnz_u);

	CHECK_CUSOLVER(cusolverSpScsrluExtractHost(handle, P.data(), Q.data(), descr_L, L_data.data(), L_indptr.data(),
											   L_cols.data(), descr_U, U_data.data(), U_indptr.data(), U_cols.data(),
											   info, buffer.data()));

	M_indptr.resize(orig_n + 1);
	thrust::for_each_n(thrust::host, thrust::make_counting_iterator<index_t>(0), orig_n + 1,
					   [&](index_t i) { M_indptr[i] = L_indptr[i] + U_indptr[i]; });

	M_indices.resize(M_indptr.back());
	M_data.resize(M_indptr.back());

	thrust::for_each_n(thrust::host, thrust::make_counting_iterator<index_t>(0), orig_n, [&](index_t i) {
		auto begin = M_indptr[i];

		auto L_begin = L_indptr[i];
		auto U_begin = U_indptr[i];

		auto L_end = L_indptr[i + 1];
		auto U_end = U_indptr[i + 1];

		thrust::copy(L_cols.begin() + L_begin, L_cols.begin() + L_end, M_indices.begin() + begin);
		thrust::copy(U_cols.begin() + U_begin, U_cols.begin() + U_end, M_indices.begin() + begin + (L_end - L_begin));

		thrust::copy(L_data.begin() + L_begin, L_data.begin() + L_end, M_data.begin() + begin);
		thrust::copy(U_data.begin() + U_begin, U_data.begin() + U_end, M_data.begin() + begin + (L_end - L_begin));
	});

	// turn back
	if (big_rows.size())
	{
		thrust::copy(big_rows.begin(), big_rows.end(), map.begin() + orig_n);

		thrust::transform_if(
			M_indices.begin(), M_indices.end(), M_indices.begin(), [&](index_t x) { return map[x]; },
			[orig_n](index_t x) { return x >= orig_n; });
	}

	CHECK_CUSPARSE(cusparseDestroyMatDescr(descr));
	CHECK_CUSPARSE(cusparseDestroyMatDescr(descr_L));
	CHECK_CUSPARSE(cusparseDestroyMatDescr(descr_U));
	CHECK_CUSOLVER(cusolverSpDestroyCsrluInfoHost(info));

	// print("scc indptr ", d_idxvec(indptr));
	// print("scc indice ", d_idxvec(rows));
	// print("scc data   ", d_datvec(data));


	// print("L indptr ", d_idxvec(L_indptr));
	// print("L indice ", d_idxvec(L_cols));
	// print("L data   ", d_datvec(L_data));

	// print("U indptr ", d_idxvec(U_indptr));
	// print("U indice ", d_idxvec(U_cols));
	// print("U data   ", d_datvec(U_data));

	// print("M indptr ", d_idxvec(M_indptr));
	// print("M indice ", d_idxvec(M_indices));
	// print("M data   ", d_datvec(M_data));
}


index_t partition_sccs(const d_idxvec& scc_offsets, d_idxvec& partitioned_scc_sizes, d_idxvec& partitioned_scc_offsets)
{
	d_idxvec scc_sizes(scc_offsets.size());

	thrust::adjacent_difference(scc_offsets.begin(), scc_offsets.end(), scc_sizes.begin());

	partitioned_scc_sizes.assign(scc_sizes.begin() + 1, scc_sizes.end());
	partitioned_scc_offsets.assign(scc_offsets.begin(), scc_offsets.end() - 1);

	auto part_point = thrust::stable_partition(
		thrust::make_zip_iterator(partitioned_scc_sizes.begin(), partitioned_scc_offsets.begin()),
		thrust::make_zip_iterator(partitioned_scc_sizes.end(), partitioned_scc_offsets.end()),
		[] __device__(thrust::tuple<index_t, index_t> x) { return thrust::get<0>(x) <= big_scc_threshold; });

	index_t small_sccs = thrust::get<1>(part_point.get_iterator_tuple()) - partitioned_scc_offsets.begin();

	/*big_scc_sizes.resize(thrust::get<0>(big_scc_end.get_iterator_tuple()) - big_scc_sizes.begin());
	big_scc_offsets.resize(big_scc_sizes.size() - 1);*/

	// we need to do this because of terminals that were stored before nonterminals
	index_t base_offset = scc_offsets.front();
	thrust::transform(partitioned_scc_offsets.begin(), partitioned_scc_offsets.end(), partitioned_scc_offsets.begin(),
					  [base_offset] __device__(index_t x) { return x - base_offset; });

	thrust::inclusive_scan(partitioned_scc_sizes.begin(), partitioned_scc_sizes.begin() + small_sccs,
						   partitioned_scc_sizes.begin());
	thrust::inclusive_scan(partitioned_scc_sizes.begin() + small_sccs, partitioned_scc_sizes.end(),
						   partitioned_scc_sizes.begin() + small_sccs);

	// const index_t big_scc_rows = big_scc_sizes.back();
	// std::cout << "splu big sccs " << big_scc_sizes.size() - 1 << std::endl;
	// std::cout << "splu big scc rows " << big_scc_rows << std::endl;
	/*print("scc offs ", scc_offsets);
	print("par offs ", partitioned_scc_offsets);
	print("par size ", partitioned_scc_sizes);*/

	return small_sccs;
}

struct lu_t
{
	thrust::host_vector<index_t> M_indptr;
	thrust::host_vector<index_t> M_indices;
	thrust::host_vector<float> M_data;
};

std::vector<lu_t> lu_big_nnz(cusolverSpHandle_t handle, index_t big_scc_start, const d_idxvec& scc_sizes,
							 const d_idxvec& scc_offsets, const d_idxvec& A_indptr, const d_idxvec& A_indices,
							 const d_datvec& A_data, d_idxvec& As_indptr)
{
	thrust::host_vector<index_t> indptr = A_indptr;
	thrust::host_vector<index_t> indices = A_indices;
	thrust::host_vector<real_t> data = A_data;

	std::vector<lu_t> lu_vec;
	lu_vec.reserve(scc_sizes.size() - big_scc_start);

	thrust::for_each(thrust::host, thrust::make_counting_iterator<index_t>(big_scc_start),
					 thrust::make_counting_iterator<index_t>(scc_sizes.size()), [&](index_t i) {
						 const index_t scc_offset = scc_offsets[i];
						 const index_t scc_size = (i == big_scc_start) ? scc_sizes[i] : scc_sizes[i] - scc_sizes[i - 1];

						 // create indptr
						 thrust::host_vector<index_t> scc_indptr(indptr.begin() + scc_offset,
																 indptr.begin() + scc_offset + scc_size + 1);
						 const index_t base = scc_indptr[0];
						 thrust::transform(scc_indptr.begin(), scc_indptr.end(), scc_indptr.begin(),
										   [base](index_t x) { return x - base; });

						 const index_t scc_nnz = scc_indptr.back();

						 thrust::host_vector<index_t> scc_indices(indices.begin() + base,
																  indices.begin() + base + scc_nnz);
						 thrust::transform(scc_indices.begin(), scc_indices.end(), scc_indices.begin(),
										   [scc_offset](index_t x) { return x - scc_offset; });

						 thrust::host_vector<real_t> scc_data(data.begin() + base, data.begin() + base + scc_nnz);

						 lu_t lu;
						 host_lu(handle, scc_indptr, scc_indices, scc_data, lu.M_indptr, lu.M_indices, lu.M_data);

						 thrust::transform(lu.M_indices.begin(), lu.M_indices.end(), lu.M_indices.begin(),
										   [scc_offset](index_t x) { return x + scc_offset; });

						 thrust::host_vector<index_t> sizes(scc_size);
						 thrust::adjacent_difference(lu.M_indptr.begin() + 1, lu.M_indptr.end(), sizes.begin());

						 CHECK_CUDA(cudaMemcpy(As_indptr.data().get() + scc_offset + 1, sizes.data(),
											   sizeof(index_t) * scc_size, cudaMemcpyHostToDevice));

						 lu_vec.emplace_back(std::move(lu));
					 });

	return lu_vec;
}

void lu_big_populate(cusolverSpHandle_t handle, index_t big_scc_start, const d_idxvec& scc_offsets,
					 const d_idxvec& As_indptr, d_idxvec& As_indices, d_datvec& As_data, const std::vector<lu_t>& lus)
{
	for (size_t i = 0; i < lus.size(); i++)
	{
		const index_t scc_offset = scc_offsets[big_scc_start + i];
		const index_t scc_size = scc_offsets[big_scc_start + i];

		const index_t begin = As_indptr[scc_offset];

		CHECK_CUDA(cudaMemcpy(As_indices.data().get() + begin, lus[i].M_indices.data(),
							  sizeof(index_t) * lus[i].M_indices.size(), cudaMemcpyHostToDevice));

		CHECK_CUDA(cudaMemcpy(As_data.data().get() + begin, lus[i].M_data.data(),
							  sizeof(index_t) * lus[i].M_data.size(), cudaMemcpyHostToDevice));
	}
}

/**
 * Sparse LU Factorization, using a left-looking algorithm on the columns of A.  Based on
 * the symbolic factorization from Rose, Tarjan's fill2 and numeric factorization in SFLU.
 */
void splu(cu_context& context, const d_idxvec& scc_offsets, const d_idxvec& A_indptr, const d_idxvec& A_indices,
		  const d_datvec& A_data, d_idxvec& As_indptr, d_idxvec& As_indices, d_datvec& As_data)
{
	d_idxvec part_scc_sizes, part_scc_offsets;
	auto small_sccs_size = partition_sccs(scc_offsets, part_scc_sizes, part_scc_offsets);
	auto big_sccs_size = scc_offsets.size() - 1 - small_sccs_size;

	const index_t small_scc_rows = small_sccs_size == 0 ? 0 : part_scc_sizes[small_sccs_size - 1];
	const index_t big_scc_rows = big_sccs_size == 0 ? 0 : part_scc_sizes.back();

	// std::cout << "splu big scc rows " << big_scc_rows << std::endl;
	// std::cout << "splu big sccs " << big_sccs_size << std::endl;

	// std::cout << "splu small scc rows " << small_scc_rows << std::endl;
	// std::cout << "splu small sccs " << small_sccs_size << std::endl;

	// std::cout << "splu rows " << A_indptr.size() - 1 << std::endl;

	const int threads_per_block = 512;

	As_indptr.resize(A_indptr.size());
	As_indptr[0] = 0;

	// first we count nnz of triv
	{
		cuda_kernel_splu_symbolic_fact_triv_nnz<<<(small_scc_rows + threads_per_block - 1) / threads_per_block,
												  threads_per_block>>>(
			small_scc_rows, small_sccs_size, part_scc_sizes.data().get(), part_scc_offsets.data().get(),
			A_indices.data().get(), A_indptr.data().get(), As_indptr.data().get() + 1);

		std::cout << "splu triv nnz done" << std::endl;
	}

	// without waiting we compute nnz of non triv
	auto lus = lu_big_nnz(context.cusolver_handle, small_sccs_size, part_scc_sizes, part_scc_offsets, A_indptr,
						  A_indices, A_data, As_indptr);

	std::cout << "splu big nnz done" << std::endl;

	// we allocate required space
	{
		thrust::inclusive_scan(As_indptr.begin(), As_indptr.end(), As_indptr.begin());
		index_t As_nnz = As_indptr.back();

		As_indices.resize(As_nnz);
		As_data.resize(As_nnz);
	}

	std::cout << "splu nnz allocated" << std::endl;

	// we populate  triv
	{
		cuda_kernel_splu_symbolic_fact_triv_populate<<<(small_scc_rows + threads_per_block - 1) / threads_per_block,
													   threads_per_block>>>(
			small_scc_rows, small_sccs_size, part_scc_sizes.data().get(), part_scc_offsets.data().get(),
			A_indptr.data().get(), A_indices.data().get(), A_data.data().get(), As_indptr.data().get(),
			As_indices.data().get(), As_data.data().get());

		std::cout << "splu triv populate done" << std::endl;
	}

	// we populate non triv
	lu_big_populate(context.cusolver_handle, small_sccs_size, part_scc_offsets, As_indptr, As_indices, As_data, lus);

	std::cout << "splu big populate done" << std::endl;

	CHECK_CUDA(cudaDeviceSynchronize());

	// print("A indptr ", A_indptr, 20);
	// print("A indice ", A_indices, 78);
	// print("A data   ", A_data, 78);

	// std::cout << "A nnz " << A_data.size() << " M nnz " << As_data.size() << std::endl;
	// std::cout << "A indptr " << A_indptr.size() << " M indptr " << As_indptr.size() << std::endl;

	// print("M indptr ", As_indptr, 20);
	// print("M indice ", As_indices, 113);
	// print("M data   ", As_data, 113);
}