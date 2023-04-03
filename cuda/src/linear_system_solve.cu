#include <cooperative_groups.h>

#include <thrust/adjacent_difference.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/partition.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include "kernels/kernels.h"
#include "linear_system_solve.h"
#include "sparse_utils.h"

namespace cg = cooperative_groups;

constexpr size_t big_scc_threshold = 2;

host_sparse_csr_matrix host_lu_wrapper(cusolverSpHandle_t handle, h_idxvec&& indptr, h_idxvec&& rows, h_datvec&& data)
{
	host_sparse_csr_matrix M;

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

	host_sparse_csr_matrix h(std::move(indptr), std::move(rows), std::move(data));

	host_sparse_csr_matrix l, u;

	host_lu(handle, h, l, u);

	M.indptr.resize(orig_n + 1);
	thrust::for_each_n(thrust::host, thrust::make_counting_iterator<index_t>(0), orig_n + 1,
					   [&](index_t i) { M.indptr[i] = l.indptr[i] + u.indptr[i]; });

	M.indices.resize(M.indptr.back());
	M.data.resize(M.indptr.back());

	thrust::for_each_n(thrust::host, thrust::make_counting_iterator<index_t>(0), orig_n, [&](index_t i) {
		auto begin = M.indptr[i];

		auto L_begin = l.indptr[i];
		auto U_begin = u.indptr[i];

		auto L_end = l.indptr[i + 1];
		auto U_end = u.indptr[i + 1];

		thrust::copy(l.indices.begin() + L_begin, l.indices.begin() + L_end, M.indices.begin() + begin);
		thrust::copy(u.indices.begin() + U_begin, u.indices.begin() + U_end,
					 M.indices.begin() + begin + (L_end - L_begin));

		thrust::copy(l.data.begin() + L_begin, l.data.begin() + L_end, M.data.begin() + begin);
		thrust::copy(u.data.begin() + U_begin, u.data.begin() + U_end, M.data.begin() + begin + (L_end - L_begin));
	});

	// turn back
	if (big_rows.size())
	{
		thrust::copy(big_rows.begin(), big_rows.end(), map.begin() + orig_n);

		thrust::transform_if(
			M.indices.begin(), M.indices.end(), M.indices.begin(), [&](index_t x) { return map[x]; },
			[orig_n](index_t x) { return x >= orig_n; });
	}

	return M;
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


	// we need to do this because of terminals that were stored before nonterminals
	index_t base_offset = scc_offsets.front();
	thrust::transform(partitioned_scc_offsets.begin(), partitioned_scc_offsets.end(), partitioned_scc_offsets.begin(),
					  [base_offset] __device__(index_t x) { return x - base_offset; });

	thrust::inclusive_scan(partitioned_scc_sizes.begin(), partitioned_scc_sizes.begin() + small_sccs,
						   partitioned_scc_sizes.begin());
	thrust::inclusive_scan(partitioned_scc_sizes.begin() + small_sccs, partitioned_scc_sizes.end(),
						   partitioned_scc_sizes.begin() + small_sccs);

	return small_sccs;
}

std::vector<host_sparse_csr_matrix> lu_big_nnz(cusolverSpHandle_t handle, index_t big_scc_start,
											   const d_idxvec& scc_sizes, const d_idxvec& scc_offsets,
											   const d_idxvec& A_indptr, const d_idxvec& A_indices,
											   const d_datvec& A_data, d_idxvec& As_indptr)
{
	thrust::host_vector<index_t> indptr = A_indptr;
	thrust::host_vector<index_t> indices = A_indices;
	thrust::host_vector<real_t> data = A_data;

	std::vector<host_sparse_csr_matrix> lu_vec;
	lu_vec.reserve(scc_sizes.size() - big_scc_start);

	thrust::for_each(
		thrust::host, thrust::make_counting_iterator<index_t>(big_scc_start),
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

			thrust::host_vector<index_t> scc_indices(indices.begin() + base, indices.begin() + base + scc_nnz);
			thrust::transform(scc_indices.begin(), scc_indices.end(), scc_indices.begin(),
							  [scc_offset](index_t x) { return x - scc_offset; });

			thrust::host_vector<real_t> scc_data(data.begin() + base, data.begin() + base + scc_nnz);


			host_sparse_csr_matrix M =
				host_lu_wrapper(handle, std::move(scc_indptr), std::move(scc_indices), std::move(scc_data));

			thrust::transform(M.indices.begin(), M.indices.end(), M.indices.begin(),
							  [scc_offset](index_t x) { return x + scc_offset; });

			thrust::host_vector<index_t> sizes(scc_size);
			thrust::adjacent_difference(M.indptr.begin() + 1, M.indptr.end(), sizes.begin());

			CHECK_CUDA(cudaMemcpy(As_indptr.data().get() + scc_offset + 1, sizes.data(), sizeof(index_t) * scc_size,
								  cudaMemcpyHostToDevice));

			lu_vec.emplace_back(std::move(M));
		});

	return lu_vec;
}

void lu_big_populate(cusolverSpHandle_t handle, index_t big_scc_start, const d_idxvec& scc_offsets,
					 const d_idxvec& As_indptr, d_idxvec& As_indices, d_datvec& As_data,
					 const std::vector<host_sparse_csr_matrix>& lus)
{
	for (size_t i = 0; i < lus.size(); i++)
	{
		const index_t scc_offset = scc_offsets[big_scc_start + i];
		const index_t scc_size = scc_offsets[big_scc_start + i];

		const index_t begin = As_indptr[scc_offset];

		CHECK_CUDA(cudaMemcpy(As_indices.data().get() + begin, lus[i].indices.data(),
							  sizeof(index_t) * lus[i].indices.size(), cudaMemcpyHostToDevice));

		CHECK_CUDA(cudaMemcpy(As_data.data().get() + begin, lus[i].data.data(), sizeof(index_t) * lus[i].data.size(),
							  cudaMemcpyHostToDevice));
	}
}

void splu(cu_context& context, const d_idxvec& scc_offsets, const d_idxvec& A_indptr, const d_idxvec& A_indices,
		  const d_datvec& A_data, d_idxvec& As_indptr, d_idxvec& As_indices, d_datvec& As_data)
{
	d_idxvec part_scc_sizes, part_scc_offsets;
	auto small_sccs_size = partition_sccs(scc_offsets, part_scc_sizes, part_scc_offsets);
	auto big_sccs_size = scc_offsets.size() - 1 - small_sccs_size;

	const index_t small_scc_rows = small_sccs_size == 0 ? 0 : part_scc_sizes[small_sccs_size - 1];
	const index_t big_scc_rows = big_sccs_size == 0 ? 0 : part_scc_sizes.back();

	As_indptr.resize(A_indptr.size());
	As_indptr[0] = 0;

	// first we count nnz of triv
	{
		run_cuda_kernel_splu_symbolic_fact_triv_nnz(small_scc_rows, small_sccs_size, part_scc_sizes.data().get(),
													part_scc_offsets.data().get(), A_indices.data().get(),
													A_indptr.data().get(), As_indptr.data().get() + 1);
	}

	// without waiting we compute nnz of non triv
	auto lus = lu_big_nnz(context.cusolver_handle, small_sccs_size, part_scc_sizes, part_scc_offsets, A_indptr,
						  A_indices, A_data, As_indptr);

	// we allocate required space
	{
		thrust::inclusive_scan(As_indptr.begin(), As_indptr.end(), As_indptr.begin());
		index_t As_nnz = As_indptr.back();

		As_indices.resize(As_nnz);
		As_data.resize(As_nnz);
	}

	// we populate  triv
	{
		run_cuda_kernel_splu_symbolic_fact_triv_populate(
			small_scc_rows, small_sccs_size, part_scc_sizes.data().get(), part_scc_offsets.data().get(),
			A_indptr.data().get(), A_indices.data().get(), A_data.data().get(), As_indptr.data().get(),
			As_indices.data().get(), As_data.data().get());
	}

	// we populate non triv
	lu_big_populate(context.cusolver_handle, small_sccs_size, part_scc_offsets, As_indptr, As_indices, As_data, lus);

	CHECK_CUDA(cudaDeviceSynchronize());
}

sparse_csr_matrix solve_system(cu_context& context, sparse_csr_matrix&& A, const d_idxvec& scc_offsets,
							   const sparse_csr_matrix& B)
{
	index_t n = A.n();

	sort_sparse_matrix(context.cusparse_handle, A);

	d_idxvec M_indptr, M_indices;
	d_datvec M_data;

	splu(context, scc_offsets, A.indptr, A.indices, A.data, M_indptr, M_indices, M_data);

	cusparseMatDescr_t descr_L = 0;
	cusparseMatDescr_t descr_U = 0;
	bsrsv2Info_t info_L = 0;
	bsrsv2Info_t info_U = 0;
	int pBufferSize_L;
	int pBufferSize_U;
	const float alpha = 1.;
	const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
	const cusparseSolvePolicy_t policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
	const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE;
	const cusparseOperation_t trans_U = CUSPARSE_OPERATION_NON_TRANSPOSE;

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
	CHECK_CUSPARSE(cusparseSbsrsv2_bufferSize(context.cusparse_handle, CUSPARSE_DIRECTION_ROW, trans_L, n,
											  M_data.size(), descr_L, M_data.data().get(), M_indptr.data().get(),
											  M_indices.data().get(), 1, info_L, &pBufferSize_L));
	CHECK_CUSPARSE(cusparseSbsrsv2_bufferSize(context.cusparse_handle, CUSPARSE_DIRECTION_ROW, trans_U, n,
											  M_data.size(), descr_U, M_data.data().get(), M_indptr.data().get(),
											  M_indices.data().get(), 1, info_U, &pBufferSize_U));

	thrust::device_vector<char> buffer_L(pBufferSize_L), buffer_U(pBufferSize_U);

	CHECK_CUSPARSE(cusparseSbsrsv2_analysis(context.cusparse_handle, CUSPARSE_DIRECTION_ROW, trans_L, n, M_data.size(),
											descr_L, M_data.data().get(), M_indptr.data().get(), M_indices.data().get(),
											1, info_L, policy_L, buffer_L.data().get()));

	CHECK_CUSPARSE(cusparseSbsrsv2_analysis(context.cusparse_handle, CUSPARSE_DIRECTION_ROW, trans_U, n, M_data.size(),
											descr_U, M_data.data().get(), M_indptr.data().get(), M_indices.data().get(),
											1, info_U, policy_U, buffer_U.data().get()));

	sparse_csr_matrix X;

	thrust::host_vector<index_t> hb_indptr = B.indptr;
	thrust::host_vector<index_t> hb_indices = B.indices;

	thrust::device_vector<float> x_vec(n);
	thrust::device_vector<float> z_vec(n);

	thrust::host_vector<index_t> hx_indptr(B.indptr.size());
	hx_indptr[0] = 0;

	for (int b_idx = 0; b_idx < B.indptr.size() - 1; b_idx++)
	{
		thrust::device_vector<float> b_vec(n, 0.f);
		auto start = hb_indptr[b_idx];
		auto end = hb_indptr[b_idx + 1];
		thrust::copy(B.data.begin() + start, B.data.begin() + end,
					 thrust::make_permutation_iterator(b_vec.begin(), B.indices.begin() + start));

		// step 6: solve L*z = x
		CHECK_CUSPARSE(cusparseSbsrsv2_solve(context.cusparse_handle, CUSPARSE_DIRECTION_ROW, trans_L, n, M_data.size(),
											 &alpha, descr_L, M_data.data().get(), M_indptr.data().get(),
											 M_indices.data().get(), 1, info_L, b_vec.data().get(), z_vec.data().get(),
											 policy_L, buffer_L.data().get()));

		CHECK_CUSPARSE(cusparseSbsrsv2_solve(context.cusparse_handle, CUSPARSE_DIRECTION_ROW, trans_U, n, M_data.size(),
											 &alpha, descr_U, M_data.data().get(), M_indptr.data().get(),
											 M_indices.data().get(), 1, info_U, z_vec.data().get(), x_vec.data().get(),
											 policy_U, buffer_U.data().get()));


		auto x_nnz = thrust::count_if(x_vec.begin(), x_vec.end(), [] __device__(float x) { return x != 0.f; });

		auto size_before = X.indices.size();
		X.indices.resize(X.indices.size() + x_nnz);
		X.data.resize(X.data.size() + x_nnz);

		hx_indptr[b_idx + 1] = hx_indptr[b_idx] + x_nnz;

		thrust::copy_if(thrust::make_zip_iterator(x_vec.begin(), thrust::make_counting_iterator<index_t>(0)),
						thrust::make_zip_iterator(x_vec.end(), thrust::make_counting_iterator<index_t>(x_vec.size())),
						thrust::make_zip_iterator(X.data.begin() + size_before, X.indices.begin() + size_before),
						[] __device__(thrust::tuple<float, index_t> x) { return thrust::get<0>(x) != 0.f; });
	}

	X.indptr = hx_indptr;

	// step 6: free resources
	CHECK_CUSPARSE(cusparseDestroyMatDescr(descr_L));
	CHECK_CUSPARSE(cusparseDestroyMatDescr(descr_U));
	CHECK_CUSPARSE(cusparseDestroyBsrsv2Info(info_L));
	CHECK_CUSPARSE(cusparseDestroyBsrsv2Info(info_U));

	return X;
}
