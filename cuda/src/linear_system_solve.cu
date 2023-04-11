#include <cusolverRf.h>

#include <thrust/adjacent_difference.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/partition.h>
#include <thrust/sequence.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include "diagnostics.h"
#include "kernels/kernels.h"
#include "linear_system_solve.h"
#include "sparse_utils.h"
#include "utils.h"

constexpr size_t big_scc_threshold = 2;

sparse_csr_matrix dense_lu_wrapper(cu_context& context, cudaStream_t stream, index_t n, index_t nnz, index_t* indptr,
								   index_t* indices, real_t* data)
{
	size_t big_rows_size = nnz;
	index_t* big_rows;
	CHECK_CUDA(cudaMallocAsync(&big_rows, sizeof(index_t) * big_rows_size, stream));

	// modify
	{
		auto end = thrust::copy_if(thrust::cuda::par.on(stream), indices, indices + nnz, big_rows,
								   [n] __device__(index_t x) { return x >= n; });

		big_rows_size = end - big_rows;

		if (big_rows_size)
		{
			thrust::sort(thrust::cuda::par.on(stream), big_rows, big_rows + big_rows_size);
			end = thrust::unique(thrust::cuda::par.on(stream), big_rows, big_rows + big_rows_size);

			big_rows_size = end - big_rows;

			index_t* map;
			CHECK_CUDA(cudaMallocAsync(&map, sizeof(index_t) * big_rows_size, stream));

			thrust::for_each_n(thrust::cuda::par.on(stream), thrust::counting_iterator<index_t>(0), big_rows_size,
							   [map, big_rows, n] __device__(index_t i) { map[big_rows[i]] = n + i; });

			thrust::transform_if(
				thrust::cuda::par.on(stream), indices, indices + nnz, indices,
				[map] __device__(index_t x) { return map[x]; }, [n] __device__(index_t x) { return x >= n; });

			CHECK_CUDA(cudaFreeAsync(map, stream));
		}
	}

	index_t rows = n;
	index_t cols = n + big_rows_size;

	CHECK_CUDA(cudaStreamSynchronize(stream));

	d_datvec dense_M = sparse2dense(context.cusparse_handle, n, nnz, rows, cols, indptr, indices, data);

	dense_lu(context.cusolver_dn_handle, dense_M, rows, cols);

	sparse_csr_matrix M = dense2sparse(context.cusparse_handle, dense_M, rows, cols);

	sort_sparse_matrix(context.cusparse_handle, M);

	if (big_rows_size)
	{
		thrust::transform_if(
			thrust::cuda::par.on(stream), M.indices.begin(), M.indices.end(), M.indices.begin(),
			[map = big_rows, n] __device__(index_t x) { return map[x - n]; },
			[n] __device__(index_t x) { return x >= n; });
	}

	CHECK_CUDA(cudaFreeAsync(big_rows, stream));
	return M;
}

host_sparse_csr_matrix host_lu_wrapper(cusolverSpHandle_t handle, h_idxvec&& indptr, h_idxvec&& rows, h_datvec&& data)
{
	host_sparse_csr_matrix M;

	auto orig_n = indptr.size() - 1;
	auto nnz = rows.size();

	h_idxvec big_rows(nnz);
	h_idxvec map;

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

std::vector<sparse_csr_matrix> lu_big_nnz(cu_context& context, index_t big_scc_start, const h_idxvec& scc_sizes,
										  const h_idxvec& scc_offsets, const d_idxvec& A_indptr, d_idxvec& A_indices,
										  d_datvec& A_data, d_idxvec& As_indptr)
{
	std::vector<sparse_csr_matrix> lu_vec;
	lu_vec.reserve(scc_sizes.size() - big_scc_start);

	std::array<cudaStream_t, 5> streams;

	for (size_t i = 0; i < streams.size(); i++)
		CHECK_CUDA(cudaStreamCreate(streams.data() + i));

	thrust::for_each(
		thrust::host, thrust::make_counting_iterator<index_t>(big_scc_start),
		thrust::make_counting_iterator<index_t>(scc_sizes.size()), [&](index_t i) {
			const index_t scc_offset = scc_offsets[i];
			const index_t scc_size = (i == big_scc_start) ? scc_sizes[i] : scc_sizes[i] - scc_sizes[i - 1];
			cudaStream_t stream = streams[i % streams.size()];

			if constexpr (diags_enabled)
			{
				printf("\r                                                                  ");
				printf("\rLU (big nnz): %i/%i with size %i", (i + 1) - big_scc_start,
					   (index_t)scc_sizes.size() - big_scc_start, scc_size);
				if (i == scc_sizes.size() - 1)
					printf("\n");
			}

			// create indptr
			index_t* scc_indptr;
			CHECK_CUDA(cudaMallocAsync(&scc_indptr, sizeof(index_t) * (scc_size + 1), stream));

			thrust::copy(thrust::cuda::par.on(stream), A_indptr.begin() + scc_offset,
						 A_indptr.begin() + scc_offset + scc_size + 1, scc_indptr);

			thrust::transform(thrust::cuda::par.on(stream), scc_indptr, scc_indptr + scc_size + 1, scc_indptr,
							  [base = A_indptr.data().get() + scc_offset] __device__(index_t x) { return x - *base; });

			const index_t base = A_indptr[scc_offset];
			const index_t scc_nnz = A_indptr[scc_offset + scc_size] - base;

			thrust::transform(thrust::cuda::par.on(stream), A_indices.begin() + base,
							  A_indices.begin() + base + scc_nnz, A_indices.begin() + base,
							  [scc_offset] __device__(index_t x) { return x - scc_offset; });


			sparse_csr_matrix M = dense_lu_wrapper(context, stream, scc_size, scc_nnz, scc_indptr,
												   A_indices.data().get() + base, A_data.data().get() + base);

			CHECK_CUDA(cudaFreeAsync(scc_indptr, stream));

			thrust::transform(thrust::cuda::par.on(stream), M.indices.begin(), M.indices.end(), M.indices.begin(),
							  [scc_offset] __device__(index_t x) { return x + scc_offset; });

			thrust::adjacent_difference(thrust::cuda::par.on(stream), M.indptr.begin() + 1, M.indptr.end(),
										As_indptr.begin() + scc_offset + 1);

			lu_vec.emplace_back(std::move(M));
		});

	for (size_t i = 0; i < streams.size(); i++)
	{
		CHECK_CUDA(cudaStreamSynchronize(streams[i]));
		CHECK_CUDA(cudaStreamDestroy(streams[i]));
	}

	return lu_vec;
}

void lu_big_populate(cusolverSpHandle_t handle, index_t big_scc_start, const d_idxvec& scc_offsets,
					 const d_idxvec& As_indptr, d_idxvec& As_indices, d_datvec& As_data,
					 const std::vector<sparse_csr_matrix>& lus)
{
	for (size_t i = 0; i < lus.size(); i++)
	{
		const index_t scc_offset = scc_offsets[big_scc_start + i];
		const index_t scc_size = scc_offsets[big_scc_start + i];

		const index_t begin = As_indptr[scc_offset];

		CHECK_CUDA(cudaMemcpy(As_indices.data().get() + begin, lus[i].indices.data().get(),
							  sizeof(index_t) * lus[i].indices.size(), cudaMemcpyDeviceToDevice));

		CHECK_CUDA(cudaMemcpy(As_data.data().get() + begin, lus[i].data.data().get(),
							  sizeof(index_t) * lus[i].data.size(), cudaMemcpyDeviceToDevice));
	}
}

void splu(cu_context& context, const d_idxvec& scc_offsets, const d_idxvec& A_indptr, d_idxvec& A_indices,
		  d_datvec& A_data, d_idxvec& As_indptr, d_idxvec& As_indices, d_datvec& As_data)
{
	d_idxvec part_scc_sizes, part_scc_offsets;
	auto small_sccs_size = partition_sccs(scc_offsets, part_scc_sizes, part_scc_offsets);
	auto big_sccs_size = scc_offsets.size() - 1 - small_sccs_size;

	const index_t small_scc_rows = small_sccs_size == 0 ? 0 : part_scc_sizes[small_sccs_size - 1];
	const index_t big_scc_rows = big_sccs_size == 0 ? 0 : part_scc_sizes.back();

	if constexpr (diags_enabled)
	{
		diag_print("LU: total number of vertices: ", A_indptr.size() - 1);
		diag_print("LU: small (<=", big_scc_threshold, ") sccs count is ", small_sccs_size, " with ", small_scc_rows,
				   " edges");
		diag_print("LU: big (> ", big_scc_threshold, ") sccs count is ", big_sccs_size, " with ", big_scc_rows,
				   " edges");
		print_big_scc_info(small_sccs_size, part_scc_sizes);
	}

	As_indptr.resize(A_indptr.size());
	As_indptr[0] = 0;

	Timer t;

	// first we count nnz of triv
	{
		t.Start();
		run_cuda_kernel_splu_symbolic_fact_triv_nnz(small_scc_rows, small_sccs_size, part_scc_sizes.data().get(),
													part_scc_offsets.data().get(), A_indices.data().get(),
													A_indptr.data().get(), As_indptr.data().get() + 1);
		t.Stop();
		diag_print("LU (small nnz): ", t.Millisecs(), "ms");
	}

	std::vector<sparse_csr_matrix> lus;

	// without waiting we compute nnz of non triv
	{
		t.Start();
		lus = lu_big_nnz(context, small_sccs_size, part_scc_sizes, part_scc_offsets, A_indptr, A_indices, A_data,
						 As_indptr);
		t.Stop();
		diag_print("LU (big nnz): ", t.Millisecs(), "ms");
	}

	// we allocate required space
	{
		thrust::inclusive_scan(As_indptr.begin(), As_indptr.end(), As_indptr.begin());
		index_t As_nnz = As_indptr.back();

		As_indices.resize(As_nnz);
		As_data.resize(As_nnz);
	}

	// we populate  triv
	{
		t.Start();
		run_cuda_kernel_splu_symbolic_fact_triv_populate(
			small_scc_rows, small_sccs_size, part_scc_sizes.data().get(), part_scc_offsets.data().get(),
			A_indptr.data().get(), A_indices.data().get(), A_data.data().get(), As_indptr.data().get(),
			As_indices.data().get(), As_data.data().get());
		t.Stop();
		diag_print("LU (small populate): ", t.Millisecs(), "ms");
	}

	// we populate non triv
	{
		t.Start();
		lu_big_populate(context.cusolver_handle, small_sccs_size, part_scc_offsets, As_indptr, As_indices, As_data,
						lus);
		t.Stop();
		diag_print("LU (big populate): ", t.Millisecs(), "ms");
	}

	CHECK_CUDA(cudaDeviceSynchronize());
}

sparse_csr_matrix refactor(sparse_csr_matrix&& A, const sparse_csr_matrix& B, host_sparse_csr_matrix& M)
{
	d_idxvec l_indptr(M.indptr.size());
	d_idxvec u_indptr(M.indptr.size());

	d_idxvec lu_indices(M.nnz());
	thrust::device_vector<double> lu_data(M.nnz());

	d_idxvec M_indptr = M.indptr;
	d_idxvec M_indices = M.indices;
	d_idxvec M_data = M.data;

	thrust::device_vector<double> A_data = A.data;

	// indptr
	{
		thrust::for_each_n(thrust::make_counting_iterator<index_t>(0), M.n(),
						   [M_indptr = M_indptr.data().get(), M_indices = M_indices.data().get(),
							l_indptr = l_indptr.data().get(), u_indptr = u_indptr.data().get()] __device__(index_t i) {
							   auto begin = M_indptr[i];
							   auto end = M_indptr[i + 1];

							   index_t l_size = 0;
							   for (index_t idx = begin; idx < end; idx++)
								   if (M_indices[idx] < i)
									   l_size++;

							   l_indptr[i + 1] = l_size;
							   u_indptr[i + 1] = end - begin - l_size;
						   });

		l_indptr[0] = 0;
		u_indptr[0] = 0;

		thrust::inclusive_scan(l_indptr.begin(), l_indptr.end(), l_indptr.begin());
		thrust::inclusive_scan(u_indptr.begin(), u_indptr.end(), u_indptr.begin());
	}

	size_t l_nnz = l_indptr.back();
	size_t u_nnz = u_indptr.back();

	// indices and data
	{
		thrust::for_each_n(thrust::make_counting_iterator<index_t>(0), M.n(),
						   [M_indptr = M_indptr.data().get(), M_indices = M_indices.data().get(),
							M_data = M_data.data().get(), l_indptr = l_indptr.data().get(),
							u_indptr = u_indptr.data().get(), l_indices = lu_indices.data().get(),
							u_indices = lu_indices.data().get() + l_nnz, l_data = lu_data.data().get(),
							u_data = lu_data.data().get() + l_nnz] __device__(index_t i) {
							   auto begin = M_indptr[i];
							   auto end = M_indptr[i + 1];

							   auto l_begin = l_indptr[i];
							   auto u_begin = u_indptr[i];

							   index_t l_idx = 0;
							   index_t u_idx = 0;
							   for (index_t idx = begin; idx < end; idx++)
							   {
								   index_t index = M_indices[idx];
								   if (index < i)
								   {
									   l_indices[l_begin + l_idx] = index;
									   l_data[l_begin + l_idx] = M_data[idx];
									   l_idx++;
								   }
								   else
								   {
									   u_indices[u_begin + u_idx] = index;
									   u_data[u_begin + u_idx] = M_data[idx];
									   u_idx++;
								   }
							   }
						   });
	}


	cusolverRfHandle_t cusolverRfH = NULL;
	const cusolverRfFactorization_t fact_alg = CUSOLVERRF_FACTORIZATION_ALG0;		// default
	const cusolverRfTriangularSolve_t solve_alg = CUSOLVERRF_TRIANGULAR_SOLVE_ALG1; // default
	double nzero = 0.0;
	double nboost = 0.0;

	CHECK_CUSOLVER(cusolverRfCreate(&cusolverRfH));

	// numerical values for checking "zeros" and for boosting.
	CHECK_CUSOLVER(cusolverRfSetNumericProperties(cusolverRfH, nzero, nboost));

	// choose algorithm for refactorization and solve
	CHECK_CUSOLVER(cusolverRfSetAlgs(cusolverRfH, fact_alg, solve_alg));

	// matrix mode: L and U are CSR format, and L has implicit unit diagonal
	CHECK_CUSOLVER(
		cusolverRfSetMatrixFormat(cusolverRfH, CUSOLVERRF_MATRIX_FORMAT_CSR, CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_L));

	//// fast mode for matrix assembling
	// CHECK_CUSOLVER(cusolverRfSetResetValuesFastMode(cusolverRfH, CUSOLVERRF_RESET_VALUES_FAST_MODE_ON));

	d_idxvec p(M.n()), q(M.n());
	thrust::sequence(p.begin(), p.end());
	thrust::sequence(q.begin(), q.end());

	CHECK_CUSOLVER(cusolverRfSetupDevice(
		A.n(), A.nnz(), A.indptr.data().get(), A.indices.data().get(), A_data.data().get(), l_nnz,
		l_indptr.data().get(), lu_indices.data().get(), lu_data.data().get(), u_nnz, u_indptr.data().get(),
		lu_indices.data().get() + l_nnz, lu_data.data().get() + l_nnz, p.data().get(), q.data().get(), cusolverRfH));
	CHECK_CUDA(cudaDeviceSynchronize());

	CHECK_CUSOLVER(cusolverRfAnalyze(cusolverRfH));
	CHECK_CUDA(cudaDeviceSynchronize());

	// CHECK_CUSOLVER(cusolverRfResetValues(A.n(), A.nnz(), A.indptr.data().get(), A.indices.data().get(),
	//									 A_data.data().get(), p.data().get(), q.data().get(), cusolverRfH));
	// CHECK_CUDA(cudaDeviceSynchronize());

	CHECK_CUSOLVER(cusolverRfRefactor(cusolverRfH));
	CHECK_CUDA(cudaDeviceSynchronize());

	index_t n = A.n();
	sparse_csr_matrix X;

	h_idxvec hb_indptr = B.indptr;
	h_idxvec hb_indices = B.indices;

	thrust::device_vector<double> z_vec(n);

	h_idxvec hx_indptr(B.indptr.size());
	hx_indptr[0] = 0;

	for (int b_idx = 0; b_idx < B.n(); b_idx++)
	{
		thrust::device_vector<double> b_vec(n, 0.f);
		auto start = hb_indptr[b_idx];
		auto end = hb_indptr[b_idx + 1];
		thrust::copy(B.data.begin() + start, B.data.begin() + end,
					 thrust::make_permutation_iterator(b_vec.begin(), B.indices.begin() + start));

		CHECK_CUSOLVER(cusolverRfSolve(cusolverRfH, p.data().get(), q.data().get(), 1, z_vec.data().get(), A.n(),
									   b_vec.data().get(), A.n()));

		CHECK_CUDA(cudaDeviceSynchronize());

		d_datvec x_vec = std::move(b_vec);

		auto x_nnz = thrust::count_if(x_vec.begin(), x_vec.end(), [] __device__(real_t x) { return x != 0.f; });

		auto size_before = X.indices.size();
		X.indices.resize(X.indices.size() + x_nnz);
		X.data.resize(X.data.size() + x_nnz);

		hx_indptr[b_idx + 1] = hx_indptr[b_idx] + x_nnz;

		thrust::copy_if(thrust::make_zip_iterator(x_vec.begin(), thrust::make_counting_iterator<index_t>(0)),
						thrust::make_zip_iterator(x_vec.end(), thrust::make_counting_iterator<index_t>(x_vec.size())),
						thrust::make_zip_iterator(X.data.begin() + size_before, X.indices.begin() + size_before),
						[] __device__(thrust::tuple<real_t, index_t> x) { return thrust::get<0>(x) != 0.f; });
	}

	X.indptr = hx_indptr;

	CHECK_CUSOLVER(cusolverRfDestroy(cusolverRfH));

	return X;
}

sparse_csr_matrix solve_system(cu_context& context, sparse_csr_matrix&& A, const d_idxvec& scc_offsets,
							   const sparse_csr_matrix& B, host_sparse_csr_matrix& M, bool should_refactor)
{
	index_t n = A.n();

	sort_sparse_matrix(context.cusparse_handle, A);

	if (should_refactor)
		return refactor(std::move(A), B, M);

	d_idxvec M_indptr, M_indices;
	d_datvec M_data;

	Timer t;

	t.Start();
	splu(context, scc_offsets, A.indptr, A.indices, A.data, M_indptr, M_indices, M_data);
	t.Stop();

	diag_print("Solving (LU decomposition): ", t.Millisecs(), "ms");

	cusparseMatDescr_t descr_L = 0;
	cusparseMatDescr_t descr_U = 0;
	bsrsv2Info_t info_L = 0;
	bsrsv2Info_t info_U = 0;
	int pBufferSize_L;
	int pBufferSize_U;
	const real_t alpha = 1.;
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

	diag_print("Number of right-hand sides needed to solve: ", B.indptr.size() - 1, " with ", A.n(),
			   " number of unknowns");

	t.Start();

	sparse_csr_matrix X;

	h_idxvec hb_indptr = B.indptr;
	h_idxvec hb_indices = B.indices;

	d_datvec x_vec(n);
	d_datvec z_vec(n);

	h_idxvec hx_indptr(B.indptr.size());
	hx_indptr[0] = 0;

	for (int b_idx = 0; b_idx < B.indptr.size() - 1; b_idx++)
	{
		d_datvec b_vec(n, 0.f);
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


		auto x_nnz = thrust::count_if(x_vec.begin(), x_vec.end(), [] __device__(real_t x) { return x != 0.f; });

		auto size_before = X.indices.size();
		X.indices.resize(X.indices.size() + x_nnz);
		X.data.resize(X.data.size() + x_nnz);

		hx_indptr[b_idx + 1] = hx_indptr[b_idx] + x_nnz;

		thrust::copy_if(thrust::make_zip_iterator(x_vec.begin(), thrust::make_counting_iterator<index_t>(0)),
						thrust::make_zip_iterator(x_vec.end(), thrust::make_counting_iterator<index_t>(x_vec.size())),
						thrust::make_zip_iterator(X.data.begin() + size_before, X.indices.begin() + size_before),
						[] __device__(thrust::tuple<real_t, index_t> x) { return thrust::get<0>(x) != 0.f; });
	}

	X.indptr = hx_indptr;

	t.Stop();
	diag_print("Solving (triangular solving): ", t.Millisecs(), "ms");

	// step 6: free resources
	CHECK_CUSPARSE(cusparseDestroyMatDescr(descr_L));
	CHECK_CUSPARSE(cusparseDestroyMatDescr(descr_U));
	CHECK_CUSPARSE(cusparseDestroyBsrsv2Info(info_L));
	CHECK_CUSPARSE(cusparseDestroyBsrsv2Info(info_U));

	M.indptr = M_indptr;
	M.indices = M_indices;
	M.data = M_data;

	return X;
}
