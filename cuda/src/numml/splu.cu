#include <cooperative_groups.h>
#include <device_launch_parameters.h>

#include <cooperative_groups/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/set_operations.h>

#include "../solver.h"
#include "../utils.h"
#include "splu.h"

namespace cg = cooperative_groups;

template <typename T>
__device__ T* allocate(int size)
{
	auto ptr = (T*)malloc(sizeof(T) * size);
	if (ptr == NULL)
		printf("allocation failed\n");

	return ptr;
}

template <typename groupT>
__device__ index_t get_merging_data(groupT& w, const index_t* const volatile __restrict__* __restrict__ As_indices,
									const index_t* __restrict__ As_nnz, const index_t* __restrict__ work,
									index_t* __restrict__ work_indices, const index_t work_size)
{
	index_t merging_data = INT_MAX;

	for (index_t i = w.thread_rank(); i < work_size; i += w.num_threads())
	{
		const index_t row = work[i];
		index_t idx = work_indices[i];
		const index_t size = As_nnz[row];
		const volatile index_t* __restrict__ row_indices = As_indices[row];

		const index_t data = idx != size ? row_indices[idx] : INT_MAX;
		merging_data = data < merging_data ? data : merging_data;
	}

	return cg::reduce(w, merging_data, cg::less<index_t>());
}

template <typename groupT>
__device__ index_t increment_merging_data(groupT& w,
										  const index_t* const volatile __restrict__* __restrict__ As_indices,
										  const index_t* __restrict__ As_nnz, const index_t* __restrict__ work,
										  index_t* __restrict__ work_indices, const index_t work_size,
										  const index_t target)
{
	for (index_t i = w.thread_rank(); i < work_size; i += w.num_threads())
	{
		const index_t row = work[i];
		index_t idx = work_indices[i];
		const index_t size = As_nnz[row];
		const volatile index_t* row_indices = As_indices[row];

		const index_t data = idx != size ? row_indices[idx] : INT_MAX;

		idx += data == target ? 1 : 0;
		work_indices[i] = idx;
	}
}

template <typename groupT, typename indT>
__device__ index_t increment_merging_data_small(groupT& g, indT* __restrict__ row_indices, index_t& row_idx,
												const index_t row_size, const index_t target)
{
	const index_t data = row_idx != row_size ? row_indices[row_idx] : INT_MAX;
	row_idx += data == target ? 1 : 0;
}

template <typename groupT, typename indT>
__device__ index_t increment_merging_data_small(groupT& g, indT* __restrict__ row_indices, index_t& row_idx,
												const index_t row_size, const index_t curr_data, const index_t target)
{
	const bool are_same = curr_data == target;

	row_idx += are_same ? 1 : 0;

	return are_same ? (row_idx != row_size ? row_indices[row_idx] : INT_MAX) : curr_data;
}

template <typename groupT>
__device__ void set_indices(groupT& w, const index_t* const volatile __restrict__* __restrict__ As_indices,
							const index_t* __restrict__ As_nnz, const index_t* __restrict__ work,
							index_t* __restrict__ work_indices, const index_t work_size)
{
	const volatile index_t* __restrict__ As_nnz_v = As_nnz;
	for (index_t i = w.thread_rank(); i < work_size; i += w.num_threads())
	{
		const index_t row = work[i];
		index_t idx = 0;
		const index_t row_size = As_nnz_v[row];

		const volatile index_t* row_indices = As_indices[row];

		while (idx < row_size && row_indices[idx] <= row)
		{
			idx++;
		}

		work_indices[i] = idx;
	}
}

template <typename groupT>
__device__ index_t kway_merge_size(groupT& w, const index_t this_row, const index_t* __restrict__ this_row_indices,
								   const index_t this_row_size,
								   const index_t* const volatile __restrict__* __restrict__ As_indices,
								   const index_t* __restrict__ As_nnz, const index_t* __restrict__ work,
								   index_t* __restrict__ work_indices, const index_t work_size, index_t& new_work_size)
{
	if (work_size <= 31)
		return kway_merge_size_small(w, this_row, this_row_indices, this_row_size, As_indices, As_nnz, work, work_size,
									 new_work_size);
	index_t new_row_size = 0;
	new_work_size = 0;

	index_t this_row_idx = 0;

	// set indices after row
	set_indices(w, As_indices, As_nnz, work, work_indices, work_size);

	index_t l_data = get_merging_data(w, As_indices, As_nnz, work, work_indices, work_size);

	while (this_row_idx < this_row_size && l_data != INT_MAX)
	{
		index_t r_data = this_row_indices[this_row_idx];

		if (r_data == l_data)
		{
			this_row_idx++;
			increment_merging_data(w, As_indices, As_nnz, work, work_indices, work_size, l_data);
			l_data = get_merging_data(w, As_indices, As_nnz, work, work_indices, work_size);
		}
		else if (l_data < r_data)
		{
			if (l_data < this_row)
				new_work_size++;

			increment_merging_data(w, As_indices, As_nnz, work, work_indices, work_size, l_data);
			l_data = get_merging_data(w, As_indices, As_nnz, work, work_indices, work_size);
		}
		else
		{
			this_row_idx++;
		}
		new_row_size++;
	}

	// merging rows are all merged
	if (l_data == INT_MAX)
		return new_row_size + this_row_size - this_row_idx;

	if (this_row_idx == this_row_size)
	{
		while (l_data != INT_MAX)
		{
			if (l_data < this_row)
				new_work_size++;
			new_row_size++;

			increment_merging_data(w, As_indices, As_nnz, work, work_indices, work_size, l_data);
			l_data = get_merging_data(w, As_indices, As_nnz, work, work_indices, work_size);
		}
	}

	return new_row_size;
}

template <typename groupT>
__device__ index_t kway_merge_size_small(groupT& w, const index_t this_row,
										 const index_t* __restrict__ this_row_indices, const index_t this_row_size,
										 const index_t* const volatile __restrict__* __restrict__ As_indices,
										 const volatile index_t* __restrict__ As_nnz, const index_t* __restrict__ work,
										 const index_t work_size, index_t& new_work_size)
{
	if (w.thread_rank() >= work_size + 1)
		return;

	auto g = cg::coalesced_threads();

	index_t new_row_size = 0;
	new_work_size = 0;

	index_t merging_data = INT_MAX;
	index_t merging_row_idx = 0;
	const index_t merging_row = w.thread_rank() == 0 ? this_row : work[w.thread_rank() - 1];
	const index_t merging_row_size = w.thread_rank() == 0 ? this_row_size : As_nnz[merging_row];
	const volatile index_t* merging_row_indices = w.thread_rank() == 0 ? this_row_indices : As_indices[merging_row];

	// set indices after row
	if (w.thread_rank() != 0)
	{
		while (merging_row_idx < merging_row_size && merging_row_indices[merging_row_idx] <= merging_row)
		{
			merging_row_idx++;
		}
	}

	merging_data = merging_row_indices[merging_row_idx];

	index_t l_data = cg::reduce(g, merging_data, cg::less<index_t>());

	while (l_data != INT_MAX)
	{
		if (l_data != merging_data && l_data < merging_row)
			new_work_size++;
		new_row_size++;

		merging_data = increment_merging_data_small(g, merging_row_indices, merging_row_idx, merging_row_size,
													merging_data, l_data);
		l_data = cg::reduce(g, merging_data, cg::less<index_t>());
	}

	return new_row_size;
}

template <typename groupT>
__device__ void kway_merge(groupT& w, const index_t this_row, const index_t* __restrict__ this_row_indices,
						   const index_t this_row_size, const index_t* const __restrict__* __restrict__ As_indices,
						   const index_t* __restrict__ As_nnz, const index_t* __restrict__ work,
						   index_t* __restrict__ work_indices, const index_t work_size,
						   index_t* __restrict__ new_row_indices)
{
	if (work_size <= 31)
		return kway_merge_small(w, this_row, this_row_indices, this_row_size, As_indices, As_nnz, work, work_size,
								new_row_indices);

	index_t new_row_idx = 0;
	index_t this_row_idx = 0;

	// set indices after row
	set_indices(w, As_indices, As_nnz, work, work_indices, work_size);

	index_t l_data = get_merging_data(w, As_indices, As_nnz, work, work_indices, work_size);

	while (this_row_idx < this_row_size && l_data != INT_MAX)
	{
		index_t r_data = this_row_indices[this_row_idx];

		index_t to_write;

		if (r_data == l_data)
		{
			to_write = r_data;

			this_row_idx++;
			increment_merging_data(w, As_indices, As_nnz, work, work_indices, work_size, l_data);
			l_data = get_merging_data(w, As_indices, As_nnz, work, work_indices, work_size);
		}
		else if (l_data < r_data)
		{
			to_write = l_data;

			increment_merging_data(w, As_indices, As_nnz, work, work_indices, work_size, l_data);
			l_data = get_merging_data(w, As_indices, As_nnz, work, work_indices, work_size);
		}
		else
		{
			to_write = r_data;

			this_row_idx++;
		}

		if (w.thread_rank() == 0)
			new_row_indices[new_row_idx] = to_write;

		new_row_idx++;
	}

	// merging rows are all merged
	if (l_data == INT_MAX)
	{
		for (index_t i = this_row_idx + w.thread_rank(); i < this_row_size; i += w.num_threads())
			new_row_indices[new_row_idx + i - this_row_idx] = this_row_indices[i];
	}

	if (this_row_idx == this_row_size)
	{
		while (l_data != INT_MAX)
		{
			if (w.thread_rank() == 0)
				new_row_indices[new_row_idx++] = l_data;

			increment_merging_data(w, As_indices, As_nnz, work, work_indices, work_size, l_data);
			l_data = get_merging_data(w, As_indices, As_nnz, work, work_indices, work_size);
		}
	}
}

template <typename groupT>
__device__ void kway_merge_small(groupT& w, const index_t this_row, const index_t* __restrict__ this_row_indices,
								 const index_t this_row_size,
								 const index_t* const __restrict__* __restrict__ As_indices,
								 const index_t* __restrict__ As_nnz, const index_t* __restrict__ work,
								 const index_t work_size, index_t* __restrict__ new_row_indices)
{
	if (w.thread_rank() >= work_size + 1)
		return;

	auto g = cg::coalesced_threads();

	index_t new_row_idx = 0;

	index_t merging_data = INT_MAX;
	index_t merging_row_idx = 0;
	const index_t merging_row = w.thread_rank() == 0 ? this_row : work[w.thread_rank() - 1];
	const index_t merging_row_size = w.thread_rank() == 0 ? this_row_size : As_nnz[merging_row];
	const index_t* merging_row_indices = w.thread_rank() == 0 ? this_row_indices : As_indices[merging_row];

	// set indices after row
	if (w.thread_rank() != 0)
	{
		while (merging_row_idx < merging_row_size && merging_row_indices[merging_row_idx] <= merging_row)
		{
			merging_row_idx++;
		}
	}

	merging_data = merging_row_indices[merging_row_idx];

	index_t l_data = cg::reduce(g, merging_data, cg::less<index_t>());

	while (l_data != INT_MAX)
	{
		if (w.thread_rank() == 0)
			new_row_indices[new_row_idx++] = l_data;

		merging_data = increment_merging_data_small(g, merging_row_indices, merging_row_idx, merging_row_size,
													merging_data, l_data);
		l_data = cg::reduce(g, merging_data, cg::less<index_t>());
	}
}

__device__ void find_new_work(const index_t row, const index_t* __restrict__ new_indices,
							  const index_t new_indices_size, const index_t* __restrict__ old_indices,
							  const index_t old_indices_size, index_t* __restrict__ new_work)
{
	index_t new_idx = 0;
	index_t work_idx = 0;

	for (index_t old_idx = 0; old_idx < old_indices_size; old_idx++)
	{
		const index_t c = old_indices[old_idx];

		if (c > row)
			return;

		while (true)
		{
			const index_t nc = new_indices[new_idx++];

			if (nc == c)
				break;

			new_work[work_idx++] = nc;
		}
	}
}

__global__ void cuda_kernel_splu_symbolic_fact(const index_t A_rows, const index_t* __restrict__ A_indices,
											   const index_t* __restrict__ A_indptr, index_t* __restrict__ As_nnz,
											   index_t* __restrict__* __restrict__ As_indices,
											   volatile index_t* __restrict__ degree)
{
	const index_t row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;

	// printf("thread %i row %i\n", blockIdx.x * blockDim.x + threadIdx.x, row);

	auto warp = cg::tiled_partition<32>(cg::this_thread_block());

	if (row >= A_rows)
		return;

	// if (warp.thread_rank() == 0)
	//	printf("row %i started\n", row);

	index_t row_size;
	index_t* row_indices;

	index_t scratchpad_alloc_size = 0;
	index_t scratchpad_size = 0;
	index_t* scratchpad = nullptr;

	{
		const index_t row_indices_begin = A_indptr[row];
		row_size = A_indptr[row + 1] - row_indices_begin;

		if (warp.thread_rank() == 0)
		{
			row_indices = allocate<index_t>(row_size);
		}

		row_indices = warp.shfl(row_indices, 0);

		for (index_t i = warp.thread_rank(); i < row_size; i += warp.num_threads())
		{
			index_t col = (A_indices + row_indices_begin)[i];
			row_indices[i] = col;
			if (col == row)
				scratchpad_alloc_size = i;
		}

		warp.sync();

		auto mask = warp.ballot(scratchpad_alloc_size);

		// if (scratchpad_alloc_size && mask == 0)
		//	printf("error\n");

		if (mask != 0)
		{
			int lane_id = -1;
			while (mask)
			{
				mask >>= 1;
				lane_id++;
			}

			scratchpad_alloc_size = warp.shfl(scratchpad_alloc_size, lane_id);
		}

		// if (bef != 0 && scratchpad_alloc_size == 0)
		//	printf("error\n");


		/*	if (scratchpad_alloc_size != 0)
				printf("thread %i scs\n", (int)warp.thread_rank());*/

		scratchpad_size = scratchpad_alloc_size;

		if (scratchpad_alloc_size)
		{
			if (warp.thread_rank() == 0)
				scratchpad = allocate<index_t>(scratchpad_size * 2);
			scratchpad = warp.shfl(scratchpad, 0);

			for (index_t i = warp.thread_rank(); i < scratchpad_size; i += warp.num_threads())
				scratchpad[i] = row_indices[i];
		}
	}

	warp.sync();

	// if (warp.thread_rank() == 0)
	//{
	//	for (index_t i = 0; i < scratchpad_size; i++)
	//		if (scratchpad[i] != row_indices[i])
	//			printf("error\n");
	// }

	index_t iteration = 0;

	while (scratchpad_size)
	{
		iteration++;

		while (true)
		{
			bool has_degree = false;
			for (index_t i = warp.thread_rank(); i < scratchpad_size; i += warp.num_threads())
			{
				const index_t index = scratchpad[i];

				has_degree |= degree[index] != 0;
			}

			has_degree = warp.any(has_degree);

			if (!has_degree)
				break;
		}

		index_t new_scratchpad_size;
		index_t new_size = kway_merge_size(warp, row, row_indices, row_size, As_indices, As_nnz, scratchpad,
										   scratchpad + scratchpad_size, scratchpad_size, new_scratchpad_size);

		new_size = warp.shfl(new_size, 0);
		new_scratchpad_size = warp.shfl(new_scratchpad_size, 0);

		if (new_size == row_size)
			break;

		// update row
		{
			index_t* row_indices_new;
			if (warp.thread_rank() == 0)
				row_indices_new = allocate<index_t>(new_size);
			row_indices_new = warp.shfl(row_indices_new, 0);

			kway_merge(warp, row, row_indices, row_size, As_indices, As_nnz, scratchpad, scratchpad + scratchpad_size,
					   scratchpad_size, row_indices_new);

			warp.sync();

			// update scratchpad
			{
				if (new_scratchpad_size > scratchpad_alloc_size)
				{
					scratchpad_alloc_size = new_scratchpad_size;

					if (warp.thread_rank() == 0)
					{
						free(scratchpad);
						scratchpad = allocate<index_t>(scratchpad_alloc_size * 2);
					}
					scratchpad = warp.shfl(scratchpad, 0);
				}

				if (warp.thread_rank() == 0)
					find_new_work(row, row_indices_new, new_size, row_indices, row_size, scratchpad);

				warp.sync();

				if (warp.thread_rank() == 0)
					printf("iteration %i row %i new work size %i old work size %i\n", iteration, row,
						   new_scratchpad_size, scratchpad_size);

				scratchpad_size = new_scratchpad_size;
			}

			/*if (warp.thread_rank() == 0)
				printf("row %i after new work \n", row);*/

			// update row size and indices
			row_size = new_size;
			if (warp.thread_rank() == 0)
				free(row_indices);
			row_indices = row_indices_new;
		}
	}

	if (warp.thread_rank() == 0)
	{
		if (scratchpad)
			free(scratchpad);

		As_nnz[row] = row_size;
		As_indices[row] = row_indices;

		__threadfence();
		degree[row] = 0;

		// printf("row %i finished\n", row);
	}
}


__global__ void cuda_kernel_splu_symbolic_populate(const index_t A_rows, const index_t* __restrict__ A_indices,
												   const index_t* __restrict__ A_indptr,
												   const real_t* __restrict__ A_data, index_t* __restrict__ As_indices,
												   const index_t* __restrict__ As_indptr, real_t* __restrict__ As_data,
												   index_t* const* const __restrict__ As_indices_by_row)
{
	const index_t row = blockIdx.x * blockDim.x + threadIdx.x;

	if (row >= A_rows)
		return;


	const index_t out_begin = As_indptr[row];
	const index_t row_size = As_indptr[row + 1] - out_begin;
	index_t* row_indices = As_indices_by_row[row];
	As_indices += out_begin;
	As_data += out_begin;

	for (index_t i = 0; i < row_size; i++)
	{
		const index_t index = row_indices[i];
		As_indices[i] = index;
		As_data[i] = 0.f;
	}

	free(row_indices);

	const index_t orig_begin = A_indptr[row];
	const index_t orig_row_size = A_indptr[row + 1] - orig_begin;
	A_indices += orig_begin;
	A_data += orig_begin;

	index_t As_idx = 0;
	for (index_t i = 0; i < orig_row_size; i++)
	{
		const index_t v = A_indices[i];

		while (As_indices[As_idx] != v)
			As_idx++;

		// printf("row %i wrote data %f at col %i\n", row, A_data[v_i], As_idx);

		As_data[As_idx] = A_data[i];
	}
}


/**
 * Count the number of upper-triangular nonzeros for each column of a CSC matrix.
 * This is inclusive of the main diagonal.
 *
 * Indexed on columns of A.
 */
__global__ void cuda_kernel_count_U_nnz(const index_t A_rows, const index_t A_cols,
										const index_t* __restrict__ At_indices, const index_t* __restrict__ At_indptr,
										index_t* __restrict__ U_col_nnz)
{
	const index_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if (j >= A_cols)
		return;

	index_t nnz = 0;
	for (index_t i_i = At_indptr[j]; i_i < At_indptr[j + 1]; i_i++)
	{
		const index_t i = At_indices[i_i];
		if (i <= j)
			nnz++;
	}

	if (nnz == 0)
		printf("L has some zeros\n");

	U_col_nnz[j] = nnz;
}


/**
 * Performs a binary search on an array between i_start and i_end (inclusive).
 */
static __device__ index_t kernel_indices_binsearch(index_t i_start, index_t i_end, const index_t i_search,
												   const index_t* __restrict__ indices)
{
	index_t i_mid;
	while (i_start <= i_end)
	{
		i_mid = (i_start + i_end) / 2;
		if (indices[i_mid] < i_search)
		{
			i_start = i_mid + 1;
		}
		else if (indices[i_mid] > i_search)
		{
			i_end = i_mid - 1;
		}
		else if (indices[i_mid] == i_search)
		{
			return i_mid;
		}
	}
	return -1;
}

/**
 * The sparse numeric LU factorization from SFLU:
 * "SFLU: Synchronization-Free Sparse LU Factorization for Fast Circuit Simulation on GPUs", J. Zhao, Y. Luo, Z. Jin, Z.
 * Zhou.
 *
 * Indexed on columns of As, where As is given in CSC format and has fill-ins represented by explicit zeros.
 */
template <typename scalar_t>
__global__ void cuda_kernel_splu_numeric_sflu(const index_t A_rows, const index_t A_cols,
											  real_t* __restrict__ As_col_data,
											  const index_t* __restrict__ As_col_indices,
											  const index_t* __restrict__ As_col_indptr,
											  volatile index_t* __restrict__ degree)
{
	const index_t k = blockIdx.x * blockDim.x + threadIdx.x;
	if (k >= A_cols)
	{
		return;
	}

	index_t diag_idx;
	const index_t col_end = As_col_indptr[k + 1];
	for (index_t i_i = As_col_indptr[k]; i_i < col_end; i_i++)
	{
		const index_t i = As_col_indices[i_i];
		if (i == k)
		{
			/* Stop once we get to the diagonal. */
			diag_idx = i_i;
			break;
		}

		/* Busy wait until intermediate results are ready */
		while (degree[i] > 0)
			;

		/* Left-looking product */
		for (index_t j_i = i_i + 1; j_i < col_end; j_i++)
		{
			const index_t j = As_col_indices[j_i];
			const index_t A_ji_i =
				kernel_indices_binsearch(As_col_indptr[i], As_col_indptr[i + 1] - 1, j, As_col_indices);
			if (A_ji_i == -1)
			{
				continue;
			}
			const scalar_t A_ji = As_col_data[A_ji_i];
			const scalar_t A_ik = As_col_data[i_i];

			/* A_{jk} \gets A_{jk} - A_{ji} A_{ik} */
			As_col_data[j_i] -= A_ji * A_ik;
		}

		// printf("thread %i decremented from %i\n", k, degree[k]);
		__threadfence();
		degree[k]--;
	}

	/* Divide column of L by diagonal entry of U */
	const scalar_t A_kk = As_col_data[diag_idx];
	for (index_t i = diag_idx + 1; i < As_col_indptr[k + 1]; i++)
	{
		As_col_data[i] /= A_kk;
	}

	// printf("thread %i decremented from %i\n", k, degree[k]);
	/* Complete the factorization and update column degree */
	__threadfence();
	degree[k]--;
}

/**
 * Sparse LU Factorization, using a left-looking algorithm on the columns of A.  Based on
 * the symbolic factorization from Rose, Tarjan's fill2 and numeric factorization in SFLU.
 */
void splu(cu_context& context, const d_idxvec& A_indptr, const d_idxvec& A_indices, const d_datvec& A_data,
		  d_idxvec& As_indptr, d_idxvec& As_indices, d_datvec& As_data)
{
	const int threads_per_block = 512;

	As_indptr.resize(A_indptr.size());
	As_indptr[0] = 0;

	index_t A_rows = A_indptr.size() - 1;
	index_t A_cols = A_indptr.size() - 1;

	std::cout << "splu start " << A_rows << std::endl;

	std::cout << "splu symbolic nnz" << std::endl;

	index_t* L_nnz;
	index_t** As_indices_by_row;

	CHECK_CUDA(cudaMalloc(&L_nnz, sizeof(index_t) * A_rows));
	CHECK_CUDA(cudaMalloc(&As_indices_by_row, sizeof(index_t*) * A_rows));
	CHECK_CUDA(cudaMemset(As_indices_by_row, 0, sizeof(index_t*) * A_rows));

	cuda_kernel_count_U_nnz<<<(A_rows + threads_per_block - 1) / threads_per_block, threads_per_block>>>(
		A_rows, A_cols, A_indices.data().get(), A_indptr.data().get(), L_nnz);

	CHECK_CUDA(cudaDeviceSynchronize());
	// print("L_nnz ", d_idxvec(L_nnz, L_nnz + A_rows));

	cuda_kernel_splu_symbolic_fact<<<(A_cols + (threads_per_block / 32) - 1) / (threads_per_block / 32),
									 threads_per_block>>>(A_rows, A_indices.data().get(), A_indptr.data().get(),
														  As_indptr.data().get() + 1, As_indices_by_row, L_nnz);

	CHECK_CUDA(cudaDeviceSynchronize());

	CHECK_CUDA(cudaFree(L_nnz));

	std::cout << "splu cumsum" << std::endl;

	thrust::inclusive_scan(As_indptr.begin(), As_indptr.end(), As_indptr.begin());
	index_t As_nnz = As_indptr.back();

	As_indices.resize(As_nnz);
	As_data.resize(As_nnz);

	std::cout << "splu nnz " << As_nnz << std::endl;

	std::cout << "splu symbolic populate" << std::endl;

	cuda_kernel_splu_symbolic_populate<<<(A_cols + threads_per_block - 1) / threads_per_block, threads_per_block>>>(
		A_rows, A_indices.data().get(), A_indptr.data().get(), A_data.data().get(), As_indices.data().get(),
		As_indptr.data().get(), As_data.data().get(), As_indices_by_row);

	CHECK_CUDA(cudaDeviceSynchronize());

	CHECK_CUDA(cudaFree(As_indices_by_row));

	d_idxvec AsT_indptr, AsT_indices;
	d_datvec AsT_data;

	/* Compute the transpose/csc representation of As so that we have easy column access. */
	solver::transpose_sparse_matrix(context.cusparse_handle, As_indptr.data().get(), As_indices.data().get(),
									As_data.data().get(), A_rows, A_cols, As_data.size(), AsT_indptr, AsT_indices,
									AsT_data);
	// print("A indptr ", A_indptr);
	// print("A indice ", A_indices);
	// print("A data   ", A_data);

	// print("As indptr ", As_indptr);
	// print("As indice ", As_indices);
	// print("As data   ", As_data);

	// print("At indptr ", AsT_indptr);
	// print("At indice ", AsT_indices);
	// print("At data   ", AsT_data);

	std::cout << "splu U nnz" << std::endl;

	index_t* U_col_nnz;
	CHECK_CUDA(cudaMalloc(&U_col_nnz, sizeof(index_t) * A_rows));

	/* Perform the numeric factorization on the CSC representation */
	cuda_kernel_count_U_nnz<<<(A_cols + threads_per_block - 1) / threads_per_block, threads_per_block>>>(
		A_rows, A_cols, AsT_indices.data().get(), AsT_indptr.data().get(), U_col_nnz);
	CHECK_CUDA(cudaDeviceSynchronize());

	// print("splu degrees ", d_idxvec(U_col_nnz, U_col_nnz + A_cols));

	std::cout << "splu numeric" << std::endl;

	cuda_kernel_splu_numeric_sflu<real_t><<<(A_cols + threads_per_block - 1) / threads_per_block, threads_per_block>>>(
		A_rows, A_cols, AsT_data.data().get(), AsT_indices.data().get(), AsT_indptr.data().get(), U_col_nnz);

	CHECK_CUDA(cudaDeviceSynchronize());
	CHECK_CUDA(cudaFree(U_col_nnz));

	/* Transpose back into CSR format */
	solver::transpose_sparse_matrix(context.cusparse_handle, AsT_indptr.data().get(), AsT_indices.data().get(),
									AsT_data.data().get(), A_cols, A_rows, AsT_data.size(), As_indptr, As_indices,
									As_data);
}