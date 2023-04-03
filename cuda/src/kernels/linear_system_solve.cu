#include <thrust/binary_search.h>

#include "kernels.h"

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

	row = scc_offset + in_scc_offset;

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

void run_cuda_kernel_splu_symbolic_fact_triv_nnz(const index_t sccs_rows, const index_t scc_count,
												 const index_t* scc_sizes, const index_t* scc_offsets,
												 const index_t* A_indices, const index_t* A_indptr, index_t* As_nnz)
{
	int grid_size = (sccs_rows + block_size - 1) / block_size;

	cuda_kernel_splu_symbolic_fact_triv_nnz<<<grid_size, block_size>>>(sccs_rows, scc_count, scc_sizes, scc_offsets,
																	   A_indices, A_indptr, As_nnz);
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

	row = scc_offset + in_scc_offset;

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

void run_cuda_kernel_splu_symbolic_fact_triv_populate(const index_t sccs_rows, const index_t scc_count,
													  const index_t* scc_sizes, const index_t* scc_offsets,
													  const index_t* A_indptr, const index_t* A_indices,
													  const real_t* A_data, index_t* As_indptr, index_t* As_indices,
													  real_t* As_data)
{
	int grid_size = (sccs_rows + block_size - 1) / block_size;

	cuda_kernel_splu_symbolic_fact_triv_populate<<<grid_size, block_size>>>(
		sccs_rows, scc_count, scc_sizes, scc_offsets, A_indptr, A_indices, A_data, As_indptr, As_indices, As_data);
}
