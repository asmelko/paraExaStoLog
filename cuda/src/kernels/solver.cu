#include "kernels.h"


__global__ void scatter_rows_data(const index_t* __restrict__ dst_indptr, index_t* __restrict__ dst_rows,
								  float* __restrict__ dst_data, const index_t* __restrict__ src_rows,
								  const index_t* __restrict__ src_indptr, const index_t* __restrict__ src_perm,
								  int perm_size, const real_t* __restrict__ rates)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx >= perm_size)
		return;

	index_t diag = src_perm[idx];
	index_t src_begin = src_indptr[diag];
	index_t size = src_indptr[diag + 1] - src_begin;

	index_t dst_begin = dst_indptr[idx];

	real_t diag_sum = 0.f;
	index_t dst_diag_idx = dst_begin + size;

	for (int i = 0; i < size; i++)
	{
		index_t r = src_rows[src_begin + i];

		if (dst_diag_idx == size && r > diag)
		{
			dst_diag_idx = dst_begin + i;
			dst_begin++;
		}

		bool up = r > diag;
		index_t state = __ffs(r ^ diag) - 1;
		real_t rate = rates[2 * state + (up ? 0 : 1)];
		dst_data[dst_begin + i] = rate;
		diag_sum += rate;


		dst_rows[dst_begin + i] = r;
	}

	dst_rows[dst_diag_idx] = diag;
	dst_data[dst_diag_idx] = -diag_sum;
}

void run_scatter_rows_data(const index_t* dst_indptr, index_t* dst_rows, float* dst_data, const index_t* src_rows,
						   const index_t* src_indptr, const index_t* src_perm, int perm_size, const real_t* rates)
{
	int grid_size = (perm_size + block_size - 1) / block_size;

	scatter_rows_data<<<grid_size, block_size>>>(dst_indptr, dst_rows, dst_data, src_rows, src_indptr, src_perm,
												 perm_size, rates);
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

void run_hstack(const index_t* out_indptr, index_t* out_indices, float* out_data, const index_t* lhs_indptr,
				const index_t* rhs_indptr, const index_t* lhs_indices, const index_t* rhs_indices,
				const float* lhs_data, const float* rhs_data, int n)
{
	int grid_size = (2 * n + block_size - 1) / block_size;

	hstack<<<grid_size, block_size>>>(out_indptr, out_indices, out_data, lhs_indptr, rhs_indptr, lhs_indices,
									  rhs_indices, lhs_data, rhs_data, n);
}
