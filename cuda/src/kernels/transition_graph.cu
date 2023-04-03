#include "kernels.h"

__global__ void topological_labelling(index_t n, const index_t* __restrict__ indptr,
									  const index_t* __restrict__ indices, index_t* __restrict__ labels,
									  index_t current_label, bool* __restrict__ changed)
{
	auto idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx >= n)
		return;

	if (labels[idx] != 0) // already labeled
		return;

	auto begin = indptr[idx];
	auto end = indptr[idx + 1];

	for (auto i = begin; i < end; i++)
	{
		auto l = labels[indices[i]];
		if (l == 0 || l == current_label) // not labelled or labelled in this round
			return;
	}

	labels[idx] = current_label;
	changed[0] = true;
}

void run_topological_labelling(index_t n, const index_t* indptr, const index_t* indices, index_t* labels,
							   index_t current_label, bool* changed)
{
	int grid_size = (n + block_size - 1) / block_size;
	topological_labelling<<<grid_size, block_size>>>(n, indptr, indices, labels, current_label, changed);
}

__global__ void reorganize(index_t scc_n, const index_t* __restrict__ original_offsets,
						   const index_t* __restrict__ new_offsets, const index_t* __restrict__ order,
						   const index_t* __restrict__ scc_groups, index_t* __restrict__ reordered)
{
	auto idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx >= scc_n)
		return;

	auto scc_idx = order[idx];
	auto out_begin = new_offsets[idx];
	auto in_begin = original_offsets[scc_idx];
	auto in_end = original_offsets[scc_idx + 1];

	for (int i = in_begin; i < in_end; i++)
		reordered[out_begin + i - in_begin] = scc_groups[i];
}

void run_reorganize(index_t scc_n, const index_t* original_offsets, const index_t* new_offsets, const index_t* order,
					const index_t* scc_groups, index_t* reordered)
{
	int grid_size = (scc_n + block_size - 1) / block_size;

	reorganize<<<grid_size, block_size>>>(scc_n, original_offsets, new_offsets, order, scc_groups, reordered);
}
