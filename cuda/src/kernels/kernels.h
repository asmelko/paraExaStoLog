#include <device_launch_parameters.h>

#include "../types.h"

constexpr size_t block_size = 512;

void run_topological_labelling(index_t n, const index_t* indptr, const index_t* indices, index_t* labels,
							   index_t current_label, bool* changed);

void run_reorganize(index_t scc_n, const index_t* original_offsets, const index_t* new_offsets, const index_t* order,
					const index_t* scc_groups, index_t* reordered);

void run_scatter_rows_data(const index_t* dst_indptr, index_t* dst_rows, float* dst_data, const index_t* src_rows,
						   const index_t* src_indptr, const index_t* src_perm, int perm_size, const real_t* rates);

void run_hstack(const index_t* out_indptr, index_t* out_indices, float* out_data, const index_t* lhs_indptr,
				const index_t* rhs_indptr, const index_t* lhs_indices, const index_t* rhs_indices,
				const float* lhs_data, const float* rhs_data, int n);

void run_cuda_kernel_splu_symbolic_fact_triv_populate(const index_t sccs_rows, const index_t scc_count,
													  const index_t* scc_sizes, const index_t* scc_offsets,
													  const index_t* A_indptr, const index_t* A_indices,
													  const real_t* A_data, index_t* As_indptr, index_t* As_indices,
													  real_t* As_data);

void run_cuda_kernel_splu_symbolic_fact_triv_nnz(const index_t sccs_rows, const index_t scc_count,
												 const index_t* scc_sizes, const index_t* scc_offsets,
												 const index_t* A_indices, const index_t* A_indptr, index_t* As_nnz);
