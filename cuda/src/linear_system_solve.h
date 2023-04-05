#pragma once

#include "sparse_utils.h"

sparse_csr_matrix solve_system(cu_context& context, sparse_csr_matrix&& A, const d_idxvec& scc_offsets,
							   const sparse_csr_matrix& B, host_sparse_csr_matrix& M, bool refactor);
