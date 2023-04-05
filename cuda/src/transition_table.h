#pragma once

#include "cu_context.h"
#include "model.h"
#include "types.h"

class transition_table
{
	cu_context& context_;
	const model_t& model_;
	h_datvec rates_;

	std::pair<d_idxvec, d_idxvec> generate_transitions(const std::vector<clause_t>& clauses, index_t variable_idx);

	std::pair<d_idxvec, d_idxvec> compute_rows_and_cols();

public:
	d_idxvec indptr;	 // CSC arrays
	d_idxvec rows, cols; // COO arrays

	transition_table(cu_context& context, const model_t& model, const d_datvec& transition_rates);

	static d_idxvec construct_transition_vector(const std::vector<index_t>& free_nodes, size_t fixed_val);

	void construct_table();
};
