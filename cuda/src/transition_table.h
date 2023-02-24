#pragma once

#include "cu_context.cuh"
#include "model.h"
#include "types.h"

class transition_table
{
	cu_context& context_;
	model_t model_;

	std::pair<d_idxvec, d_idxvec> compute_rows_and_cols();

public:
	d_idxvec indptr, indices; // CSR arrays

	transition_table(cu_context& context, model_t model);

	void construct_table();
};
