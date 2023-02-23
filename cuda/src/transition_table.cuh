#pragma once

#include <thrust/device_vector.h>

#include "cu_context.cuh"
#include "model.h"


using d_idxvec = thrust::device_vector<index_t>;

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
