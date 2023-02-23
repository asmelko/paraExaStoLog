#pragma once

#include <thrust/device_vector.h>

#include "model.h"

using d_idxvec = thrust::device_vector<index_t>;

class transition_table
{
	model_t model_;

public:
	transition_table(model_t model);

	void compute_rows_and_cols();
};