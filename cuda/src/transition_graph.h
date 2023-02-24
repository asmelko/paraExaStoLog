#pragma once

#include "types.h"

class transition_graph
{
	d_idxvec indices_, indptr_;

	size_t vertices_count_;

public:
	transition_graph(d_idxvec csr_indices, d_idxvec csr_indptr);

    void order_vertices();
};