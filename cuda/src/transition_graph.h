#pragma once

#include "types.h"

class transition_graph
{
	d_idxvec indices_, indptr_, rows_, cols_;

	size_t vertices_count_;

public:
	transition_graph(d_idxvec csr_indices, d_idxvec csr_indptr, d_idxvec rows, d_idxvec cols);

    void order_vertices();
};