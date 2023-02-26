#pragma once

#include "types.h"

class transition_graph
{
	d_idxvec rows_, cols_, indptr_;

	size_t vertices_count_;

public:
	transition_graph(d_idxvec rows, d_idxvec cols, d_idxvec indptr);

    void order_vertices();
};