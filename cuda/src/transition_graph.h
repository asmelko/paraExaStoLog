#pragma once

#include "types.h"

class transition_graph
{
	d_idxvec rows_, cols_, indptr_;

	size_t vertices_count_;

public:
	d_idxvec terminals, labels;
	size_t sccs_count;

	transition_graph(d_idxvec rows, d_idxvec cols, d_idxvec indptr);

    void find_terminals();
};
