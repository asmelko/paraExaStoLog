#pragma once

#include "types.h"

class transition_graph
{
	const d_idxvec &rows_, &cols_, &indptr_;

	size_t vertices_count_;

public:
	d_idxvec terminals, labels;
	size_t sccs_count;

	transition_graph(const d_idxvec& rows, const d_idxvec& cols, const d_idxvec& indptr);

	void find_terminals();
};
