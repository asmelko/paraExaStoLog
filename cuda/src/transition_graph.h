#pragma once

#include "types.h"

#include "cu_context.h"

class transition_graph
{
	cu_context& context_;

	const d_idxvec &rows_, &cols_, &indptr_;

	size_t vertices_count_, edges_count_;

	d_idxvec compute_sccs();

public:
	d_idxvec terminals, nonterminals, labels;
	size_t sccs_count;

	transition_graph(cu_context& context, const d_idxvec& rows, const d_idxvec& cols, const d_idxvec& indptr);

	void find_terminals();
};
