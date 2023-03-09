#pragma once

#include <thrust/host_vector.h>

#include "cu_context.h"
#include "types.h"

class transition_graph
{
	cu_context& context_;

	const d_idxvec &rows_, &cols_, &indptr_;

	size_t vertices_count_, edges_count_;

	d_idxvec compute_sccs();

	void create_metagraph(d_idxvec& meta_indptr, d_idxvec& meta_indices);
	void toposort(const d_idxvec& indptr, const d_idxvec& indices, d_idxvec& labels, d_idxvec& ordering);

public:
	d_idxvec terminals, labels;
	size_t sccs_count;

	thrust::host_vector<index_t> terminals_offsets;
	d_idxvec reordered_vertices;

	transition_graph(cu_context& context, const d_idxvec& rows, const d_idxvec& cols, const d_idxvec& indptr);

	void find_terminals();
};
