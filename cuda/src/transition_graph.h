#pragma once

#include <thrust/host_vector.h>

#include "cu_context.h"
#include "types.h"

class transition_graph
{
	cu_context& context_;

	const d_idxvec &rows_, &cols_, &indptr_;

	d_idxvec compute_sccs(const d_idxvec& indptr, const d_idxvec& indices);

	void create_metagraph(const d_idxvec& rows, const d_idxvec& cols, const d_idxvec& labels, index_t sccs_count,
						  d_idxvec& meta_indptr, d_idxvec& meta_indices);
	void toposort(const d_idxvec& indptr, const d_idxvec& indices, d_idxvec& sizes, d_idxvec& labels,
				  d_idxvec& ordering);

public:
	d_idxvec reordered_vertices_all;
	thrust::host_vector<index_t> terminals_offsets_all;

	transition_graph(cu_context& context, const d_idxvec& rows, const d_idxvec& cols, const d_idxvec& indptr);

	void reorganize_graph(const d_idxvec& indptr, const d_idxvec& rows, const d_idxvec& cols,
						  d_idxvec& reordered_vertices, d_idxvec& scc_offsets, index_t& terminals_count);

	void reorganize_all();

	void reorder_sccs(const d_idxvec& indptr, const d_idxvec& rows, const d_idxvec& cols,
					  const d_idxvec& reordered_vertices, const thrust::host_vector<index_t>& scc_offsets);
	void take_coo_subset(const d_idxvec& rows, const d_idxvec& cols, index_t v_n, index_t subset_n,
						 const index_t* vertices, d_idxvec& subset_rows, d_idxvec& subset_cols);
};
