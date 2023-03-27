#pragma once

#include <thrust/host_vector.h>

#include "initial_state.h"
#include "sparse_utils.h"
#include "transition_graph.h"
#include "transition_table.h"

class solver
{
	cu_context& context_;
	thrust::device_vector<float> initial_state_;

	const d_idxvec &rows_, &cols_; // COO
	const d_idxvec& indptr_;	   // CSC

	d_idxvec ordered_vertices_;
	thrust::host_vector<index_t> terminals_offsets_;
	d_idxvec nonterminals_offsets_;

	d_idxvec submatrix_vertex_mapping_;

public:
	d_idxvec term_indptr, term_rows;
	thrust::device_vector<float> term_data;

	d_idxvec nonterm_indptr, nonterm_cols;
	thrust::device_vector<float> nonterm_data;

	thrust::device_vector<float> final_state;

	solver(cu_context& context, const transition_table& t, transition_graph g, initial_state s);

	void create_minor(d_idxvec& indptr, d_idxvec& rows, d_datvec& data, const index_t remove_vertex);

	float determinant(const d_idxvec& indptr, const d_idxvec& rows, const thrust::device_vector<float>& data, int n,
					  int nnz);

	void take_submatrix(index_t n, d_idxvec::const_iterator vertices_subset_begin, sparse_csc_matrix& m,
						bool mapping_prefilled = false);

	void solve_system(const d_idxvec& indptr, d_idxvec& rows, thrust::device_vector<float>& data, int n, int cols,
					  int nnz, const d_idxvec& b_indptr, const d_idxvec& b_indices,
					  const thrust::device_vector<float>& b_data, d_idxvec& x_indptr, d_idxvec& x_indices,
					  thrust::device_vector<float>& x_data);

	void solve_terminal_part();

	void solve_nonterminal_part();

	void compute_final_states();

	void solve();
};
