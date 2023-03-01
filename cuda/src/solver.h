#pragma once

#include "initial_state.h"
#include "transition_graph.h"
#include "transition_table.h"

class solver
{
	cu_context& context_;
	thrust::device_vector<float> initial_state_;

	d_idxvec labels_, terminals_;
	size_t sccs_count_;

	const d_idxvec &rows_, &cols_; // COO
	const d_idxvec& indptr_;	   // CSC


	float determinant(const d_idxvec& indptr, const d_idxvec& rows, const thrust::device_vector<float>& data, int n,
					  int nnz);

public:
	d_idxvec term_indptr, term_rows;
	thrust::device_vector<float> term_data;

	solver(cu_context& context, const transition_table& t, transition_graph g, initial_state s);

	void solve_terminal_part();

	void solve();
};
