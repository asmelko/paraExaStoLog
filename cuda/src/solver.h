#pragma once

#include "initial_state.h"
#include "transition_graph.cuh"
#include "transition_table.h"

class solver
{
	thrust::device_vector<float> initial_state_;

	d_idxvec labels_, terminals_;
	size_t sccs_count_;

	const d_idxvec &rows_, &cols_; // COO
	const d_idxvec& idxptr_;	   // CSC


	void solve_terminal_part();

public:
	solver(const transition_table& t, transition_graph g, initial_state s);

	void solve();
};
