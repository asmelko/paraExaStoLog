#pragma once

#include "sparse_utils.h"

struct persistent_data
{
	// transition_table
	d_idxvec rows, cols;
	d_idxvec indptr;

	// transition_graph
	d_idxvec ordered_vertices;
	h_idxvec terminals_offsets;
	d_idxvec nonterminals_offsets;

	// transition_rates
	d_datvec rates;

	// solver
	sparse_csc_matrix solution_term;
	sparse_csr_matrix solution_nonterm;
	host_sparse_csr_matrix n_inverse;
};

class solver;

class persistent_solution
{
public:
	static void serialize(const std::string& file, const solver& s, bool no_inverse);
	static persistent_data deserialize(const std::string& file, bool no_inverse);

	static bool has_compatible_zero_rates(const persistent_data& stored, const d_datvec& new_rates);
	static bool are_same(const persistent_data& stored, const d_datvec& new_rates);

	static bool check_are_equal(const solver& s, const persistent_data& d);
};
