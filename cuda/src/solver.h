#pragma once

#include "initial_state.h"
#include "sparse_utils.h"
#include "transition_graph.h"
#include "transition_rates.h"
#include "transition_table.h"

class solver
{
	cu_context& context_;

	// initial_state
	d_datvec initial_state_;

	// transition_table
	const d_idxvec &rows_, &cols_; // COO
	const d_idxvec& indptr_;	   // CSC

	// transition_graph
	const d_idxvec ordered_vertices_;
	const h_idxvec terminals_offsets_;
	const d_idxvec nonterminals_offsets_;

	// transition_rates
	const d_datvec rates_;

	d_idxvec submatrix_vertex_mapping_;

public:
	sparse_csc_matrix solution_term;
	sparse_csr_matrix solution_nonterm;

	d_datvec final_state;

	solver(cu_context& context, const transition_table& t, transition_graph g, transition_rates r, initial_state s);

	void break_NB(sparse_csc_matrix&& NB, sparse_csc_matrix& N, sparse_csc_matrix& B);

	void take_submatrix(index_t n, d_idxvec::const_iterator vertices_subset_begin, sparse_csc_matrix& m,
						bool mapping_prefilled = false);

	void solve_terminal_part();

	void solve_nonterminal_part();

	void compute_final_states();

	void solve();
};
