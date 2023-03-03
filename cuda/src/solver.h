#pragma once

#include <thrust/host_vector.h>

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

	d_idxvec sccs_;
	thrust::host_vector<index_t> sccs_offsets_;

	d_idxvec submatrix_vertex_mapping_;

	void matmul(index_t* lhs_indptr, index_t* lhs_indices, float* lhs_data, index_t lhs_rows, index_t lhs_cols,
				index_t lhs_nnz, index_t* rhs_indptr, index_t* rhs_indices, float* rhs_data, index_t rhs_rows,
				index_t rhs_cols, index_t rhs_nnz, d_idxvec& out_indptr, d_idxvec& out_indices,
				thrust::device_vector<float>& out_data);

	void csr_csc_switch(const index_t* in_indptr, const index_t* in_indices, const float* in_data, index_t in_n,
						index_t out_n, index_t nnz, d_idxvec& out_indptr, d_idxvec& out_indices,
						thrust::device_vector<float>& out_data);


public:
	d_idxvec term_indptr, term_rows;
	thrust::device_vector<float> term_data;

	d_idxvec nonterm_indptr, nonterm_rows;
	thrust::device_vector<float> nonterm_data;

	index_t take_submatrix(index_t n, d_idxvec::const_iterator vertices_subset_begin, d_idxvec& submatrix_indptr,
						   d_idxvec& submatrix_rows, thrust::device_vector<float>& submatrix_data,
						   bool mapping_prefilled = false);

	solver(cu_context& context, const transition_table& t, transition_graph g, initial_state s);

	void reorganize_terminal_sccs();

	void solve_terminal_part();

	void solve_nonterminal_part();

	void solve();
};
