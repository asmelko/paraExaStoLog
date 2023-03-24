#pragma once

#include <thrust/host_vector.h>

#include "initial_state.h"
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

	float determinant(const d_idxvec& indptr, const d_idxvec& rows, const thrust::device_vector<float>& data, int n,
					  int nnz);


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

	d_idxvec nonterm_indptr, nonterm_cols;
	thrust::device_vector<float> nonterm_data;

	thrust::device_vector<float> final_state;

	static void transpose_sparse_matrix(cusparseHandle_t handle, const index_t* in_indptr, const index_t* in_indices,
										const float* in_data, index_t in_n, index_t out_n, index_t nnz,
										d_idxvec& out_indptr, d_idxvec& out_indices,
										thrust::device_vector<float>& out_data);

	index_t take_submatrix(index_t n, d_idxvec::const_iterator vertices_subset_begin, d_idxvec& submatrix_indptr,
						   d_idxvec& submatrix_rows, thrust::device_vector<float>& submatrix_data,
						   bool mapping_prefilled = false);

	solver(cu_context& context, const transition_table& t, transition_graph g, initial_state s);

	void solve_system(const d_idxvec& indptr, d_idxvec& rows, thrust::device_vector<float>& data, int n, int cols,
					  int nnz, const d_idxvec& b_indptr, const d_idxvec& b_indices,
					  const thrust::device_vector<float>& b_data, d_idxvec& x_indptr, d_idxvec& x_indices,
					  thrust::device_vector<float>& x_data);

	void reorganize_terminal_sccs();

	void solve_terminal_part();

	void solve_nonterminal_part();

	void compute_final_states();

	void solve();
};
