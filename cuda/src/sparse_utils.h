#pragma once

#include "cu_context.h"
#include "types.h"

enum class cs_kind
{
	CSR,
	CSC
};

template <cs_kind k, typename idx_vec_t, typename dat_vec_t>
struct sparse_cs_matrix
{
	idx_vec_t indptr;
	idx_vec_t indices;
	dat_vec_t data;

	size_t w, h;

	sparse_cs_matrix(idx_vec_t indptr, idx_vec_t indices, dat_vec_t data)
		: indptr(std::move(indptr)),
		  indices(std::move(indices)),
		  data(std::move(data)),
		  w(this->indptr.size() - 1),
		  h(w)
	{}

	sparse_cs_matrix(idx_vec_t indptr, idx_vec_t indices, dat_vec_t data, size_t w, size_t h)
		: indptr(std::move(indptr)), indices(std::move(indices)), data(std::move(data)), w(w), h(h)
	{}

	sparse_cs_matrix() : w(0), h(0) {}

	size_t nnz() const { return indices.size(); }
	size_t n() const { return indptr.size() - 1; }
};

template <cs_kind out_k, cs_kind in_k, typename idx_vec_t, typename dat_vec_t>
sparse_cs_matrix<out_k, idx_vec_t, dat_vec_t> sparse_cast(sparse_cs_matrix<in_k, idx_vec_t, dat_vec_t>&& in)
{
	return sparse_cs_matrix<out_k, idx_vec_t, dat_vec_t>(std::move(in.indptr), std::move(in.indices),
														 std::move(in.data));
}

using sparse_csc_matrix = sparse_cs_matrix<cs_kind::CSC, d_idxvec, d_datvec>;
using sparse_csr_matrix = sparse_cs_matrix<cs_kind::CSR, d_idxvec, d_datvec>;

using host_sparse_csc_matrix = sparse_cs_matrix<cs_kind::CSC, h_idxvec, h_datvec>;
using host_sparse_csr_matrix = sparse_cs_matrix<cs_kind::CSR, h_idxvec, h_datvec>;

struct sparse_coo_matrix
{
	d_idxvec rows;
	d_idxvec cols;
	d_datvec data;

	size_t w, h, nnz;
};

// COO to CSC
void coo2csc(cusparseHandle_t handle, index_t n, d_idxvec& rows, d_idxvec& cols, d_idxvec& indptr);
sparse_csc_matrix coo2csc(cusparseHandle_t handle, sparse_coo_matrix& coo);

// Sparse matrix multiplication
sparse_csr_matrix matmul(cusparseHandle_t handle, index_t* lhs_indptr, index_t* lhs_indices, real_t* lhs_data,
						 index_t lhs_rows, index_t lhs_cols, index_t lhs_nnz, index_t* rhs_indptr, index_t* rhs_indices,
						 real_t* rhs_data, index_t rhs_rows, index_t rhs_cols, index_t rhs_nnz);
sparse_csr_matrix matmul(cusparseHandle_t handle, const sparse_csr_matrix& lhs, const sparse_csr_matrix& rhs);

// CSC=>CSR & CSR=>CSC
void transpose_sparse_matrix(cusparseHandle_t handle, const index_t* in_indptr, const index_t* in_indices,
							 const real_t* in_data, index_t in_n, index_t out_n, index_t nnz, d_idxvec& out_indptr,
							 d_idxvec& out_indices, d_datvec& out_data);
sparse_csc_matrix csr2csc(cusparseHandle_t handle, const sparse_csr_matrix& in, index_t in_n, index_t out_n,
						  index_t nnz);
sparse_csr_matrix csc2csr(cusparseHandle_t handle, const sparse_csc_matrix& in, index_t in_n, index_t out_n,
						  index_t nnz);

// Sparse matrix and dense vector multiplication
d_datvec mvmul(cusparseHandle_t handle, d_idxvec& indptr, d_idxvec& indices, d_datvec& data, cs_kind k, index_t rows,
			   index_t cols, d_datvec& x);

void host_lu(cusolverSpHandle_t handle, const host_sparse_csr_matrix& h, host_sparse_csr_matrix& l,
			 host_sparse_csr_matrix& u);

double host_det(cusolverSpHandle_t handle, const host_sparse_csr_matrix& h);

void create_minor(cusparseHandle_t handle, d_idxvec& indptr, d_idxvec& indices, d_datvec& data,
				  const index_t remove_vertex);

void sort_sparse_matrix(cusparseHandle_t handle, sparse_csr_matrix& N);

void dense_lu(cusolverDnHandle_t handle, d_datvec& A, index_t rows, index_t cols);
sparse_csr_matrix dense2sparse(cusparseHandle_t handle, d_datvec& mat_dn, index_t rows, index_t cols);
d_datvec sparse2dense(cusparseHandle_t handle, sparse_csr_matrix& M);
