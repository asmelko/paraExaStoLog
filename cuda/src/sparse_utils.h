#pragma once
#include "cu_context.h"
#include "types.h"

enum class cs_kind
{
	CSR,
	CSC
};

template <cs_kind k>
struct sparse_cs_matrix
{
	d_idxvec indptr;
	d_idxvec indices;
	d_datvec data;

	size_t w, h;

	sparse_cs_matrix(d_idxvec indptr, d_idxvec indices, d_datvec data)
		: indptr(std::move(indptr)),
		  indices(std::move(indices)),
		  data(std::move(data)),
		  w(this->indptr.size() - 1),
		  h(w)
	{}

	sparse_cs_matrix(d_idxvec indptr, d_idxvec indices, d_datvec data, size_t w, size_t h)
		: indptr(std::move(indptr)), indices(std::move(indices)), data(std::move(data)), w(w), h(h)
	{}

	sparse_cs_matrix() : w(0), h(0) {}

	size_t nnz() const { return indices.size(); }
};

using sparse_csc_matrix = sparse_cs_matrix<cs_kind::CSC>;
using sparse_csr_matrix = sparse_cs_matrix<cs_kind::CSR>;

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
sparse_csr_matrix matmul(cusparseHandle_t handle, index_t* lhs_indptr, index_t* lhs_indices, float* lhs_data,
						 index_t lhs_rows, index_t lhs_cols, index_t lhs_nnz, index_t* rhs_indptr, index_t* rhs_indices,
						 float* rhs_data, index_t rhs_rows, index_t rhs_cols, index_t rhs_nnz);
sparse_csr_matrix matmul(cusparseHandle_t handle, const sparse_csr_matrix& lhs, const sparse_csr_matrix& rhs);

// CSC=>CSR & CSR=>CSC
void transpose_sparse_matrix(cusparseHandle_t handle, const index_t* in_indptr, const index_t* in_indices,
							 const float* in_data, index_t in_n, index_t out_n, index_t nnz, d_idxvec& out_indptr,
							 d_idxvec& out_indices, thrust::device_vector<float>& out_data);
sparse_csc_matrix csr2csc(cusparseHandle_t handle, const sparse_csr_matrix& in, index_t in_n, index_t out_n,
						  index_t nnz);
sparse_csr_matrix csc2csr(cusparseHandle_t handle, const sparse_csc_matrix& in, index_t in_n, index_t out_n,
						  index_t nnz);

// Sparse matrix and dense vector multiplication
d_datvec mvmul(cusparseHandle_t handle, const d_idxvec& indptr, const d_idxvec& indices, const d_datvec& data,
			   cs_kind k, index_t rows, index_t cols, const d_datvec& x);