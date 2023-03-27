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
		: indptr(std::move(indptr)),
		  indices(std::move(indices)),
		  data(std::move(data)),
		  w(w),
		  h(h)
	{}

	sparse_cs_matrix() : w(0), h(0) {}

	size_t nnz() { return indices.size(); }
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


void coo2csc(cusparseHandle_t handle, index_t n, d_idxvec& rows, d_idxvec& cols, d_idxvec& indptr);
sparse_csc_matrix coo2csc(cusparseHandle_t handle, sparse_coo_matrix& coo);
