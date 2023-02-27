#pragma once

#include "types.h"

template <typename T1, typename T2>
struct zip_take_first_ftor : public thrust::unary_function<thrust::tuple<T1, T2>, T1>
{
	__host__ __device__ T1 operator()(const thrust::tuple<T1, T2>& x) const { return thrust::get<0>(x); }
};

class transition_graph
{
	const d_idxvec& rows_, &cols_, &indptr_;

	size_t vertices_count_;

public:
	d_idxvec terminals, labels;
	size_t sccs_count;

	transition_graph(const d_idxvec& rows, const d_idxvec& cols, const d_idxvec& indptr);

    void find_terminals();
};
