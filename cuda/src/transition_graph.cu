#include "cugraph/scc_matrix.cuh"
#include "transition_graph.h"
#include "utils.cuh"

transition_graph::transition_graph(d_idxvec csr_indices, d_idxvec csr_indptr, d_idxvec rows, d_idxvec cols)
	: indices_(std::move(csr_indices)), indptr_(std::move(csr_indptr)), rows_(std::move(rows)), cols_(std::move(cols)), vertices_count_(indptr_.size() - 1)
{}

void transition_graph::order_vertices()
{
	d_idxvec labels(vertices_count_);

	SCC_Data<char, int> sccd(vertices_count_, indptr_.data().get(), indices_.data().get());
	sccd.run_scc(labels.data().get());

	print("labels: ", labels);

	auto in_begin = thrust::make_zip_iterator(rows_.begin(), cols_.begin());
	auto in_end = thrust::make_zip_iterator(rows_.end(), cols_.end());

	thrust::transform_if(in_begin, in_end, )

}