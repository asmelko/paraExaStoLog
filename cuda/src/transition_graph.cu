#include "cugraph/scc_matrix.cuh"
#include "transition_graph.h"
#include "utils.cuh"

transition_graph::transition_graph(d_idxvec csr_indices, d_idxvec csr_indptr)
	: indices_(std::move(csr_indices)), indptr_(std::move(csr_indptr)), vertices_count_(indptr_.size() - 1)
{}

void transition_graph::order_vertices()
{
	d_idxvec labels(indptr_.size() - 1);

	SCC_Data<char, int> sccd(indptr_.size() - 1, indptr_.data().get(), indices_.data().get());
	auto scc_count = sccd.run_scc(labels.data().get());

	std::cout << scc_count << std::endl;
	print("labels: ", labels);
}