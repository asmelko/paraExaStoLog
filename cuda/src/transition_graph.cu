#include <thrust/set_operations.h>
#include <thrust/sort.h>

#include "cugraph/scc_matrix.cuh"
#include "transition_graph.cuh"
#include "utils.cuh"

transition_graph::transition_graph(const d_idxvec& rows, const d_idxvec& cols, const d_idxvec& indptr)
	: indptr_(indptr), rows_(rows), cols_(cols), vertices_count_(indptr_.size() - 1)
{}

struct zip_non_equal_ftor : public thrust::unary_function<thrust::tuple<index_t, index_t>, bool>
{
	__host__ __device__ bool operator()(const thrust::tuple<index_t, index_t>& x) const
	{
		return thrust::get<0>(x) != thrust::get<1>(x);
	}
};

void transition_graph::find_terminals()
{
	labels = d_idxvec(vertices_count_);

	SCC_Data<char, int> sccd(vertices_count_, indptr_.data().get(), rows_.data().get());
	sccd.run_scc(labels.data().get());

	sccs_count = *thrust::max_element(labels.begin(), labels.end()) + 1;

	auto in_begin = thrust::make_zip_iterator(thrust::make_permutation_iterator(labels.begin(), cols_.begin()),
											  thrust::make_permutation_iterator(labels.begin(), rows_.begin()));

	auto in_end = thrust::make_zip_iterator(thrust::make_permutation_iterator(labels.begin(), cols_.end()),
											thrust::make_permutation_iterator(labels.begin(), rows_.end()));

	d_idxvec meta_src_transitions(vertices_count_);

	auto meta_src_transitions_end = thrust::transform_if(in_begin, in_end, meta_src_transitions.begin(),
														 zip_take_first_ftor<index_t, index_t>(), zip_non_equal_ftor());

	meta_src_transitions.resize(meta_src_transitions_end - meta_src_transitions.begin());

	terminals = d_idxvec(meta_src_transitions.size());

	auto terminal_end =
		thrust::set_difference(thrust::counting_iterator<index_t>(0), thrust::counting_iterator<index_t>(sccs_count),
							   meta_src_transitions.begin(), meta_src_transitions.end(), terminals.begin());

	terminals.resize(terminal_end - terminals.begin());
}
