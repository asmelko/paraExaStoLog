#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include "cugraph/scc_matrix.cuh"
#include "transition_graph.h"
#include "utils.h"

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

	std::cout << "vertices count " << vertices_count_ << std::endl;

	SCC_Data<char, int> sccd(vertices_count_, indptr_.data().get(), rows_.data().get());
	sccd.run_scc(labels.data().get());

	std::cout << "labeled " << std::endl;

	d_idxvec scc_ids = labels;
	thrust::sort(scc_ids.begin(), scc_ids.end());
	auto ids_end = thrust::unique(scc_ids.begin(), scc_ids.end());
	scc_ids.resize(ids_end - scc_ids.begin());

	sccs_count = scc_ids.size();

	std::cout << "sccs size " << sccs_count << std::endl;

	if (sccs_count == 1)
	{
		terminals = scc_ids;
		return;
	}

	auto in_begin = thrust::make_zip_iterator(thrust::make_permutation_iterator(labels.begin(), cols_.begin()),
											  thrust::make_permutation_iterator(labels.begin(), rows_.begin()));

	auto in_end = thrust::make_zip_iterator(thrust::make_permutation_iterator(labels.begin(), cols_.end()),
											thrust::make_permutation_iterator(labels.begin(), rows_.end()));

	d_idxvec meta_src_transitions(vertices_count_);

	auto meta_src_transitions_end = thrust::copy_if(
		in_begin, in_end,
		thrust::make_zip_iterator(meta_src_transitions.begin(), thrust::make_constant_iterator<index_t>(0)),
		zip_non_equal_ftor());

	std::cout << "copyif " << std::endl;

	meta_src_transitions.resize(thrust::get<0>(meta_src_transitions_end.get_iterator_tuple())
								- meta_src_transitions.begin());

	terminals = d_idxvec(meta_src_transitions.size());

	auto terminal_end = thrust::set_difference(scc_ids.begin(), scc_ids.end(), meta_src_transitions.begin(),
											   meta_src_transitions.end(), terminals.begin());

	std::cout << "diff " << std::endl;

	terminals.resize(terminal_end - terminals.begin());
}
