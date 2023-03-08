#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include "sga/scc.h"
#include "transition_graph.h"
#include "utils.h"

transition_graph::transition_graph(cu_context& context, const d_idxvec& rows, const d_idxvec& cols,
								   const d_idxvec& indptr)
	: context_(context),
	  indptr_(indptr),
	  rows_(rows),
	  cols_(cols),
	  vertices_count_(indptr_.size() - 1),
	  edges_count_(rows.size())
{}

struct zip_non_equal_ftor : public thrust::unary_function<thrust::tuple<index_t, index_t>, bool>
{
	__host__ __device__ bool operator()(const thrust::tuple<index_t, index_t>& x) const
	{
		return thrust::get<0>(x) != thrust::get<1>(x);
	}
};

d_idxvec transition_graph::compute_sccs()
{
	int n = indptr_.size() - 1;
	int nnz = rows_.size();

	auto& in_offsets = indptr_;
	auto& in_indices = rows_;

	d_idxvec out_offset;
	d_idxvec out_indices;
	{
		out_offset.resize(in_offsets.size());
		out_indices.resize(in_indices.size());

		size_t buffersize;
		CHECK_CUSPARSE(cusparseCsr2cscEx2_bufferSize(
			context_.cusparse_handle, n, n, nnz, nullptr, in_offsets.data().get(), in_indices.data().get(), nullptr,
			out_offset.data().get(), out_indices.data().get(), CUDA_R_32F, CUSPARSE_ACTION_SYMBOLIC,
			CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, &buffersize));

		thrust::device_vector<float> dummy(nnz);

		thrust::device_vector<char> buffer(buffersize);
		CHECK_CUSPARSE(cusparseCsr2cscEx2(
			context_.cusparse_handle, n, n, nnz, dummy.data().get(), in_offsets.data().get(), in_indices.data().get(),
			dummy.data().get(), out_offset.data().get(), out_indices.data().get(), CUDA_R_32F, CUSPARSE_ACTION_SYMBOLIC,
			CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, buffer.data().get()));
	}

	//print("in_offset ", in_offsets);
	//print("in_indice ", in_indices);
	//print("out_offset ", out_offset);
	//print("out_indice ", out_indices);

	thrust::host_vector<index_t> hin_offsets = in_offsets;
	thrust::host_vector<index_t> hin_indices = in_indices;
	thrust::host_vector<index_t> hout_offsets = out_offset;
	thrust::host_vector<index_t> hout_indices = out_indices;
	thrust::host_vector<index_t> h_labels(n);

	SCCSolver(n, nnz, hin_offsets.data(), hin_indices.data(), hout_offsets.data(), hout_indices.data(),
			  h_labels.data());

	return h_labels;
}

void transition_graph::find_terminals()
{
	std::cout << "vertices count " << vertices_count_ << std::endl;
	std::cout << "edges count " << edges_count_ << std::endl;

	labels = compute_sccs();

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

	d_idxvec meta_src_transitions(edges_count_);

	auto meta_src_transitions_end = thrust::copy_if(
		in_begin, in_end,
		thrust::make_zip_iterator(meta_src_transitions.begin(), thrust::make_constant_iterator<index_t>(0)),
		zip_non_equal_ftor());


	meta_src_transitions.resize(thrust::get<0>(meta_src_transitions_end.get_iterator_tuple())
								- meta_src_transitions.begin());

	thrust::sort(meta_src_transitions.begin(), meta_src_transitions.end());

	std::cout << "meta_src_transitions " << meta_src_transitions.size() << std::endl;

	terminals = d_idxvec(scc_ids.size());

	auto terminal_end = thrust::set_difference(scc_ids.begin(), scc_ids.end(), meta_src_transitions.begin(),
											   meta_src_transitions.end(), terminals.begin());

	std::cout << "diff " << std::endl;

	terminals.resize(terminal_end - terminals.begin());
}
