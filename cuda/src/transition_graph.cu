#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include "kernels/kernels.h"
#include "sga/scc.h"
#include "sparse_utils.h"
#include "transition_graph.h"
#include "utils.h"

struct zip_non_equal_ftor : public thrust::unary_function<thrust::tuple<index_t, index_t>, bool>
{
	__host__ __device__ bool operator()(const thrust::tuple<index_t, index_t>& x) const
	{
		return thrust::get<0>(x) != thrust::get<1>(x);
	}
};

struct map_ftor : public thrust::unary_function<index_t, index_t>
{
	const index_t* __restrict__ map;

	map_ftor(const index_t* map) : map(map) {}

	__host__ __device__ index_t operator()(index_t x) const { return map[x]; }
};

transition_graph::transition_graph(cu_context& context, const d_idxvec& rows, const d_idxvec& cols,
								   const d_idxvec& indptr)
	: context_(context),
	  indptr_(indptr),
	  rows_(rows),
	  cols_(cols),
	  vertices_count_(indptr_.size() - 1),
	  edges_count_(rows.size())
{}

d_idxvec transition_graph::compute_sccs()
{
	int n = indptr_.size() - 1;
	int nnz = rows_.size();

	auto& in_offsets = indptr_;
	auto& in_indices = rows_;

	d_idxvec out_offsets;
	d_idxvec out_indices;
	{
		out_offsets.resize(in_offsets.size());
		out_indices.resize(in_indices.size());

		size_t buffersize;
		CHECK_CUSPARSE(cusparseCsr2cscEx2_bufferSize(
			context_.cusparse_handle, n, n, nnz, nullptr, in_offsets.data().get(), in_indices.data().get(), nullptr,
			out_offsets.data().get(), out_indices.data().get(), CUDA_R_32F, CUSPARSE_ACTION_SYMBOLIC,
			CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, &buffersize));

		d_datvec dummy(nnz);

		thrust::device_vector<char> buffer(buffersize);
		CHECK_CUSPARSE(cusparseCsr2cscEx2(
			context_.cusparse_handle, n, n, nnz, dummy.data().get(), in_offsets.data().get(), in_indices.data().get(),
			dummy.data().get(), out_offsets.data().get(), out_indices.data().get(), CUDA_R_32F,
			CUSPARSE_ACTION_SYMBOLIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, buffer.data().get()));
	}

	d_idxvec labels(n);

	SCCSolver(n, nnz, in_offsets.data().get(), in_indices.data().get(), out_offsets.data().get(),
			  out_indices.data().get(), labels.data().get());

	return labels;
}

void transition_graph::toposort(const d_idxvec& indptr, const d_idxvec& indices, d_idxvec& sizes, d_idxvec& labels,
								d_idxvec& ordering)
{
	index_t n = indptr.size() - 1;
	labels = d_idxvec(n, 0);

	thrust::device_vector<bool> changed(1);

	index_t curr_label = 0;
	do
	{
		changed[0] = false;
		++curr_label;

		run_topological_labelling(n, indptr.data().get(), indices.data().get(), labels.data().get(), curr_label,
								  changed.data().get());

		CHECK_CUDA(cudaDeviceSynchronize());
	} while (changed.front() == true);

	ordering = d_idxvec(thrust::make_counting_iterator(0), thrust::make_counting_iterator(n));
	thrust::sort_by_key(labels.begin(), labels.end(), thrust::make_zip_iterator(ordering.begin(), sizes.begin() + 1));
}

void transition_graph::create_metagraph(const d_idxvec& labels, index_t sccs_count, d_idxvec& meta_indptr,
										d_idxvec& meta_indices)
{
	auto in_begin = thrust::make_zip_iterator(thrust::make_permutation_iterator(labels.begin(), cols_.begin()),
											  thrust::make_permutation_iterator(labels.begin(), rows_.begin()));

	auto in_end = thrust::make_zip_iterator(thrust::make_permutation_iterator(labels.begin(), cols_.end()),
											thrust::make_permutation_iterator(labels.begin(), rows_.end()));

	d_idxvec meta_cols(edges_count_);
	d_idxvec meta_rows(edges_count_);

	auto meta_end = thrust::copy_if(in_begin, in_end, thrust::make_zip_iterator(meta_cols.begin(), meta_rows.begin()),
									zip_non_equal_ftor());

	meta_cols.resize(thrust::get<0>(meta_end.get_iterator_tuple()) - meta_cols.begin());
	meta_rows.resize(meta_cols.size());

	meta_indices = std::move(meta_rows);

	coo2csc(context_.cusparse_handle, sccs_count, meta_indices, meta_cols, meta_indptr);
}

void transition_graph::find_terminals()
{
	auto labels = compute_sccs();

	d_idxvec scc_ids_tmp = labels;
	d_idxvec vertices_ordered_by_scc(thrust::make_counting_iterator<index_t>(0),
									 thrust::make_counting_iterator<index_t>(vertices_count_));
	thrust::sort_by_key(scc_ids_tmp.begin(), scc_ids_tmp.end(), vertices_ordered_by_scc.begin());

	d_idxvec scc_ids(vertices_count_), scc_sizes(vertices_count_ + 1);
	auto scc_end =
		thrust::reduce_by_key(scc_ids_tmp.begin(), scc_ids_tmp.end(), thrust::make_constant_iterator<index_t>(1),
							  scc_ids.begin(), scc_sizes.begin() + 1);

	scc_sizes.resize(scc_end.second - scc_sizes.begin());
	scc_ids.resize(scc_sizes.size() - 1);

	d_idxvec original_sccs_offsets(scc_sizes.size());
	scc_sizes[0] = 0;
	thrust::inclusive_scan(scc_sizes.begin(), scc_sizes.end(), original_sccs_offsets.begin());

	auto sccs_count = scc_ids.size();

	// make labels start from 0
	{
		d_idxvec mapping(vertices_count_);

		thrust::copy(thrust::make_counting_iterator<intptr_t>(0), thrust::make_counting_iterator<intptr_t>(sccs_count),
					 thrust::make_permutation_iterator(mapping.begin(), scc_ids.begin()));

		thrust::transform(labels.begin(), labels.end(), labels.begin(), map_ftor(mapping.data().get()));
	}

	if (sccs_count == 1)
	{
		reordered_vertices = d_idxvec(thrust::make_counting_iterator<index_t>(0),
									  thrust::make_counting_iterator<index_t>(vertices_count_));

		terminals_count = 1;
		sccs_offsets = std::move(scc_sizes);
		return;
	}

	d_idxvec meta_indptr, meta_indices;
	create_metagraph(labels, sccs_count, meta_indptr, meta_indices);

	d_idxvec meta_labels, meta_ordering;
	toposort(meta_indptr, meta_indices, scc_sizes, meta_labels, meta_ordering);

	// get terminals
	{
		terminals_count = thrust::count(meta_labels.begin(), meta_labels.end(), 1);
	}

	// reorganize
	{
		// reverse ordering + ordering sizes such that nonterminals are sorted ascending
		thrust::reverse(scc_sizes.begin() + terminals_count + 1, scc_sizes.end());
		thrust::reverse(meta_ordering.begin() + terminals_count, meta_ordering.end());

		thrust::inclusive_scan(scc_sizes.begin(), scc_sizes.end(), scc_sizes.begin());

		sccs_offsets = std::move(scc_sizes);

		reordered_vertices.resize(vertices_count_);

		run_reorganize(sccs_count, original_sccs_offsets.data().get(), sccs_offsets.data().get(),
					   meta_ordering.data().get(), vertices_ordered_by_scc.data().get(),
					   reordered_vertices.data().get());

		CHECK_CUDA(cudaDeviceSynchronize());
	}
}
