#include <device_launch_parameters.h>

#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include "sga/scc.h"
#include "transition_graph.h"
#include "transition_table.h"
#include "utils.h"

__global__ void topological_labelling(index_t n, const index_t* __restrict__ indptr,
									  const index_t* __restrict__ indices, index_t* __restrict__ labels,
									  index_t current_label, bool* __restrict__ changed)
{
	auto idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx >= n)
		return;

	if (labels[idx] != 0) // already labeled
		return;

	auto begin = indptr[idx];
	auto end = indptr[idx + 1];

	for (auto i = begin; i < end; i++)
	{
		if (labels[indices[begin]] == 0) // not labelled
			return;
	}

	labels[idx] = current_label;
	changed[0] = true;
}

__global__ void reorganize(index_t v_n, index_t scc_n, index_t t_n, const index_t* __restrict__ offsets,
						   const index_t* __restrict__ order, const index_t* __restrict__ labels,
						   index_t* __restrict__ reordered)
{
	auto idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx >= scc_n)
		return;

	idx = (idx >= t_n) ? (scc_n - 1) - (idx - t_n) : idx;

	auto begin = offsets[idx];
	auto scc_idx = order[idx];

	for (int v_idx = 0; v_idx < v_n; v_idx++)
	{
		if (labels[v_idx] == scc_idx)
			reordered[begin++] = v_idx;
	}
}

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

	// print("in_offset ", in_offsets);
	// print("in_indice ", in_indices);
	// print("out_offset ", out_offset);
	// print("out_indice ", out_indices);

	thrust::host_vector<index_t> hin_offsets = in_offsets;
	thrust::host_vector<index_t> hin_indices = in_indices;
	thrust::host_vector<index_t> hout_offsets = out_offset;
	thrust::host_vector<index_t> hout_indices = out_indices;
	thrust::host_vector<index_t> h_labels(n);

	SCCSolver(n, nnz, hin_offsets.data(), hin_indices.data(), hout_offsets.data(), hout_indices.data(),
			  h_labels.data());

	return h_labels;
}

void transition_graph::toposort(const d_idxvec& indptr, const d_idxvec& indices, d_idxvec& labels, d_idxvec& ordering)
{
	index_t n = indptr.size() - 1;
	labels = d_idxvec(n, 0);

	thrust::device_vector<bool> changed(1, false);

	index_t curr_label = 0;
	do
	{
		++curr_label;
		int blocksize = 256;
		int gridsize = (n + blocksize - 1) / blocksize;
		topological_labelling<<<gridsize, blocksize>>>(n, indptr.data().get(), indices.data().get(),
													   labels.data().get(), curr_label, changed.data().get());

		CHECK_CUDA(cudaDeviceSynchronize());
	} while (changed.front() == true);

	ordering = d_idxvec(thrust::make_counting_iterator(0), thrust::make_counting_iterator(n));
	thrust::sort_by_key(labels.begin(), labels.end(), ordering.begin());
}

void transition_graph::create_metagraph(d_idxvec& meta_indptr, d_idxvec& meta_indices)
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

	meta_indices = std::move(meta_cols);

	transition_table::coo2csc(context_.cusparse_handle, sccs_count, meta_rows, meta_indices, meta_indptr);
}

void transition_graph::find_terminals()
{
	std::cout << "vertices count " << vertices_count_ << std::endl;
	std::cout << "edges count " << edges_count_ << std::endl;

	labels = compute_sccs();

	std::cout << "labeled " << std::endl;

	d_idxvec scc_ids_tmp = labels;
	thrust::sort(scc_ids_tmp.begin(), scc_ids_tmp.end());

	d_idxvec scc_ids(vertices_count_), scc_sizes(vertices_count_ + 1);
	auto scc_end = thrust::reduce_by_key(scc_ids.begin(), scc_ids.end(), thrust::make_constant_iterator<index_t>(1),
										 scc_ids.begin(), scc_sizes.begin() + 1);

	scc_sizes.resize(scc_end.second - scc_sizes.begin());
	scc_ids.resize(scc_sizes.size() - 1);

	sccs_count = scc_ids.size();

	// make labels start from 0
	{
		d_idxvec mapping(vertices_count_);

		thrust::copy(thrust::make_counting_iterator<intptr_t>(0), thrust::make_counting_iterator<intptr_t>(sccs_count),
					 thrust::make_permutation_iterator(mapping.begin(), scc_ids.begin()));

		thrust::transform(labels.begin(), labels.end(), labels.begin(),
						  [map = mapping.data().get()] __device__(index_t x) { return map[x]; });
	}

	std::cout << "sccs size " << sccs_count << std::endl;

	if (sccs_count == 1)
	{
		terminals = d_idxvec(1, 0);
		reordered_vertices = d_idxvec(thrust::make_counting_iterator<index_t>(0),
									  thrust::make_counting_iterator<index_t>(vertices_count_));

		terminals_offsets.resize(2);
		terminals_offsets[0] = 0;
		terminals_offsets[1] = vertices_count_;
		return;
	}

	d_idxvec meta_indptr, meta_indices;
	create_metagraph(meta_indptr, meta_indices);

	d_idxvec meta_labels, meta_ordering;
	toposort(meta_indptr, meta_indices, meta_labels, meta_ordering);


	// get terminals
	{
		auto terminals_count = thrust::count(meta_labels.begin(), meta_labels.end(), 1);
		terminals.resize(terminals_count);
		thrust::copy(meta_ordering.begin(), meta_ordering.begin() + terminals_count, terminals.begin());

		terminals_offsets.resize(terminals_count + 1);
		terminals_offsets[0] = 0;

		thrust::copy(thrust::make_permutation_iterator(scc_sizes.begin() + 1, terminals.begin()),
					 thrust::make_permutation_iterator(scc_sizes.begin() + 1, terminals.end()),
					 terminals_offsets.begin());

		thrust::inclusive_scan(terminals_offsets.begin(), terminals_offsets.end(), terminals_offsets.begin());
	}

	// reorganize
	{
		scc_sizes[0] = 0;
		thrust::inclusive_scan(scc_sizes.begin(), scc_sizes.end(), scc_sizes.begin());

		reordered_vertices.resize(vertices_count_);

		int blocksize = 256;
		int gridsize = (sccs_count + blocksize - 1) / blocksize;

		reorganize<<<blocksize, gridsize>>>(vertices_count_, sccs_count, terminals_offsets.size() - 1,
											scc_sizes.data().get(), meta_ordering.data().get(), labels.data().get(),
											reordered_vertices.data().get());
	}
}
