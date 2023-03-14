#include <device_launch_parameters.h>

#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include "sga/scc.h"
#include "transition_graph.h"
#include "transition_table.h"
#include "utils.h"

constexpr size_t BIG_SCC_SIZE = 100'000;

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
		auto l = labels[indices[i]];
		if (l == 0 || l == current_label) // not labelled or labelled in this round
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

	// idx = (idx >= t_n) ? (scc_n - 1) - (idx - t_n) : idx;

	auto scc_idx = order[idx];
	auto begin = offsets[idx];

	for (int v_idx = 0; v_idx < v_n; v_idx++)
	{
		if (labels[v_idx] == scc_idx)
			reordered[begin++] = v_idx;
	}
}

transition_graph::transition_graph(cu_context& context, const d_idxvec& rows, const d_idxvec& cols,
								   const d_idxvec& indptr)
	: context_(context), indptr_(indptr), rows_(rows), cols_(cols)
{}

struct zip_non_equal_ftor : public thrust::unary_function<thrust::tuple<index_t, index_t>, bool>
{
	__host__ __device__ bool operator()(const thrust::tuple<index_t, index_t>& x) const
	{
		return thrust::get<0>(x) != thrust::get<1>(x);
	}
};

d_idxvec transition_graph::compute_sccs(const d_idxvec& indptr, const d_idxvec& indices)
{
	int n = indptr.size() - 1;
	int nnz = indices.size();

	auto& in_offsets = indptr;
	auto& in_indices = indices;

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

		int blocksize = 256;
		int gridsize = (n + blocksize - 1) / blocksize;
		topological_labelling<<<gridsize, blocksize>>>(n, indptr.data().get(), indices.data().get(),
													   labels.data().get(), curr_label, changed.data().get());

		CHECK_CUDA(cudaDeviceSynchronize());
	} while (changed.front() == true);

	// print("topo labels   ", labels);

	ordering = d_idxvec(thrust::make_counting_iterator(0), thrust::make_counting_iterator(n));
	thrust::sort_by_key(labels.begin(), labels.end(), thrust::make_zip_iterator(ordering.begin(), sizes.begin() + 1));

	// print("20 topo ordering ", ordering, 20);
	// print("20 topo sizes    ", sizes, 20);
}

void transition_graph::create_metagraph(const d_idxvec& rows, const d_idxvec& cols, const d_idxvec& labels,
										index_t sccs_count, d_idxvec& meta_indptr, d_idxvec& meta_indices)
{
	auto in_begin = thrust::make_zip_iterator(thrust::make_permutation_iterator(labels.begin(), cols.begin()),
											  thrust::make_permutation_iterator(labels.begin(), rows.begin()));

	auto in_end = thrust::make_zip_iterator(thrust::make_permutation_iterator(labels.begin(), cols.end()),
											thrust::make_permutation_iterator(labels.begin(), rows.end()));

	d_idxvec meta_cols(cols.size());
	d_idxvec meta_rows(cols.size());

	auto meta_end = thrust::copy_if(in_begin, in_end, thrust::make_zip_iterator(meta_cols.begin(), meta_rows.begin()),
									zip_non_equal_ftor());

	meta_cols.resize(thrust::get<0>(meta_end.get_iterator_tuple()) - meta_cols.begin());
	meta_rows.resize(meta_cols.size());

	meta_indices = std::move(meta_rows);

	transition_table::coo2csc(context_.cusparse_handle, sccs_count, meta_indices, meta_cols, meta_indptr);
}

void transition_graph::reorganize_all()
{
	index_t terminals_count;
	d_idxvec scc_offsets;
	reorganize_graph(indptr_, rows_, cols_, reordered_vertices_all, scc_offsets, terminals_count, false);

	terminals_offsets_all.resize(terminals_count + 1);

	thrust::copy(scc_offsets.begin(), scc_offsets.begin() + terminals_offsets_all.size(),
				 terminals_offsets_all.begin());
}

void transition_graph::reorganize_graph(const d_idxvec& indptr, const d_idxvec& rows, const d_idxvec& cols,
										d_idxvec& reordered_vertices, d_idxvec& scc_offsets, index_t& terminals_count,
										bool revert_all)
{
	auto vertices_count = indptr.size() - 1;
	auto edges_count = cols.size();

	std::cout << "vertices count " << vertices_count << std::endl;
	std::cout << "edges count " << edges_count << std::endl;

	auto labels = compute_sccs(indptr, rows);

	std::cout << "labeled " << std::endl;

	d_idxvec scc_ids_tmp = labels;
	thrust::sort(scc_ids_tmp.begin(), scc_ids_tmp.end());

	d_idxvec scc_ids(vertices_count), scc_sizes(vertices_count + 1, 0);
	auto scc_end =
		thrust::reduce_by_key(scc_ids_tmp.begin(), scc_ids_tmp.end(), thrust::make_constant_iterator<index_t>(1),
							  scc_ids.begin(), scc_sizes.begin() + 1);

	scc_sizes.resize(scc_end.second - scc_sizes.begin());
	scc_ids.resize(scc_sizes.size() - 1);

	// print("labels    ", labels);
	// print("scc_ids   ", scc_ids);
	// print("scc_sizes ", scc_sizes);

	auto sccs_count = scc_ids.size();

	// make labels start from 0
	{
		d_idxvec mapping(vertices_count);

		thrust::copy(thrust::make_counting_iterator<intptr_t>(0), thrust::make_counting_iterator<intptr_t>(sccs_count),
					 thrust::make_permutation_iterator(mapping.begin(), scc_ids.begin()));

		thrust::transform(labels.begin(), labels.end(), labels.begin(),
						  [map = mapping.data().get()] __device__(index_t x) { return map[x]; });
	}

	std::cout << "sccs size " << sccs_count << std::endl;

	if (sccs_count == 1)
	{
		reordered_vertices = d_idxvec(thrust::make_counting_iterator<index_t>(0),
									  thrust::make_counting_iterator<index_t>(vertices_count));

		terminals_count = 1;
		scc_offsets = std::move(scc_sizes);
		return;
	}

	d_idxvec meta_indptr, meta_indices;
	create_metagraph(rows, cols, labels, sccs_count, meta_indptr, meta_indices);

	d_idxvec meta_labels, meta_ordering;
	toposort(meta_indptr, meta_indices, scc_sizes, meta_labels, meta_ordering);

	// get terminals
	terminals_count = thrust::count(meta_labels.begin(), meta_labels.end(), 1);

	std::cout << "terminals count " << terminals_count << std::endl;

	// reorganize
	{
		scc_sizes[0] = 0;

		// reverse ordering + ordering sizes such that nonterminals are sorted ascending
		auto reverse_begin = revert_all ? 0 : terminals_count;
		thrust::reverse(scc_sizes.begin() + reverse_begin + 1, scc_sizes.end());
		thrust::reverse(meta_ordering.begin() + reverse_begin, meta_ordering.end());

		thrust::inclusive_scan(scc_sizes.begin(), scc_sizes.end(), scc_sizes.begin());

		// print("20 scc sizes ", scc_sizes, 20);

		scc_offsets = std::move(scc_sizes);

		reordered_vertices.resize(vertices_count);

		int blocksize = 256;
		int gridsize = (sccs_count + blocksize - 1) / blocksize;

		reorganize<<<gridsize, blocksize>>>(vertices_count, sccs_count, terminals_count, scc_offsets.data().get(),
											meta_ordering.data().get(), labels.data().get(),
											reordered_vertices.data().get());

		CHECK_CUDA(cudaDeviceSynchronize());
	}

	// print("20 reordered_vertices ", reordered_vertices, 20);
}

void transition_graph::take_coo_subset(const d_idxvec& rows, const d_idxvec& cols, index_t v_n, index_t subset_n,
									   const index_t* vertices, d_idxvec& subset_rows, d_idxvec& subset_cols)
{
	subset_rows.resize(rows.size());
	subset_cols.resize(rows.size());

	auto b = thrust::make_zip_iterator(rows.begin(), cols.begin());
	auto e = thrust::make_zip_iterator(rows.end(), cols.end());

	auto out_b = thrust::make_zip_iterator(subset_rows.begin(), subset_cols.begin());

	d_idxvec mask(v_n, 0);

	thrust::for_each_n(thrust::make_counting_iterator<index_t>(0), subset_n,
					   [vertices, mask = mask.data().get()] __device__(index_t x) { mask[vertices[x]] = 1; });

	auto out_e = thrust::copy_if(b, e, out_b, [mask = mask.data().get()] __device__(thrust::tuple<index_t, index_t> x) {
		return mask[thrust::get<0>(x)] || mask[thrust::get<1>(x)];
	});

	subset_rows.resize(thrust::get<0>(out_e.get_iterator_tuple()) - subset_rows.begin());
	subset_cols.resize(subset_rows.size());

	out_b = thrust::make_zip_iterator(subset_rows.begin(), subset_cols.begin());
	out_e = thrust::make_zip_iterator(subset_rows.end(), subset_cols.end());

	// now transform the vertices so they start from 0
	{
		thrust::for_each_n(thrust::make_counting_iterator<index_t>(0), subset_n,
						   [vertices, mask = mask.data().get()] __device__(index_t x) { mask[vertices[x]] = x; });
	}

	thrust::transform(out_b, out_e, out_b, [map = mask.data().get()] __device__(thrust::tuple<index_t, index_t> x) {
		return thrust::make_tuple(map[thrust::get<0>(x)], map[thrust::get<1>(x)]);
	});
}

void transition_graph::reorder_sccs(const d_idxvec& indptr, const d_idxvec& rows, const d_idxvec& cols,
									const d_idxvec& reordered_vertices, const thrust::host_vector<index_t>& scc_offsets)
{
	auto sccs_n = scc_offsets.size() - 1;

	for (int i = 0; i < sccs_n; i++)
	{
		auto scc_size = scc_offsets[i + 1] - scc_offsets[i];

		if (scc_size < BIG_SCC_SIZE)
			continue;

		std::cout << "REORDERING scc " << i << " with size " << scc_size << std::endl;

		d_idxvec scc_rows, scc_cols;

		// now we must take one vertex from the scc to break it down into (hopefully) multiple smaller sccs
		take_coo_subset(rows, cols, indptr.size() - 1, scc_size - 1, reordered_vertices.data().get() + scc_offsets[i],
						scc_rows, scc_cols);

		// now coo to csc
		d_idxvec scc_indptr;
		transition_table::coo2csc(context_.cusparse_handle, scc_size - 1, scc_rows, scc_cols, scc_indptr);
	}
}
