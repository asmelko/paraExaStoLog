#include <thrust/host_vector.h>
#include <thrust/set_operations.h>

#include "transition_table.h"

#include "utils.h"

struct transition_ftor : public thrust::unary_function<index_t, index_t>
{
	index_t free_vars[32] = { 0 };
	index_t fixed = 0;

	transition_ftor(const std::vector<index_t>& free_v, index_t fixed)
	{
		size_t i = 0;
		for (; i < free_v.size(); i++)
			free_vars[i] = 1 << free_v[i];

		for (; i < 32; i++)
			free_vars[i] = 0;

		this->fixed = fixed;
	}
	__host__ __device__ index_t operator()(index_t x) const
	{
		index_t ret = fixed;
		for (int i = 0; i < 32; i++)
			ret += (x >> i) % 2 ? free_vars[i] : 0;

		return ret;
	}
};

d_idxvec transition_table::construct_transition_vector(const std::vector<index_t>& free_nodes, size_t fixed_val)
{
	auto c_b = thrust::make_counting_iterator(0);
	auto c_e = c_b + (1ULL << free_nodes.size());

	auto b = thrust::make_transform_iterator(c_b, transition_ftor(free_nodes, fixed_val));
	auto e = thrust::make_transform_iterator(c_e, transition_ftor(free_nodes, fixed_val));

	return d_idxvec(b, e);
}

d_idxvec generate_transitions(const std::vector<clause_t>& clauses)
{
	d_idxvec transitions;

	for (const auto& c : clauses)
	{
		auto free_vars = c.get_free_variables();
		auto fixed = c.get_fixed_part();

		d_idxvec single_clause_transitions = transition_table::construct_transition_vector(free_vars, fixed);

		d_idxvec tmp(transitions.size() + single_clause_transitions.size());

		auto tmp_end = thrust::set_union(transitions.begin(), transitions.end(), single_clause_transitions.begin(),
										 single_clause_transitions.end(), tmp.begin());

		tmp.resize(tmp_end - tmp.begin());

		std::swap(tmp, transitions);
	}

	return transitions;
}

struct flip_ftor : public thrust::unary_function<index_t, index_t>
{
	index_t mask;

	flip_ftor(index_t mask) : mask(mask) {}
	__host__ __device__ index_t operator()(index_t x) const { return x ^ mask; }
};

void transition_table::coo2csc(cusparseHandle_t handle, index_t n, d_idxvec& rows, d_idxvec& cols, d_idxvec& indptr)
{
	size_t buffersize;
	CHECK_CUSPARSE(cusparseXcscsort_bufferSizeExt(handle, n, n, (int)cols.size(), rows.data().get(), cols.data().get(),
												  &buffersize));

	thrust::device_vector<char> buffer(buffersize);

	d_idxvec P(cols.size());
	CHECK_CUSPARSE(cusparseCreateIdentityPermutation(handle, P.size(), P.data().get()));

	CHECK_CUSPARSE(cusparseXcoosortByColumn(handle, n, n, (int)cols.size(), rows.data().get(), cols.data().get(),
											P.data().get(), buffer.data().get()));


	indptr.resize(n + 1);

	CHECK_CUSPARSE(cusparseXcoo2csr(handle, cols.data().get(), (int)rows.size(), n, indptr.data().get(),
									CUSPARSE_INDEX_BASE_ZERO));


	print("coo2csc cols   ", cols, 20);
	print("coo2csc rows   ", rows, 20);
	print("coo2csc indptr ", indptr, 20);
}

void transition_table::construct_table()
{
	auto p = compute_rows_and_cols();

	cols = std::move(p.first);
	rows = std::move(p.second);

	int matrix_size = (int)(1ULL << model_.nodes.size());
	coo2csc(context_.cusparse_handle, matrix_size, rows, cols, indptr);
}

std::pair<d_idxvec, d_idxvec> transition_table::compute_rows_and_cols()
{
	std::vector<d_idxvec> ups, downs;

	for (const auto& f : model_.dnfs)
	{
		ups.emplace_back(generate_transitions(f.activations));
		downs.emplace_back(generate_transitions(f.deactivations));
	}

	size_t transitions_count = 0;

	for (size_t i = 0; i < ups.size(); i++)
		transitions_count += ups[i].size() + downs[i].size();

	d_idxvec trans_src(transitions_count), trans_dst(transitions_count);

	auto src_begin = trans_src.begin();
	for (size_t i = 0; i < ups.size(); i++)
	{
		src_begin = thrust::copy(ups[i].begin(), ups[i].end(), src_begin);
		src_begin = thrust::copy(downs[i].begin(), downs[i].end(), src_begin);
	}

	auto dst_begin = trans_dst.begin();
	for (size_t i = 0; i < ups.size(); i++)
	{
		dst_begin = thrust::copy(thrust::make_transform_iterator(ups[i].begin(), flip_ftor(1ULL << i)),
								 thrust::make_transform_iterator(ups[i].end(), flip_ftor(1ULL << i)), dst_begin);

		dst_begin = thrust::copy(thrust::make_transform_iterator(downs[i].begin(), flip_ftor(1ULL << i)),
								 thrust::make_transform_iterator(downs[i].end(), flip_ftor(1ULL << i)), dst_begin);
	}

	return std::make_pair(std::move(trans_src), std::move(trans_dst));
}

transition_table::transition_table(cu_context& context, model_t model) : context_(context), model_(std::move(model)) {}
