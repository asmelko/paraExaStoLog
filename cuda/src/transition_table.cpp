#include "transition_table.h"

#include <thrust/set_operations.h>

using d_idxvec = thrust::device_vector<index_t>;

struct transition_ftor : public thrust::unary_function<index_t, index_t>
{
	index_t free_vars[32] = { 0 };
	index_t fixed = 0;

	transition_ftor(const std::vector<index_t>& free_v, index_t fixed)
	{
		for (size_t i = 0; i < free_v.size(); i++)
			free_vars[i] = 1ULL << free_v[i];

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

d_idxvec generate_transitions(const std::vector<clause_t>& clauses)
{
	d_idxvec transitions(0);

	for (const auto& c : clauses)
	{
		auto free_vars = c.get_free_variables();
		auto fixed = c.get_fixed_part();

		auto c_b = thrust::make_counting_iterator(0);
		auto c_e = c_b + (1ULL << free_vars.size());

		auto b = thrust::make_transform_iterator(c_b, transition_ftor(free_vars, fixed));
		auto e = thrust::make_transform_iterator(c_e, transition_ftor(free_vars, fixed));

		d_idxvec single_clause_transitions(b, e);

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

void transition_table::compute_rows_and_cols()
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
}

transition_table::transition_table(model_t model) : model_(std::move(model)) {}