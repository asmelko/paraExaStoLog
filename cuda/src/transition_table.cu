#include <thrust/copy.h>
#include <thrust/set_operations.h>

#include "sparse_utils.h"
#include "transition_table.h"

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

struct evaluation_ftor : public thrust::unary_function<index_t, bool>
{
	index_t p_mask = 0;
	index_t n_mask = 0;

	evaluation_ftor(index_t p_mask, index_t n_mask) : p_mask(p_mask), n_mask(n_mask) {}

	__host__ __device__ bool operator()(index_t x) const { return (x & p_mask) == p_mask && (x & n_mask) == 0; }
};

struct var_trans_ftor : public thrust::unary_function<thrust::tuple<index_t, bool>, bool>
{
	index_t var_mask;
	bool transition;

	var_trans_ftor(index_t var_mask, bool transition) : var_mask(var_mask), transition(transition) {}

	__host__ __device__ index_t operator()(thrust::tuple<index_t, bool> x) const
	{
		return thrust::get<1>(x) == transition && ((thrust::get<0>(x) & var_mask) != 0) != transition;
	}
};

struct flip_ftor : public thrust::unary_function<index_t, index_t>
{
	index_t mask;

	flip_ftor(index_t mask) : mask(mask) {}
	__host__ __device__ index_t operator()(index_t x) const { return x ^ mask; }
};

d_idxvec transition_table::construct_transition_vector(const std::vector<index_t>& free_nodes, size_t fixed_val)
{
	auto c_b = thrust::make_counting_iterator(0);
	auto c_e = c_b + (1ULL << free_nodes.size());

	auto b = thrust::make_transform_iterator(c_b, transition_ftor(free_nodes, fixed_val));
	auto e = thrust::make_transform_iterator(c_e, transition_ftor(free_nodes, fixed_val));

	return d_idxvec(b, e);
}

std::pair<d_idxvec, d_idxvec> transition_table::generate_transitions(const std::vector<clause_t>& clauses,
																	 index_t variable_idx)
{
	bool zero_up_rate = rates_[2 * variable_idx] == 0.f;
	bool zero_down_rate = rates_[2 * variable_idx + 1] == 0.f;

	if (zero_up_rate && zero_down_rate)
		return {};

	index_t states_n = (index_t)(1ULL << model_.nodes.size());
	thrust::device_vector<bool> evaluation(states_n, false);

	// Evaluate DNF
	for (const auto& c : clauses)
	{
		auto p_mask = c.get_positive_mask();
		auto n_mask = c.get_negative_mask();

		thrust::transform_if(thrust::make_counting_iterator(0), thrust::make_counting_iterator(states_n),
							 evaluation.begin(), evaluation.begin(), evaluation_ftor(p_mask, n_mask),
							 thrust::logical_not<bool>());
	}

	auto b = thrust::make_zip_iterator(thrust::make_counting_iterator(0), evaluation.begin());
	auto e = thrust::make_zip_iterator(thrust::make_counting_iterator(states_n), evaluation.end());

	d_idxvec up_transition(zero_up_rate ? 0 : states_n);
	if (!zero_up_rate)
	{
		auto up_out = thrust::make_zip_iterator(up_transition.begin(), thrust::make_constant_iterator(0));
		auto up_end = thrust::copy_if(b, e, up_out, var_trans_ftor(1 << variable_idx, true));
		up_transition.resize(thrust::get<0>(up_end.get_iterator_tuple()) - up_transition.begin());
	}

	d_idxvec down_transition(zero_down_rate ? 0 : states_n);
	if (!zero_down_rate)
	{
		auto down_out = thrust::make_zip_iterator(down_transition.begin(), thrust::make_constant_iterator(0));
		auto down_end = thrust::copy_if(b, e, down_out, var_trans_ftor(1 << variable_idx, false));
		down_transition.resize(thrust::get<0>(down_end.get_iterator_tuple()) - down_transition.begin());
	}

	return std::make_pair(std::move(up_transition), std::move(down_transition));
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

	for (size_t i = 0; i < model_.dnfs.size(); i++)
	{
		auto p = generate_transitions(model_.dnfs[i].activations, i);
		ups.emplace_back(std::move(p.first));
		downs.emplace_back(std::move(p.second));
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

transition_table::transition_table(cu_context& context, const model_t& model, const d_datvec& transition_rates)
	: context_(context), model_(std::move(model)), rates_(transition_rates)
{}
