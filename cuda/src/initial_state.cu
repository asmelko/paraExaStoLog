#include <thrust/transform.h>

#include "initial_state.h"
#include "transition_table.h"
#include "utils.h"

struct const_transform_ftor : public thrust::unary_function<real_t, void>
{
	real_t value;

	const_transform_ftor(real_t value) : value(value) {}

	__host__ __device__ void operator()(real_t& x) const { x = value; }
};

initial_state::initial_state(const std::vector<std::string>& node_names,
							 const std::vector<std::string>& fixed_node_names,
							 const std::vector<bool>& fixed_node_values, real_t fixed_probability)

{
	if (fixed_node_names.empty())
	{
		state = d_datvec(1ULL << node_names.size(), 1.f / (1ULL << node_names.size()));

		return;
	}

	std::vector<index_t> fixed_nodes;
	for (const auto& fn : fixed_node_names)
	{
		auto it = std::find(node_names.begin(), node_names.end(), fn);

		if (it == node_names.end())
		{
			std::runtime_error("invalid initial node name");
		}

		fixed_nodes.push_back(std::distance(node_names.begin(), it));
	}

	d_idxvec fixed_indices;

	{
		size_t fixed_val = 0;

		for (size_t i = 0; i < fixed_nodes.size(); i++)
			fixed_val += (1 << fixed_nodes[i]) * fixed_node_values[i];

		std::vector<index_t> free_nodes;

		for (size_t i = 0; i < node_names.size(); i++)
			if (std::find(fixed_nodes.begin(), fixed_nodes.end(), i) == fixed_nodes.end())
				free_nodes.push_back(i);

		fixed_indices = transition_table::construct_transition_vector(free_nodes, fixed_val);
	}

	size_t fixed_states = fixed_indices.size();
	size_t nonfixed_states = (1ULL << node_names.size()) - fixed_indices.size();

	state = d_datvec(fixed_states + nonfixed_states, (1.f - fixed_probability) / nonfixed_states);

	thrust::for_each(thrust::make_permutation_iterator(state.begin(), fixed_indices.begin()),
					 thrust::make_permutation_iterator(state.begin(), fixed_indices.end()),
					 const_transform_ftor(fixed_probability / (real_t)fixed_states));
}
