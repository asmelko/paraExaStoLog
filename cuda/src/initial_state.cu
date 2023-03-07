#include <thrust/transform.h>

#include "initial_state.h"
#include "transition_table.h"

struct const_transform_ftor : public thrust::unary_function<float, float>
{
	float value;

	const_transform_ftor(float value) : value(value) {}

	__host__ __device__ index_t operator()(float) const { return value; }
};

initial_state::initial_state(const std::vector<std::string>& node_names,
							 const std::vector<std::string>& fixed_node_names,
							 const std::vector<bool>& fixed_node_values, float fixed_probability)

{
	std::cout << "node names ";
	for (int i = 0; i < node_names.size(); i++)
		std::cout << node_names[i] << " ";
	std::cout << std::endl;

	if (fixed_node_names.empty())
	{
		state = thrust::device_vector<float>(1ULL << node_names.size(), 1.f / (1ULL << node_names.size()));

		return;
	}

	std::vector<index_t> fixed_nodes;
	for (const auto& fn : fixed_node_names)
	{
		std::cout << fn << std::endl;
		auto it = std::find(node_names.begin(), node_names.end(), fn);

		if (it == node_names.end())
		{
			std::runtime_error("invalid initial node name");
		}

		fixed_nodes.push_back(std::distance(node_names.begin(), it));
	}

	std::cout << "fixed nodes ";
	for (int i = 0; i < fixed_nodes.size(); i++)
		std::cout << fixed_nodes[i] << " ";
	std::cout << std::endl;

	d_idxvec fixed_indices;

	{
		size_t fixed_val = 0;

		for (size_t i = 0; i < fixed_nodes.size(); i++)
			fixed_val += (1 << fixed_nodes[i]) * fixed_node_values[i];

		std::cout << "fixed_val" << fixed_val << std::endl;

		std::vector<index_t> free_nodes;

		for (size_t i = 0; i < node_names.size(); i++)
			if (std::find(fixed_nodes.begin(), fixed_nodes.end(), i) == fixed_nodes.end())
				free_nodes.push_back(i);

		std::cout << "free nodes ";
		for (int i = 0; i < free_nodes.size(); i++)
			std::cout << free_nodes[i] << " ";
		std::cout << std::endl;

		fixed_indices = transition_table::construct_transition_vector(free_nodes, fixed_val);
	}

	size_t fixed_states = fixed_indices.size();
	size_t nonfixed_states = (1ULL << node_names.size()) - fixed_indices.size();

	state = thrust::device_vector<float>(fixed_states + nonfixed_states, (1.f - fixed_probability) / nonfixed_states);

	thrust::transform(thrust::make_permutation_iterator(state.begin(), fixed_indices.begin()),
					  thrust::make_permutation_iterator(state.begin(), fixed_indices.end()),
					  thrust::make_permutation_iterator(state.begin(), fixed_indices.begin()),
					  const_transform_ftor(fixed_probability / (float)fixed_states));
}
