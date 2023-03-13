#pragma once

#include "types.h"

struct initial_state
{
	thrust::device_vector<float> state;

	initial_state(const std::vector<std::string>& node_names, const std::vector<std::string>& fixed_node_names = {},
				  const std::vector<bool>& fixed_node_values = {}, float fixed_probability = 1.f);
};
