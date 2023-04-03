#pragma once

#include "types.h"

struct initial_state
{
	d_datvec state;

	initial_state(const std::vector<std::string>& node_names, const std::vector<std::string>& fixed_node_names = {},
				  const std::vector<bool>& fixed_node_values = {}, real_t fixed_probability = 1.f);
};
