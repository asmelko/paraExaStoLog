#pragma once

#include <string>
#include <utility>

#include "model.h"
#include "types.h"

using ptrans_t = std::pair<std::string, real_t>;

class transition_rates
{
	const model_t& model_;

public:
	transition_rates(const model_t& model);

	d_datvec generate_uniform(const std::vector<ptrans_t>& up_transition_rates = {},
							  const std::vector<ptrans_t>& down_transition_rates = {});

	d_datvec generate_normal(real_t mean, real_t std, const std::vector<ptrans_t>& up_transition_rates = {},
							 const std::vector<ptrans_t>& down_transition_rates = {})
};