#pragma once

#include <map>
#include <string>
#include <utility>

#include "model.h"
#include "types.h"

using ptrans_t = std::pair<std::string, real_t>;

class transition_rates
{
	const model_t& model_;

	std::map<index_t, real_t> transform(const std::vector<ptrans_t>& up_transition_rates);

	d_datvec generate(std::function<real_t()> generator, const std::vector<ptrans_t>& up_transition_rates,
					  const std::vector<ptrans_t>& down_transition_rates);

public:
	d_datvec rates;

	transition_rates(const model_t& model);

	void generate_uniform(const std::vector<ptrans_t>& up_transition_rates = {},
						  const std::vector<ptrans_t>& down_transition_rates = {});

	void generate_normal(real_t mean, real_t std, const std::vector<ptrans_t>& up_transition_rates = {},
						 const std::vector<ptrans_t>& down_transition_rates = {});
};
