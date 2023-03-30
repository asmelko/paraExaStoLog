#include "transition_rates.h"

#include <random>

#include "thrust/fill.h"

transition_rates::transition_rates(const model_t& model) : model_(model), rates(model_.nodes.size() * 2) {}

d_datvec transition_rates::generate_uniform(const std::vector<ptrans_t>& up_transition_rates,
											const std::vector<ptrans_t>& down_transition_rates)
{
	return d_datvec(model_.nodes.size() * 2, 1.f);
}

d_datvec transition_rates::generate_normal(real_t mean, real_t std, const std::vector<ptrans_t>& up_transition_rates,
										   const std::vector<ptrans_t>& down_transition_rates)
{
	std::default_random_engine generator;
	std::normal_distribution<real_t> distribution(mean, std);

	auto generate_nonnegative = [&]() {
		real_t num;
		size_t count = 0;
		do
		{
			num = distribution(generator);
			count++;
		} while (num < 0 && count < 1000);

		if (num < 0)
			throw std::runtime_error("could not generate transition rates");

		return num;
	};

	std::vector<real_t> rates;
	rates.resize(model_.nodes.size() * 2);

	for (size_t i = 0; i < model_.nodes.size() * 2; ++i)
	{
		rates[i] = generate_nonnegative();
	}

	return rates;
}