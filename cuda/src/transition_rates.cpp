#include "transition_rates.h"

#include <random>

#include "diagnostics.h"

std::map<index_t, real_t> transition_rates::transform(const std::vector<ptrans_t>& transition_rates)
{
	std::map<index_t, real_t> rates_map;
	for (const auto& rate : transition_rates)
	{
		auto it = std::find(model_.nodes.begin(), model_.nodes.end(), rate.first);
		if (it == model_.nodes.end())
			throw std::runtime_error("node " + rate.first + " not found");
		if (rate.second < 0)
			throw std::runtime_error("transition rate must be non-negative");

		rates_map[(index_t)std::distance(model_.nodes.begin(), it)] = rate.second;
	}

	return rates_map;
}

d_datvec transition_rates::generate(std::function<real_t()> generator, const std::vector<ptrans_t>& up_transition_rates,
									const std::vector<ptrans_t>& down_transition_rates)
{
	std::vector<real_t> h_rates;
	h_rates.resize(model_.nodes.size() * 2);

	auto up_rates = transform(up_transition_rates);
	auto down_rates = transform(down_transition_rates);

	for (index_t i = 0; i < (index_t)model_.nodes.size(); i++)
	{
		auto up_rate = up_rates.find(i);
		auto down_rate = down_rates.find(i);

		if (up_rate != up_rates.end())
		{
			h_rates[i * 2] = up_rate->second;
		}
		else
		{
			h_rates[i * 2] = generator();
		}

		if (down_rate != down_rates.end())
		{
			h_rates[i * 2 + 1] = down_rate->second;
		}
		else
		{
			h_rates[i * 2 + 1] = generator();
		}
	}

	return h_rates;
}

transition_rates::transition_rates(const model_t& model) : model_(model) {}

void transition_rates::generate_uniform(const std::vector<ptrans_t>& up_transition_rates,
										const std::vector<ptrans_t>& down_transition_rates)
{
	Timer t;
	t.Start();

	auto generator = []() { return 1.f; };
	rates = generate(generator, up_transition_rates, down_transition_rates);

	t.Stop();
	diag_print("Transition rates creation: ", t.Millisecs(), "ms");
}

void transition_rates::generate_normal(real_t mean, real_t std, const std::vector<ptrans_t>& up_transition_rates,
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
			throw std::runtime_error("could not generate postive transition rates with the provided mean and std");

		return num;
	};

	rates = generate(generate_nonnegative, up_transition_rates, down_transition_rates);
}
