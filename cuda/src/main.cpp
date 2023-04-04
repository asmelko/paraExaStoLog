#include <string_view>

#include "cli/options.h"
#include "solver.h"

struct trans_rate_arg
{
	std::vector<ptrans_t> up_transition_rates;
	std::vector<ptrans_t> down_transition_rates;
};

template <>
struct mbas::value_type<trans_rate_arg>
{
	static bool parse(const std::string& value, trans_rate_arg& result)
	{
		size_t element_offset = 0;
		while (true)
		{
			size_t end = value.find_first_of(',', element_offset);

			std::string_view e(value.data() + element_offset,
							   (end == std::string::npos ? value.size() : end) - element_offset);

			size_t eq_pos = e.find_first_of('=');

			if (eq_pos == std::string::npos || (std::toupper(e[0]) != 'D' && std::toupper(e[0]) != 'U') || e[1] != '_')
				return false;

			auto val = e.substr(eq_pos + 1);
			auto node = e.substr(2, eq_pos - 2);

			ptrans_t t;

			t.first = node;

			try
			{
				t.second = std::stof(std::string(val));
			}
			catch (...)
			{
				return false;
			}

			if (std::toupper(e[0]) == 'D')
				result.down_transition_rates.push_back(std::move(t));
			else
				result.up_transition_rates.push_back(std::move(t));

			if (end == std::string::npos)
				break;

			element_offset = end + 1;
		}

		return true;
	}
};

struct fixed_init_arg
{
	std::vector<std::string> nodes;
	std::vector<bool> values;
};

template <>
struct mbas::value_type<fixed_init_arg>
{
	static bool parse(const std::string& value, fixed_init_arg& result)
	{
		size_t element_offset = 0;
		while (true)
		{
			size_t end = value.find_first_of(',', element_offset);

			std::string_view e(value.data() + element_offset,
							   (end == std::string::npos ? value.size() : end) - element_offset);

			size_t eq_pos = e.find_first_of('=');

			if (eq_pos == std::string::npos || eq_pos + 1 >= e.size() || (e[eq_pos + 1] != '0' && e[eq_pos + 1] != '1'))
				return false;

			auto val = e.substr(eq_pos + 1);
			auto node = e.substr(0, eq_pos);

			result.nodes.emplace_back(node);
			result.values.push_back(e[eq_pos + 1] == '1');

			if (end == std::string::npos)
				break;

			element_offset = end + 1;
		}

		return true;
	}
};

int main(int argc, char** argv)
{
	mbas::command cmd;

	// bnet file
	cmd.add_option("f,bnet-file", "Path to the bnet file.", false)
		.add_parameter<std::string>(mbas::value_type<std::string>(), "BNET_FILE");

	// transition rate
	cmd.add_option("t,trans-distribution", "Distribution of transition rates. Can be NORMAL or UNIFORM (default).",
				   true)
		.add_parameter<std::string>(mbas::value_type<std::string>(), "T_DISTR");
	cmd.add_option("mean", "Mean value for normal transition rate distribution.", true)
		.add_parameter<float>(mbas::value_type<float>(), "MEAN");
	cmd.add_option("std", "Standard deviation value for normal transition rate distribution.", true)
		.add_parameter<float>(mbas::value_type<float>(), "STD");
	cmd.add_option("trans-rates", "Explicit transition rates. A comma separated list of (u|d)_NAME=VAL.", true)
		.add_parameter<trans_rate_arg>(mbas::value_type<trans_rate_arg>(), "TRANS_RATE");

	// initial state
	cmd.add_option("i,initial-distribution", "Distribution of initial state. Can be RANDOM or UNIFORM (default).", true)
		.add_parameter<std::string>(mbas::value_type<std::string>(), "INIT_DISTR");
	cmd.add_option("initial-fixed-prob", "Probability of fixed initial states.", true)
		.add_parameter<float>(mbas::value_type<float>(), "INIT_PROB");
	cmd.add_option("initial-fixed-list",
				   "Specify fixed parts of the initial state. A comma separated list of NAME=(0|1).", true)
		.add_parameter<fixed_init_arg>(mbas::value_type<fixed_init_arg>(), "INIT_FIXED_LIST");

	// symbolic computation
	cmd.add_option("s,symbolic",
				   "Turns on symbolic computation. After the first execution of a bnet file F, F.symb file is created "
				   "which contains symbolic information of the computation. In the further runs of F, the "
				   "program first looks up F.symb and uses it to accelerate the computation.",
				   true);

	auto parsed = cmd.parse(argc, argv);

	if (!parsed.parse_ok())
	{
		std::cerr << cmd.help();
		return 1;
	}

	auto filename = parsed["bnet-file"]->get_value<std::string>();

	bool trans_distr_uniform = true;
	if (parsed["trans-distribution"])
	{
		auto distr = parsed["trans-distribution"]->get_value<std::string>();
		if (distr == "NORMAL")
			trans_distr_uniform = false;
		if (distr != "NORMAL" && distr != "UNIFORM")
		{
			std::cerr << "Bad transition distribution type" << std::endl;
			std::cerr << cmd.help();
			return 1;
		}
	}

	float mean = 0.f;
	if (parsed["mean"])
		mean = parsed["mean"]->get_value<float>();

	float std = 1.f;
	if (parsed["std"])
		std = parsed["std"]->get_value<float>();

	trans_rate_arg tr;
	if (parsed["trans-rates"])
		tr = parsed["trans-rates"]->get_value<trans_rate_arg>();

	bool init_distr_uniform = true;
	if (parsed["initial-distribution"])
	{
		auto distr = parsed["initial-distribution"]->get_value<std::string>();
		if (distr == "RANDOM")
			init_distr_uniform = false;
		if (distr != "RANDOM" && distr != "UNIFORM")
		{
			std::cerr << "Bad initial state distribution type" << std::endl;
			std::cerr << cmd.help();
			return 1;
		}
	}

	float fixed_prob = 1.f;
	if (parsed["initial-fixed-prob"])
		std = parsed["initial-fixed-prob"]->get_value<float>();
	if (fixed_prob < 0.f || fixed_prob > 1.f)
	{
		std::cerr << "Bad fixed initial states probability" << std::endl;
		std::cerr << cmd.help();
		return 1;
	}

	fixed_init_arg fi;
	if (parsed["initial-fixed-list"])
		fi = parsed["initial-fixed-list"]->get_value<fixed_init_arg>();

	bool symbolic = parsed["symbolic"];

	// **** args done, lets get to work ****

	try
	{
		// create context
		cu_context context;

		// create model
		model_builder builder;
		auto model = builder.construct_model(filename);

		// create initial state
		initial_state st(model.nodes, fi.nodes, fi.values, fixed_prob);

		// create transition rates
		transition_rates r(model);
		if (trans_distr_uniform)
			r.generate_uniform(tr.up_transition_rates, tr.down_transition_rates);
		else
			r.generate_normal(mean, std, tr.up_transition_rates, tr.down_transition_rates);

		// create table
		transition_table table(context, model);
		table.construct_table();

		// create graph
		transition_graph g(context, table.rows, table.cols, table.indptr);
		g.find_terminals();

		// solve
		solver s(context, table, std::move(g), std::move(r), std::move(st));
		s.solve();

		s.print_final_state(model.nodes);
	}
	catch (std::exception& e)
	{
		std::cerr << "Program ended with an exception: " << e.what() << std::endl;
	}
}
