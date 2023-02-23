#include "model.h"

#include <algorithm>
#include <fstream>

#include "boolstuff/BoolExprParser.h"

std::vector<index_t> clause_t::get_free_variables() const
{
	std::vector<index_t> ret;
	for (int i = 0; i < variables_count; i++)
	{
		if (std::find(positive_variables.begin(), positive_variables.end(), i) != positive_variables.end()
			&& std::find(negative_variables.begin(), negative_variables.end(), i) != negative_variables.end())
			ret.push_back(i);
	}

	return ret;
}

index_t clause_t::get_fixed_part() const
{
	index_t fixed = 0;
	for (auto var : positive_variables)
	{
		fixed += 1ULL << var;
	}

	return fixed;
}

std::vector<clause_t> model_builder::construct_clauses(const std::string& target, const std::string& factors,
													   const std::vector<std::string>& targets, bool activate)
{
	std::vector<clause_t> clauses;

	boolstuff::BoolExpr<std::string>* expr = parser_.parse(factors);

	boolstuff::BoolExpr<std::string>* dnf = boolstuff::BoolExpr<std::string>::getDisjunctiveNormalForm(expr);

	if (dnf == NULL)
		throw std::runtime_error("bad input");

	typedef std::vector<const boolstuff::BoolExpr<std::string>*> V;
	typedef V::const_iterator IT;

	V termRoots;
	dnf->getDNFTermRoots(std::inserter(termRoots, termRoots.end()));
	for (IT it = termRoots.begin(); it != termRoots.end(); it++)
	{
		clause_t clause;
		clause.variables_count = (int)targets.size();

		const boolstuff::BoolExpr<std::string>* term = *it;
		std::set<std::string> positives, negatives;
		term->getTreeVariables(positives, negatives);

		auto indexize = [&targets](const std::set<std::string>& targets_set, std::vector<index_t>& target_indices) {
			for (int i = 0; i < targets.size(); i++)
			{
				if (targets_set.find(targets[i]) != targets_set.end())
					target_indices.push_back(i);
			}
		};

		auto test_skip_and_modify = [&target](const std::set<std::string>& to_be_absent,
											  std::set<std::string>& to_be_present) {
			if (to_be_absent.find(target) != to_be_absent.end())
				return true;

			to_be_present.insert(target);

			return false;
		};

		bool skip;
		if (activate)
			skip = test_skip_and_modify(positives, negatives);
		else
			skip = test_skip_and_modify(negatives, positives);

		if (skip)
			continue;

		indexize(positives, clause.positive_variables);
		indexize(negatives, clause.negative_variables);

		clauses.emplace_back(std::move(clause));
	}

	delete dnf;

	return clauses;
}

model_t model_builder::construct_model(const std::string& file)
{
	std::ifstream f(file);

	std::string line;
	std::getline(f, line);

	// skip the first line
	if (line == "targets, factors")
		std::getline(f, line);

	boolstuff::BoolExprParser parser;

	std::vector<std::string> factors;
	std::vector<std::string> targets;

	while (!f.eof())
	{
		auto target_end = line.find(',');
		auto target = line.substr(0, target_end);

		auto factor_start = line.find_first_not_of(' ', target_end + 1);
		auto factor = line.substr(factor_start);

		targets.emplace_back(std::move(target));
		factors.emplace_back(std::move(factor));

		std::getline(f, line);
	}

	model_t model;

	for (size_t i = 0; i < factors.size(); ++i)
	{
		transition_formulae_t t;

		t.activations = construct_clauses(targets[i], factors[i], targets, true);
		t.deactivations = construct_clauses(targets[i], "!(" + factors[i] + ")", targets, false);

		model.dnfs.emplace_back(std::move(t));
	}

	return model;
}
