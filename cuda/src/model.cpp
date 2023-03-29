#include "model.h"

#include <algorithm>
#include <fstream>

#include "boolstuff/BoolExprParser.h"

index_t get_part(const std::vector<index_t>& vars)
{
	index_t fixed = 0;
	for (auto var : vars)
	{
		fixed += 1ULL << var;
	}

	return fixed;
}

index_t clause_t::get_positive_mask() const { return get_part(positive_variables); }

index_t clause_t::get_negative_mask() const { return get_part(negative_variables); }

void clause_t::print() const
{
	std::cout << "Clause printout:" << std::endl;
	std::cout << "Vars: " << variables_count << ", positives: ";
	for (auto p : positive_variables)
		std::cout << p << " ";
	std::cout << ", negatives: ";
	for (auto n : negative_variables)
		std::cout << n << " ";
	std::cout << std::endl;
}

std::vector<clause_t> model_builder::construct_clauses(const std::string& target, const std::string& factors,
													   const std::vector<std::string>& targets)
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
			for (int i = 0; i < (int)targets.size(); i++)
			{
				if (targets_set.find(targets[i]) != targets_set.end())
					target_indices.push_back(i);
			}
		};

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
	if (line.rfind("targets", 0) == 0)
		std::getline(f, line);

	boolstuff::BoolExprParser parser;

	std::vector<std::string> targets, factors;

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
	model.nodes = targets;

	for (size_t i = 0; i < factors.size(); ++i)
	{
		transition_formula_t t;

		t.activations = construct_clauses(targets[i], factors[i], targets);

		model.dnfs.emplace_back(std::move(t));
	}

	return model;
}
