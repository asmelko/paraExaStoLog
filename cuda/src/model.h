#pragma once

#include <string>
#include <utility>
#include <vector>

#include "boolstuff/BoolExprParser.h"
#include "types.h"

struct clause_t
{
	int variables_count;

	std::vector<index_t> positive_variables;
	std::vector<index_t> negative_variables;

	index_t get_positive_mask() const;
	index_t get_negative_mask() const;

	void print() const;
};

struct transition_formula_t
{
	std::vector<clause_t> activations;
};

struct model_t
{
	std::vector<std::string> nodes;
	std::vector<transition_formula_t> dnfs;
};

class model_builder
{
	boolstuff::BoolExprParser parser_;

	std::vector<clause_t> construct_clauses(const std::string& target, const std::string& factors,
											const std::vector<std::string>& targets);

public:
	model_t construct_model(const std::string& file);
};
