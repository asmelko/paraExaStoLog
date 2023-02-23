#pragma once

#include <string>
#include <utility>
#include <vector>

#include "boolstuff/BoolExprParser.h"

using index_t = int;

struct clause_t
{
	int variables_count;

	std::vector<index_t> positive_variables;
	std::vector<index_t> negative_variables;

	std::vector<index_t> get_free_variables() const;
	index_t get_fixed_part() const;
};

struct transition_formulae_t
{
	std::vector<clause_t> activations;
	std::vector<clause_t> deactivations;
};

struct model_t
{
	std::vector<std::string> nodes;
	std::vector<transition_formulae_t> dnfs;
};

class model_builder
{
	boolstuff::BoolExprParser parser_;

	std::vector<clause_t> construct_clauses(const std::string& target, const std::string& factors,
											const std::vector<std::string>& targets, bool activate);

public:
	model_t construct_model(const std::string& file);
};
