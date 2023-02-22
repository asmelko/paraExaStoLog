#pragma once

#include <string>
#include <utility>
#include <vector>

#include <boost/variant.hpp>
#include <boost/variant/recursive_wrapper.hpp>

struct op_or
{};
struct op_and
{};
struct op_not
{};

typedef std::string var;
template <typename tag>
struct binop;
template <typename tag>
struct unop;

typedef boost::variant<var, boost::recursive_wrapper<unop<op_not>>, boost::recursive_wrapper<binop<op_and>>,
					   boost::recursive_wrapper<binop<op_or>>>
	bool_expr;

template <typename tag>
struct binop
{
	explicit binop(const bool_expr& l, const bool_expr& r) : oper1(l), oper2(r) {}
	bool_expr oper1, oper2;
};

template <typename tag>
struct unop
{
	explicit unop(const bool_expr& o) : oper1(o) {}
	bool_expr oper1;
};

using taget_t = std::string;
using factor_t = bool_expr;

class bnet_parser
{
public:
	static std::vector<std::pair<taget_t, factor_t>> parse(const std::string& file);
};
