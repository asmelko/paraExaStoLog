#include "bnet_parser.h"

#include <fstream>

#include <boost/spirit/include/phoenix.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>
#include <boost/spirit/include/qi.hpp>


namespace qi = boost::spirit::qi;
namespace phx = boost::phoenix;

struct printer : boost::static_visitor<void>
{
	printer(std::ostream& os) : _os(os) {}
	std::ostream& _os;

	//
	void operator()(const var& v) const { _os << v; }

	void operator()(const binop<op_and>& b) const { print(" & ", b.oper1, b.oper2); }
	void operator()(const binop<op_or>& b) const { print(" | ", b.oper1, b.oper2); }

	void print(const std::string& op, const bool_expr& l, const bool_expr& r) const
	{
		_os << "(";
		boost::apply_visitor(*this, l);
		_os << op;
		boost::apply_visitor(*this, r);
		_os << ")";
	}

	void operator()(const unop<op_not>& u) const
	{
		_os << "(";
		_os << "!";
		boost::apply_visitor(*this, u.oper1);
		_os << ")";
	}
};

std::ostream& operator<<(std::ostream& os, const bool_expr& e)
{
	boost::apply_visitor(printer(os), e);
	return os;
}

template <typename It, typename Skipper = qi::space_type>
struct parser : qi::grammar<It, bool_expr(), Skipper>
{
	parser() : parser::base_type(expr_)
	{
		using namespace qi;

		expr_ = or_.alias();

		not_ = ("!" > simple)[_val = phx::construct<unop<op_not>>(_1)] | simple[_val = _1];
#ifdef RIGHT_ASSOCIATIVE
		or_ = (and_ >> "|" >> or_)[_val = phx::construct<binop<op_or>>(_1, _2)] | xor_[_val = _1];
		and_ = (not_ >> "&" >> and_)[_val = phx::construct<binop<op_and>>(_1, _2)] | not_[_val = _1];
#else
		or_ = and_[_val = _1] >> *("|" >> and_[_val = phx::construct<binop<op_or>>(_val, _1)]);
		and_ = not_[_val = _1] >> *("&" >> not_[_val = phx::construct<binop<op_and>>(_val, _1)]);
#endif

		simple = (('(' > expr_ > ')') | var_);
		var_ = qi::lexeme[+alpha];

		BOOST_SPIRIT_DEBUG_NODE(expr_);
		BOOST_SPIRIT_DEBUG_NODE(or_);
		BOOST_SPIRIT_DEBUG_NODE(and_);
		BOOST_SPIRIT_DEBUG_NODE(not_);
		BOOST_SPIRIT_DEBUG_NODE(simple);
		BOOST_SPIRIT_DEBUG_NODE(var_);
	}

private:
	qi::rule<It, var(), Skipper> var_;
	qi::rule<It, bool_expr(), Skipper> not_, and_, or_, simple, expr_;
};

std::vector<std::pair<taget_t, factor_t>> bnet_parser::parse(const std::string& file)
{
	std::vector<std::pair<taget_t, factor_t>> ret;

	std::ifstream f(file);

	std::string line;
	std::getline(f, line);

	// skip the first line
	if (line == "targets, factors")
		std::getline(f, line);

	do
	{
		auto target_end = line.find(',');
		auto target = line.substr(0, target_end);

		auto factors_start = line.find_first_not_of(' ', target_end + 1);
		auto factors = std::string_view(line).substr(factors_start);

		auto b(std::begin(factors)), e(std::end(factors));
		parser<decltype(b)> p;

		bool_expr expr;
		bool ok = qi::phrase_parse(b, e, p, qi::space, expr);

		if (!ok)
			throw std::runtime_error("invalid input");

		if (b != e)
			throw std::runtime_error("unparsed: '" + std::string(b, e));

		ret.emplace_back(std::make_pair(std::move(target), std::move(expr)));

		std::getline(f, line);
	} while (!f.eof());

	return ret;
}
