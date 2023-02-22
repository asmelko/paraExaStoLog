#include "bnet_parser.h"

int main(int argc, char** argv)
{
	std::vector<std::string> args(argv + 1, argv + argc);
	auto model = bnet_parser::parse(args[0]);
}