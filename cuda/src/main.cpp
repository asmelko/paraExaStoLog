#include "model.h"

int main(int argc, char** argv)
{
	std::vector<std::string> args(argv + 1, argv + argc);

	model_builder builder;
	auto model = builder.construct_model(args[0]);
}