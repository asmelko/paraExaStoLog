#include <gtest/gtest.h>

#include "transition_table.cuh"

TEST(model, valid)
{
	model_builder builder;
	auto model = builder.construct_model("data/toy.bnet");
	ASSERT_EQ(model.dnfs.size(), 3);
}

TEST(trans_table, toy)
{
	model_builder builder;
	auto model = builder.construct_model("data/toy.bnet");

	cu_context context;

	transition_table table(context, std::move(model));

	table.construct_table();
}
