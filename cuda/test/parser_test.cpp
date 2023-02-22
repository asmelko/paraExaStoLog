#include <gtest/gtest.h>

#include "model.h"

TEST(parser, valid)
{
	model_builder builder;
	auto model = builder.construct_model("data/toy.bnet");
	ASSERT_EQ(model.dnfs.size(), 3);
}
