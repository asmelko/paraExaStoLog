#include <gtest/gtest.h>

#include "bnet_parser.h"

TEST(parser, valid)
{
	auto res = bnet_parser::parse("data/toy.bnet");
	ASSERT_EQ(res.size(), 3);
}
