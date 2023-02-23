#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <thrust/host_vector.h>

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

	thrust::host_vector<index_t> indptr = table.indptr;
	thrust::host_vector<index_t> indices = table.indices;

	ASSERT_THAT(indptr, ::testing::ElementsAre(0, 0, 2, 4, 5, 7, 8, 8, 8));
	ASSERT_THAT(indices, ::testing::ElementsAre(3, 5, 0, 6, 7, 0, 6, 7));
}
