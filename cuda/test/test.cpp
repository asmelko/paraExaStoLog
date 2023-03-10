#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <thrust/host_vector.h>

#include "transition_table.cuh"

TEST(model, valid)
{
	model_builder builder;
	auto model = builder.construct_model("data/toy.bnet");
	ASSERT_EQ(model.dnfs.size(), 3);

	ASSERT_THAT(model.nodes, ::testing::ElementsAre("A", "C", "D"));

	ASSERT_EQ(model.dnfs[0].activations.size(), 0);
	ASSERT_EQ(model.dnfs[0].deactivations.size(), 0);

	ASSERT_EQ(model.dnfs[1].activations.size(), 1);
	ASSERT_EQ(model.dnfs[1].activations[0].positive_variables.size(), 0);
	ASSERT_THAT(model.dnfs[1].activations[0].negative_variables, ::testing::ElementsAre(0,1,2));

	ASSERT_EQ(model.dnfs[1].deactivations.size(), 2);
	ASSERT_THAT(model.dnfs[1].deactivations[0].positive_variables, ::testing::ElementsAre(0, 1));
	ASSERT_THAT(model.dnfs[1].deactivations[0].negative_variables, ::testing::ElementsAre());
	ASSERT_THAT(model.dnfs[1].deactivations[1].positive_variables, ::testing::ElementsAre(1, 2));
	ASSERT_THAT(model.dnfs[1].deactivations[1].negative_variables, ::testing::ElementsAre());

	ASSERT_EQ(model.dnfs[2].activations.size(), 1);
	ASSERT_EQ(model.dnfs[2].activations[0].positive_variables.size(), 0);
	ASSERT_THAT(model.dnfs[2].activations[0].negative_variables, ::testing::ElementsAre(0,1,2));

	ASSERT_EQ(model.dnfs[2].deactivations.size(), 2);
	ASSERT_THAT(model.dnfs[2].deactivations[0].positive_variables, ::testing::ElementsAre(0, 2));
	ASSERT_THAT(model.dnfs[2].deactivations[0].negative_variables, ::testing::ElementsAre());
	ASSERT_THAT(model.dnfs[2].deactivations[1].positive_variables, ::testing::ElementsAre(1, 2));
	ASSERT_THAT(model.dnfs[2].deactivations[1].negative_variables, ::testing::ElementsAre());
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
