#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <thrust/host_vector.h>

#include "solver.h"

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
	ASSERT_THAT(model.dnfs[1].activations[0].negative_variables, ::testing::ElementsAre(0, 1, 2));

	ASSERT_EQ(model.dnfs[1].deactivations.size(), 2);
	ASSERT_THAT(model.dnfs[1].deactivations[0].positive_variables, ::testing::ElementsAre(0, 1));
	ASSERT_THAT(model.dnfs[1].deactivations[0].negative_variables, ::testing::ElementsAre());
	ASSERT_THAT(model.dnfs[1].deactivations[1].positive_variables, ::testing::ElementsAre(1, 2));
	ASSERT_THAT(model.dnfs[1].deactivations[1].negative_variables, ::testing::ElementsAre());

	ASSERT_EQ(model.dnfs[2].activations.size(), 1);
	ASSERT_EQ(model.dnfs[2].activations[0].positive_variables.size(), 0);
	ASSERT_THAT(model.dnfs[2].activations[0].negative_variables, ::testing::ElementsAre(0, 1, 2));

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
	thrust::host_vector<index_t> indices = table.cols;

	ASSERT_THAT(indptr, ::testing::ElementsAre(0, 2, 2, 2, 3, 3, 4, 6, 8));
	ASSERT_THAT(indices, ::testing::ElementsAre(0, 0, 3, 5, 6, 6, 7, 7));
}

TEST(trans_graph, toy)
{
	model_builder builder;
	auto model = builder.construct_model("data/toy.bnet");

	cu_context context;

	transition_table table(context, std::move(model));

	table.construct_table();

	transition_graph g(table.rows, table.cols, table.indptr);

	g.find_terminals();

	thrust::host_vector<index_t> labels = g.labels;
	thrust::host_vector<index_t> terminals = g.terminals;

	ASSERT_EQ(g.sccs_count, 8);
	ASSERT_THAT(labels, ::testing::ElementsAre(0, 1, 2, 3, 4, 5, 6, 7));
	ASSERT_THAT(terminals, ::testing::ElementsAre(1, 2, 4));
}

TEST(initial_value, toy)
{
	model_builder builder;
	auto model = builder.construct_model("data/toy.bnet");

	initial_state s(model.nodes, { "A", "C", "D" }, { false, false, false }, 1.f);

	thrust::host_vector<float> state = s.state;

	ASSERT_THAT(state, ::testing::ElementsAre(1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f));
}

TEST(solver, toy)
{
	model_builder builder;
	auto model = builder.construct_model("data/toy.bnet");

	cu_context context;

	transition_table table(context, model);

	table.construct_table();

	transition_graph g(table.rows, table.cols, table.indptr);

	g.find_terminals();

	thrust::host_vector<index_t> labels = g.labels;
	thrust::host_vector<index_t> terminals = g.terminals;

	ASSERT_EQ(g.sccs_count, 8);
	ASSERT_THAT(terminals, ::testing::ElementsAre(1, 2, 4));

	initial_state st(model.nodes, { "A" }, { true }, 1.f);

	solver s(context, table, std::move(g), std::move(st));

	s.solve();

	thrust::host_vector<index_t> term_indptr = s.term_indptr;
	thrust::host_vector<index_t> term_rows = s.term_rows;
	thrust::host_vector<float> term_data = s.term_data;

	ASSERT_THAT(term_indptr, ::testing::ElementsAre(0, 1, 2, 3));
	ASSERT_THAT(term_rows, ::testing::ElementsAre(1, 2, 4));
	ASSERT_THAT(term_data, ::testing::Each(::testing::Eq(1)));

	thrust::host_vector<index_t> nonterm_indptr = s.nonterm_indptr;
	thrust::host_vector<index_t> nonterm_rows = s.nonterm_rows;
	thrust::host_vector<float> nonterm_data = s.nonterm_data;

	ASSERT_THAT(nonterm_indptr, ::testing::ElementsAre(0, 4, 7, 10));
	ASSERT_THAT(nonterm_rows, ::testing::ElementsAre(1, 3,5,7,2, 0,6,4,0,6));
	ASSERT_THAT(nonterm_data, ::testing::ElementsAre(1,1,1,1,1,0.5,0.5,1,0.5,0.5));
}



TEST(solver, toy3)
{
	model_builder builder;
	auto model = builder.construct_model("data/toy3.bnet");

	cu_context context;

	transition_table table(context, model);

	table.construct_table();

	transition_graph g(table.rows, table.cols, table.indptr);

	g.find_terminals();

	thrust::host_vector<index_t> labels = g.labels;
	thrust::host_vector<index_t> terminals = g.terminals;

	ASSERT_EQ(g.sccs_count, 8);
	ASSERT_THAT(terminals, ::testing::ElementsAre(1, 2, 4));

	initial_state st(model.nodes, { "A" }, { true }, 1.f);

	solver s(context, table, std::move(g), std::move(st));

	s.solve();

	thrust::host_vector<index_t> term_indptr = s.term_indptr;
	thrust::host_vector<index_t> term_rows = s.term_rows;
	thrust::host_vector<float> term_data = s.term_data;

	ASSERT_THAT(term_indptr, ::testing::ElementsAre(0, 1, 2, 3));
	ASSERT_THAT(term_rows, ::testing::ElementsAre(1, 2, 4));
	ASSERT_THAT(term_data, ::testing::Each(::testing::Eq(1)));

	thrust::host_vector<index_t> nonterm_indptr = s.nonterm_indptr;
	thrust::host_vector<index_t> nonterm_rows = s.nonterm_rows;
	thrust::host_vector<float> nonterm_data = s.nonterm_data;

	ASSERT_THAT(nonterm_indptr, ::testing::ElementsAre(0, 4, 7, 10));
	ASSERT_THAT(nonterm_rows, ::testing::ElementsAre(1, 3,5,7,2, 0,6,4,0,6));
	ASSERT_THAT(nonterm_data, ::testing::ElementsAre(1,1,1,1,1,0.5,0.5,1,0.5,0.5));
}

TEST(solver, toy2)
{
	model_builder builder;
	auto model = builder.construct_model("data/toy2.bnet");

	cu_context context;

	transition_table table(context, model);

	table.construct_table();

	transition_graph g(table.rows, table.cols, table.indptr);

	g.find_terminals();

	thrust::host_vector<index_t> labels = g.labels;
	thrust::host_vector<index_t> terminals = g.terminals;

	ASSERT_EQ(g.sccs_count, 3);
	ASSERT_THAT(labels, ::testing::ElementsAre(0, 0, 0, 3, 4, 0, 0, 0));
	ASSERT_THAT(terminals, ::testing::ElementsAre(0));

	initial_state st(model.nodes, { "A" }, { true }, 1.f);

	solver s(context, table, std::move(g), std::move(st));

	s.solve();

	thrust::host_vector<index_t> term_indptr = s.term_indptr;
	thrust::host_vector<index_t> term_rows = s.term_rows;
	thrust::host_vector<float> term_data = s.term_data;

	ASSERT_THAT(term_indptr, ::testing::ElementsAre(0, 6));
	ASSERT_THAT(term_rows, ::testing::ElementsAre(0, 1, 2, 5, 6, 7));
	ASSERT_THAT(term_data, ::testing::Each(::testing::Eq(1.f / 6.f)));

	thrust::host_vector<index_t> nonterm_indptr = s.nonterm_indptr;
	thrust::host_vector<index_t> nonterm_rows = s.nonterm_rows;
	thrust::host_vector<float> nonterm_data = s.nonterm_data;

	ASSERT_THAT(nonterm_indptr, ::testing::ElementsAre(0, 8));
	ASSERT_THAT(nonterm_rows, ::testing::ElementsAre(0, 1, 2, 5, 6, 7, 3, 4));
	ASSERT_THAT(nonterm_data, ::testing::Each(::testing::Eq(1.f)));
}