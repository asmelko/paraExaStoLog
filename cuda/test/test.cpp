#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <thrust/copy.h>
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

	transition_graph g(context, table.rows, table.cols, table.indptr);

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

	transition_graph g(context, table.rows, table.cols, table.indptr);

	g.find_terminals();

	thrust::host_vector<index_t> labels = g.labels;
	thrust::host_vector<index_t> terminals = g.terminals;

	ASSERT_EQ(g.sccs_count, 8);
	ASSERT_THAT(terminals, ::testing::ElementsAre(1, 2, 4));

	initial_state st(model.nodes, { "A", "C", "D" }, { false, false, false }, 1.f);

	solver s(context, table, std::move(g), std::move(st));

	s.solve();

	thrust::host_vector<index_t> term_indptr = s.term_indptr;
	thrust::host_vector<index_t> term_rows = s.term_rows;
	thrust::host_vector<float> term_data = s.term_data;

	ASSERT_THAT(term_indptr, ::testing::ElementsAre(0, 1, 2, 3));
	ASSERT_THAT(term_rows, ::testing::ElementsAre(1, 2, 4));
	ASSERT_THAT(term_data, ::testing::Each(::testing::Eq(1)));

	thrust::host_vector<index_t> nonterm_indptr = s.nonterm_indptr;
	thrust::host_vector<index_t> nonterm_cols = s.nonterm_cols;
	thrust::host_vector<float> nonterm_data = s.nonterm_data;

	ASSERT_THAT(nonterm_indptr, ::testing::ElementsAre(0, 4, 7, 10));
	ASSERT_THAT(nonterm_cols, ::testing::ElementsAre(1, 3, 5, 7, 2, 0, 6, 4, 0, 6));
	ASSERT_THAT(nonterm_data, ::testing::ElementsAre(1, 1, 1, 1, 1, 0.5, 0.5, 1, 0.5, 0.5));

	thrust::host_vector<float> final_state = s.final_state;

	thrust::host_vector<index_t> nonzero_indices(labels.size());
	thrust::host_vector<float> nonzero_data(labels.size());

	auto i_end = thrust::copy_if(thrust::make_counting_iterator<index_t>(0),
								 thrust::make_counting_iterator<index_t>(labels.size()), final_state.begin(),
								 nonzero_indices.begin(), thrust::identity<float>());
	nonzero_indices.resize(i_end - nonzero_indices.begin());

	auto d_end = thrust::copy(thrust::make_permutation_iterator(final_state.begin(), nonzero_indices.begin()),
							  thrust::make_permutation_iterator(final_state.begin(), nonzero_indices.end()),
							  nonzero_data.begin());
	nonzero_data.resize(d_end - nonzero_data.begin());

	ASSERT_THAT(nonzero_indices, ::testing::ElementsAre(2, 4));
	ASSERT_THAT(nonzero_data, ::testing::Each(::testing::Eq(0.5f)));
}

TEST(solver, toy2)
{
	model_builder builder;
	auto model = builder.construct_model("data/toy2.bnet");

	cu_context context;

	transition_table table(context, model);

	table.construct_table();

	transition_graph g(context, table.rows, table.cols, table.indptr);

	g.find_terminals();

	thrust::host_vector<index_t> labels = g.labels;
	thrust::host_vector<index_t> terminals = g.terminals;

	ASSERT_EQ(g.sccs_count, 3);
	ASSERT_THAT(labels, ::testing::ElementsAre(0, 0, 0, 3, 4, 0, 0, 0));
	ASSERT_THAT(terminals, ::testing::ElementsAre(0));

	initial_state st(model.nodes, { "A", "B", "C" }, { false, false, false }, 1.f);

	solver s(context, table, std::move(g), std::move(st));

	s.solve();

	thrust::host_vector<index_t> term_indptr = s.term_indptr;
	thrust::host_vector<index_t> term_rows = s.term_rows;
	thrust::host_vector<float> term_data = s.term_data;

	ASSERT_THAT(term_indptr, ::testing::ElementsAre(0, 6));
	ASSERT_THAT(term_rows, ::testing::ElementsAre(0, 1, 2, 5, 6, 7));
	ASSERT_THAT(term_data, ::testing::Each(::testing::Eq(1.f / 6.f)));

	thrust::host_vector<index_t> nonterm_indptr = s.nonterm_indptr;
	thrust::host_vector<index_t> nonterm_cols = s.nonterm_cols;
	thrust::host_vector<float> nonterm_data = s.nonterm_data;

	ASSERT_THAT(nonterm_indptr, ::testing::ElementsAre(0, 8));
	ASSERT_THAT(nonterm_cols, ::testing::ElementsAre(0, 1, 2, 5, 6, 7, 3, 4));
	ASSERT_THAT(nonterm_data, ::testing::Each(::testing::Eq(1.f)));

	thrust::host_vector<float> final_state = s.final_state;

	thrust::host_vector<index_t> nonzero_indices(labels.size());
	thrust::host_vector<float> nonzero_data(labels.size());

	auto i_end = thrust::copy_if(thrust::make_counting_iterator<index_t>(0),
								 thrust::make_counting_iterator<index_t>(labels.size()), final_state.begin(),
								 nonzero_indices.begin(), thrust::identity<float>());
	nonzero_indices.resize(i_end - nonzero_indices.begin());

	auto d_end = thrust::copy(thrust::make_permutation_iterator(final_state.begin(), nonzero_indices.begin()),
							  thrust::make_permutation_iterator(final_state.begin(), nonzero_indices.end()),
							  nonzero_data.begin());
	nonzero_data.resize(d_end - nonzero_data.begin());

	ASSERT_THAT(nonzero_indices, ::testing::ElementsAre(0, 1, 2, 5, 6, 7));
	ASSERT_THAT(nonzero_data, ::testing::Each(::testing::Eq(1.f / 6.f)));
}

TEST(solver, toy3)
{
	model_builder builder;
	auto model = builder.construct_model("data/toy3.bnet");

	cu_context context;

	transition_table table(context, model);

	table.construct_table();

	transition_graph g(context, table.rows, table.cols, table.indptr);

	g.find_terminals();

	thrust::host_vector<index_t> labels = g.labels;
	thrust::host_vector<index_t> terminals = g.terminals;

	ASSERT_EQ(g.sccs_count, 1);
	ASSERT_THAT(terminals, ::testing::ElementsAre(0));

	initial_state st(model.nodes, { "A", "B" }, { false, false }, 1.f);

	solver s(context, table, std::move(g), std::move(st));

	s.solve();

	thrust::host_vector<index_t> term_indptr = s.term_indptr;
	thrust::host_vector<index_t> term_rows = s.term_rows;
	thrust::host_vector<float> term_data = s.term_data;

	ASSERT_THAT(term_indptr, ::testing::ElementsAre(0, 4));
	ASSERT_THAT(term_rows, ::testing::ElementsAre(0, 1, 2, 3));
	ASSERT_THAT(term_data, ::testing::Each(::testing::Eq(1.f / 4.f)));

	thrust::host_vector<index_t> nonterm_indptr = s.nonterm_indptr;
	thrust::host_vector<index_t> nonterm_cols = s.nonterm_cols;
	thrust::host_vector<float> nonterm_data = s.nonterm_data;

	ASSERT_THAT(nonterm_indptr, ::testing::ElementsAre(0, 4));
	ASSERT_THAT(nonterm_cols, ::testing::ElementsAre(0, 1, 2, 3));
	ASSERT_THAT(nonterm_data, ::testing::Each(::testing::Eq(1)));

	thrust::host_vector<float> final_state = s.final_state;

	thrust::host_vector<index_t> nonzero_indices(labels.size());
	thrust::host_vector<float> nonzero_data(labels.size());

	auto i_end = thrust::copy_if(thrust::make_counting_iterator<index_t>(0),
								 thrust::make_counting_iterator<index_t>(labels.size()), final_state.begin(),
								 nonzero_indices.begin(), thrust::identity<float>());
	nonzero_indices.resize(i_end - nonzero_indices.begin());

	auto d_end = thrust::copy(thrust::make_permutation_iterator(final_state.begin(), nonzero_indices.begin()),
							  thrust::make_permutation_iterator(final_state.begin(), nonzero_indices.end()),
							  nonzero_data.begin());
	nonzero_data.resize(d_end - nonzero_data.begin());

	ASSERT_THAT(nonzero_indices, ::testing::ElementsAre(0, 1, 2, 3));
	ASSERT_THAT(nonzero_data, ::testing::Each(::testing::Eq(0.25f)));
}
//
//TEST(solver, kras)
//{
//	model_builder builder;
//	auto model = builder.construct_model("data/krasmodel15vars.bnet");
//
//	cu_context context;
//
//	transition_table table(context, model);
//
//	table.construct_table();
//
//	std::cout << "after table" << std::endl;
//
//	transition_graph g(context, table.rows, table.cols, table.indptr);
//
//	g.find_terminals();
//
//	std::cout << "after graph" << std::endl;
//
//	initial_state st(model.nodes, { "cc", "KRAS", "DSB", "cell_death" }, { true, true, true, false }, 1.f);
//
//	solver s(context, table, std::move(g), std::move(st));
//
//	s.solve();
//
//	auto n = 1 << model.nodes.size();
//
//	thrust::host_vector<float> final_state = s.final_state;
//
//	thrust::host_vector<index_t> nonzero_indices(n);
//	thrust::host_vector<float> nonzero_data(n);
//	auto i_end = thrust::copy_if(thrust::make_counting_iterator<index_t>(0), thrust::make_counting_iterator<index_t>(n),
//								 final_state.begin(), nonzero_indices.begin(), thrust::identity<float>());
//	nonzero_indices.resize(i_end - nonzero_indices.begin());
//
//	auto d_end = thrust::copy(thrust::make_permutation_iterator(final_state.begin(), nonzero_indices.begin()),
//							  thrust::make_permutation_iterator(final_state.begin(), nonzero_indices.end()),
//							  nonzero_data.begin());
//	nonzero_data.resize(d_end - nonzero_data.begin());
//
//	ASSERT_THAT(nonzero_indices, ::testing::ElementsAre(35, 67, 291, 323, 16159, 16387, 16643));
//	ASSERT_THAT(nonzero_data, ::testing::Pointwise(::testing::FloatNear(128 * std::numeric_limits<float>::epsilon()),
//												   { 0.16982302, 0.01069429, 0.25817667, 0.01918749, 0.3131778,
//													 0.09025865, 0.1386820 }));
//}

TEST(solver, cohen)
{
	model_builder builder;
	auto model = builder.construct_model("data/EMT_cohen_ModNet.bnet");

	cu_context context;

	transition_table table(context, model);

	table.construct_table();

	std::cout << "after table" << std::endl;

	transition_graph g(context, table.rows, table.cols, table.indptr);

	g.find_terminals();

	std::cout << "after graph" << std::endl;

	initial_state st(
		model.nodes,
		{ "ECMicroenv", "DNAdamage", "Metastasis", "Migration", "Invasion", "EMT", "Apoptosis", "Notch_pthw", "p53" },
		{ true, true, false, false, false, false, false, true, false }, 1.f);

	solver s(context, table, std::move(g), std::move(st));

	s.solve();

	auto n = 1 << model.nodes.size();

	thrust::host_vector<float> final_state = s.final_state;

	thrust::host_vector<index_t> nonzero_indices(n);
	thrust::host_vector<float> nonzero_data(n);
	auto i_end = thrust::copy_if(thrust::make_counting_iterator<index_t>(0), thrust::make_counting_iterator<index_t>(n),
								 final_state.begin(), nonzero_indices.begin(), thrust::identity<float>());
	nonzero_indices.resize(i_end - nonzero_indices.begin());

	auto d_end = thrust::copy(thrust::make_permutation_iterator(final_state.begin(), nonzero_indices.begin()),
							  thrust::make_permutation_iterator(final_state.begin(), nonzero_indices.end()),
							  nonzero_data.begin());
	nonzero_data.resize(d_end - nonzero_data.begin());

	ASSERT_THAT(nonzero_indices, ::testing::ElementsAre(35, 67, 291, 323, 16159, 16387, 16643));
	ASSERT_THAT(nonzero_data, ::testing::Pointwise(::testing::FloatNear(128 * std::numeric_limits<float>::epsilon()),
												   { 0.16982302, 0.01069429, 0.25817667, 0.01918749, 0.3131778,
													 0.09025865, 0.1386820 }));
}
