#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <thrust/copy.h>
#include <thrust/host_vector.h>

#include "solver.h"

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

	thrust::host_vector<index_t> vertices = g.reordered_vertices;
	thrust::host_vector<index_t> offsets(g.sccs_offsets.begin(), g.sccs_offsets.begin() + g.terminals_count + 1);

	ASSERT_THAT(vertices, ::testing::ElementsAre(1, 2, 4, 7, 6, 5, 3, 0));
	ASSERT_THAT(offsets, ::testing::ElementsAre(0, 1, 2, 3));
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

	thrust::host_vector<index_t> vertices = g.reordered_vertices;
	thrust::host_vector<index_t> offsets(g.sccs_offsets.begin(), g.sccs_offsets.begin() + g.terminals_count + 1);

	ASSERT_THAT(vertices, ::testing::ElementsAre(1, 2, 4, 7, 6, 5, 3, 0));
	ASSERT_THAT(offsets, ::testing::ElementsAre(0, 1, 2, 3));

	initial_state st(model.nodes, { "A", "C", "D" }, { false, false, false }, 1.f);

	transition_rates r(model);
	r.generate_uniform();

	solver s(context, table, std::move(g), std::move(r), std::move(st));

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
	ASSERT_THAT(nonterm_cols, ::testing::ElementsAre(1, 7, 5, 3, 2, 6, 0, 4, 6, 0));
	ASSERT_THAT(nonterm_data, ::testing::ElementsAre(1, 1, 1, 1, 1, 0.5, 0.5, 1, 0.5, 0.5));

	thrust::host_vector<float> final_state = s.final_state;

	thrust::host_vector<index_t> nonzero_indices(final_state.size());
	thrust::host_vector<float> nonzero_data(final_state.size());

	auto i_end = thrust::copy_if(thrust::make_counting_iterator<index_t>(0),
								 thrust::make_counting_iterator<index_t>((index_t)final_state.size()),
								 final_state.begin(), nonzero_indices.begin(), thrust::identity<float>());
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

	thrust::host_vector<index_t> vertices = g.reordered_vertices;
	thrust::host_vector<index_t> offsets(g.sccs_offsets.begin(), g.sccs_offsets.begin() + g.terminals_count + 1);

	ASSERT_THAT(vertices, ::testing::ElementsAre(0, 1, 2, 5, 6, 7, 4, 3));
	ASSERT_THAT(offsets, ::testing::ElementsAre(0, 6));

	initial_state st(model.nodes, { "A", "B", "C" }, { false, false, false }, 1.f);

	transition_rates r(model);
	r.generate_uniform();

	solver s(context, table, std::move(g), std::move(r), std::move(st));

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
	ASSERT_THAT(nonterm_cols, ::testing::ElementsAre(0, 1, 2, 5, 6, 7, 4, 3));
	ASSERT_THAT(nonterm_data, ::testing::Each(::testing::Eq(1.f)));

	thrust::host_vector<float> final_state = s.final_state;

	thrust::host_vector<index_t> nonzero_indices(final_state.size());
	thrust::host_vector<float> nonzero_data(final_state.size());

	auto i_end = thrust::copy_if(thrust::make_counting_iterator<index_t>(0),
								 thrust::make_counting_iterator<index_t>((index_t)final_state.size()),
								 final_state.begin(), nonzero_indices.begin(), thrust::identity<float>());
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

	thrust::host_vector<index_t> vertices = g.reordered_vertices;
	thrust::host_vector<index_t> offsets(g.sccs_offsets.begin(), g.sccs_offsets.begin() + g.terminals_count + 1);

	ASSERT_THAT(vertices, ::testing::ElementsAre(0, 1, 2, 3));
	ASSERT_THAT(offsets, ::testing::ElementsAre(0, 4));

	initial_state st(model.nodes, { "A", "B" }, { false, false }, 1.f);

	transition_rates r(model);
	r.generate_uniform();

	solver s(context, table, std::move(g), std::move(r), std::move(st));

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

	thrust::host_vector<index_t> nonzero_indices(final_state.size());
	thrust::host_vector<float> nonzero_data(final_state.size());

	auto i_end = thrust::copy_if(thrust::make_counting_iterator<index_t>(0),
								 thrust::make_counting_iterator<index_t>((index_t)final_state.size()),
								 final_state.begin(), nonzero_indices.begin(), thrust::identity<float>());
	nonzero_indices.resize(i_end - nonzero_indices.begin());

	auto d_end = thrust::copy(thrust::make_permutation_iterator(final_state.begin(), nonzero_indices.begin()),
							  thrust::make_permutation_iterator(final_state.begin(), nonzero_indices.end()),
							  nonzero_data.begin());
	nonzero_data.resize(d_end - nonzero_data.begin());

	ASSERT_THAT(nonzero_indices, ::testing::ElementsAre(0, 1, 2, 3));
	ASSERT_THAT(nonzero_data, ::testing::Each(::testing::Eq(0.25f)));
}

TEST(solver, kras)
{
	model_builder builder;
	auto model = builder.construct_model("data/krasmodel15vars.bnet");

	cu_context context;

	transition_table table(context, model);

	table.construct_table();

	transition_graph g(context, table.rows, table.cols, table.indptr);

	g.find_terminals();

	initial_state st(model.nodes, { "cc", "KRAS", "DSB", "cell_death" }, { true, true, true, false }, 1.f);

	transition_rates r(model);
	r.generate_uniform();

	solver s(context, table, std::move(g), std::move(r), std::move(st));

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

TEST(solver, cohen)
{
	model_builder builder;
	auto model = builder.construct_model("data/EMT_cohen_ModNet.bnet");

	cu_context context;

	transition_table table(context, model);

	table.construct_table();

	transition_graph g(context, table.rows, table.cols, table.indptr);

	g.find_terminals();

	initial_state st(
		model.nodes,
		{ "ECMicroenv", "DNAdamage", "Metastasis", "Migration", "Invasion", "EMT", "Apoptosis", "Notch_pthw", "p53" },
		{ true, true, false, false, false, false, false, true, false }, 1.f);

	transition_rates r(model);
	r.generate_uniform();

	solver s(context, table, std::move(g), std::move(r), std::move(st));

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

	ASSERT_THAT(nonzero_indices, ::testing::ElementsAre(206719, 790915, 803203));
	ASSERT_THAT(nonzero_data, ::testing::Pointwise(::testing::FloatNear(128 * std::numeric_limits<float>::epsilon()),
												   { 0.66441368, 0.1986147, 0.13697163 }));
}

TEST(solver, zanudo)
{
	model_builder builder;
	auto model = builder.construct_model("data/zanudo_expanded.bnet");

	cu_context context;

	transition_table table(context, model);

	table.construct_table();

	transition_graph g(context, table.rows, table.cols, table.indptr);

	g.find_terminals();

	initial_state st(model.nodes, { "Alpelisib", "Everolimus", "PIM", "Proliferation", "Apoptosis" },
					 { false, true, false, false, false }, 1.f);

	transition_rates r(model);
	r.generate_uniform();

	solver s(context, table, std::move(g), std::move(r), std::move(st));

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

	ASSERT_THAT(nonzero_indices, ::testing::ElementsAre(2229634, 2229642, 4326786, 4326794));
	ASSERT_THAT(nonzero_data, ::testing::Pointwise(::testing::FloatNear(128 * std::numeric_limits<float>::epsilon()),
												   { 0.28328907, 0.29057509, 0.21671093, 0.20942491 }));
}

TEST(solver, mammal)
{
	model_builder builder;
	auto model = builder.construct_model("data/mammalian_cc.bnet");

	cu_context context;

	transition_table table(context, model);

	table.construct_table();

	transition_graph g(context, table.rows, table.cols, table.indptr);

	g.find_terminals();

	initial_state st(model.nodes, { "CycE", "CycA", "CycB", "Cdh1", "Rb_b1", "Rb_b2", "p27_b1", "p27_b2" },
					 { false, false, false, true, true, true, true, true }, 1.f);

	transition_rates r(model);
	r.generate_uniform();

	solver s(context, table, std::move(g), std::move(r), std::move(st));

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

	std::vector<index_t> expected_indices = {
		1414, 4129, 4131, 4137, 4145, 4147, 4153, 4155, 4193, 4195, 4201, 4203, 4209, 4211, 4217, 4219, 4257,
		4259, 4265, 4267, 4273, 4275, 4281, 4283, 4321, 4323, 4329, 4331, 4337, 4339, 4345, 4347, 4705, 4707,
		4713, 4715, 4721, 4723, 4729, 4731, 4833, 4835, 4841, 4843, 4849, 4851, 4857, 4859, 5121, 5123, 5129,
		5131, 5137, 5139, 5145, 5147, 5153, 5155, 5161, 5169, 5171, 5177, 5179, 5249, 5251, 5257, 5259, 5265,
		5267, 5273, 5275, 5281, 5283, 5289, 5291, 5297, 5299, 5305, 5307, 6145, 6147, 6153, 6155, 6161, 6163,
		6169, 6171, 6177, 6179, 6185, 6187, 6193, 6195, 6201, 6203, 6209, 6211, 6217, 6219, 6225, 6227, 6233,
		6235, 6241, 6243, 6249, 6251, 6257, 6259, 6265, 6267, 6273, 6275, 6281, 6283, 6289, 6291, 6297, 6299,
		6305, 6307, 6313, 6315, 6321, 6323, 6329, 6331, 6337, 6339, 6345, 6347, 6353, 6355, 6361, 6363, 6369,
		6371, 6377, 6379, 6385, 6387, 6393, 6395, 6657, 6659, 6665, 6667, 6673, 6675, 6681, 6683, 6689, 6691,
		6697, 6699, 6705, 6707, 6713, 6715, 6721, 6723, 6729, 6731, 6737, 6739, 6745, 6747, 6753, 6755, 6761,
		6763, 6769, 6771, 6777, 6779, 6785, 6787, 6793, 6795, 6801, 6803, 6809, 6811, 6817, 6819, 6825, 6827,
		6833, 6835, 6841, 6843, 6849, 6851, 6857, 6859, 6865, 6867, 6873, 6875, 6881, 6883, 6889, 6891, 6897,
		6899, 6905, 6907, 7169, 7171, 7177, 7179, 7185, 7187, 7193, 7195, 7233, 7235, 7241, 7243, 7249, 7251,
		7257, 7259, 7297, 7299, 7305, 7307, 7313, 7315, 7321, 7323, 7361, 7363, 7369, 7371, 7377, 7379, 7385,
		7387, 7681, 7683, 7689, 7691, 7697, 7699, 7705, 7707, 7745, 7747, 7753, 7755, 7761, 7763, 7769, 7771,
		7809, 7811, 7817, 7819, 7825, 7827, 7833, 7835, 7873, 7875, 7881, 7883, 7889, 7891, 7897, 7899
	};

	std::vector<double> expected_probs = {
		5.00000000e-01, 8.15564577e-03, 9.23763705e-06, 2.45418759e-05, 5.78264819e-03, 1.60878332e-05, 6.92910955e-03,
		1.64874938e-05, 5.45880790e-03, 3.00486805e-05, 8.21152370e-06, 1.38403649e-06, 2.70515093e-03, 8.76189341e-06,
		2.31847497e-03, 7.94113460e-06, 6.17476638e-05, 1.75789740e-04, 4.15210946e-05, 2.76807297e-05, 1.76759530e-05,
		2.46136832e-05, 4.38933506e-05, 3.84891436e-05, 1.85588945e-05, 7.07624744e-05, 6.92018243e-06, 6.92018243e-06,
		5.56773193e-06, 1.10186058e-05, 1.01627066e-05, 1.38930082e-05, 7.44962655e-03, 4.05001250e-05, 3.48686100e-06,
		8.65022803e-07, 1.93947949e-03, 6.75494967e-06, 1.16493949e-03, 5.03046899e-06, 7.35217848e-06, 4.33315971e-05,
		1.38403649e-06, 2.07605473e-06, 2.11360442e-06, 6.47248660e-06, 2.88668577e-06, 6.28524955e-06, 4.32380913e-04,
		8.97219904e-04, 2.94502511e-04, 1.30210316e-03, 1.19485873e-04, 3.44675224e-06, 4.11012378e-02, 2.40079945e-02,
		1.04948638e-02, 1.16250779e-05, 9.81675036e-05, 1.03850712e-02, 2.32501559e-05, 2.07024059e-02, 2.74608314e-05,
		8.75499190e-04, 5.19251437e-03, 8.30421891e-04, 1.28527498e-02, 7.13693062e-08, 1.72647470e-06, 5.04177160e-04,
		1.80931607e-02, 1.26045944e-04, 2.37537404e-04, 2.07605473e-04, 6.92018243e-05, 4.44864146e-05, 4.22896363e-05,
		1.77945658e-04, 8.23824942e-05, 1.95793264e-03, 1.13380811e-03, 6.57715532e-04, 7.43861914e-04, 2.93453160e-04,
		8.06004360e-06, 1.84041570e-03, 7.37917603e-04, 2.85292822e-02, 4.75825831e-05, 2.36534369e-04, 1.78793584e-07,
		6.09768871e-03, 3.20448208e-05, 4.91101782e-03, 2.96156620e-05, 1.40262143e-03, 5.14902344e-04, 9.33103187e-04,
		5.86304685e-04, 9.85376593e-05, 2.12083549e-06, 1.93212372e-03, 8.97597410e-04, 4.15674105e-02, 2.40670821e-04,
		4.14804236e-04, 8.29062366e-06, 6.84867751e-03, 3.94519597e-05, 4.83280877e-03, 3.38580405e-05, 1.08797819e-03,
		1.33092151e-03, 5.69520777e-04, 1.34332525e-03, 2.10275757e-07, 2.01222987e-06, 1.70235493e-04, 7.07583450e-04,
		1.26416941e-04, 4.90714493e-04, 1.53257610e-04, 6.12054699e-05, 2.81121608e-05, 4.20734055e-05, 9.26277244e-05,
		6.63486830e-05, 2.17602647e-04, 3.87233404e-04, 1.31187237e-04, 4.65436473e-04, 3.50459595e-08, 4.09455165e-07,
		6.02915552e-05, 4.11240311e-04, 7.51678339e-05, 3.55967795e-04, 5.82730060e-05, 3.15996646e-05, 2.22546602e-05,
		4.36911242e-05, 5.53387480e-05, 5.57267012e-05, 9.49621003e-03, 2.56923978e-03, 1.98836055e-03, 1.18392401e-03,
		1.16554232e-03, 3.82879881e-05, 3.66192532e-03, 7.54247401e-04, 1.39918307e-02, 5.37039147e-05, 2.71669067e-05,
		7.15174336e-07, 1.44155443e-03, 8.35756179e-06, 6.93732627e-04, 5.83201574e-06, 3.23236771e-02, 6.88757186e-04,
		4.18425833e-04, 1.77380536e-04, 3.18106520e-03, 1.73231505e-04, 2.44052134e-03, 4.27520810e-04, 2.64612362e-02,
		1.52039008e-04, 1.07952452e-04, 3.57587168e-06, 3.62042815e-03, 2.04525420e-05, 2.04507452e-03, 1.58891776e-05,
		2.39376985e-03, 1.76896601e-03, 1.10191016e-03, 1.38567154e-03, 1.26165454e-06, 9.85087357e-06, 2.81446410e-04,
		6.07177378e-04, 9.71644765e-06, 8.52289359e-05, 1.98856808e-06, 2.67807001e-06, 2.14496570e-06, 7.14568943e-06,
		3.12444458e-06, 6.72371100e-06, 6.78912744e-05, 2.48808580e-04, 2.35216796e-05, 9.95363728e-05, 5.42496156e-06,
		4.08470239e-05, 1.94893779e-05, 1.37839598e-04, 2.50158092e-05, 1.50917665e-04, 9.94284041e-06, 8.72371194e-06,
		7.60038389e-06, 1.97140810e-05, 1.36336548e-05, 2.10923294e-05, 1.17765686e-03, 1.35861214e-03, 7.45629131e-04,
		1.41248391e-03, 2.35453625e-04, 8.61378202e-06, 1.61750778e-02, 4.60928383e-03, 1.62675778e-05, 2.15446767e-05,
		2.14806725e-05, 3.47133253e-05, 3.93683970e-09, 1.38432053e-08, 3.43019301e-05, 5.76113851e-05, 1.31854610e-03,
		3.41806880e-03, 1.32126397e-03, 5.52771042e-03, 2.85477225e-07, 5.10805480e-06, 6.82038221e-04, 4.73450724e-03,
		4.35216977e-05, 1.07709540e-04, 3.49417870e-05, 1.52021950e-04, 5.84099325e-09, 8.30592316e-08, 2.38097958e-05,
		1.95718372e-04, 2.50097362e-03, 1.73421299e-03, 1.12566345e-03, 1.35463522e-03, 4.04004519e-04, 2.12731865e-05,
		8.02795538e-03, 2.26743374e-03, 1.03679929e-04, 1.11152398e-04, 1.33196454e-05, 2.01052159e-05, 2.39867836e-05,
		2.87533049e-05, 3.84084167e-05, 5.38557078e-05, 1.64619606e-03, 2.71524026e-03, 1.29515412e-03, 3.40825695e-03,
		1.21126937e-06, 1.80514529e-05, 5.30519924e-04, 2.35085597e-03, 4.26334526e-05, 2.63053998e-04, 1.16926933e-05,
		6.58127540e-05, 4.79469233e-06, 6.11436685e-05, 1.37479668e-05, 1.37706230e-04
	};

	ASSERT_EQ(nonzero_indices.size(), expected_indices.size());
	ASSERT_EQ(nonzero_data.size(), expected_probs.size());

	for (index_t i = 0; i < nonzero_indices.size(); i++)
	{
		ASSERT_EQ(nonzero_indices[i], expected_indices[i]);
	}

	for (index_t i = 0; i < nonzero_indices.size(); i++)
	{
		ASSERT_NEAR(nonzero_data[i], expected_probs[i], 1e-07f);
	}
}

TEST(rates, toy)
{
	model_builder builder;
	auto model = builder.construct_model("data/toy.bnet");

	cu_context context;

	transition_table table(context, model);

	table.construct_table();

	transition_graph g(context, table.rows, table.cols, table.indptr);

	g.find_terminals();

	thrust::host_vector<index_t> vertices = g.reordered_vertices;
	thrust::host_vector<index_t> offsets(g.sccs_offsets.begin(), g.sccs_offsets.begin() + g.terminals_count + 1);

	initial_state st(model.nodes);

	transition_rates r(model);
	r.generate_uniform({}, { { "C", 2.f } });

	solver s(context, table, std::move(g), std::move(r), std::move(st));

	s.solve();

	thrust::host_vector<float> final_state = s.final_state;

	thrust::host_vector<index_t> nonzero_indices(final_state.size());
	thrust::host_vector<float> nonzero_data(final_state.size());

	auto i_end = thrust::copy_if(thrust::make_counting_iterator<index_t>(0),
								 thrust::make_counting_iterator<index_t>((index_t)final_state.size()),
								 final_state.begin(), nonzero_indices.begin(), thrust::identity<float>());
	nonzero_indices.resize(i_end - nonzero_indices.begin());

	auto d_end = thrust::copy(thrust::make_permutation_iterator(final_state.begin(), nonzero_indices.begin()),
							  thrust::make_permutation_iterator(final_state.begin(), nonzero_indices.end()),
							  nonzero_data.begin());
	nonzero_data.resize(d_end - nonzero_data.begin());

	ASSERT_THAT(nonzero_indices, ::testing::ElementsAre(1, 2, 4));
	ASSERT_THAT(nonzero_data, ::testing::Pointwise(::testing::FloatNear(128 * std::numeric_limits<float>::epsilon()),
												   { 0.5000000000000009, 0.22916666666666666, 0.2708333333333333 }));
}
