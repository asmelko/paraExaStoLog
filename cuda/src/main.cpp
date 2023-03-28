#include "model.h"
#include "solver.h"

int main(int argc, char** argv)
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

	solver s(context, table, std::move(g), std::move(st));

	s.solve();
}