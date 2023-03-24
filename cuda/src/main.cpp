#include "model.h"
#include "solver.h"

int main(int argc, char** argv)
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

	solver s(context, table, std::move(g), std::move(st));

	s.solve();
}