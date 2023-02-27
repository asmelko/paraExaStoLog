#include <thrust/partition.h>

#include "solver.h"

struct equals_ftor : public thrust::unary_function<index_t, bool>
{
	index_t value;

	equals_ftor(index_t value) : value(value) {}

	__host__ __device__ bool operator()(index_t x) const { return x == value; }
};

solver::solver(const transition_table& t, transition_graph g, initial_state s)
	: initial_state_(std::move(s.state)),
	  labels_(std::move(g.labels)),
	  terminals_(std::move(g.terminals)),
	  rows_(t.rows),
	  cols_(t.cols),
	  idxptr_(t.indptr)
{}

void solver::solve_terminal_part()
{
	d_idxvec sccs(thrust::make_counting_iterator<index_t>(0), thrust::make_counting_iterator<index_t>(labels_.size()));

	std::vector<size_t> terminals_offsets;
	terminals_offsets.reserve(terminals_.size());

	auto partition_point = sccs.begin();
	for (auto it = terminals_.begin(); it != terminals_.end(); it++)
	{
		partition_point = thrust::stable_partition(partition_point, sccs.end(), labels_.begin() + terminals_offsets.back(), equals_ftor(*it));
		thrust::stable_partition(labels_.begin() + terminals_offsets.back(), labels_.end(), equals_ftor(*it));

		terminals_offsets.push_back(partition_point - sccs.begin());
	}

}

void solver::solve() {}