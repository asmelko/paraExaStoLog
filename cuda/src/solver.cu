#include <thrust/partition.h>

#include "diagnostics.h"
#include "kernels/kernels.h"
#include "linear_system_solve.h"
#include "solver.h"
#include "utils.h"

solver::solver(cu_context& context, const transition_table& t, transition_graph g, transition_rates r, initial_state s)
	: context_(context),
	  initial_state_(std::move(s.state)),
	  rows_(t.rows),
	  cols_(t.cols),
	  indptr_(t.indptr),
	  ordered_vertices_(std::move(g.reordered_vertices)),
	  terminals_offsets_(g.sccs_offsets.begin(), g.sccs_offsets.begin() + g.terminals_count + 1),
	  nonterminals_offsets_(g.sccs_offsets.begin() + g.terminals_count, g.sccs_offsets.end()),
	  rates_(std::move(r.rates)),
	  submatrix_vertex_mapping_(ordered_vertices_.size()),
	  symbolic_loaded_(false),
	  can_refactor_(false),
	  recompute_needed(false)
{}

solver::solver(cu_context& context, persistent_data& persisted, transition_rates r, initial_state s)
	: context_(context),
	  initial_state_(std::move(s.state)),
	  rows_(persisted.rows),
	  cols_(persisted.cols),
	  indptr_(persisted.indptr),
	  ordered_vertices_(std::move(persisted.ordered_vertices)),
	  terminals_offsets_(std::move(persisted.terminals_offsets)),
	  nonterminals_offsets_(std::move(persisted.nonterminals_offsets)),
	  rates_(std::move(r.rates)),
	  submatrix_vertex_mapping_(ordered_vertices_.size()),
	  symbolic_loaded_(true)
{
	recompute_needed = !persistent_solution::are_same(persisted, rates_);
	can_refactor_ = persisted.n_inverse.nnz() != 0;

	if (!recompute_needed)
	{
		solution_term = std::move(persisted.solution_term);
		solution_nonterm = std::move(persisted.solution_nonterm);
	}

	if (can_refactor_)
	{
		n_inverse_ = std::move(persisted.n_inverse);
	}
}

void solver::take_submatrix(index_t n, d_idxvec::const_iterator vertices_subset_begin, sparse_csc_matrix& m,
							bool mapping_prefilled)
{
	m.indptr.resize(n + 1);
	m.indptr[0] = 0;

	// this creates indptr of scc in CSC
	{
		auto scc_begins_b = thrust::make_permutation_iterator(indptr_.begin(), vertices_subset_begin);
		auto scc_begins_e = thrust::make_permutation_iterator(indptr_.begin(), vertices_subset_begin + n);

		auto scc_ends_b = thrust::make_permutation_iterator(indptr_.begin() + 1, vertices_subset_begin);
		auto scc_ends_e = thrust::make_permutation_iterator(indptr_.begin() + 1, vertices_subset_begin + n);

		// first get sizes of each col - also add 1 for diagonal part
		thrust::transform(
			thrust::make_zip_iterator(scc_begins_b, scc_ends_b), thrust::make_zip_iterator(scc_begins_e, scc_ends_e),
			m.indptr.begin() + 1,
			[] __device__(thrust::tuple<index_t, index_t> x) { return 1 + thrust::get<1>(x) - thrust::get<0>(x); });

		thrust::inclusive_scan(m.indptr.begin(), m.indptr.end(), m.indptr.begin());
	}

	index_t nnz = m.indptr.back();
	m.indices.resize(nnz);
	m.data.assign(nnz, 1.f);

	// this creates rows and data of scc
	{
		run_scatter_rows_data(m.indptr.data().get(), m.indices.data().get(), m.data.data().get(), rows_.data().get(),
							  indptr_.data().get(), (&*vertices_subset_begin).get(), n, rates_.data().get());

		CHECK_CUDA(cudaDeviceSynchronize());
	}

	// finally we transform rows so they start from 0
	{
		if (!mapping_prefilled)
		{
			// create map for scc vertices so they start from 0
			thrust::copy(thrust::make_counting_iterator<intptr_t>(0), thrust::make_counting_iterator<intptr_t>(n),
						 thrust::make_permutation_iterator(submatrix_vertex_mapping_.begin(), vertices_subset_begin));
		}

		thrust::transform(m.indices.begin(), m.indices.end(), m.indices.begin(),
						  [map = submatrix_vertex_mapping_.data().get()] __device__(index_t x) { return map[x]; });
	}
}

void solver::solve_terminal_part()
{
	print_terminal_info(terminals_offsets_);

	solution_term.indptr = terminals_offsets_;
	solution_term.indices.resize(terminals_offsets_.back());
	solution_term.data.resize(terminals_offsets_.back());

	thrust::copy(ordered_vertices_.begin(), ordered_vertices_.begin() + terminals_offsets_.back(),
				 solution_term.indices.begin());

	for (size_t terminal_scc_idx = 1; terminal_scc_idx < terminals_offsets_.size(); terminal_scc_idx++)
	{
		size_t scc_size = terminals_offsets_[terminal_scc_idx] - terminals_offsets_[terminal_scc_idx - 1];

		if constexpr (diags_enabled)
		{
			printf("\r                                                                        ");
			printf("\rSolving (terminal part): %i/%i with size %i", (index_t)terminal_scc_idx,
				   (index_t)terminals_offsets_.size() - 1, (index_t)scc_size);

			if (terminal_scc_idx == terminals_offsets_.size() - 1)
				printf("\n");
		}

		if (scc_size == 1)
		{
			solution_term.data[terminals_offsets_[terminal_scc_idx - 1]] = 1;
			continue;
		}

		sparse_csc_matrix scc;

		take_submatrix(scc_size, ordered_vertices_.begin() + terminals_offsets_[terminal_scc_idx - 1], scc);

		thrust::host_vector<double> h_minors(scc_size);
		for (size_t minor_i = 0; minor_i < scc_size; minor_i++)
		{
			// copy indptr
			d_idxvec minor_indptr = scc.indptr;
			d_idxvec minor_rows = scc.indices;
			d_datvec minor_data = scc.data;

			create_minor(context_.cusparse_handle, minor_indptr, minor_rows, minor_data, minor_i);

			host_sparse_csr_matrix h(minor_indptr, minor_rows, minor_data);

			h_minors[minor_i] = std::abs(host_det(context_.cusolver_handle, h));
		}

		thrust::device_vector<double> minors = h_minors;
		auto sum = thrust::reduce(minors.begin(), minors.end(), 0., thrust::plus<double>());

		thrust::transform(minors.begin(), minors.end(),
						  solution_term.data.begin() + terminals_offsets_[terminal_scc_idx - 1],
						  [sum] __device__(double x) { return x / sum; });
	}
}

void solver::break_NB(sparse_csc_matrix&& NB, sparse_csc_matrix& N, sparse_csc_matrix& B)
{
	const auto nnz = NB.data.size();
	const auto n = NB.indptr.size() - 1;
	const auto nonterm_n = ordered_vertices_.size() - terminals_offsets_.back();

	d_idxvec NB_decomp_indptr(nnz);

	// this decompresses indptr into cols
	CHECK_CUSPARSE(cusparseXcsr2coo(context_.cusparse_handle, NB.indptr.data().get(), nnz, n,
									NB_decomp_indptr.data().get(), CUSPARSE_INDEX_BASE_ZERO));

	auto part_point = thrust::stable_partition(
		thrust::make_zip_iterator(NB_decomp_indptr.begin(), NB.indices.begin(), NB.data.begin()),
		thrust::make_zip_iterator(NB_decomp_indptr.end(), NB.indices.end(), NB.data.end()),
		[point = nonterm_n] __device__(thrust::tuple<index_t, index_t, real_t> x) {
			return thrust::get<1>(x) < point;
		});

	auto N_size = thrust::get<0>(part_point.get_iterator_tuple()) - NB_decomp_indptr.begin();

	d_idxvec B_decomp_indptr(NB_decomp_indptr.begin() + N_size, NB_decomp_indptr.end());
	B.indices.assign(NB.indices.begin() + N_size, NB.indices.end());
	B.data.assign(NB.data.begin() + N_size, NB.data.end());

	NB_decomp_indptr.resize(N_size);
	N.indices = std::move(NB.indices);
	N.indices.resize(N_size);
	N.data = std::move(NB.data);
	N.data.resize(N_size);

	thrust::transform(B.indices.begin(), B.indices.end(), B.indices.begin(),
					  [point = nonterm_n] __device__(index_t x) { return x - point; });

	// this compresses rows back into indptr
	N.indptr.resize(nonterm_n + 1);
	CHECK_CUSPARSE(cusparseXcoo2csr(context_.cusparse_handle, NB_decomp_indptr.data().get(), NB_decomp_indptr.size(),
									nonterm_n, N.indptr.data().get(), CUSPARSE_INDEX_BASE_ZERO));

	B.indptr.resize(nonterm_n + 1);
	CHECK_CUSPARSE(cusparseXcoo2csr(context_.cusparse_handle, B_decomp_indptr.data().get(), B_decomp_indptr.size(),
									nonterm_n, B.indptr.data().get(), CUSPARSE_INDEX_BASE_ZERO));
}

void solver::solve_nonterminal_part()
{
	index_t n = ordered_vertices_.size();
	index_t terminal_vertices_n = terminals_offsets_.back();
	index_t nonterminal_vertices_n = n - terminal_vertices_n;

	if (nonterminal_vertices_n == 0)
	{
		solution_nonterm.indptr = solution_term.indptr;
		solution_nonterm.indices = solution_term.indices;
		solution_nonterm.data = d_datvec(solution_term.nnz(), 1.f);

		return;
	}

	// -U
	sparse_csr_matrix U;
	{
		U.indptr = solution_term.indptr;
		U.indices.resize(solution_term.nnz());

		thrust::copy(thrust::make_counting_iterator<intptr_t>(0),
					 thrust::make_counting_iterator<intptr_t>(terminal_vertices_n),
					 thrust::make_permutation_iterator(submatrix_vertex_mapping_.begin(), ordered_vertices_.begin()));

		thrust::transform(ordered_vertices_.begin(), ordered_vertices_.begin() + terminals_offsets_.back(),
						  U.indices.begin(),
						  [map = submatrix_vertex_mapping_.data().get()] __device__(index_t x) { return map[x]; });

		U.data.assign(solution_term.nnz(), -1.f);
	}

	// NB
	sparse_csc_matrix NB;
	{
		// custom mapping
		{
			thrust::copy(thrust::make_counting_iterator<intptr_t>(0),
						 thrust::make_counting_iterator<intptr_t>(nonterminal_vertices_n),
						 thrust::make_permutation_iterator(submatrix_vertex_mapping_.begin(),
														   ordered_vertices_.begin() + terminals_offsets_.back()));

			thrust::copy(
				thrust::make_counting_iterator<intptr_t>(nonterminal_vertices_n),
				thrust::make_counting_iterator<intptr_t>(n),
				thrust::make_permutation_iterator(submatrix_vertex_mapping_.begin(), ordered_vertices_.begin()));
		}

		// create vstack(N, B) matrix in csc
		take_submatrix(nonterminal_vertices_n, ordered_vertices_.begin() + terminals_offsets_.back(), NB, true);
	}

	// extract B
	sparse_csc_matrix N, B;
	break_NB(std::move(NB), N, B);

	auto B_t = csc2csr(context_.cusparse_handle, B, nonterminal_vertices_n, terminal_vertices_n, B.nnz());

	sparse_csr_matrix A =
		matmul(context_.cusparse_handle, U.indptr.data().get(), U.indices.data().get(), U.data.data().get(),
			   terminals_offsets_.size() - 1, terminal_vertices_n, U.indices.size(), B_t.indptr.data().get(),
			   B_t.indices.data().get(), B_t.data.data().get(), terminal_vertices_n, nonterminal_vertices_n, B.nnz());

	sparse_csr_matrix X = solve_system(context_, sparse_cast<cs_kind::CSR>(std::move(N)), nonterminals_offsets_, A,
									   n_inverse_, can_refactor_);

	solution_nonterm.indptr.resize(U.indptr.size());
	index_t nonterm_nnz = U.indptr.back() + X.indptr.back();
	solution_nonterm.indices.resize(nonterm_nnz);
	solution_nonterm.data.resize(nonterm_nnz);

	thrust::transform(U.indptr.begin(), U.indptr.end(), X.indptr.begin(),solution_nonterm.indptr.begin(), thrust::plus<index_t>());

	// -U back to U
	thrust::transform(U.data.begin(), U.data.end(), U.data.begin(), thrust::negate<real_t>());

	// nonterminal vertices from 0, ..., n_nt to actual indices
	{
		thrust::copy(ordered_vertices_.begin() + terminals_offsets_.back(), ordered_vertices_.end(),
					 submatrix_vertex_mapping_.begin());

		thrust::transform(X.indices.begin(), X.indices.end(), X.indices.begin(),
						  [map = submatrix_vertex_mapping_.data().get()] __device__(index_t x) { return map[x]; });
	}

	// hstack(U,X)
	{
		run_hstack(solution_nonterm.indptr.data().get(), solution_nonterm.indices.data().get(),
				   solution_nonterm.data.data().get(), U.indptr.data().get(), X.indptr.data().get(),
				   solution_term.indices.data().get(), X.indices.data().get(), U.data.data().get(), X.data.data().get(),
				   solution_nonterm.indptr.size() - 1);

		CHECK_CUDA(cudaDeviceSynchronize());
	}
}

void solver::compute_final_states()
{
	auto y = mvmul(context_.cusparse_handle, solution_nonterm.indptr, solution_nonterm.indices, solution_nonterm.data,
				   cs_kind::CSR, terminals_offsets_.size() - 1, ordered_vertices_.size(), initial_state_);

	final_state = mvmul(context_.cusparse_handle, solution_term.indptr, solution_term.indices, solution_term.data,
						cs_kind::CSC, ordered_vertices_.size(), terminals_offsets_.size() - 1, y);
}

void solver::solve()
{
	Timer t;
	if (!symbolic_loaded_ || (symbolic_loaded_ && recompute_needed))
	{
		t.Start();
		solve_terminal_part();
		t.Stop();

		diag_print("Solving (terminal part): ", t.Millisecs(), "ms");

		t.Start();
		solve_nonterminal_part();
		t.Stop();

		diag_print("Solving (nonterminal part): ", t.Millisecs(), "ms");
	}

	t.Start();
	compute_final_states();
	t.Stop();

	diag_print("Solving (final states): ", t.Millisecs(), "ms");
}

void solver::print_final_state(const std::vector<std::string>& model_nodes)
{
	index_t n = final_state.size();

	d_idxvec nonzero_indices(n);

	auto i_end = thrust::copy_if(thrust::make_counting_iterator<index_t>(0), thrust::make_counting_iterator<index_t>(n),
								 final_state.begin(), nonzero_indices.begin(), thrust::identity<float>());
	nonzero_indices.resize(i_end - nonzero_indices.begin());

	d_datvec nonzero_data(nonzero_indices.size());

	thrust::copy(thrust::make_permutation_iterator(final_state.begin(), nonzero_indices.begin()),
				 thrust::make_permutation_iterator(final_state.begin(), nonzero_indices.end()), nonzero_data.begin());

	h_idxvec nonzero_indices_h = std::move(nonzero_indices);
	h_datvec nonzero_data_h = std::move(nonzero_data);

	for (size_t i = 0; i < nonzero_indices_h.size(); i++)
	{
		index_t index = nonzero_indices_h[i];
		bool node_printed = false;
		for (index_t node_i = 0; node_i < model_nodes.size(); node_i++)
			if (index & (1 << node_i))
			{
				if (node_printed)
					std::cout << " --- ";
				std::cout << model_nodes[node_i];
				node_printed = true;
			}

		if (!node_printed)
			std::cout << "<nil>";

		std::cout << " " << nonzero_data_h[i] << std::endl;
	}
}
