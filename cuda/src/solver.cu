#include <thrust/partition.h>

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
	  submatrix_vertex_mapping_(ordered_vertices_.size())
{}

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
	term_indptr = terminals_offsets_;
	term_rows.resize(terminals_offsets_.back());
	term_data.resize(terminals_offsets_.back());

	thrust::copy(ordered_vertices_.begin(), ordered_vertices_.begin() + terminals_offsets_.back(), term_rows.begin());

	for (size_t terminal_scc_idx = 1; terminal_scc_idx < terminals_offsets_.size(); terminal_scc_idx++)
	{
		size_t scc_size = terminals_offsets_[terminal_scc_idx] - terminals_offsets_[terminal_scc_idx - 1];

		if (scc_size == 1)
		{
			term_data[terminals_offsets_[terminal_scc_idx - 1]] = 1;
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

		thrust::transform(minors.begin(), minors.end(), term_data.begin() + terminals_offsets_[terminal_scc_idx - 1],
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
		nonterm_indptr = term_indptr;
		nonterm_cols = term_rows;
		nonterm_data = d_datvec(term_rows.size(), 1.f);

		return;
	}

	// -U
	sparse_csr_matrix U;
	{
		U.indptr = term_indptr;
		U.indices.resize(term_rows.size());

		thrust::copy(thrust::make_counting_iterator<intptr_t>(0),
					 thrust::make_counting_iterator<intptr_t>(terminal_vertices_n),
					 thrust::make_permutation_iterator(submatrix_vertex_mapping_.begin(), ordered_vertices_.begin()));

		thrust::transform(ordered_vertices_.begin(), ordered_vertices_.begin() + terminals_offsets_.back(),
						  U.indices.begin(),
						  [map = submatrix_vertex_mapping_.data().get()] __device__(index_t x) { return map[x]; });

		U.data.assign(term_rows.size(), -1.f);
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

	sparse_csr_matrix X = solve_system(context_, sparse_cast<cs_kind::CSR>(std::move(N)), nonterminals_offsets_, A);

	nonterm_indptr.resize(U.indptr.size());
	index_t nonterm_nnz = U.indptr.back() + X.indptr.back();
	nonterm_cols.resize(nonterm_nnz);
	nonterm_data.resize(nonterm_nnz);

	thrust::transform(
		thrust::make_zip_iterator(U.indptr.begin(), X.indptr.begin()),
		thrust::make_zip_iterator(U.indptr.end(), X.indptr.end()), nonterm_indptr.begin(),
		[] __device__(thrust::tuple<index_t, index_t> x) { return thrust::get<0>(x) + thrust::get<1>(x); });


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
		run_hstack(nonterm_indptr.data().get(), nonterm_cols.data().get(), nonterm_data.data().get(),
				   U.indptr.data().get(), X.indptr.data().get(), term_rows.data().get(), X.indices.data().get(),
				   U.data.data().get(), X.data.data().get(), nonterm_indptr.size() - 1);

		CHECK_CUDA(cudaDeviceSynchronize());
	}
}

void solver::compute_final_states()
{
	auto y = mvmul(context_.cusparse_handle, nonterm_indptr, nonterm_cols, nonterm_data, cs_kind::CSR,
				   terminals_offsets_.size() - 1, ordered_vertices_.size(), initial_state_);

	final_state = mvmul(context_.cusparse_handle, term_indptr, term_rows, term_data, cs_kind::CSC,
						ordered_vertices_.size(), terminals_offsets_.size() - 1, y);
}

void solver::solve()
{
	solve_terminal_part();
	solve_nonterminal_part();

	compute_final_states();
}
