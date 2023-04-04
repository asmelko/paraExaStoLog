#include <thrust/execution_policy.h>
#include <thrust/for_each.h>

#include "bpplib/system/file.hpp"
#include "bpplib/system/mmap_file.hpp"
#include "persistent_solution.h"
#include "solver.h"

template <typename vec_t>
void write_size(bpp::File& f, const vec_t& v)
{
	size_t size = v.size();
	f.write(&size, 1);
}

template <typename T>
void write_content(bpp::File& f, const thrust::device_vector<T>& v)
{
	thrust::host_vector<T> h = v;

	write_content(f, h);
}

template <typename T>
void write_content(bpp::File& f, const thrust::host_vector<T>& h)
{
	f.write(h.data(), h.size());
}

void persistent_solution::serialize(const std::string& file, const solver& s)
{
	bpp::File f(file);
	f.open("wb");

	// first write sizes
	{
		write_size(f, s.rows_);
		write_size(f, s.cols_);
		write_size(f, s.indptr_);

		write_size(f, s.ordered_vertices_);
		write_size(f, s.terminals_offsets_);
		write_size(f, s.nonterminals_offsets_);

		write_size(f, s.rates_);

		write_size(f, s.solution_term.indptr);
		write_size(f, s.solution_term.indices);
		write_size(f, s.solution_term.data);
		write_size(f, s.solution_nonterm.indptr);
		write_size(f, s.solution_nonterm.indices);
		write_size(f, s.solution_nonterm.data);
	}

	// second write contents
	{
		write_content(f, s.rows_);
		write_content(f, s.cols_);
		write_content(f, s.indptr_);

		write_content(f, s.ordered_vertices_);
		write_content(f, s.terminals_offsets_);
		write_content(f, s.nonterminals_offsets_);

		write_content(f, s.rates_);

		write_content(f, s.solution_term.indptr);
		write_content(f, s.solution_term.indices);
		write_content(f, s.solution_term.data);
		write_content(f, s.solution_nonterm.indptr);
		write_content(f, s.solution_nonterm.indices);
		write_content(f, s.solution_nonterm.data);
	}

	f.close();
}

template <typename T, typename vec_t>
T* read_content(T* data, vec_t& v)
{
	v.assign(data, data + v.size());

	return data + v.size();
}

persistent_data persistent_solution::deserialize(const std::string& file)
{
	persistent_data d;

	bpp::MMapFile f;
	f.open(file);

	void* data = f.getData();

	size_t* sizes = reinterpret_cast<size_t*>(data);

	if (f.length() < 13 * sizeof(size_t))
		throw std::runtime_error("Persistend data file is corrupted");

	d.rows.resize(*sizes++);
	d.cols.resize(*sizes++);
	d.indptr.resize(*sizes++);

	d.ordered_vertices.resize(*sizes++);
	d.terminals_offsets.resize(*sizes++);
	d.nonterminals_offsets.resize(*sizes++);

	d.rates.resize(*sizes++);

	d.solution_term.indptr.resize(*sizes++);
	d.solution_term.indices.resize(*sizes++);
	d.solution_term.data.resize(*sizes++);
	d.solution_nonterm.indptr.resize(*sizes++);
	d.solution_nonterm.indices.resize(*sizes++);
	d.solution_nonterm.data.resize(*sizes++);

	size_t idx_size = d.rows.size() + d.cols.size() + d.indptr.size() + d.ordered_vertices.size()
					  + d.terminals_offsets.size() + d.nonterminals_offsets.size() + d.solution_term.indptr.size()
					  + d.solution_term.indices.size() + d.solution_nonterm.indptr.size()
					  + d.solution_nonterm.indices.size();
	size_t real_size = d.rates.size() + d.solution_term.data.size() + d.solution_nonterm.data.size();

	size_t total_size = idx_size * sizeof(index_t) + real_size * sizeof(real_t) + 13 * sizeof(size_t);

	if (f.length() != total_size)
		throw std::runtime_error("Persistend data file is corrupted");

	index_t* idx_data = reinterpret_cast<index_t*>(sizes);

	idx_data = read_content(idx_data, d.rows);
	idx_data = read_content(idx_data, d.cols);
	idx_data = read_content(idx_data, d.indptr);

	idx_data = read_content(idx_data, d.ordered_vertices);
	idx_data = read_content(idx_data, d.terminals_offsets);
	idx_data = read_content(idx_data, d.nonterminals_offsets);

	real_t* real_data = reinterpret_cast<real_t*>(idx_data);
	real_data = read_content(real_data, d.rates);

	idx_data = reinterpret_cast<index_t*>(real_data);

	idx_data = read_content(idx_data, d.solution_term.indptr);
	idx_data = read_content(idx_data, d.solution_term.indices);

	real_data = reinterpret_cast<real_t*>(idx_data);
	real_data = read_content(real_data, d.solution_term.data);

	idx_data = reinterpret_cast<index_t*>(real_data);

	idx_data = read_content(idx_data, d.solution_nonterm.indptr);
	idx_data = read_content(idx_data, d.solution_nonterm.indices);

	real_data = reinterpret_cast<real_t*>(idx_data);
	read_content(real_data, d.solution_nonterm.data);

	f.close();

	return d;
}

template <typename T>
bool compare(const thrust::host_vector<T>& l, const thrust::host_vector<T>& r)
{
	if (l.size() != r.size())
		return false;

	for (size_t i = 0; i < l.size(); i++)
		if (l[i] != r[i])
			return false;
	return true;
}

template <typename T>
bool compare(const thrust::device_vector<T>& l, const thrust::device_vector<T>& r)
{
	thrust::host_vector<T> hl = l;
	thrust::host_vector<T> hr = r;

	return compare(hl, hr);
}

bool persistent_solution::has_incompatible_zero_rates(const persistent_data& stored, const d_datvec& new_rates)
{
	thrust::device_vector<bool> are_compatible(1, true);

	if (stored.rates.size() != new_rates.size())
		throw std::runtime_error("symbolic data dimensions do not match model dimensions");

	thrust::for_each_n(thrust::make_counting_iterator(0), new_rates.size(),
					   [old_rates = stored.rates.data().get(), new_rates = new_rates.data().get(),
						are_compatible = are_compatible.data().get()] __device__(size_t i) {
						   if ((old_rates[i] == 0.f && new_rates[i] != 0.f)
							   || (new_rates[i] == 0.f && old_rates[i] != 0.f))
							   are_compatible[0] = false;
					   });

	return are_compatible[0];
}

bool persistent_solution::are_same(const persistent_data& stored, const d_datvec& new_rates)
{
	thrust::device_vector<bool> same(1, true);

	if (stored.rates.size() != new_rates.size())
		throw std::runtime_error("symbolic data dimensions do not match model dimensions");

	thrust::for_each_n(
		thrust::make_counting_iterator(0), new_rates.size(),
		[l = stored.rates.data().get(), r = new_rates.data().get(), same = same.data().get()] __device__(size_t i) {
			if (l[0] != r[0])
				same[0] = false;
		});

	return same[0];
}

bool persistent_solution::check_are_equal(const solver& s, const persistent_data& d)
{
	bool are_equal = true;

	are_equal &= compare(s.rows_, d.rows);
	are_equal &= compare(s.cols_, d.cols);
	are_equal &= compare(s.indptr_, d.indptr);

	are_equal &= compare(s.ordered_vertices_, d.ordered_vertices);
	are_equal &= compare(s.terminals_offsets_, d.terminals_offsets);
	are_equal &= compare(s.nonterminals_offsets_, d.nonterminals_offsets);

	are_equal &= compare(s.rates_, d.rates);

	are_equal &= compare(s.solution_term.indptr, d.solution_term.indptr);
	are_equal &= compare(s.solution_term.indices, d.solution_term.indices);
	are_equal &= compare(s.solution_term.data, d.solution_term.data);
	are_equal &= compare(s.solution_nonterm.indptr, d.solution_nonterm.indptr);
	are_equal &= compare(s.solution_nonterm.indices, d.solution_nonterm.indices);
	are_equal &= compare(s.solution_nonterm.data, d.solution_nonterm.data);

	return are_equal;
}
