#include <thrust/set_operations.h>

#include <thrust/host_vector.h>

#include "transition_table.cuh"

struct transition_ftor : public thrust::unary_function<index_t, index_t>
{
	index_t free_vars[32] = { 0 };
	index_t fixed = 0;

	transition_ftor(const std::vector<index_t>& free_v, index_t fixed)
	{
		for (size_t i = 0; i < free_v.size(); i++)
			free_vars[i] = 1ULL << free_v[i];

		this->fixed = fixed;
	}
	__host__ __device__ index_t operator()(index_t x) const
	{
		index_t ret = fixed;
		for (int i = 0; i < 32; i++)
			ret += (x >> i) % 2 ? free_vars[i] : 0;

		return ret;
	}
};

void print(const char* msg, const d_idxvec& v)
{
	thrust::host_vector<index_t> h = v;

	std::cout << msg;
	for (auto t : h)
		std::cout << t << " ";
	std::cout << std::endl;
}

d_idxvec generate_transitions(const std::vector<clause_t>& clauses)
{
	d_idxvec transitions;

	for (const auto& c : clauses)
	{
		auto free_vars = c.get_free_variables();
		auto fixed = c.get_fixed_part();

		auto c_b = thrust::make_counting_iterator(0);
		auto c_e = c_b + (1ULL << free_vars.size());

		auto b = thrust::make_transform_iterator(c_b, transition_ftor(free_vars, fixed));
		auto e = thrust::make_transform_iterator(c_e, transition_ftor(free_vars, fixed));

		d_idxvec single_clause_transitions(b, e);

		d_idxvec tmp(transitions.size() + single_clause_transitions.size());

		auto tmp_end = thrust::set_union(transitions.begin(), transitions.end(), single_clause_transitions.begin(), single_clause_transitions.end(), tmp.begin());

		tmp.resize(tmp_end - tmp.begin());

		std::swap(tmp, transitions);
	}

	return transitions;
}

struct flip_ftor : public thrust::unary_function<index_t, index_t>
{
	index_t mask;

	flip_ftor(index_t mask) : mask(mask) {}
	__host__ __device__ index_t operator()(index_t x) const { return x ^ mask; }
};

void transition_table::construct_table()
{
	auto p = compute_rows_and_cols();
	auto& trans_src = p.first;
	auto& trans_dst = p.second;

	int matrix_size = (int)(1ULL << model_.nodes.size());

	size_t buffersize;
	CHECK_CUSPARSE(cusparseXcsrsort_bufferSizeExt(context_.cusparse_handle, matrix_size, matrix_size,
												  (int)trans_src.size(), trans_dst.data().get(), trans_src.data().get(),
												  &buffersize));

	void* d_buffer;
	cudaMalloc( &d_buffer, buffersize);

	d_idxvec P(trans_src.size());
	CHECK_CUSPARSE(cusparseCreateIdentityPermutation(context_.cusparse_handle, P.size(), P.data().get()));

	CHECK_CUSPARSE(cusparseXcoosortByRow(context_.cusparse_handle, matrix_size, matrix_size, (int)trans_src.size(),
										 trans_dst.data().get(), trans_src.data().get(), P.data().get(), d_buffer));

	indices = std::move(trans_src);
	indptr = d_idxvec(matrix_size + 1);

	CHECK_CUSPARSE(cusparseXcoo2csr(context_.cusparse_handle, trans_dst.data().get(), (int)trans_dst.size(), matrix_size,
									indptr.data().get(), CUSPARSE_INDEX_BASE_ZERO));
}

std::pair<d_idxvec, d_idxvec> transition_table::compute_rows_and_cols()
{
	std::vector<d_idxvec> ups, downs;

	for (const auto& f : model_.dnfs)
	{
		ups.emplace_back(generate_transitions(f.activations));
		downs.emplace_back(generate_transitions(f.deactivations));
	}

	size_t transitions_count = 0;

	for (size_t i = 0; i < ups.size(); i++)
		transitions_count += ups[i].size() + downs[i].size();

	d_idxvec trans_src(transitions_count), trans_dst(transitions_count);

	auto src_begin = trans_src.begin();
	for (size_t i = 0; i < ups.size(); i++)
	{
		src_begin = thrust::copy(ups[i].begin(), ups[i].end(), src_begin);
		src_begin = thrust::copy(downs[i].begin(), downs[i].end(), src_begin);
	}

	auto dst_begin = trans_dst.begin();
	for (size_t i = 0; i < ups.size(); i++)
	{
		dst_begin = thrust::copy(thrust::make_transform_iterator(ups[i].begin(), flip_ftor(1ULL << i)),
								 thrust::make_transform_iterator(ups[i].end(), flip_ftor(1ULL << i)), dst_begin);

		dst_begin = thrust::copy(thrust::make_transform_iterator(downs[i].begin(), flip_ftor(1ULL << i)),
								 thrust::make_transform_iterator(downs[i].end(), flip_ftor(1ULL << i)), dst_begin);
	}

	return std::make_pair(std::move(trans_src), std::move(trans_dst));
}

transition_table::transition_table(cu_context& context, model_t model) : context_(context), model_(std::move(model)) {}