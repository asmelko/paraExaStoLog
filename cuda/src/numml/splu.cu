#include <device_launch_parameters.h>

#include <thrust/execution_policy.h>
#include <thrust/set_operations.h>

#include "../solver.h"
#include "../utils.h"
#include "splu.h"


__global__ void cuda_kernel_splu_symbolic_fact(const index_t A_rows, const index_t* __restrict__ A_indices,
											   const index_t* __restrict__ A_indptr,
											   volatile index_t* __restrict__ As_nnz,
											   index_t * volatile* __restrict__ As_indices,
											   volatile index_t* __restrict__ degree)
{
	const index_t row = blockIdx.x * blockDim.x + threadIdx.x;


	if (row >= A_rows)
		return;
	
	index_t row_size;
	index_t* row_indices;

	// fill As_indices
	{
		const index_t row_indices_begin = A_indptr[row];
		row_size = A_indptr[row + 1] - row_indices_begin;

		row_indices = (index_t*)malloc(sizeof(index_t) * row_size);
		if (row_indices == NULL)
			printf("thread %i error row size %i \n", row, row_size);


		thrust::copy(thrust::seq, A_indices + row_indices_begin, A_indices + row_indices_begin + row_size, row_indices);
	}

	// printf("thread %i coppied \n", row);

	for (index_t index_i = 0; index_i < row_size; index_i++)
	{
		const index_t index = row_indices[index_i];

		if (index >= row)
			break;

		while (degree[index] != 0)
			;

		const index_t row_to_merge_size = As_nnz[index];
		volatile index_t* row_to_merge_indices_v = As_indices[index];
		index_t* row_to_merge_indices = As_indices[index];

		for (int i = 0; i < row_to_merge_size; i++)
			row_to_merge_indices[i] = row_to_merge_indices_v[i];

		if (row_to_merge_indices == 0)
			printf("error at row %i index %i \n", row, index);
		// printf("thread %i merging row %i with size %i\n", row, last_visited_index, row_to_merge_size);

		{
			index_t row_to_merge_index_i = 0;

			// forward merging row after the "index" column
			while (row_to_merge_index_i < row_to_merge_size && row_to_merge_indices[row_to_merge_index_i] <= index)
			{
				row_to_merge_index_i++;
			}

			if (row_to_merge_index_i == row_to_merge_size)
				continue;

			// TODO: count how many data will be required
			const index_t new_merge_size_upper_bound = row_to_merge_size + row_size;

			index_t* row_indices_scratch = (index_t*)malloc(sizeof(index_t) * new_merge_size_upper_bound);

			if (row_indices_scratch == NULL)
				printf("thread %i error scratchrow size %i \n", row, new_merge_size_upper_bound);

			index_t* merge_end = thrust::set_union(thrust::seq, row_to_merge_indices + row_to_merge_index_i,
												   row_to_merge_indices + row_to_merge_size, row_indices,
												   row_indices + row_size, row_indices_scratch);

			// update row size and indices 
			row_size = merge_end - row_indices_scratch;
			free(row_indices);
			row_indices = row_indices_scratch;
		}
	}

	//printf("writing %i %p\n", row, row_indices);
	As_nnz[row] = row_size;
	As_indices[row] = row_indices;

	__threadfence();
	degree[row] = 0;
}


__global__ void cuda_kernel_splu_symbolic_populate(const index_t A_rows, const index_t* __restrict__ A_indices,
												   const index_t* __restrict__ A_indptr,
												   const real_t* __restrict__ A_data, index_t* __restrict__ As_indices,
												   const index_t* __restrict__ As_indptr, real_t* __restrict__ As_data,
												   index_t* const* const __restrict__ As_indices_by_row)
{
	const index_t row = blockIdx.x * blockDim.x + threadIdx.x;

	if (row >= A_rows)
		return;


	const index_t out_begin = As_indptr[row];
	const index_t row_size = As_indptr[row + 1] - out_begin;
	index_t* row_indices = As_indices_by_row[row];
	As_indices += out_begin;
	As_data += out_begin;

	for (index_t i = 0; i < row_size; i++)
	{
		const index_t index = row_indices[i];
		As_indices[i] = index;
		As_data[i] = 0.f;
	}

	free(row_indices);

	const index_t orig_begin = A_indptr[row];
	const index_t orig_row_size = A_indptr[row + 1] - orig_begin;
	A_indices += orig_begin;
	A_data += orig_begin;

	const index_t v_end = A_indptr[row + 1];
	index_t As_idx = 0;
	for (index_t i = 0; i < orig_row_size; i++)
	{
		const index_t v = A_indices[i];

		while (As_indices[As_idx] != v)
			As_idx++;

		// printf("row %i wrote data %f at col %i\n", row, A_data[v_i], As_idx);

		As_data[As_idx] = A_data[i];
	}
}


/**
 * Count the number of upper-triangular nonzeros for each column of a CSC matrix.
 * This is inclusive of the main diagonal.
 *
 * Indexed on columns of A.
 */
__global__ void cuda_kernel_count_U_nnz(const index_t A_rows, const index_t A_cols,
										const index_t* __restrict__ At_indices, const index_t* __restrict__ At_indptr,
										index_t* __restrict__ U_col_nnz)
{
	const index_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if (j >= A_cols)
		return;

	index_t nnz = 0;
	for (index_t i_i = At_indptr[j]; i_i < At_indptr[j + 1]; i_i++)
	{
		const index_t i = At_indices[i_i];
		if (i <= j)
			nnz++;
	}

	if (nnz == 0)
		printf("L has some zeros\n");

	U_col_nnz[j] = nnz;
}


/**
 * Performs a binary search on an array between i_start and i_end (inclusive).
 */
static __device__ index_t kernel_indices_binsearch(index_t i_start, index_t i_end, const index_t i_search,
												   const index_t* __restrict__ indices)
{
	index_t i_mid;
	while (i_start <= i_end)
	{
		i_mid = (i_start + i_end) / 2;
		if (indices[i_mid] < i_search)
		{
			i_start = i_mid + 1;
		}
		else if (indices[i_mid] > i_search)
		{
			i_end = i_mid - 1;
		}
		else if (indices[i_mid] == i_search)
		{
			return i_mid;
		}
	}
	return -1;
}

/**
 * The sparse numeric LU factorization from SFLU:
 * "SFLU: Synchronization-Free Sparse LU Factorization for Fast Circuit Simulation on GPUs", J. Zhao, Y. Luo, Z. Jin, Z.
 * Zhou.
 *
 * Indexed on columns of As, where As is given in CSC format and has fill-ins represented by explicit zeros.
 */
template <typename scalar_t>
__global__ void cuda_kernel_splu_numeric_sflu(const index_t A_rows, const index_t A_cols,
											  real_t* __restrict__ As_col_data,
											  const index_t* __restrict__ As_col_indices,
											  const index_t* __restrict__ As_col_indptr,
											  volatile index_t* __restrict__ degree)
{
	const index_t k = blockIdx.x * blockDim.x + threadIdx.x;
	if (k >= A_cols)
	{
		return;
	}

	index_t diag_idx;
	const index_t col_end = As_col_indptr[k + 1];
	for (index_t i_i = As_col_indptr[k]; i_i < col_end; i_i++)
	{
		const index_t i = As_col_indices[i_i];
		if (i == k)
		{
			/* Stop once we get to the diagonal. */
			diag_idx = i_i;
			break;
		}

		/* Busy wait until intermediate results are ready */
		while (degree[i] > 0)
			;

		/* Left-looking product */
		for (index_t j_i = i_i + 1; j_i < col_end; j_i++)
		{
			const index_t j = As_col_indices[j_i];
			const index_t A_ji_i =
				kernel_indices_binsearch(As_col_indptr[i], As_col_indptr[i + 1] - 1, j, As_col_indices);
			if (A_ji_i == -1)
			{
				continue;
			}
			const scalar_t A_ji = As_col_data[A_ji_i];
			const scalar_t A_ik = As_col_data[i_i];

			/* A_{jk} \gets A_{jk} - A_{ji} A_{ik} */
			As_col_data[j_i] -= A_ji * A_ik;
		}

		// printf("thread %i decremented from %i\n", k, degree[k]);
		__threadfence();
		degree[k]--;
	}

	/* Divide column of L by diagonal entry of U */
	const scalar_t A_kk = As_col_data[diag_idx];
	for (index_t i = diag_idx + 1; i < As_col_indptr[k + 1]; i++)
	{
		As_col_data[i] /= A_kk;
	}

	// printf("thread %i decremented from %i\n", k, degree[k]);
	/* Complete the factorization and update column degree */
	__threadfence();
	degree[k]--;
}

/**
 * Sparse LU Factorization, using a left-looking algorithm on the columns of A.  Based on
 * the symbolic factorization from Rose, Tarjan's fill2 and numeric factorization in SFLU.
 */
void splu(cu_context& context, const d_idxvec& A_indptr, const d_idxvec& A_indices, const d_datvec& A_data,
		  d_idxvec& As_indptr, d_idxvec& As_indices, d_datvec& As_data)
{
	const int threads_per_block = 512;

	As_indptr.resize(A_indptr.size());
	As_indptr[0] = 0;

	index_t A_rows = A_indptr.size() - 1;
	index_t A_cols = A_indptr.size() - 1;

	std::cout << "splu start " << A_rows << std::endl;

	std::cout << "splu symbolic nnz" << std::endl;

	index_t* L_nnz;
	index_t** As_indices_by_row;

	CHECK_CUDA(cudaMalloc(&L_nnz, sizeof(index_t) * A_rows));
	CHECK_CUDA(cudaMalloc(&As_indices_by_row, sizeof(index_t*) * A_rows));
	CHECK_CUDA(cudaMemset(As_indices_by_row, 0, sizeof(index_t*) * A_rows));

	cuda_kernel_count_U_nnz<<<(A_rows + threads_per_block - 1) / threads_per_block, threads_per_block>>>(
		A_rows, A_cols, A_indices.data().get(), A_indptr.data().get(), L_nnz);

	CHECK_CUDA(cudaDeviceSynchronize());
	// print("L_nnz ", d_idxvec(L_nnz, L_nnz + A_rows));

	cuda_kernel_splu_symbolic_fact<<<(A_cols + threads_per_block - 1) / threads_per_block, threads_per_block>>>(
		A_rows, A_indices.data().get(), A_indptr.data().get(), As_indptr.data().get() + 1, As_indices_by_row, L_nnz);

	CHECK_CUDA(cudaDeviceSynchronize());

	CHECK_CUDA(cudaFree(L_nnz));

	std::cout << "splu cumsum" << std::endl;

	thrust::inclusive_scan(As_indptr.begin(), As_indptr.end(), As_indptr.begin());
	index_t As_nnz = As_indptr.back();

	As_indices.resize(As_nnz);
	As_data.resize(As_nnz);

	std::cout << "splu symbolic populate" << std::endl;

	cuda_kernel_splu_symbolic_populate<<<(A_cols + threads_per_block - 1) / threads_per_block, threads_per_block>>>(
		A_rows, A_indices.data().get(), A_indptr.data().get(), A_data.data().get(), As_indices.data().get(),
		As_indptr.data().get(), As_data.data().get(), As_indices_by_row);

	CHECK_CUDA(cudaDeviceSynchronize());

	CHECK_CUDA(cudaFree(As_indices_by_row));

	d_idxvec AsT_indptr, AsT_indices;
	d_datvec AsT_data;

	/* Compute the transpose/csc representation of As so that we have easy column access. */
	solver::transpose_sparse_matrix(context.cusparse_handle, As_indptr.data().get(), As_indices.data().get(),
									As_data.data().get(), A_rows, A_cols, As_data.size(), AsT_indptr, AsT_indices,
									AsT_data);
	// print("A indptr ", A_indptr);
	// print("A indice ", A_indices);
	// print("A data   ", A_data);

	// print("As indptr ", As_indptr);
	// print("As indice ", As_indices);
	// print("As data   ", As_data);

	// print("At indptr ", AsT_indptr);
	// print("At indice ", AsT_indices);
	// print("At data   ", AsT_data);

	std::cout << "splu U nnz" << std::endl;

	index_t* U_col_nnz;
	CHECK_CUDA(cudaMalloc(&U_col_nnz, sizeof(index_t) * A_rows));

	/* Perform the numeric factorization on the CSC representation */
	cuda_kernel_count_U_nnz<<<(A_cols + threads_per_block - 1) / threads_per_block, threads_per_block>>>(
		A_rows, A_cols, AsT_indices.data().get(), AsT_indptr.data().get(), U_col_nnz);
	CHECK_CUDA(cudaDeviceSynchronize());

	// print("splu degrees ", d_idxvec(U_col_nnz, U_col_nnz + A_cols));

	std::cout << "splu numeric" << std::endl;

	cuda_kernel_splu_numeric_sflu<real_t><<<(A_cols + threads_per_block - 1) / threads_per_block, threads_per_block>>>(
		A_rows, A_cols, AsT_data.data().get(), AsT_indices.data().get(), AsT_indptr.data().get(), U_col_nnz);

	CHECK_CUDA(cudaDeviceSynchronize());
	CHECK_CUDA(cudaFree(U_col_nnz));

	/* Transpose back into CSR format */
	solver::transpose_sparse_matrix(context.cusparse_handle, AsT_indptr.data().get(), AsT_indices.data().get(),
									AsT_data.data().get(), A_cols, A_rows, AsT_data.size(), As_indptr, As_indices,
									As_data);
}