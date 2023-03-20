#include "splu.h"

#include <thrust/execution_policy.h>
#include <thrust/set_operations.h>

#include "../solver.h"
#include "../utils.h"

/**
 * Compute the number of nonzero entries in each column of the LU factorization.
 * Finds fill-in using a depth-first traversal of the matrix.  Based on
 * "GSoFa: Scalable Sparse Symbolic LU Factorization on GPUs",  Gaihre A, Li X, Liu H.
 *
 * This should be run with one block of some predetermined fixed size.  This is
 * run with less threads than overall columns due to memory constraints.
 */
__global__ void cuda_kernel_splu_symbolic_fact_trav_nnz(
    const index_t A_rows, const index_t A_cols,
    const index_t* __restrict__ A_indices,
    const index_t* __restrict__ A_indptr,
    index_t* __restrict__ vert_fill,
    index_t* __restrict__ vert_queue,
    index_t* __restrict__ As_nnz) {

    const index_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    index_t row = thread_idx;

    /* We'll round robin over the columns to save memory */
    while (row < A_rows) {
        index_t As_nnz_row = 0;
        index_t queue_end = 0;

        /* Set fill array */
        const index_t v_end = A_indptr[row + 1];
        for (index_t v_i = A_indptr[row]; v_i < v_end; v_i++) {
            const index_t v = A_indices[v_i];
            vert_fill[thread_idx * A_rows + v] = row;
            As_nnz_row++;

            if (v < row) {
                vert_queue[thread_idx * A_rows + queue_end++] = v;
            }
        }

        index_t queue_start = 0;

        while (queue_start != queue_end) {
            const index_t u = vert_queue[thread_idx * A_rows + (queue_start % A_rows)];
            queue_start++;

            const index_t w_end = A_indptr[u + 1];
            for (index_t w_i = A_indptr[u]; w_i < w_end; w_i++) {
                const index_t w = A_indices[w_i];
                if (vert_fill[thread_idx * A_rows + w] < row) {
                    vert_fill[thread_idx * A_rows + w] = row;
                    if (w > u) {
                        As_nnz_row++;
                        if (w < row) {
                            vert_queue[thread_idx * A_rows + (queue_end % A_rows)] = w;
                            queue_end++;
                        }
                    }
                }
            }
        }

        /* Count number of nonzeros in L and U in the current column */
        As_nnz[row] = As_nnz_row;
        row += blockDim.x * gridDim.x;
    }
}

/**
 * Given number of nonzero fill-ins in the column LU factorization, populate
 * row indices and data entries of the symbolic factorization.  Based on
 * "GSoFa: Scalable Sparse Symbolic LU Factorization on GPUs",  Gaihre A, Li X, Liu H.
 *
 * This should be run with one block of some predetermined fixed size.  This is
 * run with less threads than overall columns due to memory constraints.
 */
template <typename scalar_t>
__global__ void cuda_kernel_splu_symbolic_fact_trav_populate(
    const index_t A_rows, const index_t A_cols,
    const real_t* __restrict__ A_data,
    const index_t* __restrict__ A_indices,
    const index_t* __restrict__ A_indptr,
    index_t* __restrict__ vert_fill,
    index_t* __restrict__ vert_queue,
    real_t* __restrict__ As_row_data,
    index_t* __restrict__ As_col_indices,
    index_t* __restrict__ As_col_indices_scratch,
    const index_t* __restrict__ As_row_indptr) {

    const index_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    index_t row = thread_idx;

    index_t* __restrict__ As_col_indices_orig = As_col_indices;

    /* We'll round robin over the columns to save memory */
    while (row < A_rows) {

        const index_t row_begin = As_row_indptr[row];
        const index_t As_nnz_row = As_row_indptr[row + 1] - As_row_indptr[row];
        index_t row_end = 0;

        index_t queue_end = 0;

        {
            const index_t v_end = A_indptr[row + 1];
            for (index_t v_i = A_indptr[row]; v_i < v_end; v_i++) {
                const index_t v = A_indices[v_i];

                As_col_indices[row_begin + row_end] = v;
                As_col_indices_scratch[row_begin + row_end] = v;
                row_end++;

                if (v < row) {
                    vert_queue[thread_idx * A_rows + queue_end++] = v;
                }
            }
        }

        index_t queue_start = 0;

        while (queue_start != queue_end) {
            const index_t u = vert_queue[thread_idx * A_rows + (queue_start % A_rows)];
            queue_start++;

            index_t idx_after_row = -1;
            const index_t w_end = A_indptr[u + 1];
            for (index_t w_i = A_indptr[u]; w_i < w_end; w_i++) {
                const index_t w = A_indices[w_i];
                if (vert_fill[thread_idx * A_rows + w] < row) {
                    vert_fill[thread_idx * A_rows + w] = row;
                    if (w < row && w > u) {
                        vert_queue[thread_idx * A_rows + (queue_end % A_rows)] = w;
                        queue_end++;
                    }
                }

                if (idx_after_row == -1 && w > u)
                    idx_after_row = w_i;
            }

            if (idx_after_row != -1) {
                // merge cols together
                index_t* new_end = thrust::set_union(thrust::seq,
                    As_col_indices + row_begin, As_col_indices + row_begin + row_end,
                    A_indices + idx_after_row, A_indices + w_end,
                    As_col_indices_scratch + row_begin);

                row_end = new_end - (As_col_indices_scratch + row_begin);

                thrust::swap(As_col_indices, As_col_indices_scratch);
            }
        }

        const index_t v_end = A_indptr[row + 1];
        index_t As_idx = 0;
        for (index_t v_i = A_indptr[row]; v_i < v_end; v_i++) {
            const index_t v = A_indices[v_i];

            while (As_col_indices[row_begin + As_idx] != v)
                As_idx++;

            As_row_data[As_idx] = A_data[v_i];
        }

        thrust::copy(thrust::seq, As_col_indices + row_begin, As_col_indices + row_begin + row_end,
            As_col_indices_orig + row_begin);

        row += blockDim.x * gridDim.x;
    }
}


/**
 * Count the number of upper-triangular nonzeros for each column of a CSC matrix.
 * This is inclusive of the main diagonal.
 *
 * Indexed on columns of A.
 */
__global__ void cuda_kernel_count_U_nnz(
    const index_t A_rows, const index_t A_cols,
    const index_t* __restrict__ At_indices,
    const index_t* __restrict__ At_indptr,
    index_t* __restrict__ U_col_nnz) {

    const index_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= A_cols) {
        return;
    }

    index_t nnz = 0;
    for (index_t i_i = At_indptr[j]; i_i < At_indptr[j + 1]; i_i++) {
        const index_t i = At_indices[i_i];
        if (i <= j) {
            nnz++;
        }
    }

    U_col_nnz[j] = nnz;
}

/**
 * Performs a binary search on an array between i_start and i_end (inclusive).
 */
static __device__ index_t kernel_indices_binsearch(index_t i_start, index_t i_end, const index_t i_search,
                                                   const index_t* __restrict__ indices) {
    index_t i_mid;
    while (i_start <= i_end) {
        i_mid = (i_start + i_end) / 2;
        if (indices[i_mid] < i_search) {
            i_start = i_mid + 1;
        } else if (indices[i_mid] > i_search) {
            i_end = i_mid - 1;
        } else if (indices[i_mid] == i_search) {
            return i_mid;
        }
    }
    return -1;
}

/**
 * The sparse numeric LU factorization from SFLU:
 * "SFLU: Synchronization-Free Sparse LU Factorization for Fast Circuit Simulation on GPUs", J. Zhao, Y. Luo, Z. Jin, Z. Zhou.
 *
 * Indexed on columns of As, where As is given in CSC format and has fill-ins represented by explicit zeros.
 */
template <typename scalar_t>
__global__ void cuda_kernel_splu_numeric_sflu(
        const index_t A_rows, const index_t A_cols,
        real_t* __restrict__ As_col_data,
        const index_t* __restrict__ As_col_indices,
        const index_t* __restrict__ As_col_indptr,
        volatile index_t* __restrict__ degree) {

    const index_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= A_cols) {
        return;
    }

    index_t diag_idx;
    const index_t col_end = As_col_indptr[k + 1];
    for (index_t i_i = As_col_indptr[k]; i_i < col_end; i_i++) {
        const index_t i = As_col_indices[i_i];
        if (i == k) {
            /* Stop once we get to the diagonal. */
            diag_idx = i_i;
            break;
        }

        /* Busy wait until intermediate results are ready */
        while (degree[i] > 0);

        /* Left-looking product */
        for (index_t j_i = i_i + 1; j_i < col_end; j_i++) {
            const index_t j = As_col_indices[j_i];
            const index_t A_ji_i = kernel_indices_binsearch(As_col_indptr[i], As_col_indptr[i + 1] - 1, j, As_col_indices);
            if (A_ji_i == -1) {
                continue;
            }
            const scalar_t A_ji = As_col_data[A_ji_i];
            const scalar_t A_ik = As_col_data[i_i];

            /* A_{jk} \gets A_{jk} - A_{ji} A_{ik} */
            As_col_data[j_i] -= A_ji * A_ik;
        }

        //printf("thread %i decremented from %i\n", k, degree[k]);
        __threadfence();
        degree[k]--;
    }

    /* Divide column of L by diagonal entry of U */
    const scalar_t A_kk = As_col_data[diag_idx];
    for (index_t i = diag_idx + 1; i < As_col_indptr[k + 1]; i++) {
        As_col_data[i] /= A_kk;
    }

    //printf("thread %i decremented from %i\n", k, degree[k]);
    /* Complete the factorization and update column degree */
    __threadfence();
    degree[k]--;
}

/**
 * Sparse LU Factorization, using a left-looking algorithm on the columns of A.  Based on
 * the symbolic factorization from Rose, Tarjan's fill2 and numeric factorization in SFLU.
 */
void splu(cu_context& context, const d_idxvec& A_indptr, const d_idxvec& A_indices, const d_datvec& A_data, d_idxvec& As_indptr, d_idxvec& As_indices, d_datvec& As_data)
{
    As_indptr.resize(A_indptr.size());
    /* First, perform the symbolic factorization to determine the sparsity pattern of the filled-in LU factorization
       of A, which we will hereby denote by As.  Note that mask(As) \superset mask(A). */
    const index_t num_threads_symb = 32;
    const index_t num_blocks_symb = 8;
    const index_t total_threads_symb = num_threads_symb * num_blocks_symb;
    index_t* vert_fill;
    index_t* vert_queue;
    index_t* As_row_nnz;
    index_t* As_row_indptr_raw = As_indptr.data().get();
    index_t* U_col_nnz;
    const int threads_per_block = 512;

    index_t A_rows = A_indptr.size() - 1;
    index_t A_cols = A_indptr.size() - 1;

    std::cout << "splu start " << A_rows << std::endl;

    CHECK_CUDA(cudaMalloc(&vert_fill, sizeof(index_t) * total_threads_symb * A_rows));
    CHECK_CUDA(cudaMalloc(&vert_queue, sizeof(index_t) * total_threads_symb * A_rows));
    CHECK_CUDA(cudaMalloc(&As_row_nnz, sizeof(index_t) * A_rows));
    CHECK_CUDA(cudaMalloc(&U_col_nnz, sizeof(index_t) * A_cols));

    std::cout << "splu symbolic nnz" << std::endl;

    cudaMemset(vert_fill, 0, sizeof(index_t) * total_threads_symb * A_rows);

    /* First, find number of nonzeros in the rows of M=(L+U) (with fill) */
    cuda_kernel_splu_symbolic_fact_trav_nnz<<<num_blocks_symb, num_threads_symb>>>(
        A_rows, A_cols, A_indices.data().get(), A_indptr.data().get(),
        vert_fill, vert_queue, As_row_nnz);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::cout << "splu cumsum" << std::endl;

    /* From the row nnz, compute row pointers */
    CHECK_CUDA(cudaMemset(As_row_indptr_raw, 0, sizeof(index_t)));
    thrust::inclusive_scan(thrust::device, As_row_nnz, As_row_nnz + A_rows, As_row_indptr_raw + 1);
    CHECK_CUDA(cudaFree(As_row_nnz));

    std::cout << "splu As nnz copy" << std::endl;

    /* Allocate storage for the data and row indices arrays */
    index_t As_nnz;
    CHECK_CUDA(cudaMemcpy(&As_nnz, As_row_indptr_raw + A_rows, sizeof(index_t), cudaMemcpyDeviceToHost));

    std::cout << "splu As nnz " << As_nnz << std::endl;

    As_indices.resize(As_nnz);
    As_data.resize(As_nnz);

    index_t* As_indices_scratch;
    CHECK_CUDA(cudaMalloc(&As_indices_scratch, sizeof(index_t) * As_nnz));

    std::cout << "splu symbolic populate" << std::endl;

    /* Now, fill in As with row indices and entries of A (with explicit zeros where we are anticipating fill) */
    cuda_kernel_splu_symbolic_fact_trav_populate<real_t><<<num_blocks_symb, num_threads_symb>>>(
        A_rows, A_cols, A_data.data().get(), A_indices.data().get(), A_indptr.data().get(),
        vert_fill, vert_queue, As_data.data().get(), As_indices.data().get(), As_indices_scratch, As_row_indptr_raw);

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaFree(vert_fill));
    CHECK_CUDA(cudaFree(vert_queue));
    CHECK_CUDA(cudaFree(As_indices_scratch));

    d_idxvec AsT_indptr, AsT_indices;
    d_datvec AsT_data;

    /* Compute the transpose/csc representation of As so that we have easy column access. */
    solver::transpose_sparse_matrix(context.cusparse_handle, As_indptr.data().get(), As_indices.data().get(), As_data.data().get(),
                                    A_rows, A_cols, As_data.size(), AsT_indptr, AsT_indices, AsT_data);

    // print("A indptr ", As_indptr);
	// print("A indice ", As_indices);
	// print("A data   ", As_data);

    // print("At indptr ", AsT_indptr);
	// print("At indice ", AsT_indices);
	// print("At data   ", AsT_data);

    std::cout << "splu U nnz" << std::endl;

    /* Perform the numeric factorization on the CSC representation */
    cuda_kernel_count_U_nnz<<<(A_cols + threads_per_block - 1) / threads_per_block, threads_per_block>>>(
        A_rows, A_cols, AsT_indices.data().get(), AsT_indptr.data().get(), U_col_nnz);
    CHECK_CUDA(cudaDeviceSynchronize());

    //print("splu degrees ", d_idxvec(U_col_nnz, U_col_nnz + A_cols));

    std::cout << "splu numeric" << std::endl;

    cuda_kernel_splu_numeric_sflu<real_t><<<(A_cols + threads_per_block - 1) / threads_per_block, threads_per_block>>>(
        A_rows, A_cols,
        AsT_data.data().get(), AsT_indices.data().get(), AsT_indptr.data().get(), U_col_nnz);

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaFree(U_col_nnz));

    /* Transpose back into CSR format */
    solver::transpose_sparse_matrix(context.cusparse_handle, AsT_indptr.data().get(), AsT_indices.data().get(), AsT_data.data().get(),
                                    A_cols, A_rows, AsT_data.size(), As_indptr, As_indices, As_data);
}