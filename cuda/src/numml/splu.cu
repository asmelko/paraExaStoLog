#include "splu.h"

#include <thrust/execution_policy.h>

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
    bool* __restrict__ vert_mask,
    index_t* __restrict__ As_nnz) {

    const index_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    index_t row = thread_idx;

    /* We'll round robin over the columns to save memory */
    while (row < A_rows) {
        /* Zero out bitmap of visited nodes */
        for (index_t i = 0; i < A_cols; i++) {
            vert_fill[thread_idx * A_cols + i] = 0;
            vert_mask[thread_idx * A_cols + i] = false;
        }

        /* Set fill array */
        for (index_t v_i = A_indptr[row]; v_i < A_indptr[row + 1]; v_i++) {
            const index_t v = A_indices[v_i];
            vert_fill[thread_idx * A_cols + v] = row;
            vert_mask[thread_idx * A_cols + v] = true;
        }
        __syncthreads();

        /* Loop over "threshold" */
        for (index_t t = 0; t < row; t++) {
            if (vert_fill[thread_idx * A_rows + t] != row) {
                continue;
            }

            index_t queue_start = 0;
            index_t queue_end = 1;
            vert_queue[thread_idx * A_rows] = t;

            while (queue_start != queue_end) {
                const index_t u = vert_queue[thread_idx * A_rows + (queue_start % A_rows)];
                queue_start++;

                for (index_t w_i = A_indptr[u]; w_i < A_indptr[u + 1]; w_i++) {
                    const index_t w = A_indices[w_i];
                    if (vert_fill[thread_idx * A_rows + w] < row) {
                        vert_fill[thread_idx * A_rows + w] = row;
                        if (w > t) {
                            vert_mask[thread_idx * A_rows + w] = true;
                        } else {
                            vert_queue[thread_idx * A_rows + (queue_end % A_rows)] = w;
                            queue_end++;
                        }
                    }
                }
            }
        }
        __syncthreads();

        /* Count number of nonzeros in L and U in the current column */
        index_t As_nnz_row = 0;
        for (index_t i = 0; i < A_cols; i++) {
            if (vert_mask[thread_idx * A_rows + i]) {
                As_nnz_row++;
                vert_mask[thread_idx * A_rows + i] = false;
            }
        }
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
    bool* __restrict__ vert_mask,
    real_t* __restrict__ As_row_data,
    index_t* __restrict__ As_col_indices,
    const index_t* __restrict__ As_row_indptr) {

    const index_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    index_t row = thread_idx;

    /* We'll round robin over the columns to save memory */
    while (row < A_rows) {
        /* Zero out bitmap of visited nodes */
        for (index_t i = 0; i < A_rows; i++) {
            vert_fill[thread_idx * A_rows + i] = 0;
            vert_mask[thread_idx * A_rows + i] = false;
        }

        /* Set fill array */
        for (index_t v_i = A_indptr[row]; v_i < A_indptr[row + 1]; v_i++) {
            const index_t v = A_indices[v_i];
            vert_fill[thread_idx * A_rows + v] = row;
            vert_mask[thread_idx * A_rows + v] = true;
        }
        __syncthreads();

        /* Loop over "threshold" */
        for (index_t t = 0; t < row; t++) {
            if (vert_fill[thread_idx * A_rows + t] != row) {
                continue;
            }

            index_t queue_start = 0;
            index_t queue_end = 1;
            vert_queue[thread_idx * A_rows] = t;

            while (queue_start != queue_end) {
                const index_t u = vert_queue[thread_idx * A_rows + (queue_start % A_rows)];
                queue_start++;

                for (index_t w_i = A_indptr[u]; w_i < A_indptr[u + 1]; w_i++) {
                    const index_t w = A_indices[w_i];
                    if (vert_fill[thread_idx * A_rows + w] < row) {
                        vert_fill[thread_idx * A_rows + w] = row;
                        if (w > t) {
                            vert_mask[thread_idx * A_rows + w] = true;
                        } else {
                            vert_queue[thread_idx * A_rows + (queue_end % A_rows)] = w;
                            queue_end++;
                        }
                    }
                }
            }
        }
        __syncthreads();

        /* Insert row indices and nonzero values of At_data.
           This is essentially a union of the two columns, where entries in As *only* are explicitly zero. */

        index_t As_ptr = 0; /* Current entry in vert_visited array */
        index_t A_ptr = A_indptr[row]; /* Current index in original A */
        index_t As_out_ptr = As_row_indptr[row]; /* Current index in output As */

        const index_t As_end = A_cols;
        const index_t A_end = A_indptr[row + 1];

        while (As_ptr < As_end && A_ptr < A_end) {
            /* Make sure we actually are at a nonzero of As */
            while (!vert_mask[thread_idx * A_rows + As_ptr]) {
                As_ptr++;
            }

            const index_t As_col = As_ptr;
            const index_t A_col = A_indices[A_ptr];
            if (As_col < A_col) {
                As_row_data[As_out_ptr] = 0.;
                As_col_indices[As_out_ptr] = As_col;

                As_ptr++;
                As_out_ptr++;
            } else if (As_col > A_col) {
                /* This is probably unlikely, since A is a subset of As..?
                   Nonetheless, let's add it here just in case. */
                As_row_data[As_out_ptr] = A_data[A_ptr];
                As_col_indices[As_out_ptr] = A_col;

                A_ptr++;
                As_out_ptr++;
            } else { /* As_col == A_col */
                As_row_data[As_out_ptr] = A_data[A_ptr];
                As_col_indices[As_out_ptr] = A_col;

                A_ptr++;
                As_ptr++;
                As_out_ptr++;
            }
        }
        /* Finish off with rest of As entries */
        for (; As_ptr < As_end; As_ptr++) {
            if (vert_mask[thread_idx * A_rows + As_ptr]) {
                As_row_data[As_out_ptr] = 0.;
                As_col_indices[As_out_ptr] = As_ptr;
                As_out_ptr++;
            }
        }

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
        index_t* __restrict__ degree) {

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

        auto old = atomicSub(degree + k, 1);
        printf("degree %i decremented at %i", old, k);
    }

    /* Divide column of L by diagonal entry of U */
    const scalar_t A_kk = As_col_data[diag_idx];
    for (index_t i = diag_idx + 1; i < As_col_indptr[k + 1]; i++) {
        As_col_data[i] /= A_kk;
    }

    /* Complete the factorization and update column degree */
    auto old = atomicSub(degree + k, 1);
    printf("degree %i decremented at %i", old, k);
}

/**
 * Sparse LU Factorization, using a left-looking algorithm on the columns of A.  Based on
 * the symbolic factorization from Rose, Tarjan's fill2 and numeric factorization in SFLU.
 */
void splu(cu_context& context, const d_idxvec& A_indptr, const d_idxvec& A_indices, const d_datvec& A_data, d_idxvec& As_indptr, d_idxvec& As_indices, d_datvec& As_data)
{
    /* First, perform the symbolic factorization to determine the sparsity pattern of the filled-in LU factorization
       of A, which we will hereby denote by As.  Note that mask(As) \superset mask(A). */
    const index_t num_threads_symb = 32;
    const index_t num_blocks_symb = 8;
    const index_t total_threads_symb = num_threads_symb * num_blocks_symb;
    index_t* vert_fill;
    index_t* vert_queue;
    bool* vert_mask;
    index_t* As_row_nnz;
    index_t* As_row_indptr_raw;
    index_t* U_col_nnz;
    const int threads_per_block = 512;

    index_t A_rows = A_indptr.size() - 1;
    index_t A_cols = A_indptr.size() - 1;

    std::cout << "splu start " << A_rows << std::endl;

    CHECK_CUDA(cudaMalloc(&vert_fill, sizeof(index_t) * total_threads_symb * A_rows));
    CHECK_CUDA(cudaMalloc(&vert_queue, sizeof(index_t) * total_threads_symb * A_rows));
    CHECK_CUDA(cudaMalloc(&vert_mask, sizeof(bool) * total_threads_symb * A_rows));
    CHECK_CUDA(cudaMalloc(&As_row_nnz, sizeof(index_t) * A_rows));
    CHECK_CUDA(cudaMalloc(&As_row_indptr_raw, sizeof(index_t) * (A_rows + 1)));
    CHECK_CUDA(cudaMalloc(&U_col_nnz, sizeof(index_t) * A_cols));

    std::cout << "splu symbolic nnz" << std::endl;

    /* First, find number of nonzeros in the rows of M=(L+U) (with fill) */
    cuda_kernel_splu_symbolic_fact_trav_nnz<<<num_blocks_symb, num_threads_symb>>>(
        A_rows, A_cols, A_indices.data().get(), A_indptr.data().get(),
        vert_fill, vert_queue, vert_mask, As_row_nnz);
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

    As_indptr.resize(A_rows + 1);
    As_indices.resize(As_nnz);
    As_data.resize(As_nnz);

    std::cout << "splu symbolic populate" << std::endl;

    /* Now, fill in As with row indices and entries of A (with explicit zeros where we are anticipating fill) */
    cuda_kernel_splu_symbolic_fact_trav_populate<real_t><<<num_blocks_symb, num_threads_symb>>>(
        A_rows, A_cols, A_data.data().get(), A_indices.data().get(), A_indptr.data().get(),
        vert_fill, vert_queue, vert_mask, As_data.data().get(), As_indices.data().get(), As_row_indptr_raw);

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaFree(vert_fill));
    CHECK_CUDA(cudaFree(vert_queue));
    CHECK_CUDA(cudaFree(vert_mask));

    d_idxvec AsT_indptr, AsT_indices;
    d_datvec AsT_data;

    /* Compute the transpose/csc representation of As so that we have easy column access. */
    solver::transpose_sparse_matrix(context.cusparse_handle, A_indptr.data().get(), A_indices.data().get(), A_data.data().get(),
                                    A_rows, A_cols, A_data.size(), AsT_indptr, AsT_indices, AsT_data);

    print("At indptr ", AsT_indptr);
	print("At indice ", AsT_indices);
	print("At data   ", AsT_data);

    std::cout << "splu U nnz" << std::endl;

    /* Perform the numeric factorization on the CSC representation */
    cuda_kernel_count_U_nnz<<<(A_cols + threads_per_block - 1) / threads_per_block, threads_per_block>>>(
        A_rows, A_cols, AsT_indices.data().get(), AsT_indptr.data().get(), U_col_nnz);
    CHECK_CUDA(cudaDeviceSynchronize());

    print("splu degrees ", d_idxvec(U_col_nnz, U_col_nnz + A_cols));

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