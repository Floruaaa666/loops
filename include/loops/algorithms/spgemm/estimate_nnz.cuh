/**
 * @file estimate_nnz.cuh
 * @author 
 * @brief SpGEMM kernels.
 * @version 0.1
 * @date 2023-11-08
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once

#include <loops/stride_ranges.hxx>
#include <loops/schedule.hxx>
#include <loops/container/formats.hxx>
#include <loops/container/vector.hxx>
#include <loops/container/matrix.cuh>
#include <loops/util/launch.hxx>
#include <loops/util/device.hxx>
#include <loops/memory.hxx>
#include <iostream>

namespace loops {
namespace algorithms {
namespace spgemm {

template <typename index_t,
          typename offset_t>
__global__ void __estimate_nnz_C(const std::size_t a_rows,
                                const std::size_t a_cols,
                                const std::size_t a_nnz,
                                const offset_t* a_offsets,
                                const index_t* a_indices,
                                const std::size_t b_rows,
                                const std::size_t b_cols,
                                const std::size_t b_nnz,
                                const offset_t* b_offsets,
                                const index_t* b_indices,
                               int* nnz_C_per_row) {

    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= a_rows) return;

    extern __shared__ int shared_marker[];

    for (int i = threadIdx.x; i < b_cols; i += blockDim.x) {
        shared_marker[i] = -1;
    }
    __syncthreads();

    int nnz_count = 0;
    int start_mm_a = a_offsets[row];
    int end_mm_a = a_offsets[row + 1];

    for (int mm = start_mm_a; mm < end_mm_a; ++mm) {
        int kk_a = a_indices[mm];
        int start_nn_b = b_offsets[kk_a];
        int end_nn_b = b_offsets[kk_a + 1];
        for (int nn = start_nn_b; nn < end_nn_b; ++nn) {
            int kk_b = b_indices[nn];
            if (atomicCAS(&shared_marker[kk_b], -1, row) == -1) {
                nnz_count++;
            }
        }
    }

    nnz_C_per_row[row] = nnz_count;
    __syncthreads();
}


/**
 * @brief Estimate the nnz of output matrix C.
 *
 * @tparam index_t Type of column indices.
 * @tparam offset_t Type of row offsets.
 * @tparam type_t Type of values.
 * @param csr CSR matrix (GPU).
 * @param n Number of columns in the B-matrix.
 * @param B Input matrix B (GPU).
 * @param nnz of C (GPU).
 * @param stream CUDA stream.
 */
template <typename index_t, typename offset_t, typename type_t>
void estimate_nnz(csr_t<index_t, offset_t, type_t>& csr,
                   csc_t<index_t, offset_t, type_t>& csc,
                   int* nnz_C_per_row) {

    std::size_t block_size = 128;
    std::size_t grid_size = (csr.rows + block_size - 1) / block_size;

    __estimate_nnz_C<<<block_size, grid_size, csc.cols * sizeof(int)>>>(
      csr.rows, csr.cols, csr.nnzs,
      csr.offsets.data().get(), csr.indices.data().get(),
      csc.rows, csc.cols, csc.nnzs,
      csc.offsets.data().get(), csc.indices.data().get(), 
      nnz_C_per_row);
}

}  // namespace spgemm
}  // namespace algorithms
}  // namespace loops