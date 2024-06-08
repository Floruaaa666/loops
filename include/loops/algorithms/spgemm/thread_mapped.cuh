/**
 * @file thread_mapped.cuh
 * @author 
 * @brief SpGEMM kernels.
 * @version 0.1
 * @date 2023-10-17
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
#include <loops/container/coo.hxx>
#include <loops/util/launch.hxx>
#include <loops/util/device.hxx>
#include <loops/memory.hxx>
#include <iostream>

namespace loops {
namespace algorithms {
namespace spgemm {

template <typename setup_t,
          typename index_t,
          typename offset_t,
          typename type_t>
__global__ void __thread_mapped(setup_t config,
                                const std::size_t a_rows,
                                const std::size_t a_cols,
                                const std::size_t a_nnz,
                                const offset_t* a_offsets,
                                const index_t* a_indices,
                                const type_t* a_values,
                                const std::size_t b_rows,
                                const std::size_t b_cols,
                                const std::size_t b_nnz,
                                const offset_t* b_offsets,
                                const index_t* b_indices,
                                const type_t* b_values,
                                const offset_t* c_offsets,
                                index_t* tmp_c_indices,
                                type_t* tmp_c_values) {


  for (auto mm : config.tiles()) { //translate tileId to rowId and colId - the grid stride grid_stride_range(T begin, T end)
    int c_row_nnz = 0;
    for (auto nn :
         custom_stride_range(std::size_t(0), b_cols, std::size_t(1))) {
      type_t sum = 0;
      for (auto nz : config.atoms(mm)) {
        auto kk_a = a_indices[nz];
          for (auto nz_b = b_offsets[nn]; nz_b < b_offsets[nn + 1]; ++nz_b) {
            if (kk_a == b_indices[nz_b]) {
              sum += a_values[nz] * b_values[nz_b];
            }
          }
      }
      
      if(sum != 0){
        tmp_c_indices[c_offsets[mm] + c_row_nnz] = nn;
        tmp_c_values[c_offsets[mm] + c_row_nnz] = sum;
        ++c_row_nnz;
        // c_row_nnz = atomicAdd(&c_row_nnz, 1);
      }
    }
  }
}


template <typename setup_t,
          typename index_t,
          typename offset_t,
          typename type_t>
__global__ void __thread_mapped_v2(setup_t config,
                                const std::size_t a_rows,
                                const std::size_t a_cols,
                                const std::size_t a_nnz,
                                const offset_t* a_offsets,
                                const index_t* a_indices,
                                const type_t* a_values,
                                const std::size_t b_rows,
                                const std::size_t b_cols,
                                const std::size_t b_nnz,
                                const offset_t* b_offsets,
                                const index_t* b_indices,
                                const type_t* b_values,
                                const offset_t* c_offsets,
                                index_t* tmp_c_indices,
                                type_t* tmp_c_values) {

  for (auto mm : config.tiles()) {
    int c_row_nnz = 0;
    type_t sum = 0;
    for (auto nz : config.atoms(mm)) {
      auto kk_a = a_indices[nz];
        for (auto nn :
          custom_stride_range(std::size_t(0), b_cols, std::size_t(1))) {
          for (auto nz_b = b_offsets[nn]; nz_b < b_offsets[nn + 1]; ++nz_b) {
            if (kk_a == b_indices[nz_b]) {
              sum += a_values[nz] * b_values[nz_b];
            }
          }
          if(sum != 0){
            tmp_c_indices[c_offsets[mm] + c_row_nnz] = nn;
            tmp_c_values[c_offsets[mm] + c_row_nnz] = sum;
            ++c_row_nnz;
            // c_row_nnz = atomicAdd(&c_row_nnz, 1);
          }
        }
    }
  }
}

// Tiling A and B
template <typename setup_t,
          typename index_t,
          typename offset_t,
          typename type_t>
__global__ void __thread_mapped_v3(setup_t config,
                                const std::size_t a_rows,
                                const std::size_t a_cols,
                                const std::size_t a_nnz,
                                const offset_t* a_offsets,
                                const index_t* a_indices,
                                const type_t* a_values,
                                const std::size_t b_rows,
                                const std::size_t b_cols,
                                const std::size_t b_nnz,
                                const offset_t* b_offsets,
                                const index_t* b_indices,
                                const type_t* b_values,
                                const offset_t* c_offsets,
                                index_t* tmp_c_indices,
                                type_t* tmp_c_values) {

  for (auto mm : config.tiles()) {
    int c_row_nnz = 0;
    type_t sum = 0;
    for (auto nz : config.atoms(mm)) {
      auto kk_a = a_indices[nz];
        for (auto nn :
          custom_stride_range(std::size_t(0), b_cols, std::size_t(1))) {
          for (auto nz_b = b_offsets[nn]; nz_b < b_offsets[nn + 1]; ++nz_b) {
            if (kk_a == b_indices[nz_b]) {
              sum += a_values[nz] * b_values[nz_b];
            }
          }
          if(sum != 0){
            tmp_c_indices[c_offsets[mm] + c_row_nnz] = nn;
            tmp_c_values[c_offsets[mm] + c_row_nnz] = sum;
            ++c_row_nnz;
            // c_row_nnz = atomicAdd(&c_row_nnz, 1);
          }
        }
    }
  }
}

/*
template <typename setup_t,
          typename index_t,
          typename offset_t,
          typename type_t>
__global__ void __thread_mapped_row_col_pairs(setup_t config,
                                const std::size_t a_rows,
                                const std::size_t a_cols,
                                const std::size_t a_nnz,
                                const offset_t* a_offsets,
                                const index_t* a_indices,
                                const type_t* a_values,
                                const std::size_t b_rows,
                                const std::size_t b_cols,
                                const std::size_t b_nnz,
                                const offset_t* b_offsets,
                                const index_t* b_indices,
                                const type_t* b_values,
                                const offset_t* c_offsets,
                                index_t* tmp_c_indices,
                                type_t* tmp_c_values) {

  __shared__ index_t shared_A_cols[TILE_SIZE * TILE_SIZE];
  __shared__ type_t shared_A_values[TILE_SIZE * TILE_SIZE];

  __shared__ index_t shared_B_rows[TILE_SIZE * TILE_SIZE];
  __shared__ type_t shared_B_values[TILE_SIZE * TILE_SIZE];

  int tx = threadIdx.x, bx = blockIdx.x;

  bool found = false;

  int shared_mem_prev_col_arr_range = 0;
  // Load entire rows of A into shared memory with stride length = TILE_SIZE.x, if the row size is larger than the empty space in shared memory, then skip the current row
  for(int m0 = bx; m0 < a_rows && (a_offsets[m0 + 1] - a_offsets[m0]) <= (TILE_SIZE * TILE_SIZE - shared_mem_prev_col_arr_range); m0 += gridDim.x) //can't exploit the shared memory b/c the shared memory isn't large enough to take an entire row of A
  { // Stride over the rows of A with the stride width of gridDim.x
    auto col_arr_start = a_offsets[m0];
    auto col_arr_end = a_offsets[m0 + 1];
    auto shared_mem_curr_col_arr_range = col_arr_end - col_arr_start;

    for(int col_arr_itr = tx; col_arr_itr < shared_mem_curr_col_arr_range; col_arr_itr += TILE_SIZE){
      shared_A_cols[col_arr_itr + shared_mem_prev_col_arr_range] = a_indices[col_arr_itr + col_arr_start];
      shared_A_values[col_arr_itr + shared_mem_prev_col_arr_range] = a_values[col_arr_itr + col_arr_start];
    }
    shared_mem_prev_col_arr_range += shared_mem_curr_col_arr_range;
  }
  __syncthreads();

  // Load entire columns of B into shared memory with stride length = TILE_SIZE.x, if the column size is larger than the empty space in shared memory, then skip the current column
  for(int n0 = tx; n0 < b_cols && b_offsets[n0 + 1] <= TILE_SIZE * TILE_SIZE; n0 += TILE_SIZE)
  {
      auto row_arr_start = b_offsets[n0];
      auto row_arr_end = b_offsets[n0 + 1];
      for(int k0 = row_arr_start; k0 < row_arr_end; ++k0){
        shared_B_rows[k0] = b_indices[k0];
        shared_B_values[k0] = b_values[k0];
      }
  }
  __syncthreads();

  if(b_nnz < TILE_SIZE * TILE_SIZE){ // If the number of non-zero elements in B is less than TILE_SIZE * TILE_SIZE, pad the shared memory with -1
    int diff = TILE_SIZE * TILE_SIZE - b_nnz;
    for(int i = tx; i < diff; i += TILE_SIZE){
      shared_B_rows[b_nnz + i] = -1;
    }
  }
  __syncthreads();

  // SHARED_A:
  int prev_col_arr_range = 0;
  int m0 = bx;
  while(m0 < a_rows && (a_offsets[m0 + 1] - a_offsets[m0]) <= (TILE_SIZE * TILE_SIZE - prev_col_arr_range))
  {
    auto col_arr_start = a_offsets[m0];
    auto col_arr_end = a_offsets[m0 + 1];
    auto curr_col_arr_range = col_arr_end - col_arr_start;

    // Using SHARED B
    int n = tx;
    while(n < b_cols && b_offsets[n + 1] < TILE_SIZE * TILE_SIZE){

      auto row_arr_start = b_offsets[n];
      auto row_arr_end = b_offsets[n + 1];

      for(int row_arr_itr_b = row_arr_start; row_arr_itr_b < row_arr_end; ++row_arr_itr_b)
      { // Iterate over all the elements in nth column of B
        for(auto col_arr_itr_a = 0; col_arr_itr_a < curr_col_arr_range; ++col_arr_itr_a)
        {
          if((shared_A_cols[col_arr_itr_a + prev_col_arr_range] == shared_B_rows[row_arr_itr_b]))
          {
            // Perform the multiplication
            // Add to C_values
          }
        }
      }
      n += TILE_SIZE;
    }
    __syncthreads();

    while(n < b_cols && b_offsets[n + 1] >= TILE_SIZE * TILE_SIZE){
      auto row_arr_start = b_offsets[n];
      auto row_arr_end = b_offsets[n + 1];
      found = false;
      for(int row_arr_itr_b = row_arr_start; row_arr_itr_b < row_arr_end; ++row_arr_itr_b)
      { // Iterate over all the elements in nth column of B
        for(auto col_arr_itr_a = 0; col_arr_itr_a < curr_col_arr_range; ++col_arr_itr_a)
        {
          if((shared_A_cols[col_arr_itr_a + prev_col_arr_range] == b_indices[row_arr_itr_b]))
          {
            // Perform the multiplication
            // Add to C_values
          }
        }
      }
      n += TILE_SIZE;
    }
    __syncthreads();

    prev_col_arr_range += curr_col_arr_range;
    m0 += gridDim.x;
  }
  __syncthreads();

  while(m0 < a_rows && (a_offsets[m0 + 1] - a_offsets[m0]) > (TILE_SIZE * TILE_SIZE - prev_col_arr_range))
  {
    auto col_arr_start = a_offsets[m0];
    auto col_arr_end = a_offsets[m0 + 1];
    auto curr_col_arr_range = col_arr_end - col_arr_start;

    int n = tx;
    while(n < b_cols && b_offsets[n + 1] < TILE_SIZE * TILE_SIZE){
      auto row_arr_start = b_offsets[n];
      auto row_arr_end = b_offsets[n + 1];
      found = false;
      for(int row_arr_itr_b = row_arr_start; row_arr_itr_b < row_arr_end; ++row_arr_itr_b)
      { // Iterate over all the elements in nth column of B
        for(auto col_arr_itr_a = 0; col_arr_itr_a < curr_col_arr_range; ++col_arr_itr_a)
        {
          if((a_indices[col_arr_itr_a + col_arr_start] == shared_B_rows[row_arr_itr_b]))
          {
           
          }
        }
      }
      n += TILE_SIZE;
    }
    __syncthreads();

    while(n < b_cols && b_offsets[n + 1] >= TILE_SIZE * TILE_SIZE){
      auto row_arr_start = b_offsets[n];
      auto row_arr_end = b_offsets[n + 1];
      found = false;
      for(int row_arr_itr_b = row_arr_start; row_arr_itr_b < row_arr_end; ++row_arr_itr_b)
      { // Iterate over all the elements in nth column of B
        for(auto col_arr_itr_a = 0; col_arr_itr_a < curr_col_arr_range; ++col_arr_itr_a)
        {
          if((a_indices[col_arr_itr_a + col_arr_start] == b_indices[row_arr_itr_b]))
          {

          }
        }
      }
      n += TILE_SIZE;
    }
    __syncthreads();

    prev_col_arr_range += curr_col_arr_range;
    m0 += gridDim.x;
  }
  __syncthreads();

}
*/

/**
 * @brief Sparse-Matrix Matrix Multiplication API.
 *
 * @tparam index_t Type of column indices.
 * @tparam offset_t Type of row offsets.
 * @tparam type_t Type of values.
 * @param csr CSR matrix (GPU).
 * @param n Number of columns in the B-matrix.
 * @param B Input matrix B (GPU).
 * @param C Output matrix C (GPU).
 * @param stream CUDA stream.
 */
template <typename index_t, typename offset_t, typename type_t>
void thread_mapped(csr_t<index_t, offset_t, type_t>& csr,
                   csc_t<index_t, offset_t, type_t>& csc,
                   csr_t<index_t, offset_t, type_t>& C,
                  //  int* c_nnz_by_row,
                   index_t* tmp_c_indices,
                   type_t* tmp_c_values,
                   cudaStream_t stream = 0) {
  // Create a schedule.
  constexpr std::size_t block_size = 128;


  /*
  /// Set-up kernel launch parameters and run the kernel.

  // Create a schedule.
  using setup_t = schedule::setup<schedule::algorithms_t::thread_mapped, 1, 1,
                                  index_t, offset_t>;
  setup_t config(csr.offsets.data().get(), csr.rows, csr.nnzs);

  std::size_t grid_size = (csr.rows + block_size - 1) / block_size;
   
  launch::non_cooperative(
      stream, __thread_mapped<setup_t, index_t, offset_t, type_t>, grid_size,
      block_size, config, csr.rows, csr.cols, csr.nnzs,
      csr.offsets.data().get(), csr.indices.data().get(),
      csr.values.data().get(), csc.rows, csc.cols, csc.nnzs,
      csc.offsets.data().get(), csc.indices.data().get(),
      csc.values.data().get(), C.offsets.data().get(),
      // c_nnz_by_row, 
      tmp_c_indices, tmp_c_values);
  */

  using setup_t = schedule::setup<schedule::algorithms_t::thread_mapped, 1, 1,
                                  index_t, offset_t>;
  setup_t config(csr.offsets.data().get(), csr.rows, csr.nnzs);

  std::size_t grid_size = (csr.rows + block_size - 1) / block_size; // fix grid size to a constant if the matrix is VERY big
   
  launch::non_cooperative(
      stream, __thread_mapped_v2<setup_t, index_t, offset_t, type_t>, grid_size,
      block_size, config, csr.rows, csr.cols, csr.nnzs,
      csr.offsets.data().get(), csr.indices.data().get(),
      csr.values.data().get(), csc.rows, csc.cols, csc.nnzs,
      csc.offsets.data().get(), csc.indices.data().get(),
      csc.values.data().get(), C.offsets.data().get(),
      // c_nnz_by_row, 
      tmp_c_indices, tmp_c_values);
  cudaStreamSynchronize(stream);
}

}  // namespace spgemm
}  // namespace algorithms
}  // namespace loops