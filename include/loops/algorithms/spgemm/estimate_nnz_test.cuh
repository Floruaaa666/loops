/**
 * @file estimate_nnz_test.cuh
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
#include <stdio.h>
#include <cub/cub.cuh>

#include <array>


#define TILE_SIZE 32


namespace loops {
namespace algorithms {
namespace spgemm {

template <typename setup_t,
          typename index_t,
          typename offset_t,
          typename type_t>
__global__ void __estimate_nnz_test(setup_t config,
                                const std::size_t a_rows,
                                const std::size_t a_cols,
                                const std::size_t a_nnz,
                                const offset_t* a_offsets,
                                const index_t* a_indices,
                                const std::size_t b_rows,
                                const std::size_t b_cols,
                                const std::size_t b_nnz,
                                const offset_t* b_offsets,
                                const index_t* b_indices,
                                int* c_nnz_per_row) {
  
  for (auto mm : config.tiles()) {
    bool found = false;
    for (auto nn :
         custom_stride_range(std::size_t(0), b_cols, std::size_t(1))) {
      type_t sum = 0;
      for (auto nz : config.atoms(mm)) {
        auto kk_a = a_indices[nz];
          for (auto nz_b = b_offsets[nn]; nz_b < b_offsets[nn + 1]; ++nz_b) {
            if(kk_a == b_indices[nz_b]&&!found){
              ++c_nnz_per_row[mm];
              found = true;
            }
          }
      }
      found = false;
    }
  }
}

template <typename setup_t,
          typename index_t,
          typename offset_t,
          typename type_t>
__global__ void __estimate_nnz_test_v2(setup_t config,
                                const std::size_t a_rows,
                                const std::size_t a_cols,
                                const std::size_t a_nnz,
                                const offset_t* a_offsets,
                                const index_t* a_indices,
                                const std::size_t b_rows,
                                const std::size_t b_cols,
                                const std::size_t b_nnz,
                                const offset_t* b_offsets,
                                const index_t* b_indices,
                                int* c_nnz_per_row) {
  
  for (auto mm : config.tiles()) {
    bool found = false;
    for (auto nz : config.atoms(mm)) {
      auto kk_a = a_indices[nz];
        for (auto nn : custom_stride_range(std::size_t(0), b_cols, std::size_t(1))) {
          for (auto nz_b = b_offsets[nn]; nz_b < b_offsets[nn + 1]; ++nz_b) {
            if(kk_a == b_indices[nz_b]&&!found){
              ++c_nnz_per_row[mm];
              found = true;
            }
          }
        }
    found = false;
    }
  }
}

template <typename setup_t,
          typename index_t,
          typename offset_t,
          typename type_t>
__global__ void __estimate_nnz_test_v3(setup_t config,
                                const std::size_t a_rows,
                                const std::size_t a_cols,
                                const std::size_t a_nnz,
                                const offset_t* a_offsets,
                                const index_t* a_indices,
                                const std::size_t b_rows,
                                const std::size_t b_cols,
                                const std::size_t b_nnz,
                                const offset_t* b_offsets,
                                const index_t* b_indices,
                                int* c_nnz_per_row) {

  __shared__ index_t shared_A_cols[TILE_SIZE * TILE_SIZE];
  __shared__ index_t shared_B_rows[TILE_SIZE * TILE_SIZE];

  int tx = threadIdx.x, ty = threadIdx.y;

  for (auto mm : config.tiles()) { // stride through rows of A
    bool found = false;
    for (auto tile_itr : custom_stride_range(std::size_t(0), std::size_t((b_cols + TILE_SIZE -1) / TILE_SIZE), std::size_t(1))){
      // Load a tile of A into shared memory
      for (auto i : custom_stride_range(std::size_t(0), std::size_t(TILE_SIZE), std::size_t(1))){
        shared_A_cols[tx + i * TILE_SIZE] = a_indices[a_offsets[mm] + tx + i * TILE_SIZE];
      }
      __syncthreads();


      // Load a tile of B into shared memory
      for (auto i : custom_stride_range(std::size_t(0), std::size_t(TILE_SIZE), std::size_t(1))){
        shared_B_rows[ty + i * TILE_SIZE] = b_indices[b_offsets[tile_itr * TILE_SIZE + ty + i * TILE_SIZE]];
      }

      __syncthreads();

      for (auto ak : custom_stride_range(std::size_t(0), std::size_t(TILE_SIZE), std::size_t(1))){
        for (auto bk : custom_stride_range(std::size_t(0), std::size_t(TILE_SIZE), std::size_t(1))){
          if(shared_A_cols[tx + ak * TILE_SIZE] == shared_B_rows[ty + bk * TILE_SIZE]&&!found){
            atomicAdd(&c_nnz_per_row[mm], 1);
            found = true;
          }
        }
      }
      // __syncthreads();
      found = false;

    }
  }
}

// Tiling
template <typename setup_t,
          typename index_t,
          typename offset_t,
          typename type_t>
__global__ void __estimate_nnz_test_v4(setup_t config,
                                const std::size_t a_rows,
                                const std::size_t a_cols,
                                const std::size_t a_nnz,
                                const offset_t* a_offsets,
                                const index_t* a_indices,
                                const std::size_t b_rows,
                                const std::size_t b_cols,
                                const std::size_t b_nnz,
                                const offset_t* b_offsets,
                                const index_t* b_indices,
                                int* c_nnz_per_row) {

  __shared__ index_t shared_A_cols[TILE_SIZE * TILE_SIZE];
  __shared__ index_t shared_B_rows[TILE_SIZE * TILE_SIZE];

  int tx = threadIdx.x, ty = threadIdx.y;
  int bx = blockIdx.x, by = blockIdx.y;

  auto m_global_idx = by * TILE_SIZE + ty;

  

  for(auto m : custom_stride_range(std::size_t(m_global_idx), std::size_t(a_rows), std::size_t(TILE_SIZE))){ // Stride over the rows of A with the stride width of M0 = TILE_SIZE
    bool found = false;

    // Load a tile of A into shared memory
    auto ka_start = a_offsets[m] + tx;
    auto ka_end = a_offsets[m + 1];
    for(auto col_arr_idx : custom_stride_range(std::size_t(ka_start), std::size_t(ka_end), std::size_t(TILE_SIZE))){
      shared_A_cols[ty * TILE_SIZE + tx] = a_indices[col_arr_idx];
    }
    __syncthreads();

    
    for (auto n1 : custom_stride_range(std::size_t(0), std::size_t((b_cols + TILE_SIZE -1) / TILE_SIZE), std::size_t(1))){
      // Load a tile of B into shared memory
      for (auto i : custom_stride_range(std::size_t(0), std::size_t(TILE_SIZE), std::size_t(1))){
        shared_B_rows[ty + i * TILE_SIZE] = b_indices[b_offsets[n1 * TILE_SIZE + ty + i * TILE_SIZE]];
      }
    }
    __syncthreads();

    for (auto ak : custom_stride_range(std::size_t(0), std::size_t(TILE_SIZE), std::size_t(1))){
      for (auto bk : custom_stride_range(std::size_t(0), std::size_t(TILE_SIZE), std::size_t(1))){
        if(shared_A_cols[tx + ak * TILE_SIZE] == shared_B_rows[ty + bk * TILE_SIZE]&&!found){
          atomicAdd(&c_nnz_per_row[m], 1);
          found = true;
        }
      }
    }
      // __syncthreads();
    found = false;
  }
}

// Tile by row, col pair of the input matrices
// For input matrices with number of columns and rows <= TILE_SIZE && B_nnz < TILE_SIZE * TILE_SIZE
template <typename setup_t,
          typename index_t,
          typename offset_t,
          typename type_t>
__global__ void __estimate_nnz_row_col_pairs_v1(setup_t config,
                                const std::size_t a_rows,
                                const std::size_t a_cols,
                                const std::size_t a_nnz,
                                const offset_t* a_offsets,
                                const index_t* a_indices,
                                const std::size_t b_rows,
                                const std::size_t b_cols,
                                const std::size_t b_nnz,
                                const offset_t* b_offsets,
                                const index_t* b_indices,
                                int* c_nnz_per_row) {

  __shared__ index_t shared_A_cols[TILE_SIZE * TILE_SIZE];
  __shared__ index_t shared_B_rows[TILE_SIZE * TILE_SIZE];
  // __shared__ int found; // allocate found in shared memory so that all the threads can read and write to it
  __shared__ int C_n_nnz_per_block[TILE_SIZE];

  int tx = threadIdx.x, bx = blockIdx.x;
  // For every block: load ONE row of A into shared memory, load as much of B as possible into shared memory
  // For every thread: load (k - ty + 1)/TILE_SIZE elements of row m of A into shared memory, load ONE column of B into shared memory
  
  auto m = bx;

  bool found = false;
  C_n_nnz_per_block[tx] = 0;
  __syncthreads();

  if(m < a_rows){
    auto col_arr_start = a_offsets[m];
    auto col_arr_end = a_offsets[m + 1];
    auto range = col_arr_end - col_arr_start;

    // Every thread loads one element of the mth row of A into shared memory
    shared_A_cols[tx] = a_indices[col_arr_start + tx];
    __syncthreads();
  }

    for(int i = 0; i < gridDim.x; ++i){
    if(bx == i && tx == 0){
      auto start = a_offsets[i];
      auto end = a_offsets[i + 1];
      auto range_i = end - start;
      for(int k0 = 0; k0 < range_i; ++k0){
        // if(shared_A_cols[k0] != a_indices[start + k0]){
          printf("m%d: shared_A_cols[%d] = %d  a_indices[%d] = %d\n", m, k0, shared_A_cols[k0], k0 + start, a_indices[k0 + start]);
        // }
      }
    }
  }

  int n = tx;
  auto row_arr_start = b_offsets[n];
  auto row_arr_end = b_offsets[n + 1];
  for(int k0 = row_arr_start; k0 < row_arr_end; ++k0){
    shared_B_rows[k0] = b_indices[k0];
  }
  __syncthreads();

  // for(int i = 0; i < gridDim.x; ++i){
  //   if(bx == 0 && tx == 0){
  //     auto start = b_offsets[0];
  //     for(int k0 = 0; k0 < b_nnz; ++k0){
  //       // if(shared_B_rows[k0] != b_indices[start + k0]){
  //         printf("shared_B_rows[%d] = %d  b_indices[%d] = %d\n", k0, shared_B_rows[k0], start + k0, b_indices[start + k0]);
  //       // }
  //     }
  //   }
  // }

  std::array <int, 8> helperArray;
  if(m < a_rows){
    int n = tx;
    auto row_arr_start = b_offsets[n];
    auto row_arr_end = b_offsets[n + 1];
    for(int row_arr_itr_b = row_arr_start; row_arr_itr_b < row_arr_end; ++row_arr_itr_b){ // Iterate over all the elements in nth column of B
      auto col_arr_start = a_offsets[m];
      auto col_arr_end = a_offsets[m + 1];
      auto range = col_arr_end - col_arr_start;
      for(auto col_arr_itr_a = 0; col_arr_itr_a < range; ++col_arr_itr_a){
        if((shared_A_cols[col_arr_itr_a] == shared_B_rows[row_arr_itr_b])){
          found = true;
          C_n_nnz_per_block[n] += 1;

           if(bx == 1){
            helperArray[0] = m;
            helperArray[1] = n;
            helperArray[2] = col_arr_itr_a;
            helperArray[3] = shared_A_cols[col_arr_itr_a];
            helperArray[4] = row_arr_itr_b - row_arr_start;
            helperArray[5] = shared_B_rows[row_arr_itr_b];
            helperArray[6] = C_n_nnz_per_block[n];

            printf("m(bx): %d, n(tx): %d, col_arr_itr_a: %d\nshared_A_cols[%d]: %d, shared_B_rows[%d]: %d\nC_n_nnz_per_block[%d]: %d\n", helperArray[0], helperArray[1], helperArray[2], helperArray[2], helperArray[3], helperArray[4], helperArray[5], helperArray[1], helperArray[6]);
          }

          break;
        }
      }
      if(found) break;
    }
  }
  __syncthreads();

  int C_n_nnz = C_n_nnz_per_block[tx];

  typedef cub::BlockReduce<int, TILE_SIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int C_nnz_per_row = BlockReduce(temp_storage).Sum(C_n_nnz);

  // for(int i = 0; i < gridDim.x; ++i){
  //   if(bx == i && tx == 0){
  //     printf("bx: %d, C_nnz_per_row: %d\n", bx, C_nnz_per_row);
  //   }
  // }

  c_nnz_per_row[m] = C_nnz_per_row;
}



// For input matrices with number of columns and rows > TILE_SIZE && B_nnz < TILE_SIZE * TILE_SIZE
template <typename setup_t,
          typename index_t,
          typename offset_t,
          typename type_t>
__global__ void __estimate_nnz_row_col_pairs_v2(setup_t config,
                                const std::size_t a_rows,
                                const std::size_t a_cols,
                                const std::size_t a_nnz,
                                const offset_t* a_offsets,
                                const index_t* a_indices,
                                const std::size_t b_rows,
                                const std::size_t b_cols,
                                const std::size_t b_nnz,
                                const offset_t* b_offsets,
                                const index_t* b_indices,
                                int* c_nnz_per_row) {

  __shared__ index_t shared_A_cols[TILE_SIZE * TILE_SIZE];
  __shared__ index_t shared_B_rows[TILE_SIZE * TILE_SIZE];
  __shared__ int C_n_nnz_per_block[TILE_SIZE * TILE_SIZE];

  int tx = threadIdx.x, bx = blockIdx.x;

  auto m = bx;

  bool found = false;

  C_n_nnz_per_block[tx] = 0;
  __syncthreads();

  if(m < a_rows){
    auto col_arr_start = a_offsets[m];
    auto col_arr_end = a_offsets[m + 1];
    auto range = col_arr_end - col_arr_start;

    for(int col_arr_itr = tx; col_arr_itr < range; col_arr_itr += TILE_SIZE){
      shared_A_cols[col_arr_itr] = a_indices[col_arr_start + col_arr_itr];
    }
    __syncthreads();
  }

  // for(int i = 0; i < gridDim.x; ++i){
    // int i = 54;
    // if(bx == i && tx == 0){
    //   auto start = a_offsets[i];
    //   auto end = a_offsets[i + 1];
    //   auto range_i = end - start;
    //   for(int k0 = 0; k0 < range_i; ++k0){
    //     // if(shared_A_cols[k0] != a_indices[start + k0]){
    //       printf("m%d: shared_A_cols[%d] = %d  a_indices[%d] = %d\n", m, k0, shared_A_cols[k0], k0 + start, a_indices[k0 + start]);
    //     // }
    //   }
    // }
  // }


  for(int n0 = tx; n0 < b_cols; n0 += TILE_SIZE){ // Each tx load n0 and n0 + (b_cols / TILE_SIZE) columns of B into shared memory
    auto row_arr_start = b_offsets[n0];
    auto row_arr_end = b_offsets[n0 + 1];
    for(int k0 = row_arr_start; k0 < row_arr_end; ++k0){
      shared_B_rows[k0] = b_indices[k0];
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


  // for(int i = 0; i < gridDim.x; ++i){
  //   if(bx == i && tx == 0){
  //     printf("block: %d\n", bx);
  //     auto start = b_offsets[0];
  //     for(int k0 = 0; k0 < TILE_SIZE * TILE_SIZE; ++k0){
  //       if(shared_B_rows[k0] != b_indices[start + k0]){
  //         printf("shared_B_rows[%d] = %d  b_indices[%d] = %d\n", k0, shared_B_rows[k0], start + k0, b_indices[start + k0]);
  //       }
  //     }
  //   }
  // }


  std::array <int, 8> helperArray;
  if(m < a_rows){
    for(int n = tx; n < b_cols; n += TILE_SIZE){
      auto row_arr_start = b_offsets[n];
      auto row_arr_end = b_offsets[n + 1];
      found = false;
      for(int row_arr_itr_b = row_arr_start; row_arr_itr_b < row_arr_end; ++row_arr_itr_b){ // Iterate over all the elements in nth column of B
        auto col_arr_start = a_offsets[m];
        auto col_arr_end = a_offsets[m + 1];
        auto range = col_arr_end - col_arr_start;

        // for(auto col_arr_itr_a = col_arr_start; col_arr_itr_a < col_arr_end; ++col_arr_itr_a){
        for(auto col_arr_itr_a = 0; col_arr_itr_a < range; ++col_arr_itr_a){

          // if(bx == 10 && n == 44){
          //   helperArray[2] = col_arr_itr_a;
          //   helperArray[3] = shared_A_cols[col_arr_itr_a];
          //   helperArray[4] = row_arr_itr_b - row_arr_start;
          //   helperArray[5] = shared_B_rows[row_arr_itr_b];
          //   printf("bx: 10, tx: 44\nshared_A_cols[%d]: %d, shared_B_rows[%d]: %d\nC_n_nnz_per_block[%d]: %d\n", helperArray[2], helperArray[3], helperArray[4], helperArray[5], n, C_n_nnz_per_block[n]);

          // }
          if((shared_A_cols[col_arr_itr_a] == shared_B_rows[row_arr_itr_b])){
            found = true;

            C_n_nnz_per_block[n % TILE_SIZE] += 1;
            
            // if(bx == 10){
            //   helperArray[0] = m;
            //   helperArray[1] = n;
            //   helperArray[2] = col_arr_itr_a;
            //   helperArray[3] = shared_A_cols[col_arr_itr_a];
            //   helperArray[4] = row_arr_itr_b - row_arr_start;
            //   helperArray[5] = shared_B_rows[row_arr_itr_b];
            //   helperArray[6] = C_n_nnz_per_block[n];

            //   printf("m(bx): %d, n(tx): %d, col_arr_itr_a: %d\nshared_A_cols[%d]: %d, shared_B_rows[%d]: %d\nC_n_nnz_per_block[%d]: %d\n", helperArray[0], helperArray[1], helperArray[2], helperArray[2], helperArray[3], helperArray[4], helperArray[5], helperArray[1], helperArray[6]);
            // }

            break;
          }
        }
        if(found) break;
      }
    }
    
  }
  __syncthreads();


  // if(bx == 10 && tx == 0){
  //   for(int i = 0; i < TILE_SIZE; ++i){
  //     printf("C_n_nnz_per_block[%d]: %d\n", i, C_n_nnz_per_block[i]);
  //   }
  // }

  int C_n_nnz = C_n_nnz_per_block[tx];

  typedef cub::BlockReduce<int, TILE_SIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int C_nnz_per_row = BlockReduce(temp_storage).Sum(C_n_nnz);

  // for(int i = 0; i < gridDim.x; ++i){
    // if(bx == 10 && tx == 0){
    //   printf("bx: %d, C_nnz_per_row: %d\n", bx, C_nnz_per_row);
    // }
  // }

  c_nnz_per_row[m] = C_nnz_per_row;

}


// For input matrices with number of columns and rows > TILE_SIZE && B_nnz <= TILE_SIZE * TILE_SIZE
// Add striding to A rows
template <typename setup_t,
          typename index_t,
          typename offset_t,
          typename type_t>
__global__ void __estimate_nnz_row_col_pairs_v3(setup_t config,
                                const std::size_t a_rows,
                                const std::size_t a_cols,
                                const std::size_t a_nnz,
                                const offset_t* a_offsets,
                                const index_t* a_indices,
                                const std::size_t b_rows,
                                const std::size_t b_cols,
                                const std::size_t b_nnz,
                                const offset_t* b_offsets,
                                const index_t* b_indices,
                                int* c_nnz_per_row) {

  __shared__ index_t shared_A_cols[TILE_SIZE * TILE_SIZE];
  __shared__ index_t shared_B_rows[TILE_SIZE * TILE_SIZE];
  __shared__ int C_n_nnz_per_m0[TILE_SIZE * TILE_SIZE];

  int tx = threadIdx.x, bx = blockIdx.x;

  auto m = bx;

  bool found = false;

  C_n_nnz_per_m0[tx] = 0;
  __syncthreads();

  std::array<int, 10> test;

  int shared_mem_prev_col_arr_range = 0;
  for(int m0 = bx; m0 < a_rows; m0 += gridDim.x){ // Stride over the rows of A with the stride width of gridDim.x
    auto col_arr_start = a_offsets[m0];
    auto col_arr_end = a_offsets[m0 + 1];
    auto shared_mem_curr_col_arr_range = col_arr_end - col_arr_start;

    for(int col_arr_itr = tx; col_arr_itr < shared_mem_curr_col_arr_range; col_arr_itr += TILE_SIZE){
      shared_A_cols[col_arr_itr + shared_mem_prev_col_arr_range] = a_indices[col_arr_itr + col_arr_start];
    }
    shared_mem_prev_col_arr_range += shared_mem_curr_col_arr_range;
  }
  __syncthreads();

  for(int n0 = tx; n0 < b_cols; n0 += TILE_SIZE){ // Each tx load n0 and n0 + (b_cols / TILE_SIZE) columns of B into shared memory
    auto row_arr_start = b_offsets[n0];
    auto row_arr_end = b_offsets[n0 + 1];
    for(int k0 = row_arr_start; k0 < row_arr_end; ++k0){
      shared_B_rows[k0] = b_indices[k0];
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

  int prev_col_arr_range = 0;
  for(int m0 = bx; m0 < a_rows; m0 += gridDim.x){ //TODO: which loop order will be faster? m0->n0->kb->ka or n0->kb->m0->ka?
    auto col_arr_start = a_offsets[m0];
    auto col_arr_end = a_offsets[m0 + 1];
    auto curr_col_arr_range = col_arr_end - col_arr_start;

    for(int n = tx; n < b_cols; n += TILE_SIZE){
      auto row_arr_start = b_offsets[n];
      auto row_arr_end = b_offsets[n + 1];
      found = false;

      for(int row_arr_itr_b = row_arr_start; row_arr_itr_b < row_arr_end; ++row_arr_itr_b){ // Iterate over all the elements in nth column of B
        for(auto col_arr_itr_a = 0; col_arr_itr_a < curr_col_arr_range; ++col_arr_itr_a){
          if((shared_A_cols[col_arr_itr_a + prev_col_arr_range] == shared_B_rows[row_arr_itr_b])){
            found = true;
            C_n_nnz_per_m0[n % TILE_SIZE] += 1;
            break;
          }
        }
        if(found) break;
      }
    }
    __syncthreads();

    int C_n_nnz = C_n_nnz_per_m0[tx];
    typedef cub::BlockReduce<int, TILE_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    int C_nnz_per_row = BlockReduce(temp_storage).Sum(C_n_nnz);  
    c_nnz_per_row[m0] = C_nnz_per_row;

    C_n_nnz_per_m0[tx] = 0;
    __syncthreads();

    prev_col_arr_range += curr_col_arr_range;
  }
  __syncthreads();
}


// For input matrices with number of columns and rows > TILE_SIZE && B_nnz > TILE_SIZE * TILE_SIZE
// Add striding to A rows
template <typename setup_t,
          typename index_t,
          typename offset_t,
          typename type_t>
__global__ void __estimate_nnz_row_col_pairs_v4(setup_t config,
                                const std::size_t a_rows,
                                const std::size_t a_cols,
                                const std::size_t a_nnz,
                                const offset_t* a_offsets,
                                const index_t* a_indices,
                                const std::size_t b_rows,
                                const std::size_t b_cols,
                                const std::size_t b_nnz,
                                const offset_t* b_offsets,
                                const index_t* b_indices,
                                int* c_nnz_per_row) {

  __shared__ index_t shared_A_cols[TILE_SIZE * TILE_SIZE];
  __shared__ index_t shared_B_rows[TILE_SIZE * TILE_SIZE];
  __shared__ int C_n_nnz_per_m0[TILE_SIZE * TILE_SIZE];

  int tx = threadIdx.x, bx = blockIdx.x;

  bool found = false;

  C_n_nnz_per_m0[tx] = 0;
  __syncthreads();

  std::array<int, 10> test;

  int shared_mem_prev_col_arr_range = 0;
  // int m0 = bx;
  // while(m0 <  a_rows && (a_offsets[m0 + 1] - a_offsets[m0]) < (TILE_SIZE * TILE_SIZE - shared_mem_prev_col_arr_range))
  for(int m0 = bx; m0 < a_rows && (a_offsets[m0 + 1] - a_offsets[m0]) <= (TILE_SIZE * TILE_SIZE - shared_mem_prev_col_arr_range); m0 += gridDim.x) //can't exploit the shared memory b/c the shared memory isn't large enough to take an entire row of A
  // for(int m0 = bx; m0 < a_rows; m0 += gridDim.x)
  { // Stride over the rows of A with the stride width of gridDim.x
    auto col_arr_start = a_offsets[m0];
    auto col_arr_end = a_offsets[m0 + 1];
    auto shared_mem_curr_col_arr_range = col_arr_end - col_arr_start;
    
    // if(bx == 0 && tx == 0)
    // if(tx == 0)
    // {
    //   printf("m0: %d\na_offsets[%d]: %d, a_offsets[%d + 1]: %d\nshared_mem_prev_col_arr_range: %d, TILE_SIZE * TILE_SIZE - shared_mem_prev_col_arr_range(%d - %d) = %d, shared_mem_curr_col_arr_range: %d\n", m0, m0, col_arr_start, m0, col_arr_end, shared_mem_prev_col_arr_range, TILE_SIZE * TILE_SIZE, shared_mem_prev_col_arr_range, TILE_SIZE * TILE_SIZE - shared_mem_prev_col_arr_range, shared_mem_curr_col_arr_range);
    // }

    for(int col_arr_itr = tx; col_arr_itr < shared_mem_curr_col_arr_range; col_arr_itr += TILE_SIZE){
      shared_A_cols[col_arr_itr + shared_mem_prev_col_arr_range] = a_indices[col_arr_itr + col_arr_start];

      // if(bx == 0 && (tx == 0 || tx == 1)){
      // if(m0 == 0){
      //   test[0] = m0;
      //   test[1] = col_arr_itr;
      //   test[2] = shared_mem_prev_col_arr_range;
      //   test[3] = col_arr_itr + shared_mem_prev_col_arr_range;
      //   test[4] = shared_A_cols[col_arr_itr + shared_mem_prev_col_arr_range];
      //   test[9] = col_arr_start;
      //   test[5] = col_arr_itr + col_arr_start;
      //   test[6] = a_indices[col_arr_itr + col_arr_start];
      //   test[7] = bx;
      //   test[8] = shared_mem_curr_col_arr_range;

      //   printf("m0: %d, bx: %d\nshared_A_cols[%d + %d = %d]: %d, a_indices[%d + %d = %d]: %d\nshared_mem_prev_col_arr_range: %d, shared_mem_curr_col_arr_range: %d\n", test[0], test[7], test[1], test[2], test[3], test[4], test[1], test[9], test[5], test[6], test[2], test[8]);
      // }

    }
    shared_mem_prev_col_arr_range += shared_mem_curr_col_arr_range;
  }
  __syncthreads();

  // while(m0 < a_rows){
  //   m0 += gridDim.x
  // }

  // for(int i = 0; i < gridDim.x; ++i){
    // int i = 0;
    // if(bx == i && tx == 0){
    //   for(int k = 0; k < TILE_SIZE * TILE_SIZE; ++k){
    //     printf("shared_A_cols[%d]: %d\n", k, shared_A_cols[k]);
    //   }
    // }
  // }

  for(int n0 = tx; n0 < b_cols && b_offsets[n0 + 1] <= TILE_SIZE * TILE_SIZE; n0 += TILE_SIZE)
  {
      auto row_arr_start = b_offsets[n0];
      auto row_arr_end = b_offsets[n0 + 1];
      for(int k0 = row_arr_start; k0 < row_arr_end; ++k0){
        shared_B_rows[k0] = b_indices[k0];
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

  // if(bx == 0 && tx == 0){
  //   auto start = b_offsets[0];
  //   for(int k0 = 0; k0 < TILE_SIZE * TILE_SIZE; ++k0){
  //     // if(shared_B_rows[k0] != b_indices[start + k0]){
  //       printf("shared_B_rows[%d] = %d  b_indices[%d] = %d\n", k0, shared_B_rows[k0], start + k0, b_indices[start + k0]);
  //     // }
  //   }
  // }

  std::array <int, 8> helperArray;

  // SHARED_A:
  int prev_col_arr_range = 0;
  int m0 = bx;
  while(m0 < a_rows && (a_offsets[m0 + 1] - a_offsets[m0]) <= (TILE_SIZE * TILE_SIZE - prev_col_arr_range))
  // for(int m0 = bx; m0 < a_rows; m0 += gridDim.x)
  { //TODO: which loop order will be faster? m0->n0->kb->ka or n0->kb->m0->ka?

  // /*
    auto col_arr_start = a_offsets[m0];
    auto col_arr_end = a_offsets[m0 + 1];
    auto curr_col_arr_range = col_arr_end - col_arr_start;

    // if(bx == 56 && tx == 0){
    //   helperArray[0] = m0;
    //   helperArray[1] = col_arr_start;
    //   helperArray[2] = col_arr_end;
    //   helperArray[3] = curr_col_arr_range;
    //   helperArray[4] = prev_col_arr_range;
    //   helperArray[5] = shared_A_cols[prev_col_arr_range+1];
    //   printf("SHARED_A: m0: %d\ncol_arr_start: %d, col_arr_end: %d\ncurr_col_arr_range: %d, prev_col_arr_range: %d\nshared_A_cols[%d + 1] = %d\n", helperArray[0], helperArray[1], helperArray[2], helperArray[3], helperArray[4], helperArray[4], helperArray[5]);
    // }

    // for(int n = tx; n < b_cols; n += TILE_SIZE){
    // Using SHARED B
    int n = tx;
    while(n < b_cols && b_offsets[n + 1] < TILE_SIZE * TILE_SIZE){

      auto row_arr_start = b_offsets[n];
      auto row_arr_end = b_offsets[n + 1];
      found = false;

      for(int row_arr_itr_b = row_arr_start; row_arr_itr_b < row_arr_end; ++row_arr_itr_b){ // Iterate over all the elements in nth column of B
        for(auto col_arr_itr_a = 0; col_arr_itr_a < curr_col_arr_range; ++col_arr_itr_a){

          // if(bx == 0 && tx == 0){
          //   helperArray[0] = m0;
          //   helperArray[1] = n;
          //   helperArray[2] = col_arr_itr_a;
          //   helperArray[3] = shared_A_cols[col_arr_itr_a + prev_col_arr_range];
          //   helperArray[4] = row_arr_itr_b - row_arr_start;
          //   helperArray[5] = shared_B_rows[row_arr_itr_b];
          //   printf("bx: 0, m0: %d, n: %d\nshared_A_cols[%d]: %d, shared_B_rows[%d]: %d\nC_n_nnz_per_m0[%d]: %d\n", helperArray[0], helperArray[1], helperArray[2], helperArray[3], helperArray[4], helperArray[5], helperArray[1], C_n_nnz_per_m0[n]);
          // }

          if((shared_A_cols[col_arr_itr_a + prev_col_arr_range] == shared_B_rows[row_arr_itr_b])){
            found = true;
            C_n_nnz_per_m0[n % TILE_SIZE] += 1;
            
            // if(bx == 56){
            //   helperArray[0] = m0;
            //   helperArray[1] = n;
            //   helperArray[2] = col_arr_itr_a + prev_col_arr_range;
            //   helperArray[3] = shared_A_cols[col_arr_itr_a + prev_col_arr_range];
            //   helperArray[4] = row_arr_itr_b - row_arr_start;
            //   helperArray[5] = shared_B_rows[row_arr_itr_b];
            //   helperArray[6] = C_n_nnz_per_m0[n % TILE_SIZE];

            //   // printf("SHARED_A && SHARED_B: m0: %d, n(tx): %d, col_arr_itr_a: %d\nshared_A_cols[%d]: %d, shared_B_rows[%d]: %d\nC_n_nnz_per_m0[%d % 32 = %d ]: %d\n", helperArray[0], helperArray[1], helperArray[2], helperArray[2], helperArray[3], helperArray[4], helperArray[5], helperArray[1], helperArray[1]%32, helperArray[6]);
            //   printf("SHARED_A && SHARED_B:\nm0: %d, n(tx): %d\nshared_A_cols[%d]: %d, shared_B_rows[%d]: %d\nC_n_nnz_per_m0[%d % 32 = %d ]: %d\n", helperArray[0], helperArray[1], helperArray[2], helperArray[3], helperArray[4], helperArray[5], helperArray[1], helperArray[1]%32, helperArray[6]); 
            // }

            break;
          }
        }
        if(found) break;
      }
      n += TILE_SIZE;
    }
    __syncthreads();

    // int n = tx;
    // Using GLOBAL B
    while(n < b_cols && b_offsets[n + 1] >= TILE_SIZE * TILE_SIZE){
      auto row_arr_start = b_offsets[n];
      auto row_arr_end = b_offsets[n + 1];
      found = false;
      for(int row_arr_itr_b = row_arr_start; row_arr_itr_b < row_arr_end; ++row_arr_itr_b){ // Iterate over all the elements in nth column of B
        for(auto col_arr_itr_a = 0; col_arr_itr_a < curr_col_arr_range; ++col_arr_itr_a){
          // if(bx == 0 && tx == 0){
          //   helperArray[0] = m0;
          //   helperArray[1] = n;
          //   helperArray[2] = col_arr_itr_a;
          //   helperArray[3] = shared_A_cols[col_arr_itr_a + prev_col_arr_range];
          //   helperArray[4] = row_arr_itr_b - row_arr_start;
          //   helperArray[5] = b_indices[row_arr_itr_b];
          //   printf("bx: 0, m0: %d, n: %d\nshared_A_cols[%d]: %d, shared_B_rows[%d]: %d\nC_n_nnz_per_m0[%d]: %d\n", helperArray[0], helperArray[1], helperArray[2], helperArray[3], helperArray[4], helperArray[5], helperArray[1], C_n_nnz_per_m0[n]);
          // }
          if((shared_A_cols[col_arr_itr_a + prev_col_arr_range] == b_indices[row_arr_itr_b])){
            found = true;
            C_n_nnz_per_m0[n % TILE_SIZE] += 1;
            
            // if(bx == 56){
            //   helperArray[0] = m0;
            //   helperArray[1] = n;
            //   helperArray[2] = col_arr_itr_a + prev_col_arr_range;
            //   helperArray[3] = shared_A_cols[col_arr_itr_a + prev_col_arr_range];
            //   helperArray[4] = row_arr_itr_b - row_arr_start;
            //   helperArray[5] = b_indices[row_arr_itr_b];
            //   helperArray[6] = C_n_nnz_per_m0[n % TILE_SIZE];

            //   // printf("SHARED_A && GLOBAL_B: m0: %d, n(tx): %d, col_arr_itr_a: %d\nshared_A_cols[%d]: %d, shared_B_rows[%d]: %d\nC_n_nnz_per_m0[%d % 32 = %d ]: %d\n", helperArray[0], helperArray[1], helperArray[2], helperArray[2], helperArray[3], helperArray[4], helperArray[5], helperArray[1], helperArray[1]%32, helperArray[6]);
            //   printf("SHARED_A && GLOBAL_B:\nm0: %d, n(tx): %d\nshared_A_cols[%d]: %d, b_indices[%d]: %d\nC_n_nnz_per_m0[%d % 32 = %d ]: %d\n", helperArray[0], helperArray[1], helperArray[2], helperArray[3], helperArray[4], helperArray[5], helperArray[1], helperArray[1]%32, helperArray[6]); 
            // }

            break;
          }
        }
        if(found) break;
      }
      n += TILE_SIZE;
    }
    __syncthreads();

    // if(bx == 56 && tx == 0){
    //   for(int i = 0; i < TILE_SIZE; ++i){
    //     printf("C_n_nnz_per_m0[%d]: %d\n", i, C_n_nnz_per_m0[i % TILE_SIZE]);
    //   }
    // }

    int C_n_nnz = C_n_nnz_per_m0[tx];
    typedef cub::BlockReduce<int, TILE_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    int C_nnz_per_row = BlockReduce(temp_storage).Sum(C_n_nnz);  
    c_nnz_per_row[m0] = C_nnz_per_row;

    // if(bx == 56 && tx == 0){
    //   printf("SHARED_A: bx: %d, m0: %d, C_nnz_per_row: %d\n", bx, m0, C_nnz_per_row);
    // }

    C_n_nnz_per_m0[tx] = 0;
    __syncthreads();

    prev_col_arr_range += curr_col_arr_range;
    // */
    m0 += gridDim.x;
  }
  __syncthreads();

  // /*
  // GLOBAL A
  while(m0 < a_rows && (a_offsets[m0 + 1] - a_offsets[m0]) > (TILE_SIZE * TILE_SIZE - prev_col_arr_range))
  {
    auto col_arr_start = a_offsets[m0];
    auto col_arr_end = a_offsets[m0 + 1];
    auto curr_col_arr_range = col_arr_end - col_arr_start;

    // if(bx == 56 && tx == 0){
    //   helperArray[0] = m0;
    //   helperArray[1] = col_arr_start;
    //   helperArray[2] = col_arr_end;
    //   helperArray[3] = curr_col_arr_range;
    //   helperArray[4] = prev_col_arr_range;
    //   helperArray[5] = shared_A_cols[prev_col_arr_range+1];
    //   printf("GLOBAL_A: m0: %d\ncol_arr_start: %d, col_arr_end: %d\ncurr_col_arr_range: %d, prev_col_arr_range: %d\nshared_A_cols[%d + 1] = %d\n", helperArray[0], helperArray[1], helperArray[2], helperArray[3], helperArray[4], helperArray[4], helperArray[5]);
    // }

    // for(int n = tx; n < b_cols; n += TILE_SIZE){
    int n = tx;
    while(n < b_cols && b_offsets[n + 1] < TILE_SIZE * TILE_SIZE){

      auto row_arr_start = b_offsets[n];
      auto row_arr_end = b_offsets[n + 1];
      found = false;

      for(int row_arr_itr_b = row_arr_start; row_arr_itr_b < row_arr_end; ++row_arr_itr_b){ // Iterate over all the elements in nth column of B
        for(auto col_arr_itr_a = 0; col_arr_itr_a < curr_col_arr_range; ++col_arr_itr_a){

          // if(bx == 0 && tx == 0){
          //   helperArray[0] = m0;
          //   helperArray[1] = n;
          //   helperArray[2] = col_arr_itr_a;
          //   helperArray[3] = a_indices[col_arr_itr_a + prev_col_arr_range];
          //   helperArray[4] = row_arr_itr_b - row_arr_start;
          //   helperArray[5] = shared_B_rows[row_arr_itr_b];
          //   printf("bx: 0, m0: %d, n: %d\a_indices[%d]: %d, shared_B_rows[%d]: %d\nC_n_nnz_per_m0[%d]: %d\n", helperArray[0], helperArray[1], helperArray[2], helperArray[3], helperArray[4], helperArray[5], helperArray[1], C_n_nnz_per_m0[n]);
          // }

          // if((a_indices[col_arr_itr_a + prev_col_arr_range] == shared_B_rows[row_arr_itr_b]))
          if((a_indices[col_arr_itr_a + col_arr_start] == shared_B_rows[row_arr_itr_b]))
          {
            found = true;
            C_n_nnz_per_m0[n % TILE_SIZE] += 1;
            
          // if(bx == 56){
          //   helperArray[0] = m0;
          //     helperArray[1] = n;
          //     // helperArray[2] = col_arr_itr_a + prev_col_arr_range;
          //     // helperArray[3] = a_indices[col_arr_itr_a + prev_col_arr_range];
          //     helperArray[2] = col_arr_itr_a + col_arr_start;
          //     helperArray[3] = a_indices[col_arr_itr_a + col_arr_start];
          //     helperArray[4] = row_arr_itr_b - row_arr_start;
          //     helperArray[5] = shared_B_rows[row_arr_itr_b];
          //     helperArray[6] = C_n_nnz_per_m0[n % TILE_SIZE];

          //     // printf("GLOBAL_A && SHARED_B: m0: %d, n(tx): %d, col_arr_itr_a: %d\a_indices[%d]: %d, shared_B_rows[%d]: %d\nC_n_nnz_per_m0[%d % 32 = %d ]: %d\n", helperArray[0], helperArray[1], helperArray[2], helperArray[2], helperArray[3], helperArray[4], helperArray[5], helperArray[1], helperArray[1]%32, helperArray[6]);

          //     printf("GLOBAL_A && SHARED_B:\nm0: %d, n(tx): %d\na_indices[%d]: %d, shared_B_rows[%d]: %d\nC_n_nnz_per_m0[%d % 32 = %d ]: %d\n", helperArray[0], helperArray[1], helperArray[2], helperArray[3], helperArray[4], helperArray[5], helperArray[1], helperArray[1]%32, helperArray[6]);
          // }

            break;
          }
        }
        if(found) break;
      }
      n += TILE_SIZE;
    }
    __syncthreads();

    while(n < b_cols && b_offsets[n + 1] >= TILE_SIZE * TILE_SIZE){
      auto row_arr_start = b_offsets[n];
      auto row_arr_end = b_offsets[n + 1];
      found = false;
      for(int row_arr_itr_b = row_arr_start; row_arr_itr_b < row_arr_end; ++row_arr_itr_b){ // Iterate over all the elements in nth column of B
        for(auto col_arr_itr_a = 0; col_arr_itr_a < curr_col_arr_range; ++col_arr_itr_a){

          // if(bx == 0 && tx == 0){
          //   helperArray[0] = m0;
          //   helperArray[1] = n;
          //   helperArray[2] = col_arr_itr_a;
          //   helperArray[3] = a_indices[col_arr_itr_a + prev_col_arr_range];
          //   helperArray[4] = row_arr_itr_b - row_arr_start;
          //   helperArray[5] = b_indices[row_arr_itr_b];
          //   printf("bx: 0, m0: %d, n: %d\a_indices[%d]: %d, b_indices[%d]: %d\nC_n_nnz_per_m0[%d]: %d\n", helperArray[0], helperArray[1], helperArray[2], helperArray[3], helperArray[4], helperArray[5], helperArray[1], C_n_nnz_per_m0[n]);
          // }
          // if((a_indices[col_arr_itr_a + prev_col_arr_range] == b_indices[row_arr_itr_b]))
          if((a_indices[col_arr_itr_a + col_arr_start] == b_indices[row_arr_itr_b]))
          {
            found = true;
            C_n_nnz_per_m0[n % TILE_SIZE] += 1;
            
            /*
            if(bx == 56){
              helperArray[0] = m0;
              helperArray[1] = n;
              // helperArray[2] = col_arr_itr_a + prev_col_arr_range;
              // helperArray[3] = a_indices[col_arr_itr_a + prev_col_arr_range];
              helperArray[2] = col_arr_itr_a + col_arr_start;
              helperArray[3] = a_indices[col_arr_itr_a + col_arr_start];
              helperArray[4] = row_arr_itr_b - row_arr_start;
              helperArray[5] = b_indices[row_arr_itr_b];
              helperArray[6] = C_n_nnz_per_m0[n % TILE_SIZE];
              printf("GLOBAL_A && GLOBAL_B: m0: %d, n(tx): %d, col_arr_itr_a: %d\a_indices[%d]: %d, b_indices[%d]: %d\nC_n_nnz_per_m0[%d % 32 = %d ]: %d\n", helperArray[0], helperArray[1], helperArray[2], helperArray[2], helperArray[3], helperArray[4], helperArray[5], helperArray[1], helperArray[1]%32, helperArray[6]);
              printf("GLOBAL_A && GLOBAL_B:\nm0: %d, n(tx): %d\na_indices[%d]: %d, b_indices[%d]: %d\nC_n_nnz_per_m0[%d % 32 = %d ]: %d\n", helperArray[0], helperArray[1], helperArray[2], helperArray[3], helperArray[4], helperArray[5], helperArray[1], helperArray[1]%32, helperArray[6]);
            }
            */
           
            break;
          }
        }
        if(found) break;
      }
      n += TILE_SIZE;
    }
    __syncthreads();

    // if(bx == 56 && tx == 0){
    //   for(int i = 0; i < TILE_SIZE; ++i){
    //     printf("GLOBAL_A: C_n_nnz_per_m0[%d]: %d\n", i, C_n_nnz_per_m0[i % TILE_SIZE]);
    //   }
    // }

    int C_n_nnz = C_n_nnz_per_m0[tx];
    typedef cub::BlockReduce<int, TILE_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    int C_nnz_per_row = BlockReduce(temp_storage).Sum(C_n_nnz);  
    c_nnz_per_row[m0] = C_nnz_per_row;

    // if(bx == 56 && tx == 0){
    //   printf("GLOBAL_A: bx: %d, m0: %d, C_nnz_per_row: %d\n", bx, m0, C_nnz_per_row);
    // }

    C_n_nnz_per_m0[tx] = 0;
    __syncthreads();

    prev_col_arr_range += curr_col_arr_range;
    m0 += gridDim.x;
  }
  __syncthreads();
// */
}


// Precalculate the column indices of C
template <typename setup_t,
          typename index_t,
          typename offset_t,
          typename type_t>
__global__ void __precalculate_c_col_indices(setup_t config,
                                const std::size_t a_rows,
                                const std::size_t a_cols,
                                const std::size_t a_nnz,
                                const offset_t* a_offsets,
                                const index_t* a_indices,
                                const std::size_t b_rows,
                                const std::size_t b_cols,
                                const std::size_t b_nnz,
                                const offset_t* b_offsets,
                                const index_t* b_indices,
                                const std::size_t c_rows,
                                const std::size_t c_cols,
                                const std::size_t c_nnz,
                                const offset_t* c_offsets,
                                const index_t* c_indices) {

  __shared__ index_t shared_A_cols[TILE_SIZE * TILE_SIZE];
  __shared__ index_t shared_B_rows[TILE_SIZE * TILE_SIZE];
  __shared__ int C_n_nnz_per_m0[TILE_SIZE * TILE_SIZE];

  int tx = threadIdx.x, bx = blockIdx.x;

  bool found = false;

  C_n_nnz_per_m0[tx] = 0;
  __syncthreads();

  std::array<int, 10> test;

  int shared_mem_prev_col_arr_range = 0;
  for(int m0 = bx; m0 < a_rows && (a_offsets[m0 + 1] - a_offsets[m0]) <= (TILE_SIZE * TILE_SIZE - shared_mem_prev_col_arr_range); m0 += gridDim.x) //can't exploit the shared memory b/c the shared memory isn't large enough to take an entire row of A
  { // Stride over the rows of A with the stride width of gridDim.x
    auto col_arr_start = a_offsets[m0];
    auto col_arr_end = a_offsets[m0 + 1];
    auto shared_mem_curr_col_arr_range = col_arr_end - col_arr_start;
    

    for(int col_arr_itr = tx; col_arr_itr < shared_mem_curr_col_arr_range; col_arr_itr += TILE_SIZE){
      shared_A_cols[col_arr_itr + shared_mem_prev_col_arr_range] = a_indices[col_arr_itr + col_arr_start];

    }
    shared_mem_prev_col_arr_range += shared_mem_curr_col_arr_range;
  }
  __syncthreads();

  for(int n0 = tx; n0 < b_cols && b_offsets[n0 + 1] <= TILE_SIZE * TILE_SIZE; n0 += TILE_SIZE)
  {
      auto row_arr_start = b_offsets[n0];
      auto row_arr_end = b_offsets[n0 + 1];
      for(int k0 = row_arr_start; k0 < row_arr_end; ++k0){
        shared_B_rows[k0] = b_indices[k0];
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

  std::array <int, 8> helperArray;

  // SHARED_A:
  int prev_col_arr_range = 0;
  int m0 = bx;
  while(m0 < a_rows && (a_offsets[m0 + 1] - a_offsets[m0]) <= (TILE_SIZE * TILE_SIZE - prev_col_arr_range))
  { //TODO: which loop order will be faster? m0->n0->kb->ka or n0->kb->m0->ka?

  // /*
    auto col_arr_start = a_offsets[m0];
    auto col_arr_end = a_offsets[m0 + 1];
    auto curr_col_arr_range = col_arr_end - col_arr_start;

    // Using SHARED B
    int n = tx;
    while(n < b_cols && b_offsets[n + 1] < TILE_SIZE * TILE_SIZE){

      auto row_arr_start = b_offsets[n];
      auto row_arr_end = b_offsets[n + 1];
      found = false;

      for(int row_arr_itr_b = row_arr_start; row_arr_itr_b < row_arr_end; ++row_arr_itr_b){ // Iterate over all the elements in nth column of B
        for(auto col_arr_itr_a = 0; col_arr_itr_a < curr_col_arr_range; ++col_arr_itr_a){
          if((shared_A_cols[col_arr_itr_a + prev_col_arr_range] == shared_B_rows[row_arr_itr_b])){
            found = true;
            C_n_nnz_per_m0[n % TILE_SIZE] += 1;
            break;
          }
        }
        if(found) break;
      }
      n += TILE_SIZE;
    }
    __syncthreads();

    // int n = tx;
    // Using GLOBAL B
    while(n < b_cols && b_offsets[n + 1] >= TILE_SIZE * TILE_SIZE){
      auto row_arr_start = b_offsets[n];
      auto row_arr_end = b_offsets[n + 1];
      found = false;
      for(int row_arr_itr_b = row_arr_start; row_arr_itr_b < row_arr_end; ++row_arr_itr_b){ // Iterate over all the elements in nth column of B
        for(auto col_arr_itr_a = 0; col_arr_itr_a < curr_col_arr_range; ++col_arr_itr_a){
          if((shared_A_cols[col_arr_itr_a + prev_col_arr_range] == b_indices[row_arr_itr_b])){
            found = true;
            C_n_nnz_per_m0[n % TILE_SIZE] += 1;
            break;
          }
        }
        if(found) break;
      }
      n += TILE_SIZE;
    }
    __syncthreads();

    int C_n_nnz = C_n_nnz_per_m0[tx];
    typedef cub::BlockReduce<int, TILE_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    int C_nnz_per_row = BlockReduce(temp_storage).Sum(C_n_nnz);  
    c_nnz_per_row[m0] = C_nnz_per_row;

    C_n_nnz_per_m0[tx] = 0;
    __syncthreads();

    prev_col_arr_range += curr_col_arr_range;
    m0 += gridDim.x;
  }
  __syncthreads();

  // GLOBAL A
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

      for(int row_arr_itr_b = row_arr_start; row_arr_itr_b < row_arr_end; ++row_arr_itr_b){ // Iterate over all the elements in nth column of B
        for(auto col_arr_itr_a = 0; col_arr_itr_a < curr_col_arr_range; ++col_arr_itr_a){
          if((a_indices[col_arr_itr_a + col_arr_start] == shared_B_rows[row_arr_itr_b]))
          {
            found = true;
            C_n_nnz_per_m0[n % TILE_SIZE] += 1;
            break;
          }
        }
        if(found) break;
      }
      n += TILE_SIZE;
    }
    __syncthreads();

    while(n < b_cols && b_offsets[n + 1] >= TILE_SIZE * TILE_SIZE){
      auto row_arr_start = b_offsets[n];
      auto row_arr_end = b_offsets[n + 1];
      found = false;
      for(int row_arr_itr_b = row_arr_start; row_arr_itr_b < row_arr_end; ++row_arr_itr_b){ // Iterate over all the elements in nth column of B
        for(auto col_arr_itr_a = 0; col_arr_itr_a < curr_col_arr_range; ++col_arr_itr_a){
          if((a_indices[col_arr_itr_a + col_arr_start] == b_indices[row_arr_itr_b]))
          {
            found = true;
            C_n_nnz_per_m0[n % TILE_SIZE] += 1;
            break;
          }
        }
        if(found) break;
      }
      n += TILE_SIZE;
    }
    __syncthreads();

    int C_n_nnz = C_n_nnz_per_m0[tx];
    typedef cub::BlockReduce<int, TILE_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    int C_nnz_per_row = BlockReduce(temp_storage).Sum(C_n_nnz);  
    c_nnz_per_row[m0] = C_nnz_per_row;

    C_n_nnz_per_m0[tx] = 0;
    __syncthreads();

    prev_col_arr_range += curr_col_arr_range;
    m0 += gridDim.x;
  }
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
 * @param C Output matrix C (GPU).
 * @param stream CUDA stream.
 */
template <typename index_t, typename offset_t, typename type_t>
void estimate_nnz_test(csr_t<index_t, offset_t, type_t>& csr,
                   csc_t<index_t, offset_t, type_t>& csc,
                   int* c_nnz_per_tile,
                   cudaStream_t stream = 0) {
  // Create a schedule.
  constexpr std::size_t block_size = 128;

  // Create a schedule.
  using setup_t = schedule::setup<schedule::algorithms_t::thread_mapped, 1, 1,
                                  index_t, offset_t>;
  setup_t config(csr.offsets.data().get(), csr.rows, csr.nnzs);

  std::size_t grid_size = (csr.rows + block_size - 1) / block_size;
   
  launch::non_cooperative(
      stream, __estimate_nnz_test<setup_t, index_t, offset_t, type_t>, grid_size,
      block_size, config, csr.rows, csr.cols, csr.nnzs,
      csr.offsets.data().get(), csr.indices.data().get(),
      csc.rows, csc.cols, csc.nnzs,
      csc.offsets.data().get(), csc.indices.data().get(),
      c_nnz_per_tile);

  cudaStreamSynchronize(stream);
}

/**
 * @brief Estimate the nnz of output matrix C using tiling
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
void estimate_nnz_test_v2(csr_t<index_t, offset_t, type_t>& csr,
                   csc_t<index_t, offset_t, type_t>& csc,
                   int* c_nnz_per_tile,
                   cudaStream_t stream = 0) {


  // Create a schedule.
  constexpr std::size_t block_size = 32;
  // constexpr dim3 block_size(TILE_SIZE, TILE_SIZE, 1);

  // Create a schedule.
  using setup_t = schedule::setup<schedule::algorithms_t::thread_mapped, 1, 1,
                                  index_t, offset_t>;
  setup_t config(csr.offsets.data().get(), csr.rows, csr.nnzs);

  // dim3 grid_size((csc.cols + block_size.x - 1) / block_size.x, (csr.rows + block_size.y - 1) / block_size.y, 1);
  // dim3 grid_size((csc.cols + block_size.x - 1) / block_size.x, csr.rows, 1);
  std::size_t grid_size = csr.rows; // Assigning the number of rows in A to the grid size


  launch::non_cooperative(
      stream, __estimate_nnz_row_col_pairs_v2<setup_t, index_t, offset_t, type_t>, grid_size,
      block_size, config, csr.rows, csr.cols, csr.nnzs,
      csr.offsets.data().get(), csr.indices.data().get(),
      csc.rows, csc.cols, csc.nnzs,
      csc.offsets.data().get(), csc.indices.data().get(),
      c_nnz_per_tile);

  cudaStreamSynchronize(stream);
}

/**
 * @brief Estimate the nnz of output matrix C using tiling
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
void estimate_nnz_test_v3(csr_t<index_t, offset_t, type_t>& csr,
                   csc_t<index_t, offset_t, type_t>& csc,
                   int* c_nnz_per_tile,
                   cudaStream_t stream = 0) {


  // Create a schedule.
  constexpr std::size_t block_size = TILE_SIZE;
  // constexpr dim3 block_size(TILE_SIZE, TILE_SIZE, 1);

  // Create a schedule.
  using setup_t = schedule::setup<schedule::algorithms_t::thread_mapped, 1, 1,
                                  index_t, offset_t>;
  setup_t config(csr.offsets.data().get(), csr.rows, csr.nnzs);

  // dim3 grid_size((csc.cols + block_size.x - 1) / block_size.x, (csr.rows + block_size.y - 1) / block_size.y, 1);
  // dim3 grid_size((csc.cols + block_size.x - 1) / block_size.x, csr.rows, 1);
  std::size_t grid_size = (csr.rows + block_size - 1) / block_size;
  printf("grid_size: %ld\n", grid_size);


  launch::non_cooperative(
      stream, __estimate_nnz_row_col_pairs_v4<setup_t, index_t, offset_t, type_t>, grid_size,
      block_size, config, csr.rows, csr.cols, csr.nnzs,
      csr.offsets.data().get(), csr.indices.data().get(),
      csc.rows, csc.cols, csc.nnzs,
      csc.offsets.data().get(), csc.indices.data().get(),
      c_nnz_per_tile);

  cudaStreamSynchronize(stream);
}

/**
 * @brief Precalculate the column indices array of C
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
void precalculate_c_col_indices(csr_t<index_t, offset_t, type_t>& csr,
                   csc_t<index_t, offset_t, type_t>& csc,
                   csr_t<index_t, offset_t, type_t>& c,
                   cudaStream_t stream = 0) {


  // Create a schedule.
  constexpr std::size_t block_size = TILE_SIZE;
  // constexpr dim3 block_size(TILE_SIZE, TILE_SIZE, 1);

  // Create a schedule.
  using setup_t = schedule::setup<schedule::algorithms_t::thread_mapped, 1, 1,
                                  index_t, offset_t>;
  setup_t config(csr.offsets.data().get(), csr.rows, csr.nnzs);

  std::size_t grid_size = (csr.rows + block_size - 1) / block_size;
  printf("grid_size: %ld\n", grid_size);

  launch::non_cooperative(
      stream, __estimate_nnz_row_col_pairs_v4<setup_t, index_t, offset_t, type_t>, grid_size,
      block_size, config, csr.rows, csr.cols, csr.nnzs,
      csr.offsets.data().get(), csr.indices.data().get(),
      csc.rows, csc.cols, csc.nnzs,
      csc.offsets.data().get(), csc.indices.data().get(),
      c_nnz_per_tile);

  cudaStreamSynchronize(stream);
}

// template <typename offset_t>
void scanNnzC(int* c_nnz_per_tile, int* c_offsets, std::size_t c_rows){
  thrust::device_ptr<int> ptr_begin = thrust::device_pointer_cast(c_nnz_per_tile);
  thrust::device_ptr<int> ptr_end = thrust::device_pointer_cast(c_nnz_per_tile + c_rows + 1);
  thrust::exclusive_scan(ptr_begin, ptr_end, c_offsets);
}

// template <typename index_t, typename offset_t, typename type_t>
int sumEstimateNnzC(int* c_nnz_per_tile, std::size_t c_rows){
  thrust::device_ptr<int> ptr_begin = thrust::device_pointer_cast(c_nnz_per_tile);
  thrust::device_ptr<int> ptr_end = thrust::device_pointer_cast(c_nnz_per_tile + c_rows);
  
  int sum = thrust::reduce(ptr_begin, ptr_end, 0);
  return sum;
}

}  // namespace spgemm
}  // namespace algorithms
}  // namespace loops