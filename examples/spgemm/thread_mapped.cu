/**
 * @file thread_mapped.cu
 * @author 
 * @brief SpGEMM example
 * @version 0.1
 * @date 2023
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "helpers.hxx"
#include <loops/algorithms/spgemm/thread_mapped.cuh>
#include <loops/algorithms/spgemm/estimate_nnz.cuh>
#include <loops/algorithms/spgemm/estimate_nnz_test.cuh>
#include <loops/algorithms/spgemm/find_explicit_zeros.cuh>

#include "helpers/test_spgemm.cpp"

using namespace loops;

int main(int argc, char** argv) {
  util::timer_t timer;

  using index_t = int;
  using offset_t = int;
  using type_t = float;

  // ... I/O parameters, mtx, etc.
  parameters_t parameters(argc, argv);

  matrix_market_t<index_t, offset_t, type_t> mtx;
  csr_t<index_t, offset_t, type_t> csr(mtx.load(parameters.filename));
  csc_t<index_t, offset_t, type_t> csc(mtx.load(parameters.filename));

  // Timer for benchmarking starts here
  timer.start();
  
  int* d_c_nnz_by_row;
  cudaMalloc(&d_c_nnz_by_row, csr.rows * sizeof(int));
  cudaMemset(d_c_nnz_by_row, 0, csr.rows * sizeof(int));
  int* h_c_nnz_by_row = new int[csr.rows]();

  algorithms::spgemm::estimate_nnz_test_v3(csr, csc, d_c_nnz_by_row);
  cudaMemcpy(h_c_nnz_by_row, d_c_nnz_by_row, csr.rows * sizeof(int), cudaMemcpyDeviceToHost);

  timer.stop();

  float estimate_nnz_elapsed = timer.milliseconds();
  std::cout << "estimate_nnz_elapsed (ms):\t" << estimate_nnz_elapsed << std::endl;

  // timer.start();
  csr_t<index_t, offset_t, type_t> c(csr.rows, 0, 0);

  // prefix sum d_c_nnz_by_row to get the row offset of C
  algorithms::spgemm::scanNnzC(d_c_nnz_by_row, c.offsets.data().get(), csr.rows);
  c.nnzs = c.offsets.back();

  // allocate indices array and values array in device
  index_t* d_c_indices;
  cudaMalloc(&d_c_indices, c.nnzs * sizeof(index_t));
  cudaMemset(d_c_indices, 0, c.nnzs * sizeof(index_t));

  type_t* d_c_values;
  cudaMalloc(&d_c_values, c.nnzs * sizeof(type_t));
  cudaMemset(d_c_values, 0, c.nnzs * sizeof(type_t));

  // Test estimate_nnz
  // printDeviceArr(d_c_nnz_by_row, csr.rows);
  // printDeviceArr(c.offsets.data().get(), csr.rows+1);
  // printDeviceArr(d_c_indices, c.nnzs);

  // copyAndSumEstimateNnzToHost(d_c_nnz_by_row, csr.rows);


  // Apply SpGEMM
  /*
  // algorithms::spgemm::thread_mapped(csr, csc, c, d_c_indices, d_c_values);
  ////////// TODO: can I use c.indices.data().get() instead of d_c_indices? //////////

  // Copy back to C
  // c.indices.resize(c.nnzs);
  // c.values.resize(c.nnzs);
  // thrust::copy(d_c_indices, d_c_indices + c.nnzs, c.indices.begin());
  // thrust::copy(d_c_values, d_c_values + c.nnzs, c.values.begin());
  // Timer for benchmarking stops here
  */

  c.indices.resize(c.nnzs);
  c.values.resize(c.nnzs);

  algorithms::spgemm::thread_mapped_v2(csr, csc, c);
  timer.stop();

  float spgemm_elapsed = timer.milliseconds();
  std::cout << "spgemm_elapsed (ms):\t" << spgemm_elapsed << std::endl;

  std::cout << "Total Elapsed (ms):\t" << estimate_nnz_elapsed+spgemm_elapsed << std::endl;



  // Sanity check thrust::copy
  /*
  std::vector<index_t> h_c_indices(c.nnzs);
  std::vector<type_t> h_c_values(c.nnzs);
  try{
    thrust::copy(c.indices.begin(), c.indices.end(), h_c_indices.begin());
  } catch(thrust::system_error &e) {
    std::cerr << "Error accessing vector element: " << e.what() << std::endl;
    exit(-1);
  }

  try{
    thrust::copy(c.values.begin(), c.values.end(), h_c_values.begin());
  } catch(thrust::system_error &e) {
    std::cerr << "Error accessing vector element: " << e.what() << std::endl;
    exit(-1);
  }

  cudaMemcpy(h_c_indices, d_c_indices, c.nnzs * sizeof(index_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_c_values, d_c_values, c.nnzs * sizeof(type_t), cudaMemcpyDeviceToHost);

  // for(int i = 0; i < c.nnzs; ++i) {
  //   std::cout << h_c_indices[i] << ",";
  // }
  // std::cout << std::endl;
  // for(int i = 0; i < c.nnzs; ++i) {
  //   std::cout << h_c_indices[i] << ",";
  // }

  // std::cout << std::endl;
  // for(int i = 0; i < c.nnzs; ++i) {
  //   std::cout << h_c_values[i] << ",";
  // }
  // std::cout << std::endl;
  // for(int i = 0; i < c.nnzs; ++i) {
  //   std::cout << h_c_values[i] << ",";
  // }

  for(int i = 0; i < c.nnzs; ++i) {
    if(h_c_indices[i] != h_c_indices[i]) {
      std::cout << "index not equal" << std::endl;
      std::cout << "h_c_indices[" << i << "]: " << h_c_indices[i] << std::endl;
      std::cout << "h_c_indices[" << i << "]: " << h_c_indices[i] << std::endl;
    }
  }

  for(int i = 0; i < c.nnzs; ++i) {
    if(h_c_values[i] != h_c_values[i]) {
      std::cout << "value not equal" << std::endl;
      std::cout << "h_c_values[" << i << "]: " << h_c_values[i] << std::endl;
      std::cout << "h_c_values[" << i << "]: " << h_c_values[i] << std::endl;
    }
  }
  */

  // Run the benchmark.
  /*
  timer.start();
  // algorithms::spgemm::thread_mapped(csr, csc, C);
  algorithms::spgemm::thread_mapped(csr, csc, coo);
  timer.stop();
*/
  // std::cout << "Elapsed (ms):\t" << timer.milliseconds() << std::endl;

  // writeMtxToFile(h_coo, csr.rows, csc.cols, "/home/ychenfei/research/libs/loops/examples/spgemm/export_mtx/test.txt");

  cudaFree(d_c_nnz_by_row);
  cudaFree(d_c_indices);
  // cudaFree(d_c_values);
}