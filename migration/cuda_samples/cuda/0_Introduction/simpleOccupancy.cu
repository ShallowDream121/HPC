#include <iostream>

template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    exit(EXIT_FAILURE);
  }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

const int manualBlockSize = 32;

__global__ void square(int *array, int arrayCount) {
  extern __shared__ int dynamicSmem[];
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < arrayCount) {
    array[idx] *= array[idx];
  }
}

static double reportPotentialOccupancy(void *kernel, int blockSize,
                                       size_t dynamicSMem) {
  int device;
  cudaDeviceProp prop;

  int numBlocks;
  int activeWarps;
  int maxWarps;

  double occupancy;

  checkCudaErrors(cudaGetDevice(&device));
  checkCudaErrors(cudaGetDeviceProperties(&prop, device));

  checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocks, kernel, blockSize, dynamicSMem));

  activeWarps = numBlocks * blockSize / prop.warpSize;
  maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;

  occupancy = (double)activeWarps / maxWarps;

  return occupancy;
}

static int launchConfig(int *array, int arrayCount, bool automatic) {
  int blockSize;
  int minGridSize;
  int gridSize;
  size_t dynamicSMemUsage = 0;

  cudaEvent_t start;
  cudaEvent_t end;

  float elapsedTime;

  double potentialOccupancy;

  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&end));

  if (automatic) {
    checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(
        &minGridSize, &blockSize, (void *)square, dynamicSMemUsage,
        arrayCount));

    std::cout << "Suggested block size: " << blockSize << std::endl
              << "Minimum grid size for maximum occupancy: " << minGridSize
              << std::endl;
  } else {
    blockSize = manualBlockSize;
  }

  gridSize = (arrayCount + blockSize - 1) / blockSize;

  checkCudaErrors(cudaEventRecord(start));
  square<<<gridSize, blockSize, dynamicSMemUsage>>>(array, arrayCount);
  checkCudaErrors(cudaEventRecord(end));

  checkCudaErrors(cudaDeviceSynchronize());

  potentialOccupancy =
      reportPotentialOccupancy((void *)square, blockSize, dynamicSMemUsage);

  std::cout << "Potential occupancy: " << potentialOccupancy * 100 << "%"
            << std::endl;

  checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, end));
  std::cout << "Elapsed time: " << elapsedTime << "ms" << std::endl;

  return 0;
}

static int test(bool automaticLaunchConfig, const int count = 1000000) {
  int *array;
  int *dArray;
  int size = count * sizeof(int);

  array = new int[count];

  for (int i = 0; i < count; i += 1) {
    array[i] = i;
  }

  checkCudaErrors(cudaMalloc(&dArray, size));
  checkCudaErrors(cudaMemcpy(dArray, array, size, cudaMemcpyHostToDevice));

  for (int i = 0; i < count; i += 1) {
    array[i] = 0;
  }

  launchConfig(dArray, count, automaticLaunchConfig);

  checkCudaErrors(cudaMemcpy(array, dArray, size, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(dArray));

  for (int i = 0; i < count; i += 1) {
    if (array[i] != i * i) {
      std::cout << "element " << i << " expected " << i * i << " actual "
                << array[i] << std::endl;
      return 1;
    }
  }
  delete[] array;

  return 0;
}

int main() {
  int status;

  std::cout << "starting Simple Occupancy" << std::endl << std::endl;

  std::cout << "[ Manual configuration with " << manualBlockSize
            << " threads per block ]" << std::endl;

  status = test(false);
  if (status) {
    std::cerr << "Test failed\n" << std::endl;
    return -1;
  }

  std::cout << std::endl;

  std::cout << "[ Automatic, occupancy-based configuration ]" << std::endl;
  status = test(true);
  if (status) {
    std::cerr << "Test failed\n" << std::endl;
    return -1;
  }

  std::cout << std::endl;
  std::cout << "Test PASSED\n" << std::endl;

  return 0;
}
