#include <stdio.h>

#include <cuda_runtime.h>
#include <cuda/barrier>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    exit(EXIT_FAILURE);
  }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

inline const char* _ConvertSMVer2ArchName(int major, int minor) {
  typedef struct {
    int SM; 
    const char* name;
  } sSMtoArchName;

  sSMtoArchName nGpuArchNameSM[] = {
      {0x30, "Kepler"},
      {0x32, "Kepler"},
      {0x35, "Kepler"},
      {0x37, "Kepler"},
      {0x50, "Maxwell"},
      {0x52, "Maxwell"},
      {0x53, "Maxwell"},
      {0x60, "Pascal"},
      {0x61, "Pascal"},
      {0x62, "Pascal"},
      {0x70, "Volta"},
      {0x72, "Xavier"},
      {0x75, "Turing"},
      {0x80, "Ampere"},
      {0x86, "Ampere"},
      {0x87, "Ampere"},
      {0x89, "Ada"},
      {0x90, "Hopper"},
      {0xa0, "Blackwell"},
      {0xa1, "Blackwell"},
      {0xc0, "Blackwell"},
      {-1, "Graphics Device"}};

  int index = 0;

  while (nGpuArchNameSM[index].SM != -1) {
    if (nGpuArchNameSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchNameSM[index].name;
    }

    index++;
  }

  printf(
      "MapSMtoArchName for SM %d.%d is undefined."
      "  Default to use %s\n",
      major, minor, nGpuArchNameSM[index - 1].name);
  return nGpuArchNameSM[index - 1].name;
}

inline int _ConvertSMVer2Cores(int major, int minor) {
  typedef struct {
    int SM;
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] = {
      {0x30, 192},
      {0x32, 192},
      {0x35, 192},
      {0x37, 192},
      {0x50, 128},
      {0x52, 128},
      {0x53, 128},
      {0x60,  64},
      {0x61, 128},
      {0x62, 128},
      {0x70,  64},
      {0x72,  64},
      {0x75,  64},
      {0x80,  64},
      {0x86, 128},
      {0x87, 128},
      {0x89, 128},
      {0x90, 128},
      {0xa0, 128},
      {0xa1, 128},
      {0xc0, 128},
      {-1, -1}};

  int index = 0;

  while (nGpuArchCoresPerSM[index].SM != -1) {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchCoresPerSM[index].Cores;
    }

    index++;
  }

  printf(
      "MapSMtoCores for SM %d.%d is undefined."
      "  Default to use %d Cores/SM\n",
      major, minor, nGpuArchCoresPerSM[index - 1].Cores);
  return nGpuArchCoresPerSM[index - 1].Cores;
}

inline int gpuGetMaxGflopsDeviceId() {
  int current_device = 0, sm_per_multiproc = 0;
  int max_perf_device = 0;
  int device_count = 0;
  int devices_prohibited = 0;

  uint64_t max_compute_perf = 0;
  checkCudaErrors(cudaGetDeviceCount(&device_count));

  if (device_count == 0) {
    fprintf(stderr,
            "gpuGetMaxGflopsDeviceId() CUDA error:"
            " no devices supporting CUDA.\n");
    exit(EXIT_FAILURE);
  }

  current_device = 0;

  while (current_device < device_count) {
    int computeMode = -1, major = 0, minor = 0;
    checkCudaErrors(cudaDeviceGetAttribute(&computeMode, cudaDevAttrComputeMode, current_device));
    checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, current_device));
    checkCudaErrors(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, current_device));

    if (computeMode != cudaComputeModeProhibited) {
      if (major == 9999 && minor == 9999) {
        sm_per_multiproc = 1;
      } else {
        sm_per_multiproc =
            _ConvertSMVer2Cores(major,  minor);
      }
      int multiProcessorCount = 0, clockRate = 0;
      checkCudaErrors(cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, current_device));
      cudaError_t result = cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, current_device);
      if (result != cudaSuccess) {
        if(result == cudaErrorInvalidValue) {
          clockRate = 1;
        }
        else {
          exit(EXIT_FAILURE);
        }
      }
      uint64_t compute_perf = (uint64_t)multiProcessorCount * sm_per_multiproc * clockRate;

      if (compute_perf > max_compute_perf) {
        max_compute_perf = compute_perf;
        max_perf_device = current_device;
      }
    } else {
      devices_prohibited++;
    }

    ++current_device;
  }

  if (devices_prohibited == device_count) {
    fprintf(stderr,
            "gpuGetMaxGflopsDeviceId() CUDA error:"
            " all devices have compute mode prohibited.\n");
    exit(EXIT_FAILURE);
  }

  return max_perf_device;
}

inline int findCudaDevice(int argc, const char **argv) {
  int devID = 0;

  devID = gpuGetMaxGflopsDeviceId();
  checkCudaErrors(cudaSetDevice(devID));
  int major = 0, minor = 0;
  checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, devID));
  checkCudaErrors(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, devID));
  printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n",
         devID, _ConvertSMVer2ArchName(major, minor), major, minor);

  return devID;
}

#if __CUDA_ARCH__ >= 700
template <bool writeSquareRoot>
__device__ void reduceBlockData(
    cuda::barrier<cuda::thread_scope_block> &barrier,
    cg::thread_block_tile<32> &tile32, double &threadSum, double *result) {
  extern __shared__ double tmp[];

#pragma unroll
  for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
    threadSum += tile32.shfl_down(threadSum, offset);
  }
  if (tile32.thread_rank() == 0) {
    tmp[tile32.meta_group_rank()] = threadSum;
  }

  auto token = barrier.arrive();

  barrier.wait(std::move(token));

  if (tile32.meta_group_rank() == 0) {
    double beta = tile32.thread_rank() < tile32.meta_group_size()
                      ? tmp[tile32.thread_rank()]
                      : 0.0;

#pragma unroll
    for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
      beta += tile32.shfl_down(beta, offset);
    }

    if (tile32.thread_rank() == 0) {
      if (writeSquareRoot)
        *result = sqrt(beta);
      else
        *result = beta;
    }
  }
}
#endif

__global__ void normVecByDotProductAWBarrier(float *vecA, float *vecB,
                                             double *partialResults, int size) {
#if __CUDA_ARCH__ >= 700
#pragma diag_suppress static_var_with_dynamic_init
  cg::thread_block cta = cg::this_thread_block();
  cg::grid_group grid = cg::this_grid();
  ;
  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

  __shared__ cuda::barrier<cuda::thread_scope_block> barrier;

  if (threadIdx.x == 0) {
    init(&barrier, blockDim.x);
  }

  cg::sync(cta);

  double threadSum = 0.0;
  for (int i = grid.thread_rank(); i < size; i += grid.size()) {
    threadSum += (double)(vecA[i] * vecB[i]);
  }

  reduceBlockData<false>(barrier, tile32, threadSum,
                         &partialResults[blockIdx.x]);

  cg::sync(grid);

  if (blockIdx.x == 0) {
    threadSum = 0.0;
    for (int i = cta.thread_rank(); i < gridDim.x; i += cta.size()) {
      threadSum += partialResults[i];
    }
    reduceBlockData<true>(barrier, tile32, threadSum, &partialResults[0]);
  }

  cg::sync(grid);

  const double finalValue = partialResults[0];

  for (int i = grid.thread_rank(); i < size; i += grid.size()) {
    vecA[i] = (float)vecA[i] / finalValue;
    vecB[i] = (float)vecB[i] / finalValue;
  }
#endif
}

int runNormVecByDotProductAWBarrier(int argc, char **argv, int deviceId);

int main(int argc, char **argv) {
  printf("%s starting...\n", argv[0]);

  int dev = findCudaDevice(argc, (const char **)argv);

  int major = 0;
  checkCudaErrors(
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));

  if (major < 7) {
    printf("simpleAWBarrier requires SM 7.0 or higher.  Exiting...\n");
    exit(EXIT_WAIVED);
  }

  int supportsCooperativeLaunch = 0;
  checkCudaErrors(cudaDeviceGetAttribute(&supportsCooperativeLaunch,
                                         cudaDevAttrCooperativeLaunch, dev));

  if (!supportsCooperativeLaunch) {
    printf(
        "\nSelected GPU (%d) does not support Cooperative Kernel Launch, "
        "Waiving the run\n",
        dev);
    exit(EXIT_WAIVED);
  }

  int testResult = runNormVecByDotProductAWBarrier(argc, argv, dev);

  printf("%s completed, returned %s\n", argv[0], testResult ? "OK" : "ERROR!");
  exit(testResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

int runNormVecByDotProductAWBarrier(int argc, char **argv, int deviceId) {
  float *vecA, *d_vecA;
  float *vecB, *d_vecB;
  double *d_partialResults;
  int size = 10000000;

  checkCudaErrors(cudaMallocHost(&vecA, sizeof(float) * size));
  checkCudaErrors(cudaMallocHost(&vecB, sizeof(float) * size));

  checkCudaErrors(cudaMalloc(&d_vecA, sizeof(float) * size));
  checkCudaErrors(cudaMalloc(&d_vecB, sizeof(float) * size));

  float baseVal = 2.0;
  for (int i = 0; i < size; i++) {
    vecA[i] = vecB[i] = baseVal;
  }

  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  checkCudaErrors(cudaMemcpyAsync(d_vecA, vecA, sizeof(float) * size,
                                  cudaMemcpyHostToDevice, stream));
  checkCudaErrors(cudaMemcpyAsync(d_vecB, vecB, sizeof(float) * size,
                                  cudaMemcpyHostToDevice, stream));

  int minGridSize = 0, blockSize = 0;
  checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(
      &minGridSize, &blockSize, (void *)normVecByDotProductAWBarrier, 0, size));

  int smemSize = ((blockSize / 32) + 1) * sizeof(double);

  int numBlocksPerSm = 0;
  checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocksPerSm, normVecByDotProductAWBarrier, blockSize, smemSize));

  int multiProcessorCount = 0;
  checkCudaErrors(cudaDeviceGetAttribute(
      &multiProcessorCount, cudaDevAttrMultiProcessorCount, deviceId));

  minGridSize = multiProcessorCount * numBlocksPerSm;
  checkCudaErrors(cudaMalloc(&d_partialResults, minGridSize * sizeof(double)));

  printf(
      "Launching normVecByDotProductAWBarrier kernel with numBlocks = %d "
      "blockSize = %d\n",
      minGridSize, blockSize);

  dim3 dimGrid(minGridSize, 1, 1), dimBlock(blockSize, 1, 1);

  void *kernelArgs[] = {(void *)&d_vecA, (void *)&d_vecB,
                        (void *)&d_partialResults, (void *)&size};

  checkCudaErrors(
      cudaLaunchCooperativeKernel((void *)normVecByDotProductAWBarrier, dimGrid,
                                  dimBlock, kernelArgs, smemSize, stream));

  checkCudaErrors(cudaMemcpyAsync(vecA, d_vecA, sizeof(float) * size,
                                  cudaMemcpyDeviceToHost, stream));
  checkCudaErrors(cudaStreamSynchronize(stream));

  float expectedResult = (baseVal / sqrt(size * baseVal * baseVal));
  unsigned int matches = 0;
  for (int i = 0; i < size; i++) {
    if ((vecA[i] - expectedResult) > 0.00001) {
      printf("mismatch at i = %d\n", i);
      break;
    } else {
      matches++;
    }
  }

  printf("Result = %s\n", matches == size ? "PASSED" : "FAILED");
  checkCudaErrors(cudaFree(d_vecA));
  checkCudaErrors(cudaFree(d_vecB));
  checkCudaErrors(cudaFree(d_partialResults));

  checkCudaErrors(cudaFreeHost(vecA));
  checkCudaErrors(cudaFreeHost(vecB));
  return matches == size;
}
