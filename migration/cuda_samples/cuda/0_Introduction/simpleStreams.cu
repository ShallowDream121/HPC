const char *sSDKsample = "simpleStreams";

const char *sEventSyncMethod[] = {"cudaEventDefault", "cudaEventBlockingSync",
                                  "cudaEventDisableTiming", NULL};

const char *sDeviceSyncMethod[] = {
    "cudaDeviceScheduleAuto",         "cudaDeviceScheduleSpin",
    "cudaDeviceScheduleYield",        "INVALID",
    "cudaDeviceScheduleBlockingSync", NULL};

#include <stdio.h>
#include <assert.h>

#include <cuda_runtime.h>

#include <sys/mman.h>  

#define MEMORY_ALIGNMENT 4096
#define ALIGN_UP(x, size) (((size_t)x + (size - 1)) & (~(size - 1)))

__global__ void init_array(int *g_data, int *factor, int num_iterations) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (int i = 0; i < num_iterations; i++) {
    g_data[idx] += *factor; 
}

bool correct_data(int *a, const int n, const int c) {
  for (int i = 0; i < n; i++) {
    if (a[i] != c) {
      printf("%d: %d %d\n", i, a[i], c);
      return false;
    }
  }

  return true;
}

inline void AllocateHostMemory(bool bPinGenericMemory, int **pp_a,
                               int **ppAligned_a, int nbytes) {
#if CUDART_VERSION >= 4000
#if !defined(__arm__) && !defined(__aarch64__)
  if (bPinGenericMemory) {
    printf(
        "> mmap() allocating %4.2f Mbytes (generic page-aligned system "
        "memory)\n",
        (float)nbytes / 1048576.0f);
    *pp_a = (int *)mmap(NULL, (nbytes + MEMORY_ALIGNMENT),
                        PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);

    *ppAligned_a = (int *)ALIGN_UP(*pp_a, MEMORY_ALIGNMENT);

    printf(
        "> cudaHostRegister() registering %4.2f Mbytes of generic allocated "
        "system memory\n",
        (float)nbytes / 1048576.0f);
    checkCudaErrors(
        cudaHostRegister(*ppAligned_a, nbytes, cudaHostRegisterMapped));
  } else
#endif
#endif
  {
    printf("> cudaMallocHost() allocating %4.2f Mbytes of system memory\n",
           (float)nbytes / 1048576.0f);
    checkCudaErrors(cudaMallocHost((void **)pp_a, nbytes));
    *ppAligned_a = *pp_a;
  }
}

inline void FreeHostMemory(bool bPinGenericMemory, int **pp_a,
                           int **ppAligned_a, int nbytes) {
#if CUDART_VERSION >= 4000
#if !defined(__arm__) && !defined(__aarch64__)
  if (bPinGenericMemory) {
    checkCudaErrors(cudaHostUnregister(*ppAligned_a));
#ifdef WIN32
    VirtualFree(*pp_a, 0, MEM_RELEASE);
#else
    munmap(*pp_a, nbytes);
#endif
  } else
#endif
#endif
  {
    cudaFreeHost(*pp_a);
  }
}

static const char *sSyncMethod[] = {
    "0 (Automatic Blocking)",
    "1 (Spin Blocking)",
    "2 (Yield Blocking)",
    "3 (Undefined Blocking Method)",
    "4 (Blocking Sync Event) = low CPU utilization",
    NULL};

void printHelp() {
  printf("Usage: %s [options below]\n", sSDKsample);
  printf("\t--sync_method=n for CPU/GPU synchronization\n");
  printf("\t             n=%s\n", sSyncMethod[0]);
  printf("\t             n=%s\n", sSyncMethod[1]);
  printf("\t             n=%s\n", sSyncMethod[2]);
  printf("\t   <Default> n=%s\n", sSyncMethod[4]);
  printf(
      "\t--use_generic_memory (default) use generic page-aligned for system "
      "memory\n");
  printf(
      "\t--use_cuda_malloc_host (optional) use cudaMallocHost to allocate "
      "system memory\n");
}

#if defined(__APPLE__) || defined(MACOSX)
#define DEFAULT_PINNED_GENERIC_MEMORY false
#else
#define DEFAULT_PINNED_GENERIC_MEMORY true
#endif

int main(int argc, char **argv) {
  int cuda_device = 0;
  int nstreams = 4;              
  int nreps = 10;                
  int n = 16 * 1024 * 1024;      
  int nbytes = n * sizeof(int);  
  dim3 threads, blocks;          
  float elapsed_time, time_memcpy, time_kernel; 
  float scale_factor = 1.0f;

  bool bPinGenericMemory =
      DEFAULT_PINNED_GENERIC_MEMORY;  
  int device_sync_method =
      cudaDeviceBlockingSync;  

  int niterations;  

  printf("[ %s ]\n\n", sSDKsample);

  if (checkCmdLineFlag(argc, (const char **)argv, "help")) {
    printHelp();
    return EXIT_SUCCESS;
  }

  if ((device_sync_method = getCmdLineArgumentInt(argc, (const char **)argv,
                                                  "sync_method")) >= 0) {
    if (device_sync_method == 0 || device_sync_method == 1 ||
        device_sync_method == 2 || device_sync_method == 4) {
      printf("Device synchronization method set to = %s\n",
             sSyncMethod[device_sync_method]);
      printf("Setting reps to 100 to demonstrate steady state\n");
      nreps = 100;
    } else {
      printf("Invalid command line option sync_method=\"%d\"\n",
             device_sync_method);
      return EXIT_FAILURE;
    }
  } else {
    printHelp();
    return EXIT_SUCCESS;
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "use_generic_memory")) {
#if defined(__APPLE__) || defined(MACOSX)
    bPinGenericMemory = false;  
                                
#else
    bPinGenericMemory = true;
#endif
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "use_cuda_malloc_host")) {
    bPinGenericMemory = false;
  }

  printf("\n> ");
  cuda_device = findCudaDevice(argc, (const char **)argv);

  int num_devices = 0;
  checkCudaErrors(cudaGetDeviceCount(&num_devices));

  if (0 == num_devices) {
    printf(
        "your system does not have a CUDA capable device, waiving test...\n");
    return EXIT_WAIVED;
  }

  if (cuda_device >= num_devices) {
    printf(
        "cuda_device=%d is invalid, must choose device ID between 0 and %d\n",
        cuda_device, num_devices - 1);
    return EXIT_FAILURE;
  }

  checkCudaErrors(cudaSetDevice(cuda_device));

  // Checking for compute capabilities
  cudaDeviceProp deviceProp;
  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, cuda_device));

  niterations = 5;

  // Check if GPU can map host memory (Generic Method), if not then we override
  // bPinGenericMemory to be false
  if (bPinGenericMemory) {
    printf("Device: <%s> canMapHostMemory: %s\n", deviceProp.name,
           deviceProp.canMapHostMemory ? "Yes" : "No");

    if (deviceProp.canMapHostMemory == 0) {
      printf(
          "Using cudaMallocHost, CUDA device does not support mapping of "
          "generic host memory\n");
      bPinGenericMemory = false;
    }
  }

  // Anything that is less than 32 Cores will have scaled down workload
  scale_factor =
      max((32.0f / (_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) *
                    (float)deviceProp.multiProcessorCount)),
          1.0f);
  n = (int)rint((float)n / scale_factor);

  printf("> CUDA Capable: SM %d.%d hardware\n", deviceProp.major,
         deviceProp.minor);
  printf("> %d Multiprocessor(s) x %d (Cores/Multiprocessor) = %d (Cores)\n",
         deviceProp.multiProcessorCount,
         _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
         _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) *
             deviceProp.multiProcessorCount);

  printf("> scale_factor = %1.4f\n", 1.0f / scale_factor);
  printf("> array_size   = %d\n\n", n);

  // enable use of blocking sync, to reduce CPU usage
  printf("> Using CPU/GPU Device Synchronization method (%s)\n",
         sDeviceSyncMethod[device_sync_method]);
  checkCudaErrors(cudaSetDeviceFlags(
      device_sync_method | (bPinGenericMemory ? cudaDeviceMapHost : 0)));

  // allocate host memory
  int c = 5;            // value to which the array will be initialized
  int *h_a = 0;         // pointer to the array data in host memory
  int *hAligned_a = 0;  // pointer to the array data in host memory (aligned to
                        // MEMORY_ALIGNMENT)

  // Allocate Host memory (could be using cudaMallocHost or VirtualAlloc/mmap if
  // using the new CUDA 4.0 features
  AllocateHostMemory(bPinGenericMemory, &h_a, &hAligned_a, nbytes);

  // allocate device memory
  int *d_a = 0,
      *d_c = 0;  // pointers to data and init value in the device memory
  checkCudaErrors(cudaMalloc((void **)&d_a, nbytes));
  checkCudaErrors(cudaMemset(d_a, 0x0, nbytes));
  checkCudaErrors(cudaMalloc((void **)&d_c, sizeof(int)));
  checkCudaErrors(cudaMemcpy(d_c, &c, sizeof(int), cudaMemcpyHostToDevice));

  printf("\nStarting Test\n");

  // allocate and initialize an array of stream handles
  cudaStream_t *streams =
      (cudaStream_t *)malloc(nstreams * sizeof(cudaStream_t));

  for (int i = 0; i < nstreams; i++) {
    checkCudaErrors(cudaStreamCreate(&(streams[i])));
  }

  // create CUDA event handles
  // use blocking sync
  cudaEvent_t start_event, stop_event;
  int eventflags =
      ((device_sync_method == cudaDeviceBlockingSync) ? cudaEventBlockingSync
                                                      : cudaEventDefault);

  checkCudaErrors(cudaEventCreateWithFlags(&start_event, eventflags));
  checkCudaErrors(cudaEventCreateWithFlags(&stop_event, eventflags));

  // time memcopy from device
  checkCudaErrors(cudaEventRecord(start_event, 0));  // record in stream-0, to
                                                     // ensure that all previous
                                                     // CUDA calls have
                                                     // completed
  checkCudaErrors(cudaMemcpyAsync(hAligned_a, d_a, nbytes,
                                  cudaMemcpyDeviceToHost, streams[0]));
  checkCudaErrors(cudaEventRecord(stop_event, 0));
  checkCudaErrors(cudaEventSynchronize(
      stop_event));  // block until the event is actually recorded
  checkCudaErrors(cudaEventElapsedTime(&time_memcpy, start_event, stop_event));
  printf("memcopy:\t%.2f\n", time_memcpy);

  // time kernel
  threads = dim3(512, 1);
  blocks = dim3(n / threads.x, 1);
  checkCudaErrors(cudaEventRecord(start_event, 0));
  init_array<<<blocks, threads, 0, streams[0]>>>(d_a, d_c, niterations);
  checkCudaErrors(cudaEventRecord(stop_event, 0));
  checkCudaErrors(cudaEventSynchronize(stop_event));
  checkCudaErrors(cudaEventElapsedTime(&time_kernel, start_event, stop_event));
  printf("kernel:\t\t%.2f\n", time_kernel);

  //////////////////////////////////////////////////////////////////////
  // time non-streamed execution for reference
  threads = dim3(512, 1);
  blocks = dim3(n / threads.x, 1);
  checkCudaErrors(cudaEventRecord(start_event, 0));

  for (int k = 0; k < nreps; k++) {
    init_array<<<blocks, threads>>>(d_a, d_c, niterations);
    checkCudaErrors(
        cudaMemcpy(hAligned_a, d_a, nbytes, cudaMemcpyDeviceToHost));
  }

  checkCudaErrors(cudaEventRecord(stop_event, 0));
  checkCudaErrors(cudaEventSynchronize(stop_event));
  checkCudaErrors(cudaEventElapsedTime(&elapsed_time, start_event, stop_event));
  printf("non-streamed:\t%.2f\n", elapsed_time / nreps);

  //////////////////////////////////////////////////////////////////////
  // time execution with nstreams streams
  threads = dim3(512, 1);
  blocks = dim3(n / (nstreams * threads.x), 1);
  memset(hAligned_a, 255,
         nbytes);  // set host memory bits to all 1s, for testing correctness
  checkCudaErrors(cudaMemset(
      d_a, 0, nbytes));  // set device memory to all 0s, for testing correctness
  checkCudaErrors(cudaEventRecord(start_event, 0));

  for (int k = 0; k < nreps; k++) {
    // asynchronously launch nstreams kernels, each operating on its own portion
    // of data
    for (int i = 0; i < nstreams; i++) {
      init_array<<<blocks, threads, 0, streams[i]>>>(d_a + i * n / nstreams,
                                                     d_c, niterations);
    }

    // asynchronously launch nstreams memcopies.  Note that memcopy in stream x
    // will only
    //   commence executing when all previous CUDA calls in stream x have
    //   completed
    for (int i = 0; i < nstreams; i++) {
      checkCudaErrors(cudaMemcpyAsync(hAligned_a + i * n / nstreams,
                                      d_a + i * n / nstreams, nbytes / nstreams,
                                      cudaMemcpyDeviceToHost, streams[i]));
    }
  }

  checkCudaErrors(cudaEventRecord(stop_event, 0));
  checkCudaErrors(cudaEventSynchronize(stop_event));
  checkCudaErrors(cudaEventElapsedTime(&elapsed_time, start_event, stop_event));
  printf("%d streams:\t%.2f\n", nstreams, elapsed_time / nreps);

  // check whether the output is correct
  printf("-------------------------------\n");
  bool bResults = correct_data(hAligned_a, n, c * nreps * niterations);

  // release resources
  for (int i = 0; i < nstreams; i++) {
    checkCudaErrors(cudaStreamDestroy(streams[i]));
  }

  checkCudaErrors(cudaEventDestroy(start_event));
  checkCudaErrors(cudaEventDestroy(stop_event));

  // Free cudaMallocHost or Generic Host allocated memory (from CUDA 4.0)
  FreeHostMemory(bPinGenericMemory, &h_a, &hAligned_a, nbytes);

  checkCudaErrors(cudaFree(d_a));
  checkCudaErrors(cudaFree(d_c));

  return bResults ? EXIT_SUCCESS : EXIT_FAILURE;
}
