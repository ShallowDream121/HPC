#include <cuda_runtime.h>

#include <iostream>
#include <memory>
#include <string>

int *pArgc = NULL;
char **pArgv = NULL;

template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    exit(EXIT_FAILURE);
  }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

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

#if CUDART_VERSION < 5000

#include <cuda.h>

template <class T>
inline void getCudaAttribute(T *attribute, CUdevice_attribute device_attribute,
                             int device) {
  CUresult error = cuDeviceGetAttribute(attribute, device_attribute, device);

  if (CUDA_SUCCESS != error) {
    fprintf(
        stderr,
        "cuSafeCallNoSync() Driver API error = %04d from file <%s>, line %i.\n",
        error, __FILE__, __LINE__);

    exit(EXIT_FAILURE);
  }
}

#endif 
int main(int argc, char **argv) {
  pArgc = &argc;
  pArgv = argv;

  printf("%s Starting...\n\n", argv[0]);
  printf(
      " CUDA Device Query (Runtime API) version (CUDART static linking)\n\n");

  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  if (error_id != cudaSuccess) {
    printf("cudaGetDeviceCount returned %d\n-> %s\n",
           static_cast<int>(error_id), cudaGetErrorString(error_id));
    printf("Result = FAIL\n");
    exit(EXIT_FAILURE);
  }

  if (deviceCount == 0) {
    printf("There are no available device(s) that support CUDA\n");
  } else {
    printf("Detected %d CUDA Capable device(s)\n", deviceCount);
  }

  int dev, driverVersion = 0, runtimeVersion = 0;

  for (dev = 0; dev < deviceCount; ++dev) {
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
           driverVersion / 1000, (driverVersion % 100) / 10,
           runtimeVersion / 1000, (runtimeVersion % 100) / 10);
    printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
           deviceProp.major, deviceProp.minor);

    char msg[256];
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    sprintf_s(msg, sizeof(msg),
              "  Total amount of global memory:                 %.0f MBytes "
              "(%llu bytes)\n",
              static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
              (unsigned long long)deviceProp.totalGlobalMem);
#else
    snprintf(msg, sizeof(msg),
             "  Total amount of global memory:                 %.0f MBytes "
             "(%llu bytes)\n",
             static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
             (unsigned long long)deviceProp.totalGlobalMem);
#endif
    printf("%s", msg);

    printf("  (%03d) Multiprocessors, (%03d) CUDA Cores/MP:    %d CUDA Cores\n",
           deviceProp.multiProcessorCount,
           _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
           _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) *
               deviceProp.multiProcessorCount);
    printf(
        "  GPU Max Clock rate:                            %.0f MHz (%0.2f "
        "GHz)\n",
        deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);

#if CUDART_VERSION >= 5000
    printf("  Memory Clock rate:                             %.0f Mhz\n",
           deviceProp.memoryClockRate * 1e-3f);
    printf("  Memory Bus Width:                              %d-bit\n",
           deviceProp.memoryBusWidth);

    if (deviceProp.l2CacheSize) {
      printf("  L2 Cache Size:                                 %d bytes\n",
             deviceProp.l2CacheSize);
    }

#else
    int memoryClock;
    getCudaAttribute<int>(&memoryClock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
                          dev);
    printf("  Memory Clock rate:                             %.0f Mhz\n",
           memoryClock * 1e-3f);
    int memBusWidth;
    getCudaAttribute<int>(&memBusWidth,
                          CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev);
    printf("  Memory Bus Width:                              %d-bit\n",
           memBusWidth);
    int L2CacheSize;
    getCudaAttribute<int>(&L2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev);

    if (L2CacheSize) {
      printf("  L2 Cache Size:                                 %d bytes\n",
             L2CacheSize);
    }

#endif

    printf(
        "  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, "
        "%d), 3D=(%d, %d, %d)\n",
        deviceProp.maxTexture1D, deviceProp.maxTexture2D[0],
        deviceProp.maxTexture2D[1], deviceProp.maxTexture3D[0],
        deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
    printf(
        "  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n",
        deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1]);
    printf(
        "  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d "
        "layers\n",
        deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1],
        deviceProp.maxTexture2DLayered[2]);

    printf("  Total amount of constant memory:               %zu bytes\n",
           deviceProp.totalConstMem);
    printf("  Total amount of shared memory per block:       %zu bytes\n",
           deviceProp.sharedMemPerBlock);
    printf("  Total shared memory per multiprocessor:        %zu bytes\n",
           deviceProp.sharedMemPerMultiprocessor);
    printf("  Total number of registers available per block: %d\n",
           deviceProp.regsPerBlock);
    printf("  Warp size:                                     %d\n",
           deviceProp.warpSize);
    printf("  Maximum number of threads per multiprocessor:  %d\n",
           deviceProp.maxThreadsPerMultiProcessor);
    printf("  Maximum number of threads per block:           %d\n",
           deviceProp.maxThreadsPerBlock);
    printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
           deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
           deviceProp.maxThreadsDim[2]);
    printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
           deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
           deviceProp.maxGridSize[2]);
    printf("  Maximum memory pitch:                          %zu bytes\n",
           deviceProp.memPitch);
    printf("  Texture alignment:                             %zu bytes\n",
           deviceProp.textureAlignment);
    printf(
        "  Concurrent copy and kernel execution:          %s with %d copy "
        "engine(s)\n",
        (deviceProp.deviceOverlap ? "Yes" : "No"), deviceProp.asyncEngineCount);
    printf("  Run time limit on kernels:                     %s\n",
           deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
    printf("  Integrated GPU sharing Host Memory:            %s\n",
           deviceProp.integrated ? "Yes" : "No");
    printf("  Support host page-locked memory mapping:       %s\n",
           deviceProp.canMapHostMemory ? "Yes" : "No");
    printf("  Alignment requirement for Surfaces:            %s\n",
           deviceProp.surfaceAlignment ? "Yes" : "No");
    printf("  Device has ECC support:                        %s\n",
           deviceProp.ECCEnabled ? "Enabled" : "Disabled");
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    printf("  CUDA Device Driver Mode (TCC or WDDM):         %s\n",
           deviceProp.tccDriver ? "TCC (Tesla Compute Cluster Driver)"
                                : "WDDM (Windows Display Driver Model)");
#endif
    printf("  Device supports Unified Addressing (UVA):      %s\n",
           deviceProp.unifiedAddressing ? "Yes" : "No");
    printf("  Device supports Managed Memory:                %s\n",
           deviceProp.managedMemory ? "Yes" : "No");
    printf("  Device supports Compute Preemption:            %s\n",
           deviceProp.computePreemptionSupported ? "Yes" : "No");
    printf("  Supports Cooperative Kernel Launch:            %s\n",
           deviceProp.cooperativeLaunch ? "Yes" : "No");
    printf("  Supports MultiDevice Co-op Kernel Launch:      %s\n",
           deviceProp.cooperativeMultiDeviceLaunch ? "Yes" : "No");
    printf("  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n",
           deviceProp.pciDomainID, deviceProp.pciBusID, deviceProp.pciDeviceID);

    const char *sComputeMode[] = {
        "Default (multiple host threads can use ::cudaSetDevice() with device "
        "simultaneously)",
        "Exclusive (only one host thread in one process is able to use "
        "::cudaSetDevice() with this device)",
        "Prohibited (no host thread can use ::cudaSetDevice() with this "
        "device)",
        "Exclusive Process (many threads in one process is able to use "
        "::cudaSetDevice() with this device)",
        "Unknown", NULL};
    printf("  Compute Mode:\n");
    printf("     < %s >\n", sComputeMode[deviceProp.computeMode]);
  }

  if (deviceCount >= 2) {
    cudaDeviceProp prop[64];
    int gpuid[64]; 
    int gpu_p2p_count = 0;

    for (int i = 0; i < deviceCount; i++) {
      checkCudaErrors(cudaGetDeviceProperties(&prop[i], i));

      if ((prop[i].major >= 2)
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
          && prop[i].tccDriver
#endif
          ) {
        gpuid[gpu_p2p_count++] = i;
      }
    }

    int can_access_peer;

    if (gpu_p2p_count >= 2) {
      for (int i = 0; i < gpu_p2p_count; i++) {
        for (int j = 0; j < gpu_p2p_count; j++) {
          if (gpuid[i] == gpuid[j]) {
            continue;
          }
          checkCudaErrors(
              cudaDeviceCanAccessPeer(&can_access_peer, gpuid[i], gpuid[j]));
          printf("> Peer access from %s (GPU%d) -> %s (GPU%d) : %s\n",
                 prop[gpuid[i]].name, gpuid[i], prop[gpuid[j]].name, gpuid[j],
                 can_access_peer ? "Yes" : "No");
        }
      }
    }
  }

  printf("\n");
  std::string sProfileString = "deviceQuery, CUDA Driver = CUDART";
  char cTemp[16];

  sProfileString += ", CUDA Driver Version = ";
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
  sprintf_s(cTemp, 10, "%d.%d", driverVersion / 1000,
            (driverVersion % 100) / 10);
#else
  snprintf(cTemp, sizeof(cTemp), "%d.%d", driverVersion / 1000,
           (driverVersion % 100) / 10);
#endif
  sProfileString += cTemp;

  sProfileString += ", CUDA Runtime Version = ";
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
  sprintf_s(cTemp, 10, "%d.%d", runtimeVersion / 1000,
            (runtimeVersion % 100) / 10);
#else
  snprintf(cTemp, sizeof(cTemp), "%d.%d", runtimeVersion / 1000,
           (runtimeVersion % 100) / 10);
#endif
  sProfileString += cTemp;

  sProfileString += ", NumDevs = ";
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
  sprintf_s(cTemp, 10, "%d", deviceCount);
#else
  snprintf(cTemp, sizeof(cTemp), "%d", deviceCount);
#endif
  sProfileString += cTemp;
  sProfileString += "\n";
  printf("%s", sProfileString.c_str());

  printf("Result = PASS\n");

  exit(EXIT_SUCCESS);
}
