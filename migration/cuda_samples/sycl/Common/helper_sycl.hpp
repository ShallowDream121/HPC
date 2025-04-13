// helper_sycl.hpp
#ifndef COMMON_HELPER_SYCL_H_
#define COMMON_HELPER_SYCL_H_

#pragma once

#include <CL/sycl.hpp>
#include <iostream>
#include <cstdlib>
#include <string>

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

// SYCL Error Handling
template <typename T>
void check(T error, const char* func, const char* file, int line) {
  if (error != T::success) {
    std::cerr << "SYCL error at " << file << ":" << line 
              << " code=" << static_cast<int>(error) 
              << "(\"" << sycl::sycl_category().message(static_cast<int>(error)) 
              << "\") in \"" << func << "\"\n";
    std::exit(EXIT_FAILURE);
  }
}

#define CHECK_SYCL(func) check(func, #func, __FILE__, __LINE__)

// SYCL Device Management
inline sycl::device selectSyclDevice(bool require_gpu = true) {
  auto devices = sycl::device::get_devices();
  
  if (devices.empty()) {
    std::cerr << "No SYCL devices found!\n";
    std::exit(EXIT_FAILURE);
  }

  // Priority: GPU > CPU > Host
  sycl::device* selected = nullptr;
  for (auto& dev : devices) {
    if (dev.is_gpu()) {
      selected = &dev;
      break;
    } else if (!selected && dev.is_cpu()) {
      selected = &dev;
    }
  }

  if (!selected) {
    if (require_gpu) {
      std::cerr << "No GPU device available!\n";
      std::exit(EXIT_FAILURE);
    }
    selected = &devices[0];
  }

  std::cout << "Selected device: " 
            << selected->get_info<sycl::info::device::name>()
            << "\n";
  return *selected;
}

// SYCL Device Information
inline void printDeviceInfo(const sycl::device& dev) {
  std::cout << "Device Info:\n"
            << "  Name: " << dev.get_info<sycl::info::device::name>() << "\n"
            << "  Vendor: " << dev.get_info<sycl::info::device::vendor>() << "\n"
            << "  Version: " << dev.get_info<sycl::info::device::version>() << "\n"
            << "  Global Memory: " 
            << dev.get_info<sycl::info::device::global_mem_size>() / (1024 * 1024)
            << " MB\n"
            << "  Local Memory: "
            << dev.get_info<sycl::info::device::local_mem_size>() / 1024 
            << " KB\n"
            << "  Compute Units: " 
            << dev.get_info<sycl::info::device::max_compute_units>() << "\n";
}

// SYCL Buffer Dimension Check
template <int Dims>
void checkBufferDims(const sycl::range<Dims>& r) {
  static_assert(Dims <= 3, "SYCL supports up to 3D buffers");
}

// SYCL Kernel Configuration Helper
inline size_t calculateBlockSize(size_t problemSize, size_t maxBlockSize = 256) {
  return std::min(problemSize, maxBlockSize);
}

// SYCL Version Check
inline bool checkSyclVersion(int major, int minor) {
#ifdef SYCL_LANGUAGE_VERSION
  const int version = SYCL_LANGUAGE_VERSION;
  const int actual_major = version / 100;
  const int actual_minor = (version % 100) / 10;
  return (actual_major > major) || 
         (actual_major == major && actual_minor >= minor);
#else
  return false;
#endif
}

// SYCL Exception Handler
inline auto exceptionHandler = [](sycl::exception_list exceptions) {
  for (std::exception_ptr const& e : exceptions) {
    try {
      std::rethrow_exception(e);
    } catch (sycl::exception const& e) {
      std::cerr << "SYCL Exception caught: " << e.what() << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
};

// SYCL Queue Management
inline sycl::queue createSyclQueue(bool enable_profiling = false) {
  sycl::property_list props = enable_profiling ?
    sycl::property_list{sycl::property::queue::enable_profiling()} :
    sycl::property_list{};
  
  return sycl::queue(selectSyclDevice(), exceptionHandler, props);
}

#endif // COMMON_HELPER_SYCL_H_