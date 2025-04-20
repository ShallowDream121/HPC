#include <CL/sycl.hpp>
#include <string>
#include <vector>

using namespace sycl;

// SYCL 设备架构信息查询函数
std::string get_device_arch_name(const device& dev) {
    // 架构映射表（兼容多厂商）
    struct ArchMap {
        int arch_id;
        std::string name;
    };

    // NVIDIA GPU 架构映射（保留原始CUDA逻辑）
    static const std::vector<ArchMap> nvidia_arch_map = {
        {0x30, "Kepler"},    {0x32, "Kepler"},    {0x35, "Kepler"},
        {0x37, "Kepler"},    {0x50, "Maxwell"},   {0x52, "Maxwell"},
        {0x53, "Maxwell"},   {0x60, "Pascal"},    {0x61, "Pascal"},
        {0x62, "Pascal"},    {0x70, "Volta"},     {0x72, "Xavier"},
        {0x75, "Turing"},    {0x80, "Ampere"},    {0x86, "Ampere"},
        {0x87, "Ampere"},    {0x89, "Ada"},       {0x90, "Hopper"},
        {0xa0, "Blackwell"}, {0xa1, "Blackwell"}, {0xc0, "Blackwell"},
        {-1, "NVIDIA Graphics Device"}
    };

    // 获取设备供应商
    const std::string vendor = dev.get_info<info::device::vendor>();

    if (vendor.find("NVIDIA") != std::string::npos) {
        // 针对 NVIDIA 设备的处理
#if defined(__SYCL_IMPLICIT_NVIDIA__) || defined(SYCL_EXT_ONEAPI_CUDA)
        // 使用扩展获取计算能力（需启用CUDA后端）
        int major = dev.get_info<sycl::ext::oneapi::experimental::info::device::compute_capability_major>();
        int minor = dev.get_info<sycl::ext::oneapi::experimental::info::device::compute_capability_minor>();
        
        const int sm_version = (major << 4) + minor;
        for (const auto& entry : nvidia_arch_map) {
            if (entry.arch_id == sm_version) 
                return entry.name;
            if (entry.arch_id == -1) 
                break;
        }
#endif
        return "NVIDIA Unknown Architecture";
    } else if (vendor.find("Intel") != std::string::npos) {
        // Intel GPU 架构判断
        const std::string name = dev.get_info<info::device::name>();
        if (name.find("Arc") != std::string::npos) return "Intel Arc";
        if (name.find("Gen") != std::string::npos) return "Intel Gen";
        return "Intel Graphics";
    } else if (vendor.find("AMD") != std::string::npos) {
        return "AMD Radeon";
    }

    return "Generic GPU";
}

// 使用示例
int main() {
    queue q(gpu_selector_v);
    device dev = q.get_device();
    
    std::cout << "Device Architecture: " 
              << get_device_arch_name(dev) << "\n";
    return 0;
}