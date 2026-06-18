#include "hip_raytracer.h"

#include <OSL/amdgpu_kernel.h>

#include <algorithm>
#include <iostream>

#include <hip/hip_runtime.h>

namespace {

bool
is_loadable_hip_module_format(const std::string& format)
{
    return format == "hsaco" || format == "amdgpu_code_object";
}

std::string
base_gfx_architecture(const std::string& architecture)
{
    size_t feature_separator = architecture.find(':');
    return feature_separator == std::string::npos
               ? architecture
               : architecture.substr(0, feature_separator);
}

bool
report_hip_error(hipError_t status, int line, const char* file)
{
    if (status == hipSuccess)
        return true;

    std::cerr << "HIP Error: " << hipGetErrorString(status)
              << " w linii " << line << " w pliku " << file << std::endl;
    return false;
}

}  // namespace

#define HIP_CHECK(command) \
    do { \
        if (!report_hip_error(command, __LINE__, __FILE__)) \
            return false; \
    } while (0)

bool
HipRaytracer::init()
{
    std::cout << "\n=== [Inicjalizacja AMD HIP] ===\n";

    int deviceCount = 0;
    if (hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount == 0) {
        std::cerr << "BLAD: Nie znaleziono urzadzen obslugujacych HIP!"
                  << std::endl;
        return false;
    }

    std::cout << "Znaleziono " << deviceCount << " urzadzenie/a HIP.\n";

    for (int i = 0; i < deviceCount; ++i) {
        hipDeviceProp_t deviceProp;
        HIP_CHECK(hipGetDeviceProperties(&deviceProp, i));
        if (i == 0)
            m_device_architecture = deviceProp.gcnArchName;

        std::cout << "Urzadzenie [" << i << "]: " << deviceProp.name << "\n";
        std::cout << "  Architektura (GCN/RDNA): " << deviceProp.gcnArchName
                  << "\n";
        std::cout << "  Calkowita pamiec VRAM: "
                  << deviceProp.totalGlobalMem / (1024 * 1024) << " MB\n";
        std::cout << "  Max watkow na blok: " << deviceProp.maxThreadsPerBlock
                  << "\n";
    }

    HIP_CHECK(hipSetDevice(0));
    std::cout << "Pomyslnie podpieto do GPU 0.\n";
    std::cout << "===============================\n\n";

    return true;
}

HipRaytracer::~HipRaytracer()
{
    if (m_module) {
        hipModuleUnload(m_module);
        m_module = nullptr;
        m_kernel = nullptr;
        std::cout << "[HIP] Zwolniono modul z pamieci VRAM.\n";
    }
}

bool
HipRaytracer::load_shader(const GPUShaderModuleDesc& desc)
{
    std::cout << "[HIP] Wywolano load_shader()...\n";

    if (!desc.data_ptr || desc.data_size == 0) {
        std::cerr << "BLAD: Otrzymano pusty wskaznik na bajty shadera!\n";
        return false;
    }

    std::cout << "[HIP] Artefakt: architektura=" << desc.architecture
              << ", format=" << desc.format
              << ", rozmiar=" << desc.data_size << " bajtow\n";

    std::string device_base_arch = base_gfx_architecture(m_device_architecture);
    std::string artifact_base_arch = base_gfx_architecture(desc.architecture);
    if (!device_base_arch.empty() && !artifact_base_arch.empty()
        && artifact_base_arch != device_base_arch) {
        std::cerr << "[HIP] Architektura artefaktu (" << desc.architecture
                  << ") rozni sie od architektury GPU ("
                  << m_device_architecture << ").\n";
        return false;
    }

    if (!is_loadable_hip_module_format(desc.format)) {
        std::cerr << "[HIP] Artefakt nie jest gotowym modulem HIP/HSACO. "
                  << "Pomijam hipModuleLoadData dla formatu: " << desc.format
                  << "\n";
        return false;
    }

    if (m_module) {
        hipModuleUnload(m_module);
        m_module = nullptr;
        m_kernel = nullptr;
    }

    HIP_CHECK(hipModuleLoadData(&m_module, desc.data_ptr));

    hipError_t status = hipModuleGetFunction(
        &m_kernel, m_module, OSL::OSL_AMDGPU_RENDER_KERNEL_NAME);
    if (status != hipSuccess) {
        std::cerr << "HIP Error: " << hipGetErrorString(status)
                  << " podczas szukania kernela "
                  << OSL::OSL_AMDGPU_RENDER_KERNEL_NAME << "\n";
        hipModuleUnload(m_module);
        m_module = nullptr;
        m_kernel = nullptr;
        return false;
    }

    std::cout << "[HIP] Sukces: wgrano modul shadera na GPU.\n";
    std::cout << "[HIP] Znaleziono kernel: "
              << OSL::OSL_AMDGPU_RENDER_KERNEL_NAME << "\n";
    return true;
}

void
HipRaytracer::render(int width, int height)
{
    if (!m_module) {
        std::cerr << "[HIP] Brak poprawnie zaladowanego modulu. "
                  << "Pomijam render GPU.\n";
        return;
    }
    if (!m_kernel) {
        std::cerr << "[HIP] Modul nie zawiera poprawnego kernela AMDGPU.\n";
        return;
    }

    std::cout << "[HIP] Rozpoczecie renderowania. Rozdzielczosc: " << width
              << "x" << height << "\n";

    size_t output_bytes = static_cast<size_t>(std::max(1, width))
                          * static_cast<size_t>(std::max(1, height))
                          * 4 * sizeof(float);
    void* output_base_ptr = nullptr;
    hipError_t status = hipMalloc(&output_base_ptr, output_bytes);
    if (status != hipSuccess) {
        std::cerr << "HIP Error: " << hipGetErrorString(status)
                  << " podczas hipMalloc bufora wyjsciowego\n";
        return;
    }

    void* groupdata_ptr = nullptr;
    void* userdata_base_ptr = nullptr;
    void* interactive_params_ptr = nullptr;
    void* args[] = {
        &output_base_ptr,
        &width,
        &height,
        &groupdata_ptr,
        &userdata_base_ptr,
        &interactive_params_ptr,
    };

    status = hipModuleLaunchKernel(m_kernel, 1, 1, 1, 1, 1, 1, 0, nullptr,
                                   args, nullptr);
    if (status == hipSuccess)
        status = hipDeviceSynchronize();

    hipFree(output_base_ptr);

    if (status != hipSuccess) {
        std::cerr << "HIP Error: " << hipGetErrorString(status)
                  << " podczas uruchamiania kernela "
                  << OSL::OSL_AMDGPU_RENDER_KERNEL_NAME << "\n";
        return;
    }

    std::cout << "[HIP] Kernel zakonczyl prace poprawnie.\n";
}
