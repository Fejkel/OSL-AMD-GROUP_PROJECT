// New - Ka
#include "hip_raytracer.h"
#include <iostream>
#include <hip/hip_runtime.h>

// Makro do wygodnego sprawdzania, czy funkcje HIP nie zwracają błędów
#define HIP_CHECK(command) \
{ \
    hipError_t status = command; \
    if (status != hipSuccess) { \
        std::cerr << "HIP Error: " << hipGetErrorString(status) \
                  << " w linii " << __LINE__ << " w pliku " << __FILE__ << std::endl; \
        return false; \
    } \
}

// -------------------------------------------------------------------------
// 1. Inicjalizacja środowiska i karty graficznej
// -------------------------------------------------------------------------
bool HipRaytracer::init() {
    std::cout << "\n=== [Inicjalizacja AMD HIP] ===\n";
    
    int deviceCount = 0;
    if (hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount == 0) {
        std::cerr << "BŁĄD: Nie znaleziono żadnych urządzeń obsługujących HIP!" << std::endl;
        return false;
    }

    std::cout << "Znaleziono " << deviceCount << " urządzenie/a HIP.\n";

    for (int i = 0; i < deviceCount; ++i) {
        hipDeviceProp_t deviceProp;
        HIP_CHECK(hipGetDeviceProperties(&deviceProp, i));
        
        std::cout << "Urządzenie [" << i << "]: " << deviceProp.name << "\n";
        std::cout << "  Architektura (GCN/RDNA): " << deviceProp.gcnArchName << "\n";
        std::cout << "  Całkowita pamięć VRAM: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB\n";
        std::cout << "  Max wątków na blok: " << deviceProp.maxThreadsPerBlock << "\n";
    }
    
    // Wybieramy domyślną kartę (indeks 0)
    HIP_CHECK(hipSetDevice(0));
    std::cout << "Pomyślnie podpięto do GPU 0.\n";
    std::cout << "===============================\n\n";
    
    return true;
}

// -------------------------------------------------------------------------
// 2. Ładowanie skompilowanego modułu OSL (pliku binarnego ELF dla AMD)
// -------------------------------------------------------------------------
bool HipRaytracer::load_shader(const GPUShaderModuleDesc& desc) {
    std::cout << "[HIP] Wywołano load_shader()...\n";
    
    // Tutaj w kolejnym etapie dodamy kod, który bierze bajty z desc
    // i ładuje je na kartę graficzną za pomocą:
    // hipModule_t module;
    // HIP_CHECK(hipModuleLoadData(&module, desc.skompilowany_kod_z_osl));
    
    return true;
}

// -------------------------------------------------------------------------
// 3. Uruchomienie kernela renderującego na GPU
// -------------------------------------------------------------------------
void HipRaytracer::render(int width, int height) {
    std::cout << "[HIP] Rozpoczęcie renderowania. Rozdzielczość: " 
              << width << "x" << height << "\n";
              
    // Tutaj w przyszłości:
    // 1. Zaalokujemy bufory obrazu (hipMalloc)
    // 2. Odpalimy kernel funkcji głównej (hipLaunchKernelGGL / hipModuleLaunchKernel)
    // 3. Skopiujemy wyrenderowany obraz z powrotem do RAMu (hipMemcpy)
}