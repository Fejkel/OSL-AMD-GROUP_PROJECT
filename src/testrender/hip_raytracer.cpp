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
// Destruktor (Zwalnianie pamięci z karty graficznej) - Na
// -------------------------------------------------------------------------
HipRaytracer::~HipRaytracer() {
    if (m_module) {
        hipModuleUnload(m_module);
        std::cout << "[HIP] Zwalnianie pamięci: Usunięto moduł z pamięci VRAM.\n";
    }
}

// -------------------------------------------------------------------------
// 2. Ładowanie skompilowanego modułu OSL (pliku binarnego ELF dla AMD) - Na
// -------------------------------------------------------------------------
bool HipRaytracer::load_shader(const GPUShaderModuleDesc& desc) {
    std::cout << "[HIP] Wywołano load_shader()...\n";

    if (!desc.data_ptr || desc.data_size == 0) {
        std::cerr << "BŁĄD: Otrzymano pusty wskaźnik na bajty shadera!\n";
        return false;
    }

    // Wgrywamy bajty z pamięci RAM (od LLVM) wprost do pamięci karty graficznej AMD!
    HIP_CHECK(hipModuleLoadData(&m_module, desc.data_ptr));
    
    std::cout << "[HIP] Sukces: Wgrano moduł shadera na kartę graficzną AMD!\n";
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

