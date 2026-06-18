// New - Ka
#include "hip_raytracer.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <cstdlib>
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

// NOWE makro dla funkcji zwracających void (render)
#define HIP_CHECK_VOID(command) \
{ \
    hipError_t status = command; \
    if (status != hipSuccess) { \
        std::cerr << "HIP Error: " << hipGetErrorString(status) \
                  << " w linii " << __LINE__ << " w pliku " << __FILE__ << std::endl; \
        return; \
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
        // TU JEST TRUE A MA BYC FALSE; TYLKO W CELACH TESTOW NA LAPTOPIE OLKA
    }

    std::cout << "Znaleziono " << deviceCount << " urządzenie/a HIP.\n";

    /*for (int i = 0; i < deviceCount; ++i) {
        hipDeviceProp_t deviceProp;
        HIP_CHECK(hipGetDeviceProperties(&deviceProp, i));
        
        std::cout << "Urządzenie [" << i << "]: " << deviceProp.name << "\n";
        std::cout << "  Architektura (GCN/RDNA): " << deviceProp.gcnArchName << "\n";
        std::cout << "  Całkowita pamięć VRAM: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB\n";
        std::cout << "  Max wątków na blok: " << deviceProp.maxThreadsPerBlock << "\n";
    }*/
    
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

    std::string hsaco_filename = "/tmp/osl_temp_shader.hsaco";
    
    // Zrzucamy gotowy plik od LLVM lub z cache prosto na dysk (dla dowodu działania)
    std::ofstream hsaco_file(hsaco_filename, std::ios::binary);
    hsaco_file.write(static_cast<const char*>(desc.data_ptr), desc.data_size);
    hsaco_file.close();
    
    std::cout << "[HIP] Sukces (Dry Run): Kod binarny zapisano bezpiecznie na dysk (" << hsaco_filename << ").\n";

  
    //HIP_CHECK(hipModuleLoadData(&m_module, desc.data_ptr));
    HIP_CHECK(hipModuleLoad(&m_module, "/tmp/osl_linked_shader.hsaco"));
    HIP_CHECK(hipModuleGetFunction(&m_kernel, m_module, "osl_kernel"));
    
    return true;
}



// -------------------------------------------------------------------------
// 3. Uruchomienie kernela renderującego na GPU
// -------------------------------------------------------------------------
void HipRaytracer::render(int width, int height) {
    std::cout << "[HIP] Rozpoczęcie renderowania. Rozdzielczość: " 
              << width << "x" << height << "\n";
    
    if (!m_kernel) {
        std::cerr << "[Błąd HIP] Kernel nie jest załadowany! Przerywam renderowanie.\n";
        return;
    }
              
              
              
    // 1. Alokacja pamięci VRAM na GPU dla bufora obrazu wyjściowego (RGBA, 4x float)
    size_t buffer_size = width * height * 4 * sizeof(float);
    hipDeviceptr_t d_output;
    HIP_CHECK_VOID(hipMalloc((void**)&d_output, buffer_size));

    // 2. Skonfigurowanie bloków i siatki (Grid) wątków
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    // 3. Przygotowanie argumentów do przekazania do kernela
    // WAŻNE: Struktura args musi ściśle odpowiadać sygnaturze funkcji w wygenerowanym pliku .bc!
    // Poniższe działa, jeśli kernel to: void my_kernel(float* d_output, int width, int height)
    void* args[] = { &d_output, &width, &height };

    // 4. Uruchomienie kernela na fizycznej karcie RDNA
    HIP_CHECK_VOID(hipModuleLaunchKernel(
        m_kernel,
        gridSize.x, gridSize.y, gridSize.z,
        blockSize.x, blockSize.y, blockSize.z,
        0, 0, // Pamięć współdzielona i strumień domyślny
        args, nullptr
    ));

    // 5. Czekamy aż karta skończy renderować
    HIP_CHECK_VOID(hipDeviceSynchronize());
    std::cout << "[HIP] Kernel zakończył pracę.\n";

    // 6. Skopiowanie wyniku z VRAM (GPU) z powrotem do RAM (Host)
    std::vector<float> h_output(width * height * 4);
    HIP_CHECK_VOID(hipMemcpy(h_output.data(), (void*)d_output, buffer_size, hipMemcpyDeviceToHost));

    std::cout << "[HIP] Renderowanie i pobieranie danych zakończone sukcesem!\n";
    // (Opcjonalnie: Zapisz wektor h_output do pliku .png używając OpenImageIO z OSL)

    // 7. Sprzątanie VRAM
    HIP_CHECK_VOID(hipFree((void*)d_output));
    
    

}

