// New - Ka
#include "hip_raytracer.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <cstring>

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
    return true;
    std::cout << "\n=== [Inicjalizacja AMD HIP] ===\n";
    
    int deviceCount = 0;
    if (hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount == 0) {
        std::cerr << "BŁĄD: Nie znaleziono żadnych urządzeń obsługujących HIP!" << std::endl;
        return false;
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
//NEWNEW - KB
// -------------------------------------------------------------------------
// 2. Ładowanie skompilowanego modułu OSL (pliku binarnego ELF dla AMD) - Na
// -------------------------------------------------------------------------
bool HipRaytracer::load_shader(const GPUShaderModuleDesc& desc) {
    if (!desc.data_ptr || desc.data_size == 0) {
        std::cerr << "[Błąd HIP] Pusty moduł przekazany do load_shader()!\n";
        return false;
    }

    const std::string hsaco_input  = "/tmp/osl_temp_shader.o";
    const std::string hsaco_output = "/tmp/gotowy_shader.hsaco";

    // 1. Zapisz surowy obiekt ELF na dysk
    {
        std::ofstream f(hsaco_input, std::ios::binary);
        if (!f) {
            std::cerr << "BŁĄD: Nie można otworzyć " << hsaco_input << " do zapisu!\n";
            return false;
        }
        f.write(static_cast<const char*>(desc.data_ptr), desc.data_size);
    }
    std::cout << "[HIP] Zapisano obiekt (" << desc.data_size << " B) do " << hsaco_input << "\n";

    // 2. Linkowanie object file -> hsaco przez ld.lld
    const std::string link_cmd =
        "/opt/rocm-7.2.4/lib/llvm/bin/ld.lld -shared "
        + hsaco_input + " -o " + hsaco_output;
    std::cout << "[HIP] Linkowanie: " << link_cmd << "\n";

    if (std::system(link_cmd.c_str()) != 0) {
        std::cerr << "BŁĄD: Linkowanie ld.lld nie powiodło się!\n";
        return false;
    }

    // 3. Załaduj moduł GPU
    std::cout << "[HIP] Ładowanie modułu GPU z: " << hsaco_output << "\n";
    HIP_CHECK(hipModuleLoad(&m_module, hsaco_output.c_str()));

    // 4. Pobierz wskaźnik na kernel (w tej wersji wymuszamy "osl_kernel")
    std::cout << "[HIP] Szukam kernela: osl_kernel\n";
    hipError_t err = hipModuleGetFunction(&m_kernel, m_module, "osl_kernel");
    if (err != hipSuccess) {
        std::cerr << "BŁĄD: Nie znaleziono kernela 'osl_kernel'. HIP: " << hipGetErrorString(err) << "\n";
        return false;
    }

    std::cout << "[HIP] Załadowano kernel: osl_kernel\n";
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
             
    // 1. Alokacja pamięci dla obrazka wyjściowego (Teraz float x3 (RGB), zgodnie z nowym LLVM)
    size_t buffer_size = width * height * 3 * sizeof(float);
    hipDeviceptr_t d_output;
    HIP_CHECK_VOID(hipMalloc((void**)&d_output, buffer_size));

    // 2. Alokacja zerowych ShaderGlobals na GPU (aby OSL nie scrashował gdy do nich sięgnie)
    size_t sg_size = width * height * 256; 
    void* d_shaderglobals = nullptr;
    HIP_CHECK_VOID(hipMalloc(&d_shaderglobals, sg_size));
    HIP_CHECK_VOID(hipMemset(d_shaderglobals, 0, sg_size));

    // 3. Skonfigurowanie bloków i siatki (Grid) wątków
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    // 4. PRZYGOTOWANIE ARGUMENTÓW ZGODNYCH Z NOWYM WRAPPEREM LLVM (7 sztuk)
    void* d_groupdata = nullptr;
    void* userdata_base_ptr = nullptr;
    int shadeindex = 0;
    void* interactive_params_ptr = nullptr;

    void* args[] = {
        &d_shaderglobals,        // 1. ShaderGlobals*
        &d_groupdata,            // 2. void* groupdata
        &userdata_base_ptr,      // 3. void* userdata
        &d_output,               // 4. void* output_base
        &shadeindex,             // 5. int shadeindex
        &interactive_params_ptr, // 6. void* interactive_params
        &width                   // 7. int width
    };

    // 5. Uruchomienie kernela
    HIP_CHECK_VOID(hipModuleLaunchKernel(
        m_kernel,
        gridSize.x, gridSize.y, gridSize.z,
        blockSize.x, blockSize.y, blockSize.z,
        0, 0, // Pamięć współdzielona i strumień domyślny
        args, nullptr
    ));

    // 6. Czekamy aż karta skończy renderować
    HIP_CHECK_VOID(hipDeviceSynchronize());
    std::cout << "[HIP] Kernel zakończył pracę.\n";

    // 7. Sprzątanie VRAM
    HIP_CHECK_VOID(hipFree((void*)d_output));
    HIP_CHECK_VOID(hipFree(d_shaderglobals));
}