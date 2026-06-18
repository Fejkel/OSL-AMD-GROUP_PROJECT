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
//NEWNEW - KB
// -------------------------------------------------------------------------
// 2. Ładowanie skompilowanego modułu OSL (pliku binarnego ELF dla AMD) - Na
// -------------------------------------------------------------------------
bool HipRaytracer::load_shader(const GPUShaderModuleDesc& desc) {
    std::cout << "[HIP] Wywołano load_shader()...\n";

    if (!desc.data_ptr || desc.data_size == 0) {
        std::cerr << "BŁĄD: Otrzymano pusty wskaźnik na bajty shadera!\n";
        return false;
    }

    if (desc.kernel_name.empty()) {
        std::cerr << "BŁĄD: Brak nazwy kernela w GPUShaderModuleDesc!\n";
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

    // 4. Pobierz wskaźnik na kernel
    std::cout << "[HIP] Szukam kernela: " << desc.kernel_name << "\n";
    hipError_t err = hipModuleGetFunction(&m_kernel, m_module, desc.kernel_name.c_str());
    if (err != hipSuccess) {
        std::cerr << "BŁĄD: Nie znaleziono kernela '" << desc.kernel_name
                  << "'. HIP: " << hipGetErrorString(err) << "\n";
        // Wypisz dostępne symbole dla debugowania
        std::system("nm /tmp/gotowy_shader.hsaco 2>&1 | grep ' T ' || "
                    "nm /tmp/osl_temp_shader.o 2>&1 | grep ' T '");
        return false;
    }

    std::cout << "[HIP] Załadowano kernel: " << desc.kernel_name << "\n";
    return true;
}
// -------------------------------------------------------------------------
// 3. Uruchomienie kernela renderującego na GPU
// -------------------------------------------------------------------------
void HipRaytracer::render(int width, int height, int groupdata_size, float* host_output_buffer) {
    std::cout << "[HIP] Rozpoczęcie renderowania sprzętowego. Rozdzielczość: " 
              << width << "x" << height << "\n";

    // 1. OBLICZENIE ROZMIARÓW I ALOKACJA STRUKTUR PER-PIKSEL
    // Zakładamy bezpieczny rozmiar struktury ShaderGlobals z OSL.
    // Musimy mieć osobną strukturę dla każdego piksela!
    size_t sg_single_size = 256; 
    size_t total_pixels = width * height;
    size_t sg_array_size = total_pixels * sg_single_size;

    // Tworzymy bufor tymczasowy na CPU, aby uzupełnić współrzędne u i v
    std::vector<char> cpu_sg_buffer(sg_array_size, 0);

    // Iterujemy po pikselach, aby ustawić u i v w strukturach
    // UWAGA: Mapowanie pól u i v zależy od definicji struktury ShaderGlobals w OSL.
    // Zazwyczaj u i v to pierwsze dwa lub jedne z pierwszych pól typu float.
    // Dla testu zakładamy standardowy układ OSL, gdzie 'u' i 'v' są na początku (bądź blisko początku) struktury.
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            size_t pixel_index = y * width + x;
            char* current_sg_ptr = cpu_sg_buffer.data() + (pixel_index * sg_single_size);

            float u_val = (static_cast<float>(x) + 0.5f) / static_cast<float>(width);
            float v_val = (static_cast<float>(y) + 0.5f) / static_cast<float>(height);

            // Wskaźniki na float wewnątrz surowego bufora bajtów (zakładamy początek struktury dla testu)
            // Jeśli OSL na GPU mapuje to inaczej, przesunięcie (offset) może wymagać dostosowania do definicji struktury z OSL
            float* sg_floats = reinterpret_cast<float*>(current_sg_ptr);
            sg_floats[0] = u_val; // przypisanie do u
            sg_floats[1] = v_val; // przypisanie do v
        }
    }

    // 2. ALOKACJA PAMIĘCI NA GPU (VRAM)
    void* d_shaderglobals = nullptr;
    void* d_groupdata     = nullptr;
    void* d_output        = nullptr;

    // Alokujemy tablicę ShaderGlobals na GPU i kopiujemy tam przygotowane dane z CPU
    HIP_CHECK_VOID(hipMalloc(&d_shaderglobals, sg_array_size));
    HIP_CHECK_VOID(hipMemcpy(d_shaderglobals, cpu_sg_buffer.data(), sg_array_size, hipMemcpyHostToDevice));

    // Alokujemy groupdata używając rozmiaru przekazanego z OSL
    if (groupdata_size > 0) {
        HIP_CHECK_VOID(hipMalloc(&d_groupdata, groupdata_size));
        HIP_CHECK_VOID(hipMemset(d_groupdata, 0, groupdata_size));
    }

    // Alokujemy bufor wyjściowy (3 kanały: RGB, float)
    size_t output_buffer_size = width * height * 3 * sizeof(float);
    HIP_CHECK_VOID(hipMalloc(&d_output, output_buffer_size));
    HIP_CHECK_VOID(hipMemset(d_output, 0, output_buffer_size));

    // 3. PRZYGOTOWANIE ARGUMENTÓW DLA OSL
    void* userdata_base_ptr      = nullptr;
    int   shadeindex             = 0;       
    void* interactive_params_ptr = nullptr;

    void* kernelArgs[] = {
        &d_shaderglobals,
        &d_groupdata,
        &userdata_base_ptr,
        &d_output,             
        &shadeindex,
        &interactive_params_ptr
    };

    // 4. KONFIGURACJA SIATKI WĄTKÓW
    dim3 blockSize(16, 16, 1);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                  (height + blockSize.y - 1) / blockSize.y, 1);

    std::cout << "[HIP] Uruchamianie kernela. Siatka: " 
              << gridSize.x << "x" << gridSize.y << ", Bloki: 16x16\n";

    // 5. WYWOŁANIE KERNELA
    HIP_CHECK_VOID(hipModuleLaunchKernel(
        m_kernel,
        gridSize.x, gridSize.y, gridSize.z,
        blockSize.x, blockSize.y, blockSize.z,
        0, nullptr, kernelArgs, nullptr
    ));

    // 6. SYNCHRONIZACJA I POBRANIE WYNIKU Z GPU DO CPU
    HIP_CHECK_VOID(hipDeviceSynchronize());
    
    if (host_output_buffer != nullptr) {
        HIP_CHECK_VOID(hipMemcpy(host_output_buffer, d_output, output_buffer_size, hipMemcpyDeviceToHost));
        std::cout << "[HIP] Pomyślnie skopiowano gotowy obraz z VRAM do RAM.\n";
        // --- TESTOWE WYMUSZENIE KOLORU (DOPISZ TO) ---
        // for (int i = 0; i < width * height * 3; i += 3) {
        //     host_output_buffer[i]     = 1.0f; // Czerwony
        //     host_output_buffer[i + 1] = 0.5f; // Zielony
        //     host_output_buffer[i + 2] = 0.5f; // Niebieski
        // }
        // ----------------------------------------------
    }

    // 7. SPRZĄTANIE VRAM
    HIP_CHECK_VOID(hipFree(d_shaderglobals));
    if (d_groupdata) HIP_CHECK_VOID(hipFree(d_groupdata));
    HIP_CHECK_VOID(hipFree(d_output));
}