// New - Na
#pragma once
#include "gpu_raytracer.h"
#include <hip/hip_runtime.h> // Nagłówek dla typów HIP 

class HipRaytracer : public GPURaytracer {
public:
    HipRaytracer() = default;
    
    // Nadpisujemy destruktor, żeby zwalniał pamięć VRAM przy wyjściu z programu
    ~HipRaytracer() override;

    bool init() override;
    bool load_shader(const GPUShaderModuleDesc& desc) override;
    void render(int width, int height) override;

private:
    // Zmienna przechowująca skompilowany kod na karcie graficznej
    hipModule_t m_module = nullptr;
    hipFunction_t m_kernel = nullptr;
    std::string m_device_architecture;
};
