// New - Na
#pragma once
#include "gpu_raytracer.h"

class HipRaytracer : public GPURaytracer {
public:
    HipRaytracer() = default;
    ~HipRaytracer() override = default;

    bool init() override;
    bool load_shader(const GPUShaderModuleDesc& desc) override;
    void render(int width, int height) override;
};
