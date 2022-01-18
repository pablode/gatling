#ifndef SHADERGEN_H
#define SHADERGEN_H

#include <cstdint>
#include <string_view>
#include <vector>
#include <string>
#include <memory>

namespace sg
{
  struct Material;

  class ShaderGen
  {
  public:
    struct InitParams
    {
      std::string_view resourcePath;
      std::string_view shaderPath;
      std::string_view mtlxlibPath;
      std::string_view mtlxmdlPath;
    };

    bool init(const InitParams& params);
    ~ShaderGen();

  public:
    struct Material* createMaterialFromMtlx(std::string_view docStr);
    void destroyMaterial(struct Material* mat);

  public:
    struct MainShaderParams
    {
      uint32_t numThreadsX;
      uint32_t numThreadsY;
      uint32_t maxStackSize;
      uint32_t spp;
      uint32_t maxBounces;
      uint32_t rrBounceOffset;
      float rrInvMinTermProb;
      float maxSampleValue;
      float bgColor[4];
      std::vector<Material*> materials;
    };

    bool generateMainShader(const struct MainShaderParams* params,
                            std::vector<uint8_t>& spv,
                            std::string& entryPoint);

  private:
    class MdlRuntime* m_mdlRuntime = nullptr;
    class MdlMaterialCompiler* m_mdlMaterialCompiler = nullptr;
    class MdlHlslCodeGen* m_mdlHlslCodeGen = nullptr;
    class MtlxMdlCodeGen* m_mtlxMdlCodeGen = nullptr;
    class IShaderCompiler* m_shaderCompiler = nullptr;
    std::string m_shaderPath;
  };
}

#endif
