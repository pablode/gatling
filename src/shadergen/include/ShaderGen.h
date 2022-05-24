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
    Material* createMaterialFromMtlx(std::string_view docStr);
    Material* createMaterialFromMdlFile(std::string_view filePath, std::string_view subIdentifier);
    void destroyMaterial(Material* mat);
    bool isMaterialEmissive(const struct Material* mat);

  public:
    struct MainShaderParams
    {
      uint32_t aovId;
      uint32_t numThreadsX;
      uint32_t numThreadsY;
      float postponeRatio;
      uint32_t maxStackSize;
      std::vector<Material*> materials;
      bool trianglePostponing;
      bool nextEventEstimation;
      uint32_t emissiveFaceCount;
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
