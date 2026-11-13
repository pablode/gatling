//
// Copyright (C) 2026 Pablo Delgado Krämer
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.
//

#pragma once

#include <stdint.h>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <gtl/cgpu/Cgpu.h>
#include <gtl/gi/Gi.h>
#include <gtl/mc/Material.h>
#include <gtl/mc/MtlxDocOps.h>

#include "GlslShaderGen.h"
#include "TextureHeap.h"
#include "TextureManager.h"

namespace gtl
{
  class GbThreadPool;
  class GgpuStager;
  class GgpuDeleteQueue;
  class McFrontend;
  class McMtlxDocumentPatcher;
  struct GiScene;

  struct GiMaterialGpuData
  {
    GiTextureHeapAllocation texOffsetAllocation;
    GiTextureHeapAllocation textureInfoOffsetAllocation;
    std::vector<GiImagePtr> images;
    std::vector<McTextureParameterBinding> textureParameterBindings;
    std::vector<uint32_t> textureParameterInfoIndices;
    bool useTextureInfoLut = false;
  };

  struct GiBaseMaterial
  {
    uint64_t topoHash = 0;
    MaterialX::DocumentPtr doc;
    MaterialX::NodePtr surfaceShader;
    McMaterial* mcMat = nullptr;
    std::vector<McParameter> parameters;
    uint32_t materialInstanceCount = 0;
    std::vector<std::string> parameterNames;
    std::vector<MaterialX::InputPtr> parameterInputs;
    std::vector<std::vector<uint8_t>> parameterDefaultData;
    std::vector<std::string> parameterDefaultStringValues;
    std::vector<uint32_t> freeInstances;
    std::vector<uint32_t> instanceRefCounts;
    std::vector<std::string> stringConstants;

    CgpuBuffer parameterBuffer;
    uint64_t parameterBufferAddress = 0;
    uint64_t parameterBufferSize = 0;
    uint32_t textureCount = 0;
  };

  using GiBaseMaterialPtr = std::shared_ptr<GiBaseMaterial>;

  struct GiMaterial
  {
    GiBaseMaterialPtr base;
    std::string name;
    GiScene* scene;
    std::optional<GiMaterialGpuData> gpuData;
    uint32_t materialInstance = 0;
  };

  struct GiMaterialCompileInfo
  {
    GiMaterial* material = nullptr;
    GiGlslShaderGen::MaterialGenInfo genInfo;
  };

  class GiMaterialManager
  {
  public:
    GiMaterialManager(CgpuContext* ctx,
                      GgpuStager& stager,
                      GgpuDeleteQueue& deleteQueue,
                      McFrontend& mcFrontend,
                      McMtlxDocumentPatcher& mtlxDocPatcher,
                      GiGlslShaderGen& shaderGen,
                      GiTextureManager& texSys,
                      GiTextureHeap& textureHeap,
                      GbThreadPool& threadPool);

    void destroy();

    GiMaterial* createMaterialFromMtlxDoc(GiScene* scene,
                                          const char* name,
                                          const std::shared_ptr<void/*MaterialX::Document*/> doc,
                                          bool usdPreviewSurfacePresenceOverride,
                                          bool& shadersChanged);
    GiMaterial* createMaterialFromMdlFile(GiScene* scene,
                                          const char* name,
                                          const char* filePath,
                                          const char* subIdentifier,
                                          const GiMaterialParameters& params);
    void destroyMaterial(GiMaterial* material);

    bool compileMaterials(const std::vector<GiMaterial*>& materials,
                          bool needsAuxOutput,
                          std::vector<GiMaterialCompileInfo>& materialInfos);

    bool updateParameterBuffers(std::vector<GiBaseMaterialPtr>& changedBases);

  private:
    GiMaterial* createMaterial(GiScene* scene,
                               const GiBaseMaterialPtr& base,
                               const char* name,
                               uint32_t materialInstance);

    bool createBaseMaterialFromMtlxDoc(GiBaseMaterial& base,
                                       const MaterialX::DocumentPtr& doc,
                                       const MaterialX::NodePtr& surfaceShader,
                                       uint32_t& materialInstance);
    void destroyMaterialInstance(GiBaseMaterial& base, uint32_t materialInstance);
    McParameter getParameters(const GiBaseMaterial& base, uint32_t materialInstance) const;

    GiBaseMaterialPtr makeBaseMaterial();
    GiBaseMaterialPtr getOrCreateBaseMaterial(uint64_t topoHash);
    void invalidateGpuData(const GiBaseMaterialPtr& base);

    bool createMaterialGpuData(const GiMaterialCompileInfo& info,
                               const std::vector<GiTextureResource>& textureResources,
                               uint32_t resourceStart,
                               uint32_t resourceCount,
                               GiMaterialGpuData& gpuData);
    void destroyMaterialGpuData(GiMaterialGpuData& gpuData);

  private:
    CgpuContext* m_ctx;
    GgpuStager& m_stager;
    GgpuDeleteQueue& m_deleteQueue;
    McFrontend& m_mcFrontend;
    McMtlxDocumentPatcher& m_mtlxDocPatcher;
    GiGlslShaderGen& m_shaderGen;
    GiTextureManager& m_texSys;
    GiTextureHeap& m_textureHeap;
    GbThreadPool& m_threadPool;

    std::mutex m_mutex;
    mutable std::mutex m_baseMaterialMutex;
    std::unordered_set<GiMaterial*> m_materials;
    std::unordered_map<uint64_t, std::weak_ptr<GiBaseMaterial>> m_baseMaterials;
  };
}
