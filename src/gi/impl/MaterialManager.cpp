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

#include "MaterialManager.h"

#include <gtl/gb/Log.h>
#include <gtl/gb/ThreadPool.h>
#include <gtl/ggpu/DeleteQueue.h>
#include <gtl/ggpu/Stager.h>
#include <gtl/mc/Frontend.h>
#include <gtl/mc/MtlxDocOps.h>
#include <gtl/mc/MtlxDocPatch.h>

#include <MaterialXCore/Document.h>

#include <assert.h>
#include <atomic>
#include <algorithm>
#include <string.h>

namespace mx = MaterialX;

namespace gtl
{
  namespace rp = shader_interface::rp_main;

  int _FindParameter(const std::vector<std::string>& names, const std::string& name)
  {
    auto it = std::find(names.begin(), names.end(), name);

    return it == names.end() ? -1 : int(it - names.begin());
  }

  bool _ParametersEqual(const McParameter& a, const McParameter& b)
  {
    return a.names == b.names && a.data == b.data && a.stringValues == b.stringValues;
  }

  bool _MakeParameterData(const mx::InputPtr& input,
                          const mx::ValuePtr& value,
                          std::vector<uint8_t>& data,
                          std::string& stringValue)
  {
    const std::string& type = input->getType();

    data.assign(16, 0);
    stringValue.clear();

    if (type == mx::FILENAME_TYPE_STRING || type == mx::STRING_TYPE_STRING)
    {
      stringValue = (value && value->isA<std::string>()) ? value->asA<std::string>() : input->getValueString();
      return true;
    }

    bool supported = (type == "float" || type == "integer" || type == "vector2" || type == "vector3" ||
                      type == "vector4" || type == "color3" || type == "color4");
    if (!supported)
    {
      return false;
    }

    if (!value)
    {
      return true;
    }

    if (type == "float" && value->isA<float>())
    {
      float v = value->asA<float>();
      memcpy(data.data(), &v, sizeof(v));
    }
    else if (type == "integer" && value->isA<int>())
    {
      int v = value->asA<int>();
      memcpy(data.data(), &v, sizeof(v));
    }
    else if (type == "vector2" && value->isA<mx::Vector2>())
    {
      const mx::Vector2& v = value->asA<mx::Vector2>();
      memcpy(data.data(), &v[0], sizeof(float) * 2);
    }
    else if (type == "vector3" && value->isA<mx::Vector3>())
    {
      const mx::Vector3& v = value->asA<mx::Vector3>();
      memcpy(data.data(), &v[0], sizeof(float) * 3);
    }
    else if (type == "color3" && value->isA<mx::Color3>())
    {
      const mx::Color3& v = value->asA<mx::Color3>();
      memcpy(data.data(), &v[0], sizeof(float) * 3);
    }
    else if (type == "vector4" && value->isA<mx::Vector4>())
    {
      const mx::Vector4& v = value->asA<mx::Vector4>();
      memcpy(data.data(), &v[0], sizeof(float) * 4);
    }
    else if (type == "color4" && value->isA<mx::Color4>())
    {
      const mx::Color4& v = value->asA<mx::Color4>();
      memcpy(data.data(), &v[0], sizeof(float) * 4);
    }
    else
    {
      return false;
    }

    return true;
  }

  McParameter _MakeDefaultParameters(const GiBaseMaterial& base)
  {
    McParameter parameters;
    parameters.names = base.parameterNames;
    parameters.data.assign(parameters.names.size() * 16, 0);
    parameters.stringValues = base.parameterDefaultStringValues;

    for (size_t i = 0; i < parameters.names.size(); i++)
    {
      memcpy(&parameters.data[i * 16], base.parameterDefaultData[i].data(), 16);
    }

    return parameters;
  }

  McParameter _RepackParameters(const GiBaseMaterial& base, const McParameter& oldParameters)
  {
    McParameter parameters = _MakeDefaultParameters(base);

    for (size_t i = 0; i < parameters.names.size(); i++)
    {
      int oldIndex = _FindParameter(oldParameters.names, parameters.names[i]);
      if (oldIndex < 0)
      {
        continue;
      }

      memcpy(&parameters.data[i * 16], &oldParameters.data[oldIndex * 16], 16);
      parameters.stringValues[i] = oldParameters.stringValues[oldIndex];
    }

    return parameters;
  }

  McParameter _MakeInstanceParameters(const GiBaseMaterial& base, const std::vector<McMtlxNetworkDiff>& diff)
  {
    McParameter parameters = _MakeDefaultParameters(base);

    for (const McMtlxNetworkDiff& networkDiff : diff)
    {
      int index = _FindParameter(parameters.names, McMtlxGetParameterName(networkDiff.input));
      if (index < 0)
      {
        continue;
      }

      std::vector<uint8_t> data;
      std::string stringValue;
      if (!_MakeParameterData(networkDiff.input, networkDiff.value, data, stringValue))
      {
        continue;
      }

      memcpy(&parameters.data[index * 16], data.data(), 16);
      parameters.stringValues[index] = std::move(stringValue);
    }

    return parameters;
  }

  bool _MergeMtlxParameters(GiBaseMaterial& base, const std::vector<McMtlxNetworkDiff>& diff)
  {
    bool parametersAdded = false;

    for (const McMtlxNetworkDiff& networkDiff : diff)
    {
      std::string name = McMtlxGetParameterName(networkDiff.input);
      if (_FindParameter(base.parameterNames, name) >= 0)
      {
        continue;
      }

      std::vector<uint8_t> defaultData;
      std::string defaultStringValue;
      if (!_MakeParameterData(networkDiff.input, networkDiff.input->getValue(), defaultData, defaultStringValue))
      {
        GB_DEBUG("input '{}' ({}) is not parameterizable; keeping reference value",
                 networkDiff.input->getNamePath(), networkDiff.input->getType());
        continue;
      }

      base.parameterNames.push_back(std::move(name));
      base.parameterInputs.push_back(networkDiff.input);
      base.parameterDefaultData.push_back(std::move(defaultData));
      base.parameterDefaultStringValues.push_back(std::move(defaultStringValue));
      parametersAdded = true;
    }

    if (parametersAdded)
    {
      for (uint32_t i = 0; i < base.materialInstanceCount; i++)
      {
        base.parameters[i] = _RepackParameters(base, base.parameters[i]);
      }
    }

    return parametersAdded;
  }

  GiMaterialManager::GiMaterialManager(CgpuContext* ctx,
                                       GgpuStager& stager,
                                       GgpuDeleteQueue& deleteQueue,
                                       McFrontend& mcFrontend,
                                       McMtlxDocumentPatcher& mtlxDocPatcher,
                                       GiGlslShaderGen& shaderGen,
                                       GiTextureManager& texSys,
                                       GiTextureHeap& textureHeap,
                                       GbThreadPool& threadPool)
    : m_ctx(ctx)
    , m_stager(stager)
    , m_deleteQueue(deleteQueue)
    , m_mcFrontend(mcFrontend)
    , m_mtlxDocPatcher(mtlxDocPatcher)
    , m_shaderGen(shaderGen)
    , m_texSys(texSys)
    , m_textureHeap(textureHeap)
    , m_threadPool(threadPool)
  {
  }

  void GiMaterialManager::destroy()
  {
    std::vector<GiMaterial*> materials;
    {
      std::lock_guard guard(m_mutex);
      materials.assign(m_materials.begin(), m_materials.end());
      m_materials.clear();
    }

    for (GiMaterial* material : materials)
    {
      destroyMaterial(material);
    }
  }

  GiMaterial* GiMaterialManager::createMaterialFromMtlxDoc(GiScene* scene,
                                                          const char* name,
                                                          const std::shared_ptr<void/*MaterialX::Document*/> doc,
                                                          bool usdPreviewSurfacePresenceOverride,
                                                          bool& shadersChanged)
  {
    mx::DocumentPtr resolvedDoc = std::static_pointer_cast<mx::Document>(doc);
    if (!resolvedDoc)
    {
      return nullptr;
    }

    mx::DocumentPtr patchedDoc = m_mtlxDocPatcher.patch(resolvedDoc, usdPreviewSurfacePresenceOverride);
    if (!patchedDoc)
    {
      return nullptr;
    }

    mx::NodePtr surfaceShader = McMtlxFindSurfaceShader(patchedDoc, name);
    if (!surfaceShader)
    {
      return nullptr;
    }

    GbHash topoHash = McMtlxHashNetworkTopological(surfaceShader);
    GB_DEBUG("material {} topoHash: {}", name, topoHash.val);

    GiBaseMaterialPtr base = topoHash.val != 0 ? getOrCreateBaseMaterial(topoHash.val) : makeBaseMaterial();

    McMaterial* oldMcMat = base->mcMat;

    uint32_t materialInstance = 0;
    if (!createBaseMaterialFromMtlxDoc(*base, patchedDoc, surfaceShader, materialInstance))
    {
      return nullptr;
    }

    shadersChanged = (base->mcMat != oldMcMat);

    GiMaterial* material = createMaterial(scene, base, name, materialInstance);
    if (shadersChanged)
    {
      invalidateGpuData(base);
    }

    return material;
  }

  bool GiMaterialManager::createBaseMaterialFromMtlxDoc(GiBaseMaterial& base,
                                                        const mx::DocumentPtr& doc,
                                                        const mx::NodePtr& surfaceShader,
                                                        uint32_t& materialInstance)
  {
    std::lock_guard guard(m_baseMaterialMutex);

    std::vector<McMtlxNetworkDiff> diff;

    if (!base.surfaceShader)
    {
      base.doc = doc->copy();

      mx::ElementPtr shaderElement = base.doc->getDescendant(surfaceShader->getNamePath());
      base.surfaceShader = shaderElement ? shaderElement->asA<mx::Node>() : nullptr;
      if (!base.surfaceShader)
      {
        return false;
      }

      base.parameters.resize(1024);
    }
    else
    {
      diff = McMtlxDiffTopoEqualNetworks(base.surfaceShader, surfaceShader);
    }

    if (base.freeInstances.empty() && base.materialInstanceCount >= base.parameters.size())
    {
      return false;
    }

    bool parametersAdded = _MergeMtlxParameters(base, diff);

    if (!base.mcMat || parametersAdded)
    {
      std::vector<McMtlxNetworkDiff> unionDiff;
      unionDiff.reserve(base.parameterInputs.size());
      for (const mx::InputPtr& parameterInput : base.parameterInputs)
      {
        unionDiff.push_back({ parameterInput, nullptr });
      }

      mx::DocumentPtr parameterizedDoc;
      std::vector<McMtlxTextureParameter> textureParameters;
      if (!McMtlxInsertParameterNodes(base.surfaceShader, unionDiff, parameterizedDoc, textureParameters))
      {
        return false;
      }

      mx::ElementPtr shaderElement = parameterizedDoc->getDescendant(base.surfaceShader->getNamePath());
      mx::NodePtr parameterizedShader = shaderElement ? shaderElement->asA<mx::Node>() : nullptr;
      if (!parameterizedShader)
      {
        return false;
      }

      McMaterial* newTemplateMaterial = m_mcFrontend.createMtlxTemplate(
        parameterizedDoc, parameterizedShader, std::move(textureParameters));
      if (!newTemplateMaterial)
      {
        return false;
      }

      delete base.mcMat;
      base.mcMat = newTemplateMaterial;
    }

    McParameter parameters = _MakeInstanceParameters(base, diff);

    int existingInstance = -1;
    for (uint32_t i = 0; i < base.materialInstanceCount; i++)
    {
      if (base.instanceRefCounts[i] > 0 && _ParametersEqual(base.parameters[i], parameters))
      {
        existingInstance = int(i);
        break;
      }
    }

    if (existingInstance >= 0)
    {
      materialInstance = uint32_t(existingInstance);
    }
    else
    {
      if (!base.freeInstances.empty())
      {
        materialInstance = base.freeInstances.back();
        base.freeInstances.pop_back();
      }
      else
      {
        materialInstance = base.materialInstanceCount++;
      }

      if (materialInstance >= base.instanceRefCounts.size())
      {
        base.instanceRefCounts.resize(materialInstance + 1, 0);
      }

      base.parameters[materialInstance] = std::move(parameters);
    }

    base.instanceRefCounts[materialInstance]++;

    return true;
  }

  void GiMaterialManager::destroyMaterialInstance(GiBaseMaterial& base, uint32_t materialInstance)
  {
    std::lock_guard guard(m_baseMaterialMutex);

    if (materialInstance >= base.instanceRefCounts.size() || base.instanceRefCounts[materialInstance] == 0)
    {
      return;
    }

    if (--base.instanceRefCounts[materialInstance] == 0)
    {
      base.freeInstances.push_back(materialInstance);
    }
  }

  McParameter GiMaterialManager::getParameters(const GiBaseMaterial& base, uint32_t materialInstance) const
  {
    std::lock_guard guard(m_baseMaterialMutex);

    assert(materialInstance < base.materialInstanceCount);

    return base.parameters[materialInstance];
  }

  GiMaterial* GiMaterialManager::createMaterialFromMdlFile(GiScene* scene,
                                                           const char* name,
                                                           const char* filePath,
                                                           const char* subIdentifier,
                                                           const GiMaterialParameters& params)
  {
    McMaterial* mcMat = m_mcFrontend.createFromMdlFile(filePath, subIdentifier, params);
    if (!mcMat)
    {
      return nullptr;
    }

    GiBaseMaterialPtr base = makeBaseMaterial();
    base->mcMat = mcMat;

    return createMaterial(scene, base, name, 0);
  }

  GiMaterial* GiMaterialManager::createMaterial(GiScene* scene,
                                                const GiBaseMaterialPtr& base,
                                                const char* name,
                                                uint32_t materialInstance)
  {
    GiMaterial* material = new GiMaterial {
      .base = base,
      .name = name,
      .scene = scene,
      .gpuData = std::nullopt,
      .materialInstance = materialInstance
    };

    {
      std::lock_guard guard(m_mutex);
      m_materials.insert(material);
    }

    return material;
  }

  void GiMaterialManager::destroyMaterial(GiMaterial* material)
  {
    {
      std::lock_guard guard(m_mutex);
      m_materials.erase(material);
    }

    if (material->gpuData)
    {
      destroyMaterialGpuData(*material->gpuData);
      material->gpuData.reset();
    }

    if (material->base->topoHash != 0)
    {
      destroyMaterialInstance(*material->base, material->materialInstance);
    }

    delete material;
  }

  GiBaseMaterialPtr GiMaterialManager::makeBaseMaterial()
  {
    return GiBaseMaterialPtr(new GiBaseMaterial, [this](GiBaseMaterial* base)
    {
      delete base->mcMat;

      if (base->parameterBuffer.handle)
      {
        m_deleteQueue.pushBack(base->parameterBuffer);
      }

      delete base;
    });
  }

  GiBaseMaterialPtr GiMaterialManager::getOrCreateBaseMaterial(uint64_t topoHash)
  {
    std::lock_guard guard(m_mutex);

    auto it = m_baseMaterials.find(topoHash);
    if (it != m_baseMaterials.end())
    {
      if (GiBaseMaterialPtr base = it->second.lock())
      {
        return base;
      }
    }

    GiBaseMaterialPtr base = makeBaseMaterial();
    base->topoHash = topoHash;

    m_baseMaterials[topoHash] = base;

    return base;
  }

  void GiMaterialManager::invalidateGpuData(const GiBaseMaterialPtr& base)
  {
    std::lock_guard guard(m_mutex);

    for (GiMaterial* material : m_materials)
    {
      if (material->base == base && material->gpuData)
      {
        destroyMaterialGpuData(*material->gpuData);
        material->gpuData.reset();
      }
    }
  }

  bool GiMaterialManager::updateParameterBuffers(std::vector<GiBaseMaterialPtr>& changedBases)
  {
    std::lock_guard guard(m_mutex);

    bool uploaded = false;

    for (auto& entry : m_baseMaterials)
    {
      GiBaseMaterialPtr base = entry.second.lock();
      if (!base)
      {
        continue;
      }

      uint32_t instanceCount = base->materialInstanceCount;
      if (instanceCount == 0)
      {
        continue;
      }

      if (instanceCount > rp::MAX_MATERIAL_INSTANCE_COUNT)
      {
        GB_ERROR("material instance count {} exceeds maximum {}", instanceCount, rp::MAX_MATERIAL_INSTANCE_COUNT);
        return false;
      }

      uint32_t slotCount = uint32_t(base->parameterNames.size()) + base->textureCount;
      uint64_t bufferSize = uint64_t(slotCount) * rp::MAX_MATERIAL_INSTANCE_COUNT * rp::SCENE_INFO_PARAMETER_ALIGNMENT;
      if (bufferSize == 0)
      {
        continue;
      }

      if (base->parameterBuffer.handle && base->parameterBufferSize != bufferSize)
      {
        m_deleteQueue.pushBack(base->parameterBuffer);
        base->parameterBuffer = {};
        base->parameterBufferAddress = 0;
        base->parameterBufferSize = 0;
      }

      if (!base->parameterBuffer.handle)
      {
        if (!cgpuCreateBuffer(m_ctx, {
                                .usage = CgpuBufferUsage::ShaderDeviceAddress | CgpuBufferUsage::TransferDst,
                                .memoryProperties = CgpuMemoryProperties::DeviceLocal,
                                .size = bufferSize,
                                .debugName = "MaterialParameters"
                              }, &base->parameterBuffer))
        {
          GB_ERROR("failed to create material parameter buffer");
          return false;
        }

        base->parameterBufferSize = bufferSize;
        changedBases.push_back(base);
      }

      std::vector<uint8_t> parameterData(bufferSize, 0);

      for (uint32_t materialInstance = 0; materialInstance < instanceCount; materialInstance++)
      {
        const McParameter parameters = getParameters(*base, materialInstance);

        for (size_t i = 0; i < base->parameterNames.size(); i++)
        {
          auto nameIt = std::find(parameters.names.begin(), parameters.names.end(), base->parameterNames[i]);
          if (nameIt == parameters.names.end())
          {
            continue;
          }

          size_t parameterIndex = nameIt - parameters.names.begin();
          uint64_t offset = (uint64_t(i) * rp::MAX_MATERIAL_INSTANCE_COUNT + materialInstance) * rp::SCENE_INFO_PARAMETER_ALIGNMENT;
          memcpy(&parameterData[offset], &parameters.data[parameterIndex * 16], 16);
        }
      }

      // Texture parameter slots are appended after the scalar parameter slots and are
      // indexed by the texture index used in the generated shader.
      for (uint32_t materialInstance = 0; materialInstance < rp::MAX_MATERIAL_INSTANCE_COUNT; materialInstance++)
      {
        for (uint32_t slot = uint32_t(base->parameterNames.size()); slot < slotCount; slot++)
        {
          uint64_t offset = (uint64_t(slot) * rp::MAX_MATERIAL_INSTANCE_COUNT + materialInstance) * rp::SCENE_INFO_PARAMETER_ALIGNMENT;
          uint32_t invalidIndex = UINT32_MAX;
          memcpy(&parameterData[offset], &invalidIndex, sizeof(invalidIndex));
        }
      }

      for (const GiMaterial* material : m_materials)
      {
        if (material->base != base || !material->gpuData)
        {
          continue;
        }

        const GiMaterialGpuData& gpuData = *material->gpuData;
        for (size_t b = 0; b < gpuData.textureParameterBindings.size(); b++)
        {
          uint32_t textureInfoIndex = gpuData.textureParameterInfoIndices[b];
          if (textureInfoIndex == UINT32_MAX)
          {
            continue;
          }

          uint32_t textureIndex = gpuData.textureParameterBindings[b].textureIndex;
          if (textureIndex < 1 || textureIndex > base->textureCount)
          {
            continue;
          }

          uint32_t slot = uint32_t(base->parameterNames.size()) + textureIndex - 1;
          uint64_t offset = (uint64_t(slot) * rp::MAX_MATERIAL_INSTANCE_COUNT + material->materialInstance) * rp::SCENE_INFO_PARAMETER_ALIGNMENT;
          memcpy(&parameterData[offset], &textureInfoIndex, sizeof(textureInfoIndex));
        }
      }

      if (!m_stager.stageToBuffer(parameterData.data(), parameterData.size(), base->parameterBuffer))
      {
        GB_ERROR("failed to upload material parameter buffer");
        return false;
      }

      base->parameterBufferAddress = cgpuGetBufferGpuAddress(m_ctx, base->parameterBuffer);
      if (base->parameterBufferAddress == 0)
      {
        GB_ERROR("failed to get material parameter buffer address");
        return false;
      }

      uploaded = true;
    }

    if (uploaded)
    {
      m_stager.flush();
    }

    return true;
  }

  bool GiMaterialManager::compileMaterials(const std::vector<GiMaterial*>& materials,
                                           bool needsAuxOutput,
                                           std::vector<GiMaterialCompileInfo>& materialInfos)
  {
    materialInfos.clear();

    std::vector<GiMaterialGpuData> gpuDatas;
    std::atomic_bool threadWorkFailed = false;

    // 1. Generate GLSL from MDL in parallel.
    {
      std::mutex materialInfosMutex;
      materialInfos.reserve(materials.size());

      m_threadPool.parallelFor((uint32_t) materials.size(), [&](uint32_t i) {
        GiMaterial* material = materials[i];

        GiGlslShaderGen::MaterialGenInfo genInfo;
        if (!m_shaderGen.generateMaterialInfo(*material->base->mcMat, genInfo, needsAuxOutput))
        {
          GB_ERROR("failed to generate code for material {}", material->name);
          threadWorkFailed = true;
          return;
        }

        GiMaterialCompileInfo info;
        info.material = material;
        info.genInfo = std::move(genInfo);

        std::lock_guard guard(materialInfosMutex);
        materialInfos.push_back(std::move(info));
      });
    }
    if (threadWorkFailed)
    {
      return false;
    }

    {
      std::unordered_map<const GiBaseMaterial*, uint32_t> instanceCounts;
      for (const GiMaterialCompileInfo& info : materialInfos)
      {
        instanceCounts[info.material->base.get()]++;
      }

      GB_LOG("compiling {} material instance(s) across {} base material(s)", materialInfos.size(), instanceCounts.size());
      for (const auto& entry : instanceCounts)
      {
        GB_DEBUG("> base material {:016x}: {} instance(s)", entry.first->topoHash, entry.second);
      }
    }

    // 2. Collect textures to load. Parameterized textures are loaded per material instance.
    std::vector<McTextureDescription> textureDescriptions;
    for (GiMaterialCompileInfo& info : materialInfos)
    {
      if (info.material->gpuData)
      {
        continue;
      }

      for (const McTextureParameterBinding& binding : info.genInfo.textureParameterBindings)
      {
        uint32_t index = binding.textureIndex - 1;
        if (index >= info.genInfo.textureDescriptions.size())
        {
          continue;
        }

        info.genInfo.textureDescriptions[index] = McTextureDescription{
          .binding = 0,
          .is3dImage = false,
          .isFloat = false,
          .width = 1,
          .height = 1,
          .depth = 1,
          .data = std::vector<uint8_t>(4, 0),
          .filePath = ""
        };
      }

      const auto& descriptions = info.genInfo.textureDescriptions;
      textureDescriptions.insert(textureDescriptions.end(), descriptions.begin(), descriptions.end());
    }

    // 3. Load textures and create per-material GPU data.
    std::vector<GiTextureResource> textureResources;
    if (!textureDescriptions.empty() && !m_texSys.loadTextureDescriptions(textureDescriptions, textureResources))
    {
      return false;
    }

    uint32_t resourceCounter = 0;
    gpuDatas.resize(materialInfos.size());
    for (size_t i = 0; i < materialInfos.size(); i++)
    {
      const GiMaterialCompileInfo& info = materialInfos[i];
      if (info.material->gpuData)
      {
        continue;
      }

      uint32_t resourceCount = uint32_t(info.genInfo.textureDescriptions.size());
      uint32_t resourceStart = resourceCounter;
      resourceCounter += resourceCount;

      if (!createMaterialGpuData(info, textureResources, resourceStart, resourceCount, gpuDatas[i]))
      {
        goto cleanup;
      }
    }

    for (size_t i = 0; i < materialInfos.size(); i++)
    {
      if (!materialInfos[i].material->gpuData)
      {
        materialInfos[i].material->gpuData = std::move(gpuDatas[i]);
      }
    }

    return true;

  cleanup:
    for (GiMaterialGpuData& gpuData : gpuDatas)
    {
      destroyMaterialGpuData(gpuData);
    }
    materialInfos.clear();
    return false;
  }

  bool GiMaterialManager::createMaterialGpuData(const GiMaterialCompileInfo& info,
                                                const std::vector<GiTextureResource>& textureResources,
                                                uint32_t resourceStart,
                                                uint32_t resourceCount,
                                                GiMaterialGpuData& gpuData)
  {
    gpuData.textureParameterBindings = info.genInfo.textureParameterBindings;
    gpuData.textureParameterInfoIndices.assign(gpuData.textureParameterBindings.size(), UINT32_MAX);
    gpuData.useTextureInfoLut = !gpuData.textureParameterBindings.empty();

    // Load per-instance textures for parameterized filename inputs.
    std::vector<McTextureDescription> parameterDescriptions;
    std::vector<uint32_t> parameterBindingIndices;
    if (!gpuData.textureParameterBindings.empty())
    {
      const McParameter parameters = getParameters(*info.material->base, info.material->materialInstance);

      for (uint32_t b = 0; b < gpuData.textureParameterBindings.size(); b++)
      {
        const McTextureParameterBinding& binding = gpuData.textureParameterBindings[b];
        auto nameIt = std::find(parameters.names.begin(), parameters.names.end(), binding.parameterName);
        if (nameIt == parameters.names.end())
        {
          continue;
        }

        uint32_t index = uint32_t(nameIt - parameters.names.begin());
        if (index >= parameters.stringValues.size() || parameters.stringValues[index].empty())
        {
          continue;
        }

        McTextureDescription desc = {};
        desc.filePath = parameters.stringValues[index];
        parameterDescriptions.push_back(std::move(desc));
        parameterBindingIndices.push_back(b);
      }
    }

    std::vector<GiTextureResource> parameterResources;
    if (!parameterDescriptions.empty() &&
        !m_texSys.loadTextureDescriptions(parameterDescriptions, parameterResources))
    {
      return false;
    }

    std::vector<GiTextureResource> resources;
    resources.reserve(resourceCount + parameterResources.size());
    for (uint32_t r = 0; r < resourceCount; r++)
    {
      resources.push_back(textureResources[resourceStart + r]);
    }
    for (GiTextureResource& resource : parameterResources)
    {
      resources.push_back(std::move(resource));
    }

    uint32_t imageCount = 0;
    for (const GiTextureResource& resource : resources)
    {
      imageCount += resource.uvtileWidth > 0 ? resource.uvtileWidth * resource.uvtileHeight : 1;
    }

    info.material->base->stringConstants = info.genInfo.stringConstants;

    std::vector<rp::TextureInfo> textureInfos(resources.size());

    if (imageCount > 0)
    {
      gpuData.texOffsetAllocation = m_textureHeap.allocateTextures(imageCount);
    }
    if (!resources.empty())
    {
      gpuData.textureInfoOffsetAllocation = m_textureHeap.allocateTextureInfos(uint32_t(resources.size()));
    }

    uint32_t imageOffset = 0;
    for (uint32_t r = 0; r < resources.size(); r++)
    {
      const GiTextureResource& resource = resources[r];

      rp::TextureInfo& texInfo = textureInfos[r];
      uint32_t imageIndex = gpuData.texOffsetAllocation.offset + imageOffset;

      if (resource.uvtileWidth > 0)
      {
        gpuData.useTextureInfoLut = true;
      }

      assert(imageIndex <= rp::TEXTURE_INFO_IMAGE_INDEX_MASK);
      assert(resource.uvtileMinU >= INT8_MIN && resource.uvtileMinU <= INT8_MAX);
      assert(resource.uvtileMinV >= INT8_MIN && resource.uvtileMinV <= INT8_MAX);
      assert(resource.uvtileWidth <= rp::TEXTURE_INFO_UVTILE_SIZE_MASK);
      assert(resource.uvtileHeight <= rp::TEXTURE_INFO_UVTILE_SIZE_MASK);

      texInfo.imageIndexAndUvtileMin =
        imageIndex |
        (uint32_t(resource.uvtileMinU & 0xFF) << rp::TEXTURE_INFO_UVTILE_MIN_U_SHIFT) |
        (uint32_t(resource.uvtileMinV & 0xFF) << rp::TEXTURE_INFO_UVTILE_MIN_V_SHIFT);

      texInfo.uvtileSizeAndParamSlot =
        resource.uvtileWidth | (resource.uvtileHeight << rp::TEXTURE_INFO_UVTILE_HEIGHT_SHIFT) |
        (rp::TEXTURE_INFO_PARAMETER_SLOT_INVALID << rp::TEXTURE_INFO_PARAMETER_SLOT_SHIFT);

      uint32_t slotCount = resource.uvtileWidth > 0 ? resource.uvtileWidth * resource.uvtileHeight : 1;
      assert(resource.images.size() == slotCount);

      for (uint32_t s = 0; s < slotCount; s++)
      {
        gpuData.images.push_back(resource.images[s]);
      }

      imageOffset += slotCount;
    }

    for (const McTextureParameterBinding& binding : gpuData.textureParameterBindings)
    {
      uint32_t textureIndex = binding.textureIndex;
      if (textureIndex < 1 || textureIndex > resourceCount)
      {
        continue;
      }

      uint32_t slot = info.material->base->parameterNames.size() + textureIndex - 1;
      if (slot >= rp::TEXTURE_INFO_PARAMETER_SLOT_MASK)
      {
        continue;
      }

      uint32_t& value = textureInfos[textureIndex - 1].uvtileSizeAndParamSlot;
      value = (value & ~(rp::TEXTURE_INFO_PARAMETER_SLOT_MASK << rp::TEXTURE_INFO_PARAMETER_SLOT_SHIFT)) |
              (slot << rp::TEXTURE_INFO_PARAMETER_SLOT_SHIFT);
    }

    if (!resources.empty() &&
        !m_textureHeap.updateTextureInfos(gpuData.textureInfoOffsetAllocation, textureInfos))
    {
      GB_ERROR("failed to update texture infos");
      return false;
    }

    for (uint32_t j = 0; j < parameterResources.size(); j++)
    {
      const McTextureParameterBinding& binding = gpuData.textureParameterBindings[parameterBindingIndices[j]];
      if (binding.textureIndex < 1 || binding.textureIndex > resourceCount)
      {
        continue;
      }

      gpuData.textureParameterInfoIndices[parameterBindingIndices[j]] =
        gpuData.textureInfoOffsetAllocation.offset + resourceCount + j;
    }

    info.material->base->textureCount = resourceCount;

    return true;
  }

  void GiMaterialManager::destroyMaterialGpuData(GiMaterialGpuData& gpuData)
  {
    m_textureHeap.freeTextures(gpuData.texOffsetAllocation);
    m_textureHeap.freeTextureInfos(gpuData.textureInfoOffsetAllocation);
    gpuData = {};
  }
}
