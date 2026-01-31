//
// Copyright (C) 2019-2022 Pablo Delgado Krämer
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
#include <stddef.h>
#include <memory>
#include <string>
#include <vector>
#include <variant>
#include <unordered_map>

#include <gtl/gb/ParamTypes.h>

namespace gtl
{
  constexpr static const uint32_t GI_MAX_AOV_COMP_SIZE = 16; // vec4

  enum class GiStatus { Ok, Error };

  enum class GiAovId
  {
    Color = 0,
    Normal,
    NEE,
    Barycentrics,
    Texcoords,
    Bounces,
    ClockCycles,
    Opacity,
    Tangents,
    Bitangents,
    ThinWalled,
    ObjectId,
    Depth,
    FaceId,
    InstanceId,
    DoubleSided,
    Albedo,
    COUNT
  };

  struct GiAsset;
  struct GiMaterial;
  struct GiMesh;
  struct GiShaderCache;
  struct GiScene;
  struct GiSphereLight;
  struct GiDistantLight;
  struct GiRectLight;
  struct GiDiskLight;
  struct GiDomeLight;
  struct GiRenderBuffer;

  enum class GiRenderBufferFormat
  {
    Int32,
    Float32,
    Float32Vec4
  };

  enum class GiPrimvarType
  {
    Float, Vec2, Vec3, Vec4, Int, Int2, Int3, Int4
  };

  enum class GiPrimvarInterpolation
  {
    Constant, Instance, Uniform, Vertex, COUNT
  };

  struct GiPrimvarData
  {
    std::string name;
    GiPrimvarType type;
    GiPrimvarInterpolation interpolation;
    std::vector<uint8_t> data;
  };

  struct GiCameraDesc
  {
    float position[3];
    float forward[3];
    float up[3];
    float vfov;
    float fStop;
    float focusDistance;
    float focalLength;
    float clipStart;
    float clipEnd;
    float exposure;
  };

  struct GiVertex
  {
    float pos[3];
    float u;
    float norm[3];
    float v;
    float tangent[3];
    float bitangentSign;
  };

  struct GiFace
  {
    uint32_t v_i[3];
  };

  struct GiMeshDesc
  {
    uint32_t                          faceCount;
    const std::vector<GiFace>&        faces;
    const std::vector<int>&           faceIds;
    int                               id;
    bool                              isDoubleSided;
    bool                              isLeftHanded;
    const char*                       name;
    uint32_t                          maxFaceId;
    const std::vector<GiPrimvarData>& primvars;
    uint32_t                          vertexCount;
    const std::vector<GiVertex>&      vertices;
  };

  struct GiRenderSettings
  {
    bool     clippingPlanes;
    bool     depthOfField;
    bool     domeLightCameraVisible;
    bool     filterImportanceSampling;
    bool     jitteredSampling;
    float    lightIntensityMultiplier;
    uint32_t maxBounces;
    float    maxSampleValue;
    uint32_t maxVolumeWalkLength;
    uint32_t mediumStackSize;
    float    metersPerSceneUnit;
    bool     nextEventEstimation;
    bool     progressiveAccumulation;
    uint32_t rrBounceOffset;
    float    rrInvMinTermProb;
    uint32_t spp;
  };

  struct GiAovBinding
  {
    GiAovId         aovId;
    uint8_t         clearValue[GI_MAX_AOV_COMP_SIZE];
    GiRenderBuffer* renderBuffer;
  };

  struct GiRenderParams
  {
    std::vector<GiAovBinding> aovBindings;
    GiCameraDesc              camera;
    GiDomeLight*              domeLight;
    GiRenderSettings          renderSettings;
    GiScene*                  scene;
  };

  struct GiInitParams
  {
    std::string_view shaderPath;
    std::string_view mdlRuntimePath;
    const std::vector<std::string>& mdlSearchPaths;
    const std::shared_ptr<void/*MaterialX::Document*/> mtlxStdLib;
    std::string mtlxCustomNodesPath;
  };

  class GiAssetReader
  {
  public:
    virtual GiAsset* open(const char* path) = 0;
    virtual size_t size(const GiAsset* asset) const = 0;
    virtual void* data(const GiAsset* asset) const = 0;
    virtual void close(GiAsset* asset) = 0;
    virtual ~GiAssetReader() = default;
  };

  using GiMaterialParameterValue = std::variant<bool, int, float, GbVec2f, GbVec3f, GbVec4f, GbColor, GbTextureAsset>;
  using GiMaterialParameters = std::unordered_map<std::string, GiMaterialParameterValue>;

  GiStatus giInitialize(const GiInitParams& params);
  void giTerminate();

  void giRegisterAssetReader(GiAssetReader* reader);

  GiMaterial* giCreateMaterialFromMtlxStr(GiScene* scene, const char* name, const char* mtlxSrc);
  GiMaterial* giCreateMaterialFromMtlxDoc(GiScene* scene, const char* name, const std::shared_ptr<void/*MaterialX::Document*/> doc);
  GiMaterial* giCreateMaterialFromMdlFile(GiScene* scene, const char* name, const char* filePath, const char* subIdentifier, const GiMaterialParameters& params = {});
  void giDestroyMaterial(GiMaterial* mat);

  GiMesh* giCreateMesh(GiScene* scene, const GiMeshDesc& desc);
  void giSetMeshTransform(GiMesh* mesh, const float* mat4x4);
  void giSetMeshInstanceTransforms(GiMesh* mesh, uint32_t count, const float(*transforms)[4][4]);
  void giSetMeshInstancerPrimvars(GiMesh* mesh, const std::vector<GiPrimvarData>& instancerPrimvars);
  void giSetMeshInstanceIds(GiMesh* mesh, uint32_t count, int* ids);
  void giSetMeshMaterial(GiMesh* mesh, GiMaterial* mat);
  void giSetMeshVisibility(GiMesh* mesh, bool visible);
  void giDestroyMesh(GiMesh* mesh);

  GiStatus giRender(const GiRenderParams& params);

  GiScene* giCreateScene();
  void giDestroyScene(GiScene* scene);

  GiSphereLight* giCreateSphereLight(GiScene* scene);
  void giDestroySphereLight(GiScene* scene, GiSphereLight* light);
  void giSetSphereLightPosition(GiSphereLight* light, float* position);
  void giSetSphereLightBaseEmission(GiSphereLight* light, float* rgb);
  void giSetSphereLightRadius(GiSphereLight* light, float radiusX, float radiusY, float radiusZ);
  void giSetSphereLightDiffuseSpecular(GiSphereLight* light, float diffuse, float specular);

  GiDistantLight* giCreateDistantLight(GiScene* scene);
  void giDestroyDistantLight(GiScene* scene, GiDistantLight* light);
  void giSetDistantLightDirection(GiDistantLight* light, float* direction);
  void giSetDistantLightBaseEmission(GiDistantLight* light, float* rgb);
  void giSetDistantLightAngle(GiDistantLight* light, float angle);
  void giSetDistantLightDiffuseSpecular(GiDistantLight* light, float diffuse, float specular);

  GiRectLight* giCreateRectLight(GiScene* scene);
  void giDestroyRectLight(GiScene* scene, GiRectLight* light);
  void giSetRectLightOrigin(GiRectLight* light, float* origin);
  void giSetRectLightTangents(GiRectLight* light, float* t0, float* t1);
  void giSetRectLightBaseEmission(GiRectLight* light, float* rgb);
  void giSetRectLightDimensions(GiRectLight* light, float width, float height);
  void giSetRectLightDiffuseSpecular(GiRectLight* light, float diffuse, float specular);

  GiDiskLight* giCreateDiskLight(GiScene* scene);
  void giDestroyDiskLight(GiScene* scene, GiDiskLight* light);
  void giSetDiskLightOrigin(GiDiskLight* light, float* origin);
  void giSetDiskLightTangents(GiDiskLight* light, float* t0, float* t1);
  void giSetDiskLightBaseEmission(GiDiskLight* light, float* rgb);
  void giSetDiskLightRadius(GiDiskLight* light, float radiusX, float radiusY);
  void giSetDiskLightDiffuseSpecular(GiDiskLight* light, float diffuse, float specular);

  GiDomeLight* giCreateDomeLight(GiScene* scene, const char* filePath);
  void giDestroyDomeLight(GiDomeLight* light);
  void giSetDomeLightRotation(GiDomeLight* light, float* quat);
  void giSetDomeLightBaseEmission(GiDomeLight* light, float* rgb);
  void giSetDomeLightDiffuseSpecular(GiDomeLight* light, float diffuse, float specular);

  GiRenderBuffer* giCreateRenderBuffer(uint32_t width, uint32_t height, GiRenderBufferFormat format);
  void giDestroyRenderBuffer(GiRenderBuffer* renderBuffer);
  void* giGetRenderBufferMem(GiRenderBuffer* renderBuffer);
}
