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

namespace gtl
{
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
    ObjectId
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
    uint32_t  faceCount;
    GiFace*   faces;
    int       id;
    bool      isLeftHanded;
    uint32_t  vertexCount;
    GiVertex* vertices;
  };

  // TODO: rename
  struct GiShaderCacheParams
  {
    GiAovId  aovId;
    bool     depthOfField;
    bool     domeLightCameraVisible;
    bool     filterImportanceSampling;
    bool     nextEventEstimation;
    bool     progressiveAccumulation;
    uint32_t mediumStackSize;
    GiScene* scene;
  };

  struct GiRenderParams
  {
    const GiShaderCacheParams* shaderCacheParams;
    GiCameraDesc    camera;
    GiRenderBuffer* renderBuffer;
    float           lightIntensityMultiplier;
    uint32_t        maxBounces;
    uint32_t        spp;
    uint32_t        rrBounceOffset;
    float           rrInvMinTermProb;
    float           maxSampleValue;
    uint32_t        maxVolumeWalkLength;
    float           backgroundColor[4];
    GiDomeLight*    domeLight;
    GiScene*        scene;
  };

  struct GiInitParams
  {
    std::string_view shaderPath;
    std::string_view mdlRuntimePath;
    const std::vector<std::string>& mdlSearchPaths;
    const std::shared_ptr<void/*MaterialX::Document*/> mtlxStdLib;
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

  GiStatus giInitialize(const GiInitParams& params);
  void giTerminate();

  void giRegisterAssetReader(GiAssetReader* reader);

  GiMaterial* giCreateMaterialFromMtlxStr(const char* str);
  GiMaterial* giCreateMaterialFromMtlxDoc(const std::shared_ptr<void/*MaterialX::Document*/> doc);
  GiMaterial* giCreateMaterialFromMdlFile(const char* filePath, const char* subIdentifier);
  void giDestroyMaterial(GiMaterial* mat);

  GiMesh* giCreateMesh(GiScene* scene, const GiMeshDesc& desc);
  void giSetMeshTransform(GiMesh* mesh, float transform[3][4]);
  void giSetMeshInstanceTransforms(GiMesh* mesh, uint32_t count, const float (*transforms)[4][4]);
  void giSetMeshMaterial(GiMesh* mesh, const GiMaterial* mat);
  void giSetMeshVisibility(GiMesh* mesh, bool visible);
  void giDestroyMesh(GiMesh* mesh);

  void giInvalidateFramebuffer();

  GiStatus giRender(const GiRenderParams& params, float* rgbaImg);

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
  void giDestroyDomeLight(GiScene* scene, GiDomeLight* light);
  void giSetDomeLightRotation(GiDomeLight* light, float* quat);
  void giSetDomeLightBaseEmission(GiDomeLight* light, float* rgb);
  void giSetDomeLightDiffuseSpecular(GiDomeLight* light, float diffuse, float specular);

  GiRenderBuffer* giCreateRenderBuffer(uint32_t width, uint32_t height);
  void giDestroyRenderBuffer(GiRenderBuffer* renderBuffer);
}
