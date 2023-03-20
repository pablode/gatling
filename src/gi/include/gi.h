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

enum GiStatus
{
  GI_OK = 0,
  GI_ERROR
};

enum GiAovId
{
  GI_AOV_ID_COLOR              = 0,
  GI_AOV_ID_NORMAL             = 1,
  GI_AOV_ID_DEBUG_NEE          = 2,
  GI_AOV_ID_DEBUG_BARYCENTRICS = 3,
  GI_AOV_ID_DEBUG_TEXCOORDS    = 4,
  GI_AOV_ID_DEBUG_BOUNCES      = 5,
  GI_AOV_ID_DEBUG_CLOCK_CYCLES = 6,
  GI_AOV_ID_DEBUG_OPACITY      = 7
};

struct GiGeomCache;
struct GiMaterial;
struct GiMesh;
struct GiMeshInstance;
struct GiShaderCache;
struct GiScene;
struct GiSphereLight;

struct GiCameraDesc
{
  float position[3];
  float forward[3];
  float up[3];
  float vfov;
};

struct GiVertex
{
  float pos[3];
  float u;
  float norm[3];
  float v;
};

struct GiFace
{
  uint32_t v_i[3];
};

struct GiMeshDesc
{
  uint32_t          face_count;
  GiFace*           faces;
  const GiMaterial* material;
  uint32_t          vertex_count;
  GiVertex*         vertices;
};

struct GiMeshInstance
{
  const GiMesh* mesh;
  float transform[3][4];
};

struct GiShaderCacheParams
{
  GiAovId            aov_id;
  uint32_t           material_count;
  const GiMaterial** materials;
};

struct GiGeomCacheParams
{
  uint32_t              mesh_instance_count;
  const GiMeshInstance* mesh_instances;
  GiShaderCache*        shader_cache;
};

struct GiRenderParams
{
  const GiCameraDesc*  camera;
  const GiGeomCache*   geom_cache;
  const GiShaderCache* shader_cache;
  uint32_t             image_width;
  uint32_t             image_height;
  uint32_t             max_bounces;
  uint32_t             spp;
  uint32_t             rr_bounce_offset;
  float                rr_inv_min_term_prob;
  float                max_sample_value;
  float                bg_color[4];
};

struct GiInitParams
{
  const char* resource_path;
  const char* shader_path;
  const char* mtlx_lib_path;
  const char* mdl_lib_path;
};

GiStatus giInitialize(const GiInitParams* params);
void giTerminate();

struct GiAsset;
class GiAssetReader
{
public:
  virtual GiAsset* open(const char* path) = 0;
  virtual size_t size(const GiAsset* asset) const = 0;
  virtual void* data(const GiAsset* asset) const = 0;
  virtual void close(GiAsset* asset) = 0;
  virtual ~GiAssetReader() = default;
};
void giRegisterAssetReader(GiAssetReader* reader);

GiMaterial* giCreateMaterialFromMtlx(const char* doc_str);
GiMaterial* giCreateMaterialFromMdlFile(const char* file_path, const char* sub_identifier);
void giDestroyMaterial(GiMaterial* mat);

GiMesh* giCreateMesh(const GiMeshDesc* desc);

GiGeomCache* giCreateGeomCache(const GiGeomCacheParams* params);
void giDestroyGeomCache(GiGeomCache* cache);

GiShaderCache* giCreateShaderCache(const GiShaderCacheParams* params);
void giDestroyShaderCache(GiShaderCache* cache);

void giInvalidateFramebuffer();

int giRender(const GiRenderParams* params, float* rgba_img);

GiScene* giCreateScene();
void giDestroyScene(GiScene* scene);

GiSphereLight* giCreateSphereLight(GiScene* scene);
void giDestroySphereLight(GiScene* scene, GiSphereLight* light);
void giSetSphereLightTransform(GiSphereLight* light, float* transform3x4);
