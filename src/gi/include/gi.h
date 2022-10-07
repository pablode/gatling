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

#ifndef GI_H
#define GI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GI_OK 0
#define GI_ERROR 1

enum gi_aov_id
{
  GI_AOV_ID_COLOR              = 0,
  GI_AOV_ID_NORMAL             = 1,
  GI_AOV_ID_DEBUG_NEE          = 2,
  GI_AOV_ID_DEBUG_BARYCENTRICS = 3,
  GI_AOV_ID_DEBUG_TEXCOORDS    = 4,
  GI_AOV_ID_DEBUG_BOUNCES      = 5,
  GI_AOV_ID_DEBUG_CLOCK_CYCLES = 6
};

struct gi_camera
{
  float position[3];
  float forward[3];
  float up[3];
  float vfov;
};

struct gi_vertex
{
  float pos[3];
  float u;
  float norm[3];
  float v;
};

struct gi_face
{
  uint32_t v_i[3];
  uint32_t mat_index;
};

struct gi_material;

struct gi_geom_cache;
struct gi_shader_cache;

struct gi_geom_cache_params
{
  bool                       next_event_estimation;
  uint32_t                   face_count;
  struct gi_face*            faces;
  uint32_t                   material_count;
  const struct gi_material** materials;
  uint32_t                   vertex_count;
  struct gi_vertex*          vertices;
};

struct gi_shader_cache_params
{
  enum gi_aov_id        aov_id;
  struct gi_geom_cache* geom_cache;
};

struct gi_render_params
{
  const struct gi_camera*       camera;
  const struct gi_geom_cache*   geom_cache;
  const struct gi_shader_cache* shader_cache;
  uint32_t                      image_width;
  uint32_t                      image_height;
  uint32_t                      max_bounces;
  uint32_t                      spp;
  uint32_t                      rr_bounce_offset;
  float                         rr_inv_min_term_prob;
  float                         max_sample_value;
  float                         bg_color[4];
};

struct gi_init_params
{
  const char* resource_path;
  const char* shader_path;
  const char* mtlx_lib_path;
  const char* mdl_lib_path;
};

int giInitialize(const struct gi_init_params* params);
void giTerminate();

struct gi_material* giCreateMaterialFromMtlx(const char* doc_str);
struct gi_material* giCreateMaterialFromMdlFile(const char* file_path, const char* sub_identifier);
void giDestroyMaterial(struct gi_material* mat);

struct gi_geom_cache* giCreateGeomCache(const struct gi_geom_cache_params* params);
void giDestroyGeomCache(struct gi_geom_cache* cache);

struct gi_shader_cache* giCreateShaderCache(const struct gi_shader_cache_params* params);
void giDestroyShaderCache(struct gi_shader_cache* cache);

void giInvalidateFramebuffer();

int giRender(const struct gi_render_params* params,
             float* rgba_img);

#ifdef __cplusplus
}
#endif

#endif
