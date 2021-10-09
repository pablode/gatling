#ifndef GI_H
#define GI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GI_OK 0
#define GI_ERROR 1

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

struct gi_geom_cache;
struct gi_shader_cache;

struct gi_geom_cache_params
{
  uint32_t                   face_count;
  struct gi_face*            faces;
  uint32_t                   material_count;
  const struct gi_material** materials;
  uint32_t                   vertex_count;
  struct gi_vertex*          vertices;
};

struct gi_shader_cache_params
{
  const struct gi_geom_cache* geom_cache;
  uint32_t                    max_bounces;
  uint32_t                    spp;
  uint32_t                    rr_bounce_offset;
  float                       rr_inv_min_term_prob;
};

struct gi_render_params
{
  const struct gi_camera*       camera;
  const struct gi_geom_cache*   geom_cache;
  uint32_t                      image_width;
  uint32_t                      image_height;
  const struct gi_shader_cache* shader_cache;
};

int giInitialize(const char* resource_path,
                 const char* shader_path,
                 const char* mtlxlib_path,
                 const char* mtlxmdl_path);

void giTerminate();

struct gi_material* giCreateMaterialFromMtlx(const char* doc);
void giDestroyMaterial(struct gi_material* mat);

struct gi_geom_cache* giCreateGeomCache(const struct gi_geom_cache_params* params);
void giDestroyGeomCache(struct gi_geom_cache* cache);

struct gi_shader_cache* giCreateShaderCache(const struct gi_shader_cache_params* params);
void giDestroyShaderCache(struct gi_shader_cache* cache);

int giRender(const struct gi_render_params* params,
             float* rgba_img);

#ifdef __cplusplus
}
#endif

#endif
