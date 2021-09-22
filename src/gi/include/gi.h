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

struct gi_material
{
  float albedo[3];
  float padding1;
  float emission[3];
  float padding2;
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

struct gi_scene_cache;

struct gi_preprocess_params
{
  uint32_t            face_count;
  struct gi_face*     faces;
  struct gi_material* materials;
  uint32_t            material_count;
  uint32_t            vertex_count;
  struct gi_vertex*   vertices;
};

struct gi_render_params
{
  struct gi_scene_cache*  scene_cache;
  const struct gi_camera* camera;
  uint32_t                image_width;
  uint32_t                image_height;
  uint32_t                max_bounces;
  uint32_t                rr_bounce_offset;
  float                   rr_inv_min_term_prob;
  uint32_t                spp;
};

int giInitialize(const char* resource_path);

void giTerminate();

int giCreateSceneCache(struct gi_scene_cache** cache);

void giDestroySceneCache(struct gi_scene_cache* cache);

int giPreprocess(const struct gi_preprocess_params* params,
                 struct gi_scene_cache* scene_cache);

int giRender(const struct gi_render_params* params,
             float* rgba_img);

#ifdef __cplusplus
}
#endif

#endif
