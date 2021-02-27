#include "gi.h"

#include "bvh.h"
#include "bvh_collapse.h"
#include "bvh_compress.h"

#include <stdlib.h>
#include <string.h>

struct gi_scene_cache
{
  gp_bvhcc             bvhcc;
  struct gi_camera     camera;
  uint32_t             face_count;
  struct gi_face*      faces;
  struct gp_material * materials;
  uint32_t             material_count;
  uint32_t             vertex_count;
  struct gi_vertex*    vertices;
};

int giInitialize()
{
}

void giTerminate()
{
}

int giCreateSceneCache(struct gi_scene_cache** cache)
{
  *cache = malloc(sizeof(struct gi_scene_cache));

  return (*cache == NULL) ? GI_ERROR : GI_OK;
}

void giDestroySceneCache(struct gi_scene_cache* cache)
{
  gp_free_bvhcc(&cache->bvhcc);
  free(cache->materials);
  free(cache->vertices);
  free(cache->faces);
  free(cache);
}

int giPreprocess(const struct gi_preprocess_params* params,
                 struct gi_scene_cache* scene_cache)
{
  /* We don't support too few faces since this would lead to the root node
   * being a leaf, requiring special handling in the traversal algorithm. */
  if (params->face_count <= 3)
  {
    return GI_ERROR;
  }

  /* Build BVH. */
  gp_bvh bvh;
  const gp_bvh_build_params bvh_params = {
    .face_batch_size          = 1,
    .face_count               = params->face_count,
    .face_intersection_cost   = 1.2f,
    .faces                    = params->faces,
    .leaf_max_face_count      = 1,
    .object_binning_mode      = GP_BVH_BINNING_MODE_FIXED,
    .object_binning_threshold = 1024,
    .object_bin_count         = 16,
    .spatial_bin_count        = 32,
    .spatial_reserve_factor   = 1.25f,
    .spatial_split_alpha      = 10e-5f,
    .vertex_count             = params->vertex_count,
    .vertices                 = params->vertices
  };

  gp_bvh_build(&bvh_params, &bvh);

  gp_bvhc bvhc;
  gp_bvh_collapse_params cparams  = {
    .bvh                    = &bvh,
    .max_leaf_size          = 3,
    .node_traversal_cost    = 1.0f,
    .face_intersection_cost = 0.3f
  };

  gp_bvh_collapse(&cparams, &bvhc);
  gp_free_bvh(&bvh);

  /* Copy vertices, materials and new faces. */
  scene_cache->face_count = bvhc.face_count;
  scene_cache->faces = malloc(scene_cache->face_count * sizeof(struct gi_face));
  memcpy(scene_cache->faces, bvhc.faces, bvhc.face_count * sizeof(struct gi_face));

  scene_cache->vertex_count = params->vertex_count;
  scene_cache->vertices = malloc(scene_cache->vertex_count * sizeof(struct gi_vertex));
  memcpy(scene_cache->vertices, params->vertices, params->vertex_count * sizeof(struct gi_vertex));

  scene_cache->material_count = params->material_count;
  scene_cache->materials = malloc(scene_cache->material_count * sizeof(struct gi_material));
  memcpy(scene_cache->materials, params->materials, params->material_count * sizeof(struct gi_material));

  gp_bvh_compress(&bvhc, &scene_cache->bvhcc);
  gp_free_bvhc(&bvhc);

  return GI_OK;
}

int giRender(const struct gi_render_params* params,
             float* rgba_img)
{
}
