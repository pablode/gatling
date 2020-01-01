#include "bvh.h"
#include "math.h"

#include <stdbool.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <pthread.h>

typedef struct face_ref {
  gp_aabb  aabb;
  uint32_t index;
} face_ref;

typedef struct bvh_range {
  uint32_t start;
  uint32_t count;
  gp_aabb  aabb;
  bool     is_leaf;
  uint32_t node_index;
} bvh_range;

typedef struct split_cand {
  float    sah_cost;
  uint32_t dim;
  uint32_t left_tri_count;
  uint32_t right_tri_count;
  gp_aabb  left_aabb;
  gp_aabb  right_aabb;
} split_cand;

typedef struct stack_item {
  bvh_range range;
  uint32_t  node_index;
} stack_item;

typedef struct thread_data {
  const gp_bvh_build_params* params;
  face_ref*                  face_refs;
  gp_aabb*                   reused_bounds;
} thread_data;

GP_INLINE int gp_bvh_sort_comp_func(
  const void* a, const void* b, uint32_t dim)
{
  const face_ref* ref_1 = (face_ref *) a;
  const face_ref* ref_2 = (face_ref *) b;
  const float aabb_length_1 = ref_1->aabb.min[dim] + ref_1->aabb.max[dim];
  const float aabb_length_2 = ref_2->aabb.min[dim] + ref_2->aabb.max[dim];
  const float aabb_center_diff = aabb_length_2 - aabb_length_1;

  if      (aabb_center_diff > 0.0f)     { return  1; }
  else if (aabb_center_diff < 0.0f)     { return -1; }
  else if (ref_1->index > ref_2->index) { return  1; }
  else if (ref_1->index < ref_2->index) { return -1; }
  else                                  { return  0; }
}

static int gp_bvh_sort_comp_func_x(const void* a, const void* b) {
  return gp_bvh_sort_comp_func(a, b, 0);
}
static int gp_bvh_sort_comp_func_y(const void* a, const void* b) {
  return gp_bvh_sort_comp_func(a, b, 1);
}
static int gp_bvh_sort_comp_func_z(const void* a, const void* b) {
  return gp_bvh_sort_comp_func(a, b, 2);
}

static void gp_bvh_sort_references(face_ref* face,
  uint32_t tri_count,
  uint32_t dim)
{
  assert(dim < 3);
  switch (dim) {
    case 0:
      qsort(face, tri_count, sizeof(face_ref), gp_bvh_sort_comp_func_x);
      break;
    case 1:
      qsort(face, tri_count, sizeof(face_ref), gp_bvh_sort_comp_func_y);
      break;
    case 2:
      qsort(face, tri_count, sizeof(face_ref), gp_bvh_sort_comp_func_z);
      break;
  }
}

GP_INLINE float gp_calc_tri_intersection_cost(
  float    base_cost,
  uint32_t batch_size,
  uint32_t num_tris)
{
  assert(num_tris > 0);
  const uint32_t rounded_to_batch_size =
    ((num_tris - 1) / batch_size) * batch_size;
  return rounded_to_batch_size * base_cost;
}

GP_INLINE float gp_calc_node_traversal_cost(
  float    base_cost,
  uint32_t batch_size,
  uint32_t num_nodes)
{
  assert(num_nodes > 0);
  const uint32_t rounded_to_batch_size =
    ((num_nodes - 1) / batch_size) * batch_size;
  return rounded_to_batch_size * base_cost;
}

static void gp_bvh_find_split(
  thread_data* thread_data,
  uint32_t     tri_offset,
  uint32_t     tri_count,
  split_cand*  split_cand)
{
  face_ref* face_refs = thread_data->face_refs;
  face_ref* base_ref  = &face_refs[tri_offset];

  float best_sah_cost  = INFINITY;
  float best_tie_break = INFINITY;
  gp_aabb right_aabb;
  gp_aabb left_aabb;

  /* Test each axis and sort triangles along it. */
  for (uint32_t dim = 0; dim < 3; ++dim)
  {
    gp_bvh_sort_references(face_refs + tri_offset, tri_count, dim);

    /* Sweep from right to left. */
    gp_aabb_make_smallest(&right_aabb);

    for (uint32_t r = tri_count; r > 0; --r)
    {
      const face_ref* ref = &base_ref[r];
      gp_aabb_merge(&right_aabb, &ref->aabb, &right_aabb);
      thread_data->reused_bounds[r - 1] = right_aabb;
    }

    /* Sweep from left to right. */
    gp_aabb_make_smallest(&left_aabb);

    for (uint32_t l = 1; l < tri_count; ++l)
    {
      const face_ref* ref = &base_ref[l - 1];
      gp_aabb_merge(&left_aabb, &ref->aabb, &left_aabb);

      /* Calculate SAH cost. */
      const uint32_t r = tri_count - l;
      const float area_l = gp_aabb_half_area(&left_aabb);
      const float area_r =
        gp_aabb_half_area(&thread_data->reused_bounds[l - 1]);
      const float sah_cost =
        gp_calc_tri_intersection_cost(
          thread_data->params->tri_intersection_cost,
          thread_data->params->tri_batch_size,
          l
        ) * area_l +
        gp_calc_tri_intersection_cost(
          thread_data->params->tri_intersection_cost,
          thread_data->params->tri_batch_size,
          r
        ) * area_r;

      /* When SAH is equal, prefer split which is more centered. */
      const float tie_break = sqrt((float)l) + sqrt((float)r);

      if (sah_cost < best_sah_cost ||
           (sah_cost == best_sah_cost && tie_break < best_tie_break))
      {
        /* Set new best split candidate. */
        split_cand->sah_cost = sah_cost;
        split_cand->dim = dim;
        split_cand->left_tri_count = l;
        split_cand->left_aabb = left_aabb;
        split_cand->right_tri_count = r;
        split_cand->right_aabb = thread_data->reused_bounds[l - 1];

        best_sah_cost = sah_cost;
        best_tie_break = tie_break;
      }
    }
  }
}

static void gp_bvh_build_range(
  thread_data* data,
  const bvh_range* range,
  bvh_range* range_left,
  bvh_range* range_right)
{
  /* Find best split candidate. */
  split_cand split;
  gp_bvh_find_split(
    data,
    range->start,
    range->count,
    &split
  );

  /* Sort triangles again in best split dimension. */
  gp_bvh_sort_references(
    data->face_refs + range->start,
    range->count,
    split.dim
  );

  /* Test if childs should be leaves. */
  const float left_leaf_sah_cost =
    gp_calc_tri_intersection_cost(
      data->params->tri_intersection_cost,
      data->params->tri_batch_size,
      split.left_tri_count
    )
    * gp_aabb_half_area(&split.left_aabb);

  const float right_leaf_sah_cost =
    gp_calc_tri_intersection_cost(
      data->params->tri_intersection_cost,
      data->params->tri_batch_size,
      split.right_tri_count
    )
    * gp_aabb_half_area(&split.right_aabb);

  const bool is_left_leaf =
    (split.left_tri_count <= data->params->min_leaf_size) ||
    (split.left_tri_count <= data->params->max_leaf_size &&
     split.sah_cost < left_leaf_sah_cost);

  const bool is_right_leaf =
    (split.right_tri_count <= data->params->min_leaf_size) ||
    (split.right_tri_count <= data->params->max_leaf_size &&
     split.sah_cost < right_leaf_sah_cost);

  /* Set new child ranges. */
  range_left->start    = range->start;
  range_left->count    = split.left_tri_count;
  range_left->aabb     = split.left_aabb;
  range_left->is_leaf  = is_left_leaf;
  range_right->start   = range->start + split.left_tri_count;
  range_right->count   = range->count - split.left_tri_count;
  range_right->aabb    = split.right_aabb;
  range_right->is_leaf = is_right_leaf;
}

GpResult gp_bvh_build(
  const gp_bvh_build_params* params,
  gp_bvh* bvh)
{
  /* Initialize the node array. We allocate memory for the worst case
     and reallocate later when we know the precise number of nodes. */

  const gp_vertex* vertices = params->vertices;
  const gp_face*      faces = params->faces;
  uint32_t vertex_count = params->vertex_count;
  uint32_t face_count   = params->face_count;

  bvh->node_count = 0;
  bvh->nodes = malloc(2 * face_count * sizeof(gp_bvh_node));

  /* Initialize triangle references, their aabbs and the root aabb. */

  gp_aabb root_aabb;
  gp_aabb_make_smallest(&root_aabb);

  face_ref *face_refs =
    (face_ref *) malloc(face_count * sizeof(face_ref));

  for (uint32_t i = 0; i < face_count; ++i)
  {
    const gp_face* face  = &faces[i];
    const gp_vertex* v_a = &vertices[face->v_i[0]];
    const gp_vertex* v_b = &vertices[face->v_i[1]];
    const gp_vertex* v_c = &vertices[face->v_i[2]];

    face_ref *face_ref = &face_refs[i];
    face_ref->index = i;

    gp_aabb_make_from_triangle(
      v_a->pos,
      v_b->pos,
      v_c->pos,
      &face_ref->aabb
    );
    gp_aabb_merge(
      &root_aabb,
      &face_ref->aabb,
      &root_aabb
    );
  }

  bvh->aabb = root_aabb;

  /* Build BVH using ranges in an iterative fashion. */

  const uint32_t max_item_count = face_count * 2;
  stack_item* items =
    (stack_item*) malloc(max_item_count * sizeof(stack_item));

  items[0].range.start   = 0;
  items[0].range.count   = face_count;
  items[0].range.aabb    = root_aabb;
  items[0].range.is_leaf = false;
  items[0].node_index    = 0;

  /* We want FIFO behaviour to have the BVH level-wise in memory. */
  uint32_t item_read_index  = 0;
  uint32_t item_write_index = 1;

  gp_aabb* reused_bounds =
    (gp_aabb*) malloc(face_count * sizeof(gp_aabb));

  uint32_t node_index = 0;

  thread_data data = {
    .params = params,
    .face_refs = face_refs,
    .reused_bounds = reused_bounds
  };

  while (item_read_index != item_write_index)
  {
    /* Dequeue range and split it. */
    const stack_item item = items[item_read_index];
    ++item_read_index;

    stack_item item_left;
    stack_item item_right;

    gp_bvh_build_range(
      &data,
      &item.range,
      &item_left.range,
      &item_right.range
    );

    /* Make node and enqueue subranges. */
    gp_bvh_node* node = &bvh->nodes[item.node_index];
    node->left_aabb   = item_left.range.aabb;
    node->right_aabb  = item_right.range.aabb;

    if (item_left.range.is_leaf) {
      node->left_child_index = item_left.range.start;
      node->left_child_count = item_left.range.count;
      node->left_child_count |= (1u << 31u);
    } else {
      node_index++;
      node->left_child_index = node_index;
      node->left_child_count = 2;
      item_left.node_index = node_index;
      items[item_write_index] = item_left;
      ++item_write_index;
    }

    if (item_right.range.is_leaf) {
      node->right_child_index = item_right.range.start;
      node->right_child_count = item_right.range.count;
      node->right_child_count |= (1u << 31u);
    } else {
      node_index++;
      node->right_child_index = node_index;
      node->right_child_count = 2;
      item_right.node_index = node_index;
      items[item_write_index] = item_right;
      ++item_write_index;
    }
  }

  free(reused_bounds);
  free(items);

  /* Reallocate node memory, copy vertices and faces. */

  bvh->node_count = node_index + 1;
  bvh->nodes = (gp_bvh_node*) realloc(bvh->nodes, bvh->node_count * sizeof(gp_bvh_node));

  bvh->vertex_count = vertex_count;
  bvh->vertices = malloc(vertex_count * sizeof(gp_vertex));
  memcpy(bvh->vertices, params->vertices, vertex_count * sizeof(gp_vertex));

  bvh->face_count = face_count;
  bvh->faces = malloc(face_count * sizeof(gp_face));

  for (uint32_t i = 0; i < face_count; ++i)
  {
    const uint32_t face_index = face_refs[i].index;
    bvh->faces[i] = params->faces[face_index];
  }

  /* Free leftovers and we're finished! */

  free(face_refs);

  return GP_OK;
}

void gp_free_bvh(gp_bvh* bvh)
{
  free(bvh->nodes);
  free(bvh->faces);
  free(bvh->vertices);
  bvh->node_count = 0;
  bvh->nodes = NULL;
  bvh->face_count = 0;
  bvh->faces = NULL;
  bvh->vertex_count = 0;
  bvh->vertices = NULL;
}
