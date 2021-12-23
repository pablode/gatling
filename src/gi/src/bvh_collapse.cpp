#include "bvh_collapse.h"

#include "gi.h"

#include <stdlib.h>
#include <math.h>
#include <assert.h>

/*
 * This file implements construction of an 8-wide BVH from a binary BVH as described
 * by Ylitie, Karras and Laine.
 * It works by first calculating SAH costs for representing the contents of each subtree
 * as a forest of at most i BVHs. By doing this bottom-up, previous results can be reused.
 * For each node and subtree count, we store the minimal cost in an N * (I-1) table, where
 * N is the number of nodes and I is the width of the BVH.
 * In a second pass, we traverse top-down and trace the decisions leading to the minimal
 * costs stored in the table. We inline DISTRIBUTE splits and combine leaf nodes. For each
 * INTERNAL split decision, we recurse further down.
 *
 * Literature:
 *   - Henri Ylitie, Tero Karras, and Samuli Laine. 2017.
 *     Efficient incoherent ray traversal on GPUs through compressed wide BVHs.
 *     In Proceedings of High Performance Graphics (HPG ’17).
 *     Association for Computing Machinery, New York, NY, USA, Article 4, 1–13.
 *     DOI: https://doi.org/10.1145/3105762.3105773
 */

enum GiBvhcSplitType
{
  GI_BVHC_SPLIT_TYPE_INVALID = 0,
  GI_BVHC_SPLIT_TYPE_INTERNAL = 1,
  GI_BVHC_SPLIT_TYPE_LEAF = 2,
  GI_BVHC_SPLIT_TYPE_DISTRIBUTE = 3
};

struct gi_bvhc_split
{
  GiBvhcSplitType split_type;
  int32_t left_count;
  int32_t right_count;
  float cost;
};

struct gi_bvhc_work_data
{
  gi_bvhc*              bvhc;
  const gi_bvhc_params* params;
  gi_bvhc_split*        splits;
};

static uint32_t gi_bvhc_count_child_faces(const gi_bvhc_work_data* wdata,
                                          uint32_t node_idx)
{
  const gi_bvh_node* node = &wdata->params->bvh->nodes[node_idx];

  if ((node->field2 & 0x80000000) == 0x80000000)
  {
    return (node->field2 & 0x7FFFFFFF);
  }

  return gi_bvhc_count_child_faces(wdata, node->field1) +
         gi_bvhc_count_child_faces(wdata, node->field2);
}

static gi_bvhc_split gi_bvhc_C(const gi_bvhc_work_data* wdata,
                               uint32_t n,
                               uint32_t i);

static gi_bvhc_split gi_bvhc_C_distribute(const gi_bvhc_work_data* wdata,
                                          uint32_t n,
                                          uint32_t j)
{
  const gi_bvh_node* node = &wdata->params->bvh->nodes[n];

  gi_bvhc_split split;
  split.split_type = GI_BVHC_SPLIT_TYPE_DISTRIBUTE;
  split.cost = INFINITY;

  for (uint32_t k = 0; k < j; ++k)
  {
    gi_bvhc_split split_left = gi_bvhc_C(wdata, node->field1, k);
    gi_bvhc_split split_right = gi_bvhc_C(wdata, node->field2, j - k - 1);
    float cost = split_left.cost + split_right.cost;

    if (cost < split.cost)
    {
      split.cost = cost;
      split.left_count = k;
      split.right_count = j - k - 1;
    }
  }

  return split;
}

static gi_bvhc_split gi_bvhc_C_internal(const gi_bvhc_work_data* wdata,
                                        uint32_t n)
{
  const gi_bvh_node* node = &wdata->params->bvh->nodes[n];
  float A_n = gml_aabb_area(&node->aabb);

  gi_bvhc_split split = gi_bvhc_C_distribute(wdata, n, 7);
  split.split_type = GI_BVHC_SPLIT_TYPE_INTERNAL;
  split.cost += A_n * wdata->params->node_traversal_cost;
  return split;
}

static gi_bvhc_split gi_bvhc_C_leaf(const gi_bvhc_work_data* wdata,
                                    uint32_t n)
{
  uint32_t p_n = gi_bvhc_count_child_faces(wdata, n);

  gi_bvhc_split split;
  split.split_type = GI_BVHC_SPLIT_TYPE_LEAF;

  if (p_n > wdata->params->max_leaf_size)
  {
    split.cost = INFINITY;
    return split;
  }

  const gi_bvh_node* node = &wdata->params->bvh->nodes[n];
  float A_n = gml_aabb_area(&node->aabb);
  split.cost = A_n * p_n * wdata->params->face_intersection_cost;

  return split;
}

static gi_bvhc_split gi_bvhc_C(const gi_bvhc_work_data* wdata,
                               uint32_t n,
                               uint32_t i)
{
  if (wdata->splits[n * 7 + i].split_type != GI_BVHC_SPLIT_TYPE_INVALID)
  {
    return wdata->splits[n * 7 + i];
  }

  if (i == 0)
  {
    gi_bvhc_split c_leaf = gi_bvhc_C_leaf(wdata, n);
    gi_bvhc_split c_internal = gi_bvhc_C_internal(wdata, n);
    return (c_leaf.cost < c_internal.cost) ? c_leaf : c_internal;
  }
  else
  {
    gi_bvhc_split c_dist = gi_bvhc_C_distribute(wdata, n, i);
    gi_bvhc_split c_recur = gi_bvhc_C(wdata, n, i - 1);
    return (c_dist.cost < c_recur.cost) ? c_dist : c_recur;
  }
}

static void gi_bvhc_calc_costs(const gi_bvhc_work_data* wdata,
                               uint32_t n)
{
  const gi_bvh_node* node = &wdata->params->bvh->nodes[n];

  if ((node->field2 & 0x80000000) == 0x80000000)
  {
    float A_n = gml_aabb_area(&node->aabb);
    uint32_t p_n = (node->field2 & 0x7FFFFFFF);
    float cost = A_n * p_n * wdata->params->face_intersection_cost;

    for (uint32_t i = 0; i < 7; ++i)
    {
      wdata->splits[n * 7 + i].split_type = GI_BVHC_SPLIT_TYPE_LEAF;
      wdata->splits[n * 7 + i].cost = cost;
    }
    return;
  }

  gi_bvhc_calc_costs(wdata, node->field1);
  gi_bvhc_calc_costs(wdata, node->field2);

  for (uint32_t i = 0; i < 7; ++i)
  {
    wdata->splits[n * 7 + i] = gi_bvhc_C(wdata, n, i);
  }
}

static void gi_bvhc_collect_childs(const gi_bvhc_work_data* wdata,
                                   uint32_t node_index,
                                   uint32_t child_index,
                                   uint32_t* child_count,
                                   uint32_t* child_indices)
{
  assert(*child_count <= 8);

  const gi_bvhc_split* split = &wdata->splits[node_index * 7 + child_index];
  const gi_bvh_node* node = &wdata->params->bvh->nodes[node_index];
  const gi_bvhc_split* left_split = &wdata->splits[node->field1 * 7 + split->left_count];
  const gi_bvhc_split* right_split = &wdata->splits[node->field2 * 7 + split->right_count];

  if (left_split->split_type == GI_BVHC_SPLIT_TYPE_DISTRIBUTE)
  {
    gi_bvhc_collect_childs(wdata, node->field1, split->left_count, child_count, child_indices);
  }
  else
  {
    child_indices[(*child_count)++] = node->field1;
  }

  if (right_split->split_type == GI_BVHC_SPLIT_TYPE_DISTRIBUTE)
  {
    gi_bvhc_collect_childs(wdata, node->field2, split->right_count, child_count, child_indices);
  }
  else
  {
    child_indices[(*child_count)++] = node->field2;
  }
}

static uint32_t gi_bvhc_push_child_leaves(const gi_bvhc_work_data* wdata,
                                          uint32_t node_idx,
                                          gml_aabb* aabb)
{
  const gi_bvh_node* node = &wdata->params->bvh->nodes[node_idx];

  if ((node->field2 & 0x80000000) == 0x80000000)
  {
    gml_aabb_merge(aabb, &node->aabb, aabb);
    uint32_t face_count = (node->field2 & 0x7FFFFFFF);

    for (uint32_t i = 0; i < face_count; ++i)
    {
      wdata->bvhc->faces[wdata->bvhc->face_count + i] = wdata->params->bvh->faces[node->field1 + i];
    }
    wdata->bvhc->face_count += face_count;

    return face_count;
  }

  return gi_bvhc_push_child_leaves(wdata, node->field1, aabb) +
         gi_bvhc_push_child_leaves(wdata, node->field2, aabb);
}

static uint32_t gi_bvhc_create_nodes(const gi_bvhc_work_data* wdata,
                                     uint32_t node_idx,
                                     gi_bvhc_node* parent_node,
                                     gml_aabb* parent_aabb)
{
  /* Inline nodes contained in distributed splits. */
  uint32_t child_node_count = 0;
  uint32_t child_node_indices[8];
  gi_bvhc_collect_childs(wdata, node_idx, 0, &child_node_count, child_node_indices);

  /* Create leaf nodes and internal node offsets. */
  parent_node->child_index = wdata->bvhc->node_count;
  parent_node->face_index = wdata->bvhc->face_count;

  for (uint32_t i = 0; i < child_node_count; ++i)
  {
    int32_t child_node_idx = child_node_indices[i];
    const gi_bvhc_split* split = &wdata->splits[child_node_idx * 7];

    if (split->split_type == GI_BVHC_SPLIT_TYPE_LEAF)
    {
      uint32_t face_offset = wdata->bvhc->face_count;
      uint32_t face_count = gi_bvhc_push_child_leaves(wdata, child_node_idx, &parent_node->aabbs[i]);

      parent_node->offsets[i] = face_offset - parent_node->face_index;
      parent_node->counts[i] = (0x80000000 | face_count);

      gml_aabb_merge(parent_aabb, &parent_node->aabbs[i], parent_aabb);
    }
    else if (split->split_type == GI_BVHC_SPLIT_TYPE_INTERNAL)
    {
      uint32_t new_node_idx = (wdata->bvhc->node_count++);
      parent_node->offsets[i] = new_node_idx - parent_node->child_index;
    }
    else
    {
      assert(false);
    }
  }

  /* Get internal node counts and AABBs by recursing into children. */
  for (uint32_t i = 0; i < child_node_count; ++i)
  {
    int32_t child_node_idx = child_node_indices[i];
    const gi_bvhc_split* split = &wdata->splits[child_node_idx * 7];

    if (split->split_type != GI_BVHC_SPLIT_TYPE_INTERNAL)
    {
      continue;
    }

    uint32_t new_node_idx = parent_node->child_index + parent_node->offsets[i];
    gi_bvhc_node* new_node = &wdata->bvhc->nodes[new_node_idx];

    for (uint32_t k = 0; k < 8; ++k)
    {
      new_node->counts[k] = 0;
      new_node->offsets[k] = 0;
      gml_aabb_make_smallest(&new_node->aabbs[k]);
    }

    parent_node->counts[i] = gi_bvhc_create_nodes(wdata, child_node_idx, new_node, &parent_node->aabbs[i]);

    gml_aabb_merge(parent_aabb, &parent_node->aabbs[i], parent_aabb);
  }

  return child_node_count;
}

void gi_bvh_collapse(const gi_bvhc_params* params,
                     gi_bvhc* bvhc)
{
  /* This would lead to a leaf node being root. We don't support it. */
  assert(params->bvh->face_count > params->max_leaf_size);

  /* Calculate cost lookup table. */
  uint32_t num_splits = params->bvh->node_count * 7;
  gi_bvhc_split* splits = new gi_bvhc_split[num_splits];

  for (uint32_t i = 0; i < num_splits; ++i)
  {
    gi_bvhc_split* split = &splits[i];
    split->split_type = GI_BVHC_SPLIT_TYPE_INVALID;
  }

  gi_bvhc_work_data work_data;
  work_data.bvhc = bvhc;
  work_data.params = params;
  work_data.splits = splits;

  gi_bvhc_calc_costs(&work_data, 0);

  /* Set up new bvh and include a root node. */
  bvhc->aabb = params->bvh->aabb;
  bvhc->node_count = 1;
  bvhc->nodes = new gi_bvhc_node[params->bvh->node_count];
  bvhc->face_count = 0;
  bvhc->faces = new gi_face[params->bvh->face_count];

  /* Clear root node. */
  gi_bvhc_node* root_node = &bvhc->nodes[0];

  for (uint32_t j = 0; j < 8; ++j)
  {
    root_node->offsets[j] = 0;
    root_node->counts[j] = 0;
    gml_aabb_make_smallest(&root_node->aabbs[j]);
  }

  /* Construct wide bvh recursively using previously calculated costs. */
  gi_bvhc_create_nodes(&work_data, 0, root_node, &bvhc->aabb);

  /* There can be less nodes than in the input BVH because we collapse leaves. */
  bvhc->nodes = (gi_bvhc_node*) realloc(bvhc->nodes, bvhc->node_count * sizeof(gi_bvhc_node));

  delete[] splits;
}

void gi_free_bvhc(gi_bvhc* bvhc)
{
  delete[] bvhc->nodes;
  delete[] bvhc->faces;
}
