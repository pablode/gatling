#include "bvh_compress.h"

#include <math.h>
#include <stdlib.h>

static void gi_bvhcc_compress_node(const struct gi_bvhc_node* in_node,
                                   const gml_aabb* parent_aabb,
                                   struct gi_bvhcc_node* out_node)
{
  const uint32_t Nq = 8;

  float B_lo_x = parent_aabb->min[0];
  float B_lo_y = parent_aabb->min[1];
  float B_lo_z = parent_aabb->min[2];
  float B_hi_x = parent_aabb->max[0];
  float B_hi_y = parent_aabb->max[1];
  float B_hi_z = parent_aabb->max[2];

  out_node->p_x = B_lo_x;
  out_node->p_y = B_lo_y;
  out_node->p_z = B_lo_z;

  float s_x = B_hi_x - B_lo_x;
  float s_y = B_hi_y - B_lo_y;
  float s_z = B_hi_z - B_lo_z;

  float e_div = 1.0f / (float) ((1 << Nq) - 1);

  int32_t e_x = -127;
  int32_t e_y = -127;
  int32_t e_z = -127;

  if (s_x > 0.0f)
  {
    e_x = (int32_t) ceilf(log2f(s_x * e_div));
    e_x = (e_x < -127) ? -127 : (e_x > 128) ? 128 : e_x;
  }

  if (s_y > 0.0f)
  {
    e_y = (int32_t) ceilf(log2f(s_y * e_div));
    e_y = (e_y < -127) ? -127 : (e_y > 128) ? 128 : e_y;
  }

  if (s_z > 0.0f)
  {
    e_z = (int32_t) ceilf(log2f(s_z * e_div));
    e_z = (e_z < -127) ? -127 : (e_z > 128) ? 128 : e_z;
  }

  out_node->e_x = (uint8_t) (e_x + 127);
  out_node->e_y = (uint8_t) (e_y + 127);
  out_node->e_z = (uint8_t) (e_z + 127);

  out_node->child_index = in_node->child_index;
  out_node->face_index = in_node->face_index;
  out_node->imask = 0;

  float b_div_x = 1.0f / exp2f((float) e_x);
  float b_div_y = 1.0f / exp2f((float) e_y);
  float b_div_z = 1.0f / exp2f((float) e_z);

  for (uint32_t i = 0; i < 8; ++i)
  {
    uint32_t child_count = in_node->counts[i] & 0x7FFFFFFF;

    if (child_count == 0)
    {
      out_node->meta[i] = 0;

      /* Let's make sure the output is deterministic. */
      out_node->q_lo_x[i] = 0;
      out_node->q_lo_y[i] = 0;
      out_node->q_lo_z[i] = 0;
      out_node->q_hi_x[i] = 0;
      out_node->q_hi_y[i] = 0;
      out_node->q_hi_z[i] = 0;
      continue;
    }

    const gml_aabb* in_aabb = &in_node->aabbs[i];
    float b_lo_x = in_aabb->min[0];
    float b_lo_y = in_aabb->min[1];
    float b_lo_z = in_aabb->min[2];
    float b_hi_x = in_aabb->max[0];
    float b_hi_y = in_aabb->max[1];
    float b_hi_z = in_aabb->max[2];

    out_node->q_lo_x[i] = (uint8_t) floorf((b_lo_x - B_lo_x) * b_div_x);
    out_node->q_lo_y[i] = (uint8_t) floorf((b_lo_y - B_lo_y) * b_div_y);
    out_node->q_lo_z[i] = (uint8_t) floorf((b_lo_z - B_lo_z) * b_div_z);
    out_node->q_hi_x[i] = (uint8_t) ceilf((b_hi_x - B_lo_x) * b_div_x);
    out_node->q_hi_y[i] = (uint8_t) ceilf((b_hi_y - B_lo_y) * b_div_y);
    out_node->q_hi_z[i] = (uint8_t) ceilf((b_hi_z - B_lo_z) * b_div_z);

    int32_t offset = in_node->offsets[i];
    bool is_internal = (in_node->counts[i] & 0x80000000) != 0x80000000;

    if (is_internal)
    {
      assert(offset <= 7);
      out_node->imask |= (1 << offset);
      out_node->meta[i] = (1 << 5) | (offset + 24);
      continue;
    }

    assert(child_count < 4);
    assert(offset <= 23);
    out_node->meta[i] = (1 << 5) | offset;
    if (child_count >= 2) { out_node->meta[i] |= (1 << 6); }
    if (child_count == 3) { out_node->meta[i] |= (1 << 7); }
  }
}

static void gi_bvhcc_compress_subtree(const struct gi_bvhc* bvhc,
                                      struct gi_bvhcc* bvhcc,
                                      const gml_aabb* node_aabb,
                                      uint32_t node_idx)
{
  const struct gi_bvhc_node* in_node = &bvhc->nodes[node_idx];
  struct gi_bvhcc_node* out_node = &bvhcc->nodes[node_idx];

  for (uint32_t i = 0; i < 8; ++i)
  {
    bool is_empty = (in_node->counts[i] == 0);
    bool is_leaf = (in_node->counts[i] & 0x80000000) == 0x80000000;

    if (is_empty || is_leaf)
    {
      continue;
    }

    uint32_t child_node_idx = in_node->child_index + in_node->offsets[i];
    gi_bvhcc_compress_subtree(bvhc, bvhcc, &in_node->aabbs[i], child_node_idx);
  }

  gi_bvhcc_compress_node(in_node, node_aabb, out_node);
}

void gi_bvh_compress(const struct gi_bvhc* bvhc,
                     struct gi_bvhcc* bvhcc)
{
  bvhcc->aabb = bvhc->aabb;

  bvhcc->node_count = bvhc->node_count;
  bvhcc->nodes = malloc(bvhc->node_count * sizeof(struct gi_bvhcc_node));

  gi_bvhcc_compress_subtree(bvhc, bvhcc, &bvhc->aabb, 0);
}

void gi_free_bvhcc(struct gi_bvhcc* bvhcc)
{
  free(bvhcc->nodes);
}
