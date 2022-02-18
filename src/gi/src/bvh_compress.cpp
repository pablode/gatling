#include "bvh_compress.h"

#include <assert.h>

using namespace gi::bvh;

namespace impl
{
  void compress_node(const BvhNode<8>& src_node,
                     const gml_aabb& parent_aabb,
                     Bvh8cNode& dest_node)
  {
    const uint32_t Nq = 8;

    float B_lo_x = parent_aabb.min[0];
    float B_lo_y = parent_aabb.min[1];
    float B_lo_z = parent_aabb.min[2];
    float B_hi_x = parent_aabb.max[0];
    float B_hi_y = parent_aabb.max[1];
    float B_hi_z = parent_aabb.max[2];

    dest_node.p_x = B_lo_x;
    dest_node.p_y = B_lo_y;
    dest_node.p_z = B_lo_z;

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

    dest_node.e_x = (uint8_t) (e_x + 127);
    dest_node.e_y = (uint8_t) (e_y + 127);
    dest_node.e_z = (uint8_t) (e_z + 127);

    dest_node.child_index = src_node.child_index;
    dest_node.face_index = src_node.face_index;
    dest_node.imask = 0;

    float b_div_x = 1.0f / exp2f((float) e_x);
    float b_div_y = 1.0f / exp2f((float) e_y);
    float b_div_z = 1.0f / exp2f((float) e_z);

    for (uint32_t i = 0; i < 8; ++i)
    {
      uint32_t child_count = src_node.counts[i] & 0x7FFFFFFF;

      if (child_count == 0)
      {
        dest_node.meta[i] = 0;

        /* Let's make sure the output is deterministic. */
        dest_node.q_lo_x[i] = 0;
        dest_node.q_lo_y[i] = 0;
        dest_node.q_lo_z[i] = 0;
        dest_node.q_hi_x[i] = 0;
        dest_node.q_hi_y[i] = 0;
        dest_node.q_hi_z[i] = 0;
        continue;
      }

      const gml_aabb& aabb = src_node.aabbs[i];
      float b_lo_x = aabb.min[0];
      float b_lo_y = aabb.min[1];
      float b_lo_z = aabb.min[2];
      float b_hi_x = aabb.max[0];
      float b_hi_y = aabb.max[1];
      float b_hi_z = aabb.max[2];

      dest_node.q_lo_x[i] = (uint8_t) floorf((b_lo_x - B_lo_x) * b_div_x);
      dest_node.q_lo_y[i] = (uint8_t) floorf((b_lo_y - B_lo_y) * b_div_y);
      dest_node.q_lo_z[i] = (uint8_t) floorf((b_lo_z - B_lo_z) * b_div_z);
      dest_node.q_hi_x[i] = (uint8_t) ceilf((b_hi_x - B_lo_x) * b_div_x);
      dest_node.q_hi_y[i] = (uint8_t) ceilf((b_hi_y - B_lo_y) * b_div_y);
      dest_node.q_hi_z[i] = (uint8_t) ceilf((b_hi_z - B_lo_z) * b_div_z);

      int32_t offset = src_node.offsets[i];
      bool is_internal = (src_node.counts[i] & 0x80000000) != 0x80000000;

      if (is_internal)
      {
        assert(offset <= 7);
        dest_node.imask |= (1 << offset);
        dest_node.meta[i] = (1 << 5) | (offset + 24);
        continue;
      }

      assert(child_count < 4);
      assert(offset <= 23);
      dest_node.meta[i] = (1 << 5) | offset;
      if (child_count >= 2) { dest_node.meta[i] |= (1 << 6); }
      if (child_count == 3) { dest_node.meta[i] |= (1 << 7); }
    }
  }

  void compress_subtree(const Bvh<8>& bvh8,
                        Bvh8c& bvh8c,
                        const gml_aabb& root_aabb,
                        uint32_t root_idx)
  {
    const BvhNode<8>& node = bvh8.nodes[root_idx];

    for (uint32_t i = 0; i < 8; i++)
    {
      bool is_empty = (node.counts[i] == 0);
      bool is_leaf = (node.counts[i] & 0x80000000) == 0x80000000;

      if (is_empty || is_leaf)
      {
        continue;
      }

      uint32_t child_idx = node.child_index + node.offsets[i];
      compress_subtree(bvh8, bvh8c, node.aabbs[i], child_idx);
    }

    compress_node(node, root_aabb, bvh8c.nodes[root_idx]);
  }
}

Bvh8c gi::bvh::compress_bvh8(const Bvh<8>& bvh8)
{
  Bvh8c bvh8c;
  bvh8c.aabb = bvh8.aabb;
  bvh8c.nodes.resize(bvh8.nodes.size());

  impl::compress_subtree(bvh8, bvh8c, bvh8.aabb, 0);

  return bvh8c;
}
