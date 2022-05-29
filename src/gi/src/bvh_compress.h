#pragma once

#include <stdint.h>
#include <vector>

#include <gml.h>

#include "bvh_collapse.h"

namespace gi
{
  namespace bvh
  {
    //
    // The memory layout of our compressed BVH nodes is the same as the one described
    // in "Efficient Incoherent Ray Traversal on GPUs Through Compressed Wide BVHs"
    // by Ylitie et al. '17.
    //
    //  ┌───────────────────────────────────┬───────────────────────────────────┐
    //  │ p_x                               │ p_y                               │
    //  ├───────────────────────────────────┼────────┬────────┬────────┬────────┤
    //  │ p_z                               │ e_x    │ e_y    │ e_z    │ imask  │
    //  ├───────────────────────────────────┼────────┴────────┴────────┴────────┤
    //  │ child node base index             │ triangle base index               │
    //  ├────────┬────────┬────────┬────────┼────────┬────────┬────────┬────────┤
    //  │ meta * │      * │      * │      * │      * │      * │      * │      * │
    //  │ q_lo,x │        │        │        │        │        │        │        │
    //  │ q_lo,y │        │        │        │        │        │        │        │
    //  │ q_lo,z │        │        │        │        │        │        │        │
    //  │ q_hi,x │        │        │        │        │        │        │        │
    //  │ q_hi,y │        │        │        │        │        │        │        │
    //  │ q_hi,z │        │        │        │        │        │        │        │
    //  └────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘
    //    child1   child2   child3   child4   child5   child6   child7   child8
    //
    // The quantization frame (p_x, p_y, p_z, e_x, e_y, e_z) is stored in 15 bytes,
    // indexing information (imask, child node base index, triangle base index,
    // all child meta fields) is stored in 17 bytes and and the quantized child
    // AABBs are stored with one byte per plane per child, resulting in 48 bytes.
    // The total size of a single compressed BVH node is therefore 80 bytes.
    //
    struct Bvh8cNode
    {
      // Quantization frame.
      float p_x;               // 4 bytes
      float p_y;               // 4 bytes
      float p_z;               // 4 bytes
      uint8_t e_x;             // 1 byte
      uint8_t e_y;             // 1 byte
      uint8_t e_z;             // 1 byte
      // Indexing info.
      uint8_t imask;           // 1 byte
      uint32_t child_index;    // 4 bytes
      uint32_t face_index;     // 4 bytes
      uint8_t meta[8];         // 8 bytes
      // Child data.
      uint8_t q_lo_x[8];       // 8 bytes
      uint8_t q_lo_y[8];       // 8 bytes
      uint8_t q_lo_z[8];       // 8 bytes
      uint8_t q_hi_x[8];       // 8 bytes
      uint8_t q_hi_y[8];       // 8 bytes
      uint8_t q_hi_z[8];       // 8 bytes
    };

    static_assert(sizeof(Bvh8cNode) == 80, "Compressed BVH node size should be 80 bytes.");

    struct Bvh8c
    {
      gml_aabb aabb;
      std::vector<Bvh8cNode> nodes;
    };

    Bvh8c compress_bvh8(const Bvh<8>& bvh8);
  }
}
