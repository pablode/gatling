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

#include "bvh_embree.h"

#include "gi.h"

#include <embree3/rtcore.h>
#include <vector>
#include <stdlib.h>
#include <assert.h>
#include <atomic>
#include <iterator>

using namespace gi;
using namespace gi::bvh;

namespace impl
{
  // Embree's BVH memory representation is slightly different from Gatling's.
  // An explicit conversion step seems like the only viable solution..
  struct EmbreeBvhNode
  {
    virtual bool isLeaf() const = 0;
  };

  struct EmbreeInnerBvhNode : public EmbreeBvhNode
  {
    gml_aabb aabbs[2];
    EmbreeBvhNode* childs[2];
    bool isLeaf() const override { return false; }
  };

  struct EmbreeLeafBvhNode : public EmbreeBvhNode
  {
    uint32_t face_index;
    gml_aabb aabb;
    bool isLeaf() const override { return true; }
  };

  static void* create_node(RTCThreadLocalAllocator alloc, unsigned int numChildren, void* userPtr)
  {
    assert(numChildren == 2);
    void* ptr = rtcThreadLocalAlloc(alloc, sizeof(EmbreeInnerBvhNode), 16);
    return new(ptr) EmbreeInnerBvhNode();
  }

  static void set_children(void* nodePtr, void** childPtr, unsigned int numChildren, void* userPtr)
  {
    assert(numChildren == 2);
    EmbreeInnerBvhNode* node = (EmbreeInnerBvhNode*) nodePtr;
    node->childs[0] = (EmbreeBvhNode*) childPtr[0];
    node->childs[1] = (EmbreeBvhNode*) childPtr[1];
  }

  static void set_node_bounds(void* nodePtr, const RTCBounds** bounds, unsigned int numChildren, void* userPtr)
  {
    assert(numChildren == 2);
    EmbreeInnerBvhNode* node = (EmbreeInnerBvhNode*) nodePtr;
    for (int i = 0; i < 2; i++)
    {
      node->aabbs[i].min[0] = bounds[i]->lower_x;
      node->aabbs[i].min[1] = bounds[i]->lower_y;
      node->aabbs[i].min[2] = bounds[i]->lower_z;
      node->aabbs[i].max[0] = bounds[i]->upper_x;
      node->aabbs[i].max[1] = bounds[i]->upper_y;
      node->aabbs[i].max[2] = bounds[i]->upper_z;
    }
  }

  static void* create_leaf(RTCThreadLocalAllocator alloc, const RTCBuildPrimitive* prims, size_t numPrims, void* userPtr)
  {
    // We can build bigger leaves in the collapsing phase, so let's just use one here.
    assert(numPrims == 1);
    void* ptr = (EmbreeLeafBvhNode*) rtcThreadLocalAlloc(alloc, sizeof(EmbreeLeafBvhNode), 16);
    EmbreeLeafBvhNode* node = new(ptr) EmbreeLeafBvhNode();
    node->face_index = prims[0].primID;
    node->aabb.min[0] = prims[0].lower_x;
    node->aabb.min[1] = prims[0].lower_y;
    node->aabb.min[2] = prims[0].lower_z;
    node->aabb.max[0] = prims[0].upper_x;
    node->aabb.max[1] = prims[0].upper_y;
    node->aabb.max[2] = prims[0].upper_z;
    return node;
  }

  void convert_bvh(Bvh2& bvh2, gi_face* faces, uint32_t bvh2_node_idx, EmbreeBvhNode* node)
  {
    BvhNode2& new_node = bvh2.nodes[bvh2_node_idx];

    if (node->isLeaf())
    {
      EmbreeLeafBvhNode* leaf_node = (EmbreeLeafBvhNode*) node;
      new_node.aabb = leaf_node->aabb;
      new_node.field1 = bvh2.faces.size();
      new_node.field2 = (0x80000000 | 1);
      bvh2.faces.push_back(faces[leaf_node->face_index]);
      return;
    }

    uint32_t left_index = bvh2.nodes.size();
    bvh2.nodes.push_back({});
    uint32_t right_index = bvh2.nodes.size();
    bvh2.nodes.push_back({});

    EmbreeInnerBvhNode* inner_node = (EmbreeInnerBvhNode*) node;
    new_node.aabb = inner_node->aabbs[0];
    gml_aabb_merge(&new_node.aabb, &inner_node->aabbs[1], &new_node.aabb);
    new_node.field1 = left_index;
    new_node.field2 = right_index;

    convert_bvh(bvh2, faces, left_index, inner_node->childs[0]);
    convert_bvh(bvh2, faces, right_index, inner_node->childs[1]);
  }

  Bvh2 build_bvh2_embree(const EmbreeBuildParams& params)
  {
    uint32_t face_count = params.face_count;

    // Get face AABBs in embree representation.
    std::vector<RTCBuildPrimitive> prims;
    prims.reserve(face_count * 2);

    for (size_t i = 0; i < face_count; i++)
    {
      const gi_face& face = params.faces[i];
      const gi_vertex& v_a = params.vertices[face.v_i[0]];
      const gi_vertex& v_b = params.vertices[face.v_i[1]];
      const gi_vertex& v_c = params.vertices[face.v_i[2]];

      RTCBuildPrimitive prim;
      prim.lower_x = fminf(fminf(v_a.pos[0], v_b.pos[0]), v_c.pos[0]);
      prim.lower_y = fminf(fminf(v_a.pos[1], v_b.pos[1]), v_c.pos[1]);
      prim.lower_z = fminf(fminf(v_a.pos[2], v_b.pos[2]), v_c.pos[2]);
      prim.geomID = 0;
      prim.upper_x = fmaxf(fmaxf(v_a.pos[0], v_b.pos[0]), v_c.pos[0]);
      prim.upper_y = fmaxf(fmaxf(v_a.pos[1], v_b.pos[1]), v_c.pos[1]);
      prim.upper_z = fmaxf(fmaxf(v_a.pos[2], v_b.pos[2]), v_c.pos[2]);
      prim.primID = i;

      prims.push_back(prim);
    }

    // Set up build params.
    RTCDevice device = rtcNewDevice(nullptr);
    RTCBVH rtcBvh = rtcNewBVH(device);

    RTCBuildArguments arguments = rtcDefaultBuildArguments();
    arguments.byteSize = sizeof(arguments);
    arguments.buildFlags = RTC_BUILD_FLAG_NONE;
    arguments.buildQuality = RTC_BUILD_QUALITY_HIGH;
    arguments.maxBranchingFactor = 2;
    arguments.maxDepth = 1024;
    arguments.sahBlockSize = params.face_batch_size;
    arguments.minLeafSize = 1;
    arguments.maxLeafSize = 1;
    arguments.traversalCost = params.node_traversal_cost;
    arguments.intersectionCost = params.face_intersection_cost;
    arguments.bvh = rtcBvh;
    arguments.primitives = prims.data();
    arguments.primitiveCount = prims.size();
    arguments.primitiveArrayCapacity = prims.capacity();
    arguments.createNode = impl::create_node;
    arguments.setNodeChildren = impl::set_children;
    arguments.setNodeBounds = impl::set_node_bounds;
    arguments.createLeaf = impl::create_leaf;
    arguments.splitPrimitive = nullptr;
    arguments.buildProgress = nullptr;
    arguments.userPtr = nullptr;

    // Build the BVH.
    EmbreeBvhNode* root = (EmbreeBvhNode*) rtcBuildBVH(&arguments);
    assert(root);
    assert(!root->isLeaf());

    // Convert it to our own representation.
    Bvh2 bvh2;
    bvh2.faces.reserve(face_count);
    bvh2.nodes.reserve(face_count * 2);
    bvh2.nodes.push_back({});

    convert_bvh(bvh2, params.faces, 0, root);
    assert(bvh2.faces.size() == face_count);

    rtcReleaseBVH(rtcBvh);
    rtcReleaseDevice(device);

    return bvh2;
  }
}

Bvh2 gi::bvh::build_bvh2_embree(const EmbreeBuildParams& params)
{
  return impl::build_bvh2_embree(params);
}
