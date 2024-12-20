#ifndef H_MDL_INTERFACE
#define H_MDL_INTERFACE

#include "common.glsl"

// See also: https://github.com/NVIDIA/MDL-SDK/blob/master/examples/mdl_sdk/dxr/content/mdl_renderer_runtime.hlsl

float apply_wrap_and_crop(float coord, int wrap, vec2 crop, int res)
{
    if (wrap == TEX_WRAP_REPEAT)
    {
        if (all(notEqual(crop, vec2(0.0, 1.0))))
        {
            return coord;
        }
        coord -= floor(coord);
    }
    else
    {
        if (wrap == TEX_WRAP_MIRRORED_REPEAT)
        {
            float tmp = floor(coord);

            if ((int(tmp) & 1) != 0)
            {
                coord = 1.0 - (coord - tmp);
            }
            else
            {
                coord -= tmp;
            }
        }

        float inv_hdim = 0.5 / float(res);
        coord = clamp(coord, inv_hdim, 1.0 - inv_hdim);
    }
    return coord * (crop.y - crop.x) + crop.x;
}

bool tex_texture_isvalid(int tex)
{
    return tex != 0;
}

vec4 tex_lookup_float4_3d(int tex, vec3 coord, int wrap_u, int wrap_v, int wrap_w, vec2 crop_u, vec2 crop_v, vec2 crop_w, float frame)
{
#if TEXTURE_COUNT_3D > 0
    if ((tex == 0) ||
        (wrap_u == TEX_WRAP_CLIP && (coord.x < 0.0 || coord.x > 1.0)) ||
        (wrap_v == TEX_WRAP_CLIP && (coord.y < 0.0 || coord.y > 1.0)) ||
        (wrap_w == TEX_WRAP_CLIP && (coord.z < 0.0 || coord.z > 1.0)))
    {
        return vec4(0, 0, 0, 0);
    }

    uint array_idx = TEXTURE_INDEX_OFFSET_3D + tex - 1;

    int mipmap_level = 0;
    ivec3 res = textureSize(textures_3d[nonuniformEXT(array_idx)], mipmap_level);
    coord.x = apply_wrap_and_crop(coord.x, wrap_u, crop_u, res.x);
    coord.y = apply_wrap_and_crop(coord.y, wrap_v, crop_v, res.y);
    coord.z = apply_wrap_and_crop(coord.z, wrap_w, crop_w, res.z);

    ASSERT(array_idx < TEXTURE_COUNT_3D, "Error: invalid texture index\n");
    return texture(sampler3D(textures_3d[nonuniformEXT(array_idx)], tex_sampler), coord);
#else
    ASSERT(tex == 0, "Error: invalid texture index\n");
    return vec4(0, 0, 0, 0);
#endif
}

vec3 tex_lookup_float3_3d(int tex, vec3 coord, int wrap_u, int wrap_v, int wrap_w, vec2 crop_u, vec2 crop_v, vec2 crop_w, float frame)
{
    return tex_lookup_float4_3d(tex, coord, wrap_u, wrap_v, wrap_w, crop_u, crop_v, crop_w, frame).xyz;
}

vec2 tex_lookup_float2_3d(int tex, vec3 coord, int wrap_u, int wrap_v, int wrap_w, vec2 crop_u, vec2 crop_v, vec2 crop_w, float frame)
{
    return tex_lookup_float4_3d(tex, coord, wrap_u, wrap_v, wrap_w, crop_u, crop_v, crop_w, frame).xy;
}

float tex_lookup_float_3d(int tex, vec3 coord, int wrap_u, int wrap_v, int wrap_w, vec2 crop_u, vec2 crop_v, vec2 crop_w, float frame)
{
    return tex_lookup_float4_3d(tex, coord, wrap_u, wrap_v, wrap_w, crop_u, crop_v, crop_w, frame).x;
}

vec3 tex_lookup_color_3d(int tex, vec3 coord, int wrap_u, int wrap_v, int wrap_w, vec2 crop_u, vec2 crop_v, vec2 crop_w, float frame)
{
    return tex_lookup_float3_3d(tex, coord, wrap_u, wrap_v, wrap_v, crop_u, crop_v, crop_w, frame);
}

vec4 tex_texel_float4_3d(int tex, ivec3 coord, float frame)
{
#if TEXTURE_COUNT_3D > 0
    if (tex == 0)
    {
        return vec4(0, 0, 0, 0);
    }

    uint array_idx = TEXTURE_INDEX_OFFSET_3D + tex - 1;

    int mipmap_level = 0;
    ivec3 res = textureSize(textures_3d[nonuniformEXT(array_idx)], mipmap_level);
    if (coord.x < 0 || coord.x >= res.x || coord.y < 0 || coord.y >= res.y || coord.z < 0 || coord.z >= res.z)
    {
        return vec4(0, 0, 0, 0);
    }

    ASSERT(array_idx < TEXTURE_COUNT_3D, "Error: invalid texture index\n");
    return texelFetch(sampler3D(textures_3d[nonuniformEXT(array_idx)], tex_sampler), coord, mipmap_level);
#else
    ASSERT(tex == 0, "Error: invalid texture index\n");
    return vec4(0, 0, 0, 0);
#endif
}

vec3 tex_texel_float3_3d(int tex, ivec3 coord, float frame)
{
    return tex_texel_float4_3d(tex, coord, frame).xyz;
}

vec2 tex_texel_float2_3d(int tex, ivec3 coord, float frame)
{
    return tex_texel_float4_3d(tex, coord, frame).xy;
}

float tex_texel_float_3d(int tex, ivec3 coord, float frame)
{
    return tex_texel_float4_3d(tex, coord, frame).x;
}

vec3 tex_texel_color_3d(int tex, ivec3 coord, float frame)
{
    return tex_texel_float3_3d(tex, coord, frame);
}

vec4 tex_lookup_float4_2d(int tex, vec2 coord, int wrap_u, int wrap_v, vec2 crop_u, vec2 crop_v, float frame)
{
#if TEXTURE_COUNT_2D > 0
    if ((tex == 0) ||
        (wrap_u == TEX_WRAP_CLIP && (coord.x < 0.0 || coord.x > 1.0)) ||
        (wrap_v == TEX_WRAP_CLIP && (coord.y < 0.0 || coord.y > 1.0)))
    {
        return vec4(0, 0, 0, 0);
    }

    uint array_idx = TEXTURE_INDEX_OFFSET_2D + tex - 1;

    int mipmap_level = 0;
    ivec2 res = textureSize(textures_2d[nonuniformEXT(array_idx)], mipmap_level);
    coord.x = apply_wrap_and_crop(coord.x, wrap_u, crop_u, res.x);
    coord.y = apply_wrap_and_crop(coord.y, wrap_v, crop_v, res.y);

    ASSERT(array_idx < TEXTURE_COUNT_2D, "Error: invalid texture index\n");
    return texture(sampler2D(textures_2d[nonuniformEXT(array_idx)], tex_sampler), coord);
#else
    ASSERT(tex == 0, "Error: invalid texture index\n");
    return vec4(0, 0, 0, 0);
#endif
}

vec3 tex_lookup_float3_2d(int tex, vec2 coord, int wrap_u, int wrap_v, vec2 crop_u, vec2 crop_v, float frame)
{
    return tex_lookup_float4_2d(tex, coord, wrap_u, wrap_v, crop_u, crop_v, frame).xyz;
}

vec2 tex_lookup_float2_2d(int tex, vec2 coord, int wrap_u, int wrap_v, vec2 crop_u, vec2 crop_v, float frame)
{
    return tex_lookup_float4_2d(tex, coord, wrap_u, wrap_v, crop_u, crop_v, frame).xy;
}

float tex_lookup_float_2d(int tex, vec2 coord, int wrap_u, int wrap_v, vec2 crop_u, vec2 crop_v, float frame)
{
    return tex_lookup_float4_2d(tex, coord, wrap_u, wrap_v, crop_u, crop_v, frame).x;
}

vec3 tex_lookup_color_2d(int tex, vec2 coord, int wrap_u, int wrap_v, vec2 crop_u, vec2 crop_v, float frame)
{
    return tex_lookup_float3_2d(tex, coord, wrap_u, wrap_v, crop_u, crop_v, frame);
}

vec4 tex_texel_float4_2d(int tex, ivec2 coord, ivec2 uv_tile, float frame)
{
#if TEXTURE_COUNT_2D > 0
    if (tex == 0)
    {
        return vec4(0, 0, 0, 0);
    }

    uint array_idx = TEXTURE_INDEX_OFFSET_2D + tex - 1;

    int mipmap_level = 0;
    ivec2 res = textureSize(textures_2d[nonuniformEXT(array_idx)], mipmap_level);
    if (coord.x < 0 || coord.x >= res.x || coord.y < 0 || coord.y >= res.y)
    {
        return vec4(0, 0, 0, 0);
    }

    ASSERT(array_idx < TEXTURE_COUNT_2D, "Error: invalid texture index\n");
    return texelFetch(sampler2D(textures_2d[nonuniformEXT(array_idx)], tex_sampler), coord, mipmap_level);
#else
    ASSERT(tex == 0, "Error: invalid texture index\n");
    return vec4(0, 0, 0, 0);
#endif
}

vec3 tex_texel_float3_2d(int tex, ivec2 coord, ivec2 uv_tile, float frame)
{
    return tex_texel_float4_2d(tex, coord, uv_tile, frame).xyz;
}

vec2 tex_texel_float2_2d(int tex, ivec2 coord, ivec2 uv_tile, float frame)
{
    return tex_texel_float4_2d(tex, coord, uv_tile, frame).xy;
}

float tex_texel_float_2d(int tex, ivec2 coord, ivec2 uv_tile, float frame)
{
    return tex_texel_float4_2d(tex, coord, uv_tile, frame).x;
}

vec3 tex_texel_color_2d(int tex, ivec2 coord, ivec2 uv_tile, float frame)
{
    return tex_texel_float3_2d(tex, coord, uv_tile, frame);
}

ivec2 tex_resolution_2d(int tex, ivec2 uv_tile, float frame)
{
#if TEXTURE_COUNT_2D > 0
    if (tex == 0)
    {
        return ivec2(0, 0);
    }

    uint array_idx = TEXTURE_INDEX_OFFSET_2D + tex - 1;

    ASSERT(array_idx < TEXTURE_COUNT_2D, "Error: invalid texture index\n");

    int mipmap_level = 0;
    return textureSize(textures_2d[nonuniformEXT(array_idx)], mipmap_level);
#else
    ASSERT(tex == 0, "Error: invalid texture index\n");
    return ivec2(0, 0);
#endif
}

int tex_width_2d(int tex, ivec2 uv_tile, float frame)
{
    return tex_resolution_2d(tex, uv_tile, frame).x;
}

int tex_height_2d(int tex, ivec2 uv_tile, float frame)
{
    return tex_resolution_2d(tex, uv_tile, frame).y;
}

// Adapt normal to fix the 'shadow terminator problem'. The problem is well-described here:
// Hanika, Hacking the Shadow Terminator. https://jo.dreggn.org/home/2021_terminator.pdf (Figure 4-3)
//
// We employ Iray's approach of bending the normal as described in:
// Keller et al. The Iray Light Transport Simulation and Rendering System.
// https://arxiv.org/pdf/1705.01263.pdf, Section A.3
vec3 mdl_adapt_normal(State state, vec3 normal)
{
#if 0
    // Disable normal mapping for debug purposes
    return state.normal;
#endif

    vec3 r = normalize(reflect(gl_WorldRayDirectionEXT, normal));

    float a = max(0.0, dot(r, -state.geom_normal));

    float b = dot(normal, state.geom_normal);

    vec3 tangent = normalize(r + (a / b) * normal);

    vec3 new_normal = normalize(-gl_WorldRayDirectionEXT + tangent);

    return new_normal;
}

#endif

// Buffer references to read different data types
layout(buffer_reference, std430, buffer_reference_align = 4 /* largest type */) buffer BufferRefInt { int data[]; };
layout(buffer_reference, std430, buffer_reference_align = 4 /* largest type */) buffer BufferRefFloat { float data[]; };
layout(buffer_reference, std430, buffer_reference_align = 8 /* largest type */) buffer BufferRefVec2 { vec2 data[]; };
layout(buffer_reference, std430, buffer_reference_align = 16/* largest type */) buffer BufferRefVec4 { vec4 data[]; };

bool scene_data_isvalid(inout State state, int scene_data_id)
{
#if SCENE_DATA_COUNT > 0
    if (scene_data_id == 0 || scene_data_id > SCENE_DATA_COUNT)
    {
      return false;
    }

    mdl_renderer_state rs = state.renderer_state;
    return rs.sceneDataInfos[scene_data_id - 1] != UINT32_MAX;
#endif
    return false;
}

#if SCENE_DATA_COUNT > 0
uvec3 get_scene_data_indices(mdl_renderer_state rs, uint sceneDataInfo, bool uniformLookup)
{
  // contains GiPrimvarInterpolation enum
  uint interpolation = (sceneDataInfo & SCENE_DATA_INTERPOLATION_MASK) >> SCENE_DATA_INTERPOLATION_OFFSET;

  if (interpolation == 2/*uniform*/)
  {
    return uvec3(gl_PrimitiveID);
  }

  if (interpolation == 1/*instance*/)
  {
    int instanceId = InstanceIds[gl_InstanceID];
    return uvec3(instanceId);
  }

  bool sceneDataConstant = (interpolation == 0/*constant*/);

  return rs.hitIndices * int(!uniformLookup && !sceneDataConstant);
}
#endif

vec4 scene_data_lookup_float4(inout State state, int scene_data_id, vec4 default_value, bool uniform_lookup)
{
#if SCENE_DATA_COUNT > 0
    if (scene_data_isvalid(state, scene_data_id))
    {
        mdl_renderer_state rs = state.renderer_state;

        uint sceneDataInfo = rs.sceneDataInfos[scene_data_id - 1];
        uint64_t address = rs.sceneDataBufferAddress + (sceneDataInfo & SCENE_DATA_OFFSET_MASK) * SCENE_DATA_ALIGNMENT;
        uint stride = (((sceneDataInfo & SCENE_DATA_STRIDE_MASK) >> SCENE_DATA_STRIDE_OFFSET) + 1) >> 2/* / 4 floats */;
        BufferRefVec4 ref = BufferRefVec4(address);

        uvec3 indices = get_scene_data_indices(rs, sceneDataInfo, uniform_lookup) * stride;
        vec4 val0 = ref.data[indices[0]];
        vec4 val1 = ref.data[indices[1]];
        vec4 val2 = ref.data[indices[2]];

        vec3 bc = vec3(1.0 - rs.hitBarycentrics.x - rs.hitBarycentrics.y, rs.hitBarycentrics.x, rs.hitBarycentrics.y);
        return val0 * bc.x + val1 * bc.y + val2 * bc.z;
    }
#endif
    return default_value;
}

vec3 scene_data_lookup_float3(inout State state, int scene_data_id, vec3 default_value, bool uniform_lookup)
{
#ifdef CAMERA_POSITION_SCENE_DATA_INDEX
    if (scene_data_id == CAMERA_POSITION_SCENE_DATA_INDEX)
    {
      return PC.cameraPosition;
    }
#endif

#if SCENE_DATA_COUNT > 0
    if (scene_data_isvalid(state, scene_data_id))
    {
        mdl_renderer_state rs = state.renderer_state;

        uint sceneDataInfo = rs.sceneDataInfos[scene_data_id - 1];
        uint64_t address = rs.sceneDataBufferAddress + (sceneDataInfo & SCENE_DATA_OFFSET_MASK) * SCENE_DATA_ALIGNMENT;
        uint stride = ((sceneDataInfo & SCENE_DATA_STRIDE_MASK) >> SCENE_DATA_STRIDE_OFFSET) + 1;
        BufferRefFloat ref = BufferRefFloat(address);

        uvec3 indices = get_scene_data_indices(rs, sceneDataInfo, uniform_lookup) * stride;
        vec3 val0 = vec3(ref.data[indices[0] + 0], ref.data[indices[0] + 1], ref.data[indices[0] + 2]);
        vec3 val1 = vec3(ref.data[indices[1] + 0], ref.data[indices[1] + 1], ref.data[indices[1] + 2]);
        vec3 val2 = vec3(ref.data[indices[2] + 0], ref.data[indices[2] + 1], ref.data[indices[2] + 2]);

        vec3 bc = vec3(1.0 - rs.hitBarycentrics.x - rs.hitBarycentrics.y, rs.hitBarycentrics.x, rs.hitBarycentrics.y);
        return val0 * bc.x + val1 * bc.y + val2 * bc.z;
    }
#endif

    return default_value;
}

vec3 scene_data_lookup_color(inout State state, int scene_data_id, vec3 default_value, bool uniform_lookup)
{
    return scene_data_lookup_float3(state, scene_data_id, default_value, uniform_lookup);
}

vec2 scene_data_lookup_float2(inout State state, int scene_data_id, vec2 default_value, bool uniform_lookup)
{
#if SCENE_DATA_COUNT > 0
    if (scene_data_isvalid(state, scene_data_id))
    {
        mdl_renderer_state rs = state.renderer_state;

        uint sceneDataInfo = rs.sceneDataInfos[scene_data_id - 1];
        uint64_t address = rs.sceneDataBufferAddress + (sceneDataInfo & SCENE_DATA_OFFSET_MASK) * SCENE_DATA_ALIGNMENT;
        uint stride = (((sceneDataInfo & SCENE_DATA_STRIDE_MASK) >> SCENE_DATA_STRIDE_OFFSET) + 1) >> 1/* / 2 floats */;
        BufferRefVec2 ref = BufferRefVec2(address);

        uvec3 indices = get_scene_data_indices(rs, sceneDataInfo, uniform_lookup) * stride;
        vec2 val0 = ref.data[indices[0]];
        vec2 val1 = ref.data[indices[1]];
        vec2 val2 = ref.data[indices[2]];

        vec3 bc = vec3(1.0 - rs.hitBarycentrics.x - rs.hitBarycentrics.y, rs.hitBarycentrics.x, rs.hitBarycentrics.y);
        return val0 * bc.x + val1 * bc.y + val2 * bc.z;
    }
#endif
    return default_value;
}

float scene_data_lookup_float(inout State state, int scene_data_id, float default_value, bool uniform_lookup)
{
#if SCENE_DATA_COUNT > 0
    if (scene_data_isvalid(state, scene_data_id))
    {
        mdl_renderer_state rs = state.renderer_state;

        uint sceneDataInfo = rs.sceneDataInfos[scene_data_id - 1];
        uint64_t address = rs.sceneDataBufferAddress + (sceneDataInfo & SCENE_DATA_OFFSET_MASK) * SCENE_DATA_ALIGNMENT;
        uint stride = ((sceneDataInfo & SCENE_DATA_STRIDE_MASK) >> SCENE_DATA_STRIDE_OFFSET) + 1;
        BufferRefFloat ref = BufferRefFloat(address);

        uvec3 indices = get_scene_data_indices(rs, sceneDataInfo, uniform_lookup) * stride;
        vec3 val = vec3(ref.data[indices[0]], ref.data[indices[1]], ref.data[indices[2]]);

        vec3 bc = vec3(1.0 - rs.hitBarycentrics.x - rs.hitBarycentrics.y, rs.hitBarycentrics.x, rs.hitBarycentrics.y);
        return val.x * bc.x + val.y * bc.y + val.z * bc.z;
    }
#endif
    return default_value;
}

int scene_data_lookup_int(inout State state, int scene_data_id, int index_offset, int default_value, bool uniform_lookup)
{
#if SCENE_DATA_COUNT > 0
    if (scene_data_isvalid(state, scene_data_id))
    {
        mdl_renderer_state rs = state.renderer_state;

        uint sceneDataInfo = rs.sceneDataInfos[scene_data_id - 1];
        uint64_t address = rs.sceneDataBufferAddress + (sceneDataInfo & SCENE_DATA_OFFSET_MASK) * SCENE_DATA_ALIGNMENT;
        uint stride = ((sceneDataInfo & SCENE_DATA_STRIDE_MASK) >> SCENE_DATA_STRIDE_OFFSET) + 1;
        BufferRefInt ref = BufferRefInt(address);

        uvec3 indices = get_scene_data_indices(rs, sceneDataInfo, uniform_lookup) * stride + index_offset;
        ivec3 val = ivec3(ref.data[indices[0]], ref.data[indices[1]], ref.data[indices[2]]);

        vec3 bc = vec3(1.0 - rs.hitBarycentrics.x - rs.hitBarycentrics.y, rs.hitBarycentrics.x, rs.hitBarycentrics.y);

        // nearest interpolation
        if (bc.x > bc.y)
        {
          return bc.x > bc.z ? val.x : val.z;
        }
        else
        {
          return bc.y > bc.z ? val.y : val.z;
        }
    }
#endif
    return default_value;
}

int scene_data_lookup_int(inout State state, int scene_data_id, int default_value, bool uniform_lookup)
{
    return scene_data_lookup_int(state, scene_data_id, 0, default_value, uniform_lookup);
}

// NOTE: the following functions are implemented in a less efficient way because MaterialX does not support them.

ivec4 scene_data_lookup_int4(inout State state, int scene_data_id, ivec4 default_value, bool uniform_lookup)
{
    return ivec4(scene_data_lookup_int(state, scene_data_id, 0, default_value.x, uniform_lookup),
                 scene_data_lookup_int(state, scene_data_id, 1, default_value.y, uniform_lookup),
                 scene_data_lookup_int(state, scene_data_id, 2, default_value.z, uniform_lookup),
                 scene_data_lookup_int(state, scene_data_id, 3, default_value.w, uniform_lookup));
}

ivec3 scene_data_lookup_int3(inout State state, int scene_data_id, ivec3 default_value, bool uniform_lookup)
{
    return ivec3(scene_data_lookup_int(state, scene_data_id, 0, default_value.x, uniform_lookup),
                 scene_data_lookup_int(state, scene_data_id, 1, default_value.y, uniform_lookup),
                 scene_data_lookup_int(state, scene_data_id, 2, default_value.z, uniform_lookup));
}

ivec2 scene_data_lookup_int2(inout State state, int scene_data_id, ivec2 default_value, bool uniform_lookup)
{
    return ivec2(scene_data_lookup_int(state, scene_data_id, 0, default_value.x, uniform_lookup),
                 scene_data_lookup_int(state, scene_data_id, 1, default_value.y, uniform_lookup));
}

mat4 scene_data_lookup_float4x4(inout State state, int scene_data_id, mat4 default_value, bool uniform_lookup)
{
    return default_value; // TODO: not implemented
}
