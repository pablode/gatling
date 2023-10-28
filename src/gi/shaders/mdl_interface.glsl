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
    ivec3 res = textureSize(textures_3d[array_idx], mipmap_level);
    coord.x = apply_wrap_and_crop(coord.x, wrap_u, crop_u, res.x);
    coord.y = apply_wrap_and_crop(coord.y, wrap_v, crop_v, res.y);
    coord.z = apply_wrap_and_crop(coord.z, wrap_w, crop_w, res.z);

    ASSERT(array_idx < TEXTURE_COUNT_3D, "Error: invalid texture index\n");
    return texture(sampler3D(textures_3d[array_idx], tex_sampler), coord);
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
    ivec3 res = textureSize(textures_3d[array_idx], mipmap_level);
    if (coord.x < 0 || coord.x >= res.x || coord.y < 0 || coord.y >= res.y || coord.z < 0 || coord.z >= res.z)
    {
        return vec4(0, 0, 0, 0);
    }

    ASSERT(array_idx < TEXTURE_COUNT_3D, "Error: invalid texture index\n");
    return texelFetch(sampler3D(textures_3d[array_idx], tex_sampler), coord, mipmap_level);
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
    ivec2 res = textureSize(textures_2d[array_idx], mipmap_level);
    coord.x = apply_wrap_and_crop(coord.x, wrap_u, crop_u, res.x);
    coord.y = apply_wrap_and_crop(coord.y, wrap_v, crop_v, res.y);

    ASSERT(array_idx < TEXTURE_COUNT_2D, "Error: invalid texture index\n");
    return texture(sampler2D(textures_2d[array_idx], tex_sampler), coord);
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
    ivec2 res = textureSize(textures_2d[array_idx], mipmap_level);
    if (coord.x < 0 || coord.x >= res.x || coord.y < 0 || coord.y >= res.y)
    {
        return vec4(0, 0, 0, 0);
    }

    ASSERT(array_idx < TEXTURE_COUNT_2D, "Error: invalid texture index\n");
    return texelFetch(sampler2D(textures_2d[array_idx], tex_sampler), coord, mipmap_level);
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
    return textureSize(textures_2d[array_idx], mipmap_level);
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

    // Calculate the perfect reflection vector
    vec3 r = reflect(gl_WorldRayDirectionEXT, normal);

    // Return if shading normal does not fall under geometric surface (angle between r and normal within 90 degrees)
    float a = dot(r, state.geom_normal);
    if (a >= 0.0)
    {
        return normal;
    }

    // Otherwise, bend normal (see above paper, A.3)
    float b = max(0.0001, dot(normal, state.geom_normal));

    vec3 tangent = normalize(r - (a / b) * normal);

    vec3 new_normal = normalize(-gl_WorldRayDirectionEXT + tangent);

    return new_normal;
}

#endif
