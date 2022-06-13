// See also: https://github.com/NVIDIA/MDL-SDK/blob/master/examples/mdl_sdk/dxr/content/mdl_renderer_runtime.hlsl

#ifdef HAS_TEXTURES

float4 tex_lookup_float4_3d(uint tex, float3 coord, int wrap_u, int wrap_v, int wrap_w, float2 crop_u, float2 crop_v, float2 crop_w, float frame)
{
    if (tex == 0)
    {
        return float4(0, 0, 0, 0);
    }

    float lod = 0.0;
    int2 offset = int2(0, 0);
    return textures[NonUniformResourceIndex(tex - 1)].SampleLevel(tex_sampler, coord.xy, lod, offset);
}

float3 tex_lookup_float3_3d(uint tex, float3 coord, int wrap_u, int wrap_v, int wrap_w, float2 crop_u, float2 crop_v, float2 crop_w, float frame)
{
    return tex_lookup_float4_3d(tex, coord, wrap_u, wrap_v, wrap_w, crop_u, crop_v, crop_w, frame).xyz;
}

float2 tex_lookup_float2_3d(uint tex, float3 coord, int wrap_u, int wrap_v, int wrap_w, float2 crop_u, float2 crop_v, float2 crop_w, float frame)
{
    return tex_lookup_float4_3d(tex, coord, wrap_u, wrap_v, wrap_w, crop_u, crop_v, crop_w, frame).xy;
}

float tex_lookup_float_3d(uint tex, float3 coord, int wrap_u, int wrap_v, int wrap_w, float2 crop_u, float2 crop_v, float2 crop_w, float frame)
{
    return tex_lookup_float4_3d(tex, coord, wrap_u, wrap_v, wrap_w, crop_u, crop_v, crop_w, frame).x;
}

float4 tex_lookup_float4_2d(uint tex, float2 coord, int wrap_u, int wrap_v, float2 crop_u, float2 crop_v, float frame)
{
    if (tex == 0)
    {
        return float4(0, 0, 0, 0);
    }

    float lod = 0.0;
    int2 offset = int2(0, 0);
    return textures[NonUniformResourceIndex(tex - 1)].SampleLevel(tex_sampler, coord.xy, lod, offset);
}

float3 tex_lookup_float3_2d(uint tex, float2 coord, int wrap_u, int wrap_v, float2 crop_u, float2 crop_v, float frame)
{
    return tex_lookup_float4_2d(tex, coord, wrap_u, wrap_v, crop_u, crop_v, frame).xyz;
}

float2 tex_lookup_float2_2d(uint tex, float2 coord, int wrap_u, int wrap_v, float2 crop_u, float2 crop_v, float frame)
{
    return tex_lookup_float4_2d(tex, coord, wrap_u, wrap_v, crop_u, crop_v, frame).xy;
}

float tex_lookup_float_2d(uint tex, float2 coord, int wrap_u, int wrap_v, float2 crop_u, float2 crop_v, float frame)
{
    return tex_lookup_float4_2d(tex, coord, wrap_u, wrap_v, crop_u, crop_v, frame).x;
}

float3 tex_lookup_color_3d(uint tex, float3 coord, int wrap_u, int wrap_v, int wrap_w, float2 crop_u, float2 crop_v, float2 crop_w, float frame)
{
    return tex_lookup_float3_3d(tex, coord, wrap_u, wrap_v, wrap_v, crop_u, crop_v, crop_w, frame);
}

float3 tex_lookup_color_2d(uint tex, float2 coord, int wrap_u, int wrap_v, float2 crop_u, float2 crop_v, float frame)
{
    return tex_lookup_float3_2d(tex, coord, wrap_u, wrap_v, crop_u, crop_v, frame);
}

float4 tex_texel_float4_3d(uint tex, int3 coord, float frame)
{
    if (tex == 0)
    {
        return float4(0, 0, 0, 0);
    }

    int mipmapLevel = 0;
    return textures[NonUniformResourceIndex(tex - 1)].Load(int3(coord.xy, mipmapLevel));
}

float3 tex_texel_float3_3d(uint tex, int3 coord, float frame)
{
    return tex_texel_float4_3d(tex, coord, frame).xyz;
}

float2 tex_texel_float2_3d(uint tex, int3 coord, float frame)
{
    return tex_texel_float4_3d(tex, coord, frame).xy;
}

float tex_texel_float_3d(uint tex, int3 coord, float frame)
{
    return tex_texel_float4_3d(tex, coord, frame).x;
}

float3 tex_texel_color_3d(uint tex, int3 coord, float frame)
{
    return tex_texel_float3_3d(tex, coord, frame);
}

float4 tex_texel_float4_2d(uint tex, int2 coord, int2 uv_tile, float frame)
{
    if (tex == 0)
    {
        return float4(0, 0, 0, 0);
    }

    int mipmapLevel = 0;
    return textures[NonUniformResourceIndex(tex - 1)].Load(int3(coord.xy, mipmapLevel));
}

float3 tex_texel_float3_2d(uint tex, int2 coord, int2 uv_tile, float frame)
{
    return tex_texel_float4_2d(tex, coord, uv_tile, frame).xyz;
}

float2 tex_texel_float2_2d(uint tex, int2 coord, int2 uv_tile, float frame)
{
    return tex_texel_float4_2d(tex, coord, uv_tile, frame).xy;
}

float tex_texel_float_2d(uint tex, int2 coord, int2 uv_tile, float frame)
{
    return tex_texel_float4_2d(tex, coord, uv_tile, frame).x;
}

float3 tex_texel_color_2d(uint tex, int2 coord, int2 uv_tile, float frame)
{
    return tex_texel_float3_2d(tex, coord, uv_tile, frame);
}

uint2 tex_resolution_2d(uint tex, int2 uv_tile, float frame)
{
    if (tex == 0)
    {
        return uint2(0, 0);
    }

    uint2 res = uint2(0, 0);
    textures[NonUniformResourceIndex(tex - 1)].GetDimensions(res.x, res.y);
    return res;
}

uint tex_width_2d(uint tex, int2 uv_tile, float frame)
{
    return tex_resolution_2d(tex, uv_tile, frame).x;
}

uint tex_height_2d(uint tex, int2 uv_tile, float frame)
{
    return tex_resolution_2d(tex, uv_tile, frame).y;
}

#endif
