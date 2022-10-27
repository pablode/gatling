#extension GL_GOOGLE_include_directive: require
#extension GL_EXT_shader_16bit_storage: require
#extension GL_EXT_samplerless_texture_functions: require
#extension GL_EXT_control_flow_attributes: require
#extension GL_KHR_shader_subgroup_ballot: require
#extension GL_EXT_shader_explicit_arithmetic_types_int16: require
#extension GL_EXT_ray_tracing: require
#extension GL_EXT_ray_query: require

#ifndef NDEBUG
#extension GL_EXT_debug_printf: enable
#define ASSERT(cond, str) if (!(cond)) debugPrintfEXT(str)
#else
#define ASSERT(cond, str)
#endif

#include "common.glsl"
#include "mdl_types.glsl"

layout(binding = 0, std430) buffer PixelsBuffer { vec4 pixels[]; };

layout(binding = 1, std430) readonly buffer FacesBuffer { face faces[]; };

#ifdef NEXT_EVENT_ESTIMATION
layout(binding = 2, std430) readonly buffer EmissiveFacesBuffer { uint emissive_face_indices[]; };
#endif

layout(binding = 3, std430) readonly buffer VerticesBuffer { fvertex vertices[]; };

#if defined(HAS_TEXTURES_2D) || defined(HAS_TEXTURES_3D)
layout(binding = 4) uniform sampler tex_sampler;

layout(binding = 5, std430) readonly buffer TexMappingsBuffer { uint16_t tex_mappings[]; };
#endif

#ifdef HAS_TEXTURES_2D
layout(binding = 6) uniform texture2D textures_2d[TEXTURE_COUNT_2D];
#endif

#ifdef HAS_TEXTURES_3D
layout(binding = 7) uniform texture3D textures_3d[TEXTURE_COUNT_3D];
#endif

layout(binding = 8) uniform accelerationStructureEXT sceneAS;

layout(push_constant) uniform PCBuffer {
    vec3  CAMERA_POSITION;
    uint  IMAGE_WIDTH;
    vec3  CAMERA_FORWARD;
    uint  IMAGE_HEIGHT;
    vec3  CAMERA_UP;
    float CAMERA_VFOV;
    vec4  BACKGROUND_COLOR;
    uint  SAMPLE_COUNT;
    uint  MAX_BOUNCES;
    float MAX_SAMPLE_VALUE;
    uint  RR_BOUNCE_OFFSET;
    float RR_INV_MIN_TERM_PROB;
    uint  SAMPLE_OFFSET;
} PC;

#include "mdl_interface.glsl"

void setup_mdl_shading_state(in uint hit_face_idx, in vec2 hit_bc, in vec3 ray_dir, out State state, out uint mat_idx)
{
  vec3 bc = vec3(1.0 - hit_bc.x - hit_bc.y, hit_bc.x, hit_bc.y);

  face f = faces[hit_face_idx];
  fvertex v_0 = vertices[f.v_0];
  fvertex v_1 = vertices[f.v_1];
  fvertex v_2 = vertices[f.v_2];

  vec3 p_0 = v_0.field1.xyz;
  vec3 p_1 = v_1.field1.xyz;
  vec3 p_2 = v_2.field1.xyz;
  vec3 pos = bc.x * p_0 + bc.y * p_1 + bc.z * p_2;
  vec3 geom_normal = normalize(cross(p_1 - p_0, p_2 - p_0));

  vec3 n_0 = v_0.field2.xyz;
  vec3 n_1 = v_1.field2.xyz;
  vec3 n_2 = v_2.field2.xyz;
  vec3 normal = normalize(bc.x * n_0 + bc.y * n_1 + bc.z * n_2);

  vec3 tangent, bitangent;
  orthonormal_basis(normal, tangent, bitangent);

  vec2 uv_0 = vec2(v_0.field1.w, v_0.field2.w);
  vec2 uv_1 = vec2(v_1.field1.w, v_1.field2.w);
  vec2 uv_2 = vec2(v_2.field1.w, v_2.field2.w);
  vec2 uv = bc.x * uv_0 + bc.y * uv_1 + bc.z * uv_2;

  if (dot(normal, ray_dir) > 0.0) { normal *= -1.0f; }
  if (dot(geom_normal, ray_dir) > 0.0) { geom_normal *= -1.0f; }

  state.normal = normal;
  state.geom_normal = geom_normal;
  state.position = pos;
  state.tangent_u[0] = tangent;
  state.tangent_v[0] = bitangent;
  state.animation_time = 0.0;
  state.text_coords[0] = vec3(uv, 0.0);

  mat_idx = f.mat_idx;
}

#pragma MDL_GENERATED_CODE

bool russian_roulette(in float random_float, inout vec3 throughput)
{
    float max_throughput = max(throughput.r, max(throughput.g, throughput.b));
    float p = min(max_throughput, PC.RR_INV_MIN_TERM_PROB);

    if (random_float > p)
    {
        return true;
    }

    throughput /= p;

    return false;
}

struct Sample_state
{
    uvec4 rng_state;
    vec3 ray_origin;
    vec3 ray_dir;
    vec3 throughput;
    vec3 radiance;
    bool inside;
};

vec3 evaluate_sample(inout uvec4 rng_state,
                     in float random_float,
                     in vec3 ray_origin,
                     in vec3 ray_dir)
{
    Sample_state sample_state;
    sample_state.ray_origin = ray_origin;
    sample_state.ray_dir    = ray_dir;
    sample_state.throughput = vec3(1.0, 1.0, 1.0);
    sample_state.radiance   = vec3(0.0, 0.0, 0.0);
    sample_state.inside     = false;

    uint bounce = 1;

    [[dont_unroll]]
    while (bounce <= PC.MAX_BOUNCES)
    {
        /* 1. Find hit. */
        RayInfo ray;
        ray.origin = sample_state.ray_origin;
        ray.tmax   = FLOAT_MAX;
        ray.dir    = sample_state.ray_dir;

        rayQueryEXT ray_query;
        rayQueryInitializeEXT(ray_query, sceneAS,
                              gl_RayFlagsOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT,
                              0xFF, ray.origin, 0.0, ray.dir, ray.tmax);

        rayQueryProceedEXT(ray_query);

        bool found_hit = (rayQueryGetIntersectionTypeEXT(ray_query, true) == gl_RayQueryCommittedIntersectionTriangleEXT);

        if (!found_hit)
        {
            sample_state.radiance += sample_state.throughput * PC.BACKGROUND_COLOR.rgb;
            break;
        }

        uint hit_face_idx = rayQueryGetIntersectionPrimitiveIndexEXT(ray_query, true);
        vec2 hit_bc = rayQueryGetIntersectionBarycentricsEXT(ray_query, true);

        /* 2. Set up shading state. */
        State shading_state; // Shading_state_material
        uint mat_idx;
        setup_mdl_shading_state(hit_face_idx, hit_bc, ray.dir, shading_state, mat_idx);

        // we keep a copy of the normal here since it can be changed within the state by *_init() functions:
        // https://github.com/NVIDIA/MDL-SDK/blob/aa9642b2546ad7b6236b5627385d882c2ed83c5d/examples/mdl_sdk/dxr/content/mdl_hit_programs.hlsl#L411
        const vec3 normal = shading_state.normal;

        const bool thin_walled = mdl_thin_walled(mat_idx, shading_state);
        const float ior1 = (sample_state.inside && !thin_walled) ? BSDF_USE_MATERIAL_IOR : 1.0;
        const float ior2 = (sample_state.inside && !thin_walled) ? 1.0 : BSDF_USE_MATERIAL_IOR;

#if AOV_ID == AOV_ID_DEBUG_NEE
        return vec3(0.0, 0.0, 0.0);
#elif AOV_ID == AOV_ID_NORMAL
        return (normal + vec3(1.0, 1.0, 1.0)) * 0.5;
#elif AOV_ID == AOV_ID_DEBUG_BARYCENTRICS
        return vec3(1.0 - hit_bc.x - hit_bc.y, hit_bc.x, hit_bc.y);
#elif AOV_ID == AOV_ID_DEBUG_TEXCOORDS
        return shading_state.text_coords[0];
#endif

        /* TODO: 3. apply volume attenuation */

        /* 4. Add Emission */
        {
            Edf_evaluate_data edf_evaluate_data;
            edf_evaluate_data.k1 = -sample_state.ray_dir;
            mdl_edf_emission_init(mat_idx, shading_state);
            mdl_edf_emission_evaluate(mat_idx, edf_evaluate_data, shading_state);

            vec3 emission_intensity = mdl_edf_emission_intensity(mat_idx, shading_state);

            sample_state.radiance += sample_state.throughput * edf_evaluate_data.edf * emission_intensity;
        }

        // reassign normal, see declaration of variable.
        shading_state.normal = normal;

        /* TODO 5. NEE light sampling */

        /* 6. Russian Roulette */
        // TODO: don't break here, need test NEE vis & add contrib - but don't do BSDF sampling
        if (bounce == PC.MAX_BOUNCES - 1)
        {
            break;
        }
        else if (bounce > PC.RR_BOUNCE_OFFSET)
        {
            if (russian_roulette(random_float, sample_state.throughput))
            {
                break;
            }
        }

        /* 7. BSDF (importance) sampling. */
        {
            Bsdf_sample_data bsdf_sample_data;
            bsdf_sample_data.ior1 = vec3(ior1);
            bsdf_sample_data.ior2 = vec3(ior2);
            bsdf_sample_data.k1 = -sample_state.ray_dir;
            bsdf_sample_data.xi = pcg4d_next(rng_state);
            mdl_bsdf_scattering_init(mat_idx, shading_state);
            mdl_bsdf_scattering_sample(mat_idx, bsdf_sample_data, shading_state);

            if (bsdf_sample_data.event_type == BSDF_EVENT_ABSORB)
            {
                // TODO: don't break here, need test NEE vis & add contrib
                break;
            }

            sample_state.throughput *= bsdf_sample_data.bsdf_over_pdf;

            bool is_transmission = (bsdf_sample_data.event_type & BSDF_EVENT_TRANSMISSION) != 0;

            if (is_transmission)
            {
                sample_state.inside = !sample_state.inside;
            }

            sample_state.ray_dir = bsdf_sample_data.k2;
            sample_state.ray_origin = offset_ray_origin(shading_state.position, shading_state.geom_normal * (is_transmission ? -1.0 : 1.0));
        }

        /* TODO 8. NEE shadow query */

        bounce++;
    }

#if AOV_ID == AOV_ID_DEBUG_BOUNCES
    return vec3(float(bounce), 0.0, 0.0);
#endif

    return clamp(sample_state.radiance, vec3(0.0, 0.0, 0.0), vec3(PC.MAX_SAMPLE_VALUE, PC.MAX_SAMPLE_VALUE, PC.MAX_SAMPLE_VALUE));
}

// Tracing rays in Morton order requires us to reorder the results for coalesced framebuffer memory accesses.
shared vec3 gs_reorder[NUM_THREADS_X * NUM_THREADS_Y];

layout(local_size_x = NUM_THREADS_X, local_size_y = NUM_THREADS_Y) in;

void main()
{
#if AOV_ID == AOV_ID_DEBUG_CLOCK_CYCLES
    uint64_t start_cycle_count = clockARB();
#endif

    // Remap to Morton order within workgroup.
    uint morton_code = uint(MORTON_2D_LUT_32x8[gl_LocalInvocationIndex]);
    uvec2 local_pixel_pos = uvec2(morton_code >> 8, morton_code & 0xFF);
    uvec2 group_base_pixel_pos = uvec2(gl_WorkGroupID.x * NUM_THREADS_X, gl_WorkGroupID.y * NUM_THREADS_Y);
    uvec2 pixel_pos = group_base_pixel_pos + local_pixel_pos;

    vec3 pixel_color = vec3(0.0, 0.0, 0.0);

    if (pixel_pos.x < PC.IMAGE_WIDTH && pixel_pos.y < PC.IMAGE_HEIGHT)
    {
        uint pixel_index = pixel_pos.x + pixel_pos.y * PC.IMAGE_WIDTH;

        vec3 camera_right = cross(PC.CAMERA_FORWARD, PC.CAMERA_UP);

        float aspect_ratio = float(PC.IMAGE_WIDTH) / float(PC.IMAGE_HEIGHT);

        float H = 1.0;
        float W = H * aspect_ratio;
        float d = H / (2.0 * tan(PC.CAMERA_VFOV * 0.5));

        float WX = W / float(PC.IMAGE_WIDTH);
        float HY = H / float(PC.IMAGE_HEIGHT);

        vec3 C = PC.CAMERA_POSITION + PC.CAMERA_FORWARD * d;
        vec3 L = C - camera_right * W * 0.5 - PC.CAMERA_UP * H * 0.5;

        float inv_sample_count = 1.0 / float(PC.SAMPLE_COUNT);

        for (uint s = 0; s < PC.SAMPLE_COUNT; ++s)
        {
            uvec4 rng_state = pcg4d_init(pixel_pos.xy, PC.SAMPLE_OFFSET + s);
            vec4 r = pcg4d_next(rng_state);

            vec3 P =
                L +
                (float(pixel_pos.x) + r.x) * camera_right * WX +
                (float(pixel_pos.y) + r.y) * PC.CAMERA_UP * HY;

            vec3 ray_origin = PC.CAMERA_POSITION;
            vec3 ray_direction = P - ray_origin;

            /* Beware: a single direction component must not be zero.
             * This is because we often take the inverse of the direction. */
            if (ray_direction.x == 0.0) ray_direction.x = FLOAT_MIN;
            if (ray_direction.y == 0.0) ray_direction.y = FLOAT_MIN;
            if (ray_direction.z == 0.0) ray_direction.z = FLOAT_MIN;

            ray_direction = normalize(ray_direction);

            /* Path trace sample and accumulate color. */
            vec3 sample_color = evaluate_sample(rng_state, r.z, ray_origin, ray_direction);
            pixel_color += sample_color * inv_sample_count;
        }
    }

#if AOV_ID == AOV_ID_DEBUG_CLOCK_CYCLES
    float cycles_elapsed_norm = float(clockARB() - start_cycle_count) / 4294967295.0;
    pixel_color = vec3(cycles_elapsed_norm, 0.0, 0.0);
#endif

    gs_reorder[gl_LocalInvocationIndex] = pixel_color;

    groupMemoryBarrier();
    barrier();

    uvec2 new_local_pixel_pos = gl_LocalInvocationID.xy;
    uvec2 new_pixel_pos = group_base_pixel_pos + new_local_pixel_pos;

    if (new_pixel_pos.x < PC.IMAGE_WIDTH && new_pixel_pos.y < PC.IMAGE_HEIGHT)
    {
        uint gs_idx = uint(MORTON_2D_LUT_32x8_REV[new_local_pixel_pos.x + new_local_pixel_pos.y * NUM_THREADS_X]);
        vec4 new_pixel_color = vec4(gs_reorder[gs_idx], 1.0);
        uint new_pixel_index = new_pixel_pos.x + new_pixel_pos.y * PC.IMAGE_WIDTH;

        if (PC.SAMPLE_OFFSET > 0)
        {
          float inv_total_sample_count = 1.0 / float(PC.SAMPLE_OFFSET + PC.SAMPLE_COUNT);

          float weight_old = float(PC.SAMPLE_OFFSET) * inv_total_sample_count;
          float weight_new = float(PC.SAMPLE_COUNT) * inv_total_sample_count;

          new_pixel_color = weight_old * pixels[new_pixel_index] + weight_new * new_pixel_color;
        }

        pixels[new_pixel_index] = new_pixel_color;
    }
}
