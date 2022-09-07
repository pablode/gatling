#extension GL_EXT_shader_16bit_storage: require
#extension GL_EXT_samplerless_texture_functions: require
#extension GL_EXT_control_flow_attributes: require
#extension GL_KHR_shader_subgroup_ballot: require
#extension GL_EXT_shader_explicit_arithmetic_types_int16: require

#ifndef NDEBUG
#extension GL_EXT_debug_printf: enable
#endif

#ifdef GL_EXT_debug_printf
#define ASSERT(cond, str) if (!(cond)) debugPrintfEXT(str)
#else
#define ASSERT(cond, str)
#endif

#include "common.glsl"
#include "mdl_types.glsl"

#pragma MDL_GENERATED_CODE

layout(binding = 0, std430) buffer PixelsBuffer { vec4 pixels[]; };

#ifdef BVH_ENABLED
layout(binding = 1, std430) readonly buffer BvhBuffer { bvh_node bvh_nodes[]; };
#endif

layout(binding = 2, std430) readonly buffer FacesBuffer { face faces[]; };

#ifdef NEXT_EVENT_ESTIMATION
layout(binding = 3, std430) readonly buffer EmissiveFacesBuffer { uint emissive_face_indices[]; };
#endif

layout(binding = 4, std430) readonly buffer VerticesBuffer { fvertex vertices[]; };

#if defined(HAS_TEXTURES_2D) || defined(HAS_TEXTURES_3D)
layout(binding = 5) uniform sampler tex_sampler;

layout(binding = 6, std430) readonly buffer TexMappingsBuffer { uint16_t tex_mappings[]; };
#endif

#ifdef HAS_TEXTURES_2D
layout(binding = 7) uniform texture2D textures_2d[TEXTURE_COUNT_2D];
#endif

#ifdef HAS_TEXTURES_3D
layout(binding = 8) uniform texture3D textures_3d[TEXTURE_COUNT_3D];
#endif

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

#include "intersection.glsl"
#include "mdl_interface.glsl"

struct Sample_state
{
    uvec4 rng_state;
    vec3 ray_origin;
    vec3 ray_dir;
    vec3 throughput;
    vec3 radiance;
    bool inside;
};

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

vec3 evaluate_sample(inout uvec4 rng_state,
                     in float random_float,
                     in vec3 ray_origin,
                     in vec3 ray_dir)
{
    Sample_state state;
    state.ray_origin = ray_origin;
    state.ray_dir    = ray_dir;
    state.throughput = vec3(1.0, 1.0, 1.0);
    state.radiance   = vec3(0.0, 0.0, 0.0);
    state.rng_state  = rng_state;
    state.inside     = false;

    uint bounce = 1;

    while (bounce <= PC.MAX_BOUNCES)
    {
        RayInfo ray;
        ray.origin = state.ray_origin;
        ray.tmax   = FLOAT_MAX;
        ray.dir    = state.ray_dir;

        Hit_info hit;

        bool found_hit = find_hit_closest(ray, hit);

#if AOV_ID == AOV_ID_DEBUG_BVH_STEPS
        return vec3(float(hit.bvh_steps), 0.0, 0.0);
#elif AOV_ID == AOV_ID_DEBUG_TRI_TESTS
        return vec3(0.0, 0.0, float(hit.tri_tests));
#endif

        if (!found_hit)
        {
            state.radiance += state.throughput * PC.BACKGROUND_COLOR.rgb;
            break;
        }

        /* Calculate hit point properties. */
        vec3 bc = vec3(1.0 - hit.bc.x - hit.bc.y, hit.bc.x, hit.bc.y);

#if AOV_ID == AOV_ID_DEBUG_BARYCENTRICS
        return bc;
#endif

        face f = faces[hit.face_idx];
        fvertex v_0 = vertices[f.v_0];
        fvertex v_1 = vertices[f.v_1];
        fvertex v_2 = vertices[f.v_2];

        vec3 p_0 = v_0.field1.xyz;
        vec3 p_1 = v_1.field1.xyz;
        vec3 p_2 = v_2.field1.xyz;

        vec3 n_0 = v_0.field2.xyz;
        vec3 n_1 = v_1.field2.xyz;
        vec3 n_2 = v_2.field2.xyz;

        vec2 uv_0 = vec2(v_0.field1.w, v_0.field2.w);
        vec2 uv_1 = vec2(v_1.field1.w, v_1.field2.w);
        vec2 uv_2 = vec2(v_2.field1.w, v_2.field2.w);

        vec3 hit_pos = bc.x * p_0 + bc.y * p_1 + bc.z * p_2;

        vec3 geom_normal = normalize(cross(p_1 - p_0, p_2 - p_0));
        if (dot(geom_normal, state.ray_dir) > 0.0) { geom_normal *= -1.0f; }

        vec3 normal = normalize(bc.x * n_0 + bc.y * n_1 + bc.z * n_2);
        if (dot(normal, state.ray_dir) > 0.0) { normal *= -1.0f; }

#if AOV_ID == AOV_ID_NORMAL
        return (normal + vec3(1.0, 1.0, 1.0)) * 0.5;
#endif

        vec2 uv = bc.x * uv_0 + bc.y * uv_1 + bc.z * uv_2;

#if AOV_ID == AOV_ID_DEBUG_TEXCOORDS
        return vec3(uv, 0.0);
#endif

        /* NEE */
#ifdef NEXT_EVENT_ESTIMATION
        {
            /* Sample light from global list. */
            vec4 random4 = pcg4d_next(state.rng_state);

            uint light_idx = min(EMISSIVE_FACE_COUNT - 1, uint(random4.x * float(EMISSIVE_FACE_COUNT)));
            uint lface_index = emissive_face_indices[light_idx];

            face fl = faces[lface_index];
            vec3 pl_0 = vertices[fl.v_0].field1.xyz;
            vec3 pl_1 = vertices[fl.v_1].field1.xyz;
            vec3 pl_2 = vertices[fl.v_2].field1.xyz;

            /* Sample point on light surface.
             * See: Ray Tracing Gems Chapter 16: Sampling Transformations Zoo 16.5.2.1 */
            float beta = 1.0 - sqrt(random4.y);
            float gamma = (1.0 - beta) * random4.z;
            float alpha = 1.0 - beta - gamma;
            vec3 P = alpha * pl_0 + beta * pl_1 + gamma * pl_2;

            vec3 to_light = P - hit_pos;
            vec3 lgeom_normal = normalize(cross(pl_1 - pl_0, pl_2 - pl_0));
            if (dot(lgeom_normal, to_light) > 0.0) { lgeom_normal *= -1.0f; }

            vec3 hit_offset = offset_ray_origin(hit_pos, geom_normal);
            vec3 light_offset = offset_ray_origin(P, lgeom_normal);

            to_light = light_offset - hit_offset;
            float light_dist = length(to_light);

            ray.origin = hit_offset;
            ray.dir = (light_offset - hit_offset) / light_dist;
            ray.tmax = light_dist;
            bool is_occluded = find_hit_any(ray);

#if AOV_ID == AOV_ID_DEBUG_NEE
            /* Occlusion debug visualization. */
            return is_occluded ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);
#endif
        }
#elif AOV_ID == AOV_ID_DEBUG_NEE
        return vec3(0.0, 0.0, 0.0);
#endif

        if (bounce == PC.MAX_BOUNCES)
        {
            break;
        }

        {
            /* Tangent and bitangent. */
            vec3 tangent, bitangent;
            orthonormal_basis(normal, tangent, bitangent);

            /* EDF evaluation. */
            State shading_state_material;
            shading_state_material.normal = normal;
            shading_state_material.geom_normal = geom_normal;
            shading_state_material.position = hit_pos;
            shading_state_material.tangent_u[0] = tangent;
            shading_state_material.tangent_v[0] = bitangent;
            shading_state_material.animation_time = 0.0;
            shading_state_material.text_coords[0] = vec3(uv, 0.0);

            Edf_evaluate_data edf_evaluate_data;
            edf_evaluate_data.k1 = -state.ray_dir;
            mdl_edf_emission_init(f.mat_idx, shading_state_material);
            mdl_edf_emission_evaluate(f.mat_idx, edf_evaluate_data, shading_state_material);

            vec3 emission_intensity = mdl_edf_emission_intensity(f.mat_idx, shading_state_material);
            bool thin_walled = mdl_thin_walled(f.mat_idx, shading_state_material);

            /* BSDF (importance) sampling. */

            /* Reassign normal - it can be changed by the _init() functions.
             * https://github.com/NVIDIA/MDL-SDK/blob/aa9642b2546ad7b6236b5627385d882c2ed83c5d/examples/mdl_sdk/dxr/content/mdl_hit_programs.hlsl#L411 */
            shading_state_material.normal = normal;

            Bsdf_sample_data bsdf_sample_data;
            bsdf_sample_data.ior1 = (state.inside && !thin_walled) ? vec3(BSDF_USE_MATERIAL_IOR) : vec3(1.0);
            bsdf_sample_data.ior2 = (state.inside && !thin_walled) ? vec3(1.0) : vec3(BSDF_USE_MATERIAL_IOR);
            bsdf_sample_data.k1 = -state.ray_dir;
            bsdf_sample_data.xi = pcg4d_next(state.rng_state);
            mdl_bsdf_scattering_init(f.mat_idx, shading_state_material);
            mdl_bsdf_scattering_sample(f.mat_idx, bsdf_sample_data, shading_state_material);

            /* Handle results. */
            state.radiance += state.throughput * edf_evaluate_data.edf * emission_intensity;

            if (bsdf_sample_data.event_type == BSDF_EVENT_ABSORB)
            {
                break;
            }

            state.throughput *= bsdf_sample_data.bsdf_over_pdf;

            /* Prepare next ray. */
            if ((bsdf_sample_data.event_type & BSDF_EVENT_TRANSMISSION) != 0)
            {
                state.inside = !state.inside;
            }

            state.ray_dir = bsdf_sample_data.k2;
            state.ray_origin = offset_ray_origin(hit_pos, geom_normal * (state.inside ? -1.0 : 1.0));
        }

        if (bounce >= PC.RR_BOUNCE_OFFSET)
        {
            if (russian_roulette(random_float, state.throughput))
            {
                break;
            }
        }

        bounce++;
    }

#if AOV_ID == AOV_ID_DEBUG_BOUNCES
    return vec3(1.0, 1.0, 1.0) * float(bounce);
#endif

    return clamp(state.radiance, vec3(0.0, 0.0, 0.0), vec3(PC.MAX_SAMPLE_VALUE, PC.MAX_SAMPLE_VALUE, PC.MAX_SAMPLE_VALUE));
}

// Tracing rays in Morton order requires us to reorder the results for coalesced framebuffer memory accesses.
shared vec3 gs_reorder[NUM_THREADS_X * NUM_THREADS_Y];

layout(local_size_x = NUM_THREADS_X, local_size_y = NUM_THREADS_Y) in;

void main()
{
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
