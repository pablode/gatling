#include "common.hlsl"
#include "bvh.hlsl"

struct PushConstants
{
  float3 CAMERA_POSITION;
  uint   IMAGE_WIDTH;
  float3 CAMERA_FORWARD;
  uint   IMAGE_HEIGHT;
  float3 CAMERA_UP;
  float  CAMERA_VFOV;
  float4 BACKGROUND_COLOR;
  uint   SAMPLE_COUNT;
  uint   MAX_BOUNCES;
  float  MAX_SAMPLE_VALUE;
  uint   RR_BOUNCE_OFFSET;
  float  RR_INV_MIN_TERM_PROB;
  uint   LIGHT_COUNT;
};

// Workaround, see https://github.com/KhronosGroup/glslang/issues/1629#issuecomment-703063873
#if defined(_DXC)
[[vk::push_constant]] PushConstants PC;
#else
[[vk::push_constant]] ConstantBuffer<PushConstants> PC;
#endif

struct Sample_state
{
    uint4 rng_state;
    float3 ray_origin;
    float3 ray_dir;
    float3 throughput;
    float3 value;
};

bool shade_hit(inout Sample_state state, in Hit_info hit)
{
    face f = faces[hit.face_idx];
    fvertex v_0 = vertices[f.v_0];
    fvertex v_1 = vertices[f.v_1];
    fvertex v_2 = vertices[f.v_2];

    /* Geometric normal. */
    float3 p_0 = v_0.field1.xyz;
    float3 p_1 = v_1.field1.xyz;
    float3 p_2 = v_2.field1.xyz;
    float3 geom_normal = normalize(cross(p_1 - p_0, p_2 - p_0));

    /* Shading normal. */
    float3 n_0 = v_0.field2.xyz;
    float3 n_1 = v_1.field2.xyz;
    float3 n_2 = v_2.field2.xyz;

    float2 bc = hit.bc;
    float3 normal = normalize(
        n_0 * (1.0 - bc.x - bc.y) +
        n_1 * bc.x +
        n_2 * bc.y
    );

#if AOV_ID == AOV_ID_NORMAL
    state.value = (normal + float3(1.0, 1.0, 1.0)) * 0.5;
    return false;
#endif

    /* Tangent and bitangent. */
    float3 L = abs(normal.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
    float3 tangent = normalize(cross(L, normal));
    float3 bitangent = cross(normal, tangent);

    /* Flip normals to side of incident ray. */
    if (dot(geom_normal, state.ray_dir) > 0.0)
    {
        geom_normal *= -1.0f;
    }
    if (dot(normal, state.ray_dir) > 0.0)
    {
        normal *= -1.0f;
    }

    /* EDF evaluation. */
    Shading_state_material shading_state_material;
    shading_state_material.normal = normal;
    shading_state_material.geom_normal = geom_normal;
    shading_state_material.position = hit.pos;
    shading_state_material.tangent_u[0] = tangent;
    shading_state_material.tangent_v[0] = bitangent;
    shading_state_material.animation_time = 0.0;

    Edf_evaluate_data edf_evaluate_data;
    edf_evaluate_data.k1 = -state.ray_dir;
    mdl_edf_emission_init(f.mat_idx, shading_state_material);
    mdl_edf_emission_evaluate(f.mat_idx, edf_evaluate_data, shading_state_material);

    float3 emission_intensity = mdl_edf_emission_intensity(f.mat_idx, shading_state_material);

    /* BSDF (importance) sampling. */
    Bsdf_sample_data bsdf_sample_data;
    bsdf_sample_data.ior1 = BSDF_USE_MATERIAL_IOR;
    bsdf_sample_data.ior2 = BSDF_USE_MATERIAL_IOR;
    bsdf_sample_data.k1 = -state.ray_dir;
    bsdf_sample_data.xi = pcg4d_next(state.rng_state);
    mdl_bsdf_scattering_init(f.mat_idx, shading_state_material);
    mdl_bsdf_scattering_sample(f.mat_idx, bsdf_sample_data, shading_state_material);

    /* Handle results. */
    state.value += state.throughput * edf_evaluate_data.edf * emission_intensity;

    if ((bsdf_sample_data.event_type & BSDF_EVENT_ABSORB) != 0)
    {
        return false;
    }

    state.throughput *= bsdf_sample_data.bsdf_over_pdf;

    /* Prepare next ray. */
    bool is_transmission = ((bsdf_sample_data.event_type & BSDF_EVENT_TRANSMISSION) != 0);

    state.ray_dir = bsdf_sample_data.k2;
    state.ray_origin = offset_ray_origin(hit.pos, geom_normal * (is_transmission ? -1.0 : 1.0));

    return true;
}

bool russian_roulette(in float random_float, inout float3 throughput)
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

float3 evaluate_sample(inout uint4 rng_state,
                       in float random_float,
                       in float3 ray_origin,
                       in float3 ray_dir)
{
    Sample_state state;
    state.ray_origin = ray_origin;
    state.ray_dir    = ray_dir;
    state.throughput = float3(1.0, 1.0, 1.0);
    state.value      = float3(0.0, 0.0, 0.0);
    state.rng_state  = rng_state;

    for (uint bounce = 0; bounce < (PC.MAX_BOUNCES + 1); bounce++)
    {
        RayInfo ray;
        ray.origin = state.ray_origin;
        ray.tmin   = 0.0;
        ray.dir    = state.ray_dir;
        ray.tmax   = FLOAT_MAX;

        Hit_info hit_info;

        bool found_hit = bvh_find_hit_closest(ray, hit_info);

#if AOV_ID == AOV_ID_DEBUG_BVH_STEPS
        state.value = float3(float(hit_info.bvh_steps), 0.0, 0.0);
        break;
#elif AOV_ID == AOV_ID_DEBUG_TRI_TESTS
        state.value = float3(0.0, 0.0, float(hit_info.tri_tests));
        break;
#endif

        if (!found_hit)
        {
            state.value += state.throughput * PC.BACKGROUND_COLOR.rgb;
            break;
        }

        /* NEE */
#ifdef NEXT_EVENT_ESTIMATION
        {
            /* Sample light from global list. */
            float4 random4 = pcg4d_next(state.rng_state);

            uint light_idx = min(PC.LIGHT_COUNT - 1, uint(random4.x * PC.LIGHT_COUNT));
            uint face_index = emissive_face_indices[light_idx];

            face f = faces[face_index];
            fvertex v0 = vertices[f.v_0];
            fvertex v1 = vertices[f.v_1];
            fvertex v2 = vertices[f.v_2];

            /* Sample point on light surface.
             * See: Ray Tracing Gems Chapter 16: Sampling Transformations Zoo 16.5.2.1 */
            float beta = 1.0 - sqrt(random4.y);
            float gamma = (1.0 - beta) * random4.z;
            float alpha = 1.0 - beta - gamma;
            float3 P = alpha * v0.field1.xyz + beta * v1.field1.xyz + gamma * v2.field1.xyz;

            float3 to_light = normalize(P - hit_info.pos);
            float3 hit_offset = offset_ray_origin(hit_info.pos, to_light);
            float3 light_offset = offset_ray_origin(P, -to_light);

            ray.origin = hit_offset;
            ray.dir = to_light;
            ray.tmin = 0.0;
            ray.tmax = length(light_offset - hit_offset);
            bool is_occluded = bvh_find_hit_any(ray);

            /* Occlusion debug visualization. */
#if AOV_ID == AOV_ID_DEBUG_NEE
            state.value = is_occluded ? float3(1.0, 0.0, 0.0) : float3(0.0, 1.0, 0.0);
            break;
#endif
        }
#elif AOV_ID == AOV_ID_DEBUG_NEE
        state.value = float3(0.0, 0.0, 0.0);
        break;
#endif

        bool continue_sampling = shade_hit(state, hit_info);

        if (!continue_sampling)
        {
            break;
        }

        if (bounce >= PC.RR_BOUNCE_OFFSET)
        {
            if (russian_roulette(random_float, state.throughput))
            {
                break;
            }
        }
    }

    float3 min_sample_value = float3(0.0, 0.0, 0.0);
    float3 max_sample_value = float3(PC.MAX_SAMPLE_VALUE, PC.MAX_SAMPLE_VALUE, PC.MAX_SAMPLE_VALUE);
    return clamp(state.value, min_sample_value, max_sample_value);
}

[numthreads(NUM_THREADS_X, NUM_THREADS_Y, 1)]
void CSMain(uint3 GlobalInvocationID : SV_DispatchThreadID)
{
    const uint2 pixel_pos = GlobalInvocationID.xy;

    if (pixel_pos.x >= PC.IMAGE_WIDTH ||
        pixel_pos.y >= PC.IMAGE_HEIGHT)
    {
        return;
    }

    const uint pixel_index = pixel_pos.x + pixel_pos.y * PC.IMAGE_WIDTH;

    float3 camera_right = cross(PC.CAMERA_FORWARD, PC.CAMERA_UP);

    const float aspect_ratio = float(PC.IMAGE_WIDTH) / float(PC.IMAGE_HEIGHT);

    const float H = 1.0;
    const float W = H * aspect_ratio;
    const float d = H / (2.0 * tan(PC.CAMERA_VFOV * 0.5));

    const float WX = W / float(PC.IMAGE_WIDTH);
    const float HY = H / float(PC.IMAGE_HEIGHT);

    const float3 C = PC.CAMERA_POSITION + PC.CAMERA_FORWARD * d;
    const float3 L = C - camera_right * W * 0.5 - PC.CAMERA_UP * H * 0.5;

    const float inv_sample_count = 1.0 / float(PC.SAMPLE_COUNT);

    float3 pixel_color = float3(0.0, 0.0, 0.0);

    for (uint s = 0; s < PC.SAMPLE_COUNT; ++s)
    {
        uint4 rng_state = pcg4d_init(pixel_pos.xy, s);
        float4 r = pcg4d_next(rng_state);

        const float3 P =
            L +
            (float(pixel_pos.x) + r.x) * camera_right * WX +
            (float(pixel_pos.y) + r.y) * PC.CAMERA_UP * HY;

        float3 ray_origin = PC.CAMERA_POSITION;
        float3 ray_direction = P - ray_origin;

        /* Beware: a single direction component must not be zero.
         * This is because we often take the inverse of the direction. */
        if (ray_direction.x == 0.0) ray_direction.x = FLOAT_MIN;
        if (ray_direction.y == 0.0) ray_direction.y = FLOAT_MIN;
        if (ray_direction.z == 0.0) ray_direction.z = FLOAT_MIN;

        ray_direction = normalize(ray_direction);

        /* Path trace sample and accumulate color. */
        const float3 sample_color = evaluate_sample(rng_state, r.z, ray_origin, ray_direction);
        pixel_color += sample_color * inv_sample_count;
    }

    pixels[pixel_index] = float4(pixel_color, 1.0);
}
