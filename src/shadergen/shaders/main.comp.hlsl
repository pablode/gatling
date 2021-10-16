#include "common.hlsl"
#include "bvh.hlsl"

struct Sample_state
{
    float3 ray_origin;
    float3 ray_dir;
    float3 throughput;
    float3 value;
    uint rng_state;
};

bool shade_hit(inout Sample_state state,
               in Hit_info hit)
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
    bsdf_sample_data.xi.x = random_float_between_0_and_1(state.rng_state);
    bsdf_sample_data.xi.y = random_float_between_0_and_1(state.rng_state);
    bsdf_sample_data.xi.z = random_float_between_0_and_1(state.rng_state);
    bsdf_sample_data.xi.w = random_float_between_0_and_1(state.rng_state);
    mdl_bsdf_scattering_init(f.mat_idx, shading_state_material);
    mdl_bsdf_scattering_sample(f.mat_idx, bsdf_sample_data, shading_state_material);

    state.value += state.throughput * edf_evaluate_data.edf * emission_intensity;

    if ((bsdf_sample_data.event_type & BSDF_EVENT_ABSORB) != 0)
    {
        return false;
    }

    /* Prepare state for next ray. */
    bool is_transmission = ((bsdf_sample_data.event_type & BSDF_EVENT_TRANSMISSION) != 0);

    state.ray_dir = bsdf_sample_data.k2;
    state.ray_origin = hit.pos + geom_normal * RAY_OFFSET_EPS * (is_transmission ? -1.0 : 1.0);

    state.throughput *= bsdf_sample_data.bsdf_over_pdf;

    return true;
}

bool russian_roulette(inout uint rng_state, inout float3 throughput)
{
    float r = random_float_between_0_and_1(rng_state);

    float max_throughput = max(throughput.r, max(throughput.g, throughput.b));
    float p = min(max_throughput, RR_INV_MIN_TERM_PROB);

    if (r > p)
    {
      return true;
    }

    throughput /= p;

    return false;
}

float3 evaluate_sample(inout uint rng_state,
                       in float3 ray_origin,
                       in float3 ray_dir)
{
    Sample_state state;
    state.ray_origin = ray_origin;
    state.ray_dir = ray_dir;
    state.throughput = float3(1.0, 1.0, 1.0);
    state.value = float3(0.0, 0.0, 0.0);
    state.rng_state = rng_state;

    for (uint bounce = 0; bounce < (MAX_BOUNCES + 1); bounce++)
    {
        Hit_info hit_info;

        bool found_hit = traverse_bvh(state.ray_origin,
                                      state.ray_dir,
                                      hit_info);

        if (!found_hit)
        {
            break;
        }

        bool continue_sampling = shade_hit(state, hit_info);

        if (!continue_sampling)
        {
            break;
        }

        if (bounce >= RR_BOUNCE_OFFSET)
        {
            if (russian_roulette(state.rng_state,
                                 state.throughput))
            {
                break;
            }
        }
    }

    float3 min_sample_value = float3(0.0, 0.0, 0.0);
    float3 max_sample_value = float3(MAX_SAMPLE_VALUE, MAX_SAMPLE_VALUE, MAX_SAMPLE_VALUE);
    return clamp(state.value, min_sample_value, max_sample_value);
}

struct PushConstants
{
  float3 CAMERA_POSITION;
  uint   IMAGE_WIDTH;
  float3 CAMERA_FORWARD;
  uint   IMAGE_HEIGHT;
  float3 CAMERA_UP;
  float  CAMERA_VFOV;
};

// Workaround, see https://github.com/KhronosGroup/glslang/issues/1629#issuecomment-703063873
#if defined(_DXC)
[[vk::push_constant]] PushConstants PC;
#else
[[vk::push_constant]] ConstantBuffer<PushConstants> PC;
#endif

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

    const float inv_sample_count = 1.0 / float(SAMPLE_COUNT);

    uint rng_state = wang_hash(pixel_index);

    float3 pixel_color = float3(0.0, 0.0, 0.0);

    for (uint s = 0; s < SAMPLE_COUNT; ++s)
    {
        const float r1 = random_float_between_0_and_1(rng_state);
        const float r2 = random_float_between_0_and_1(rng_state);

        const float3 P =
            L +
            (float(pixel_pos.x) + r1) * camera_right * WX +
            (float(pixel_pos.y) + r2) * PC.CAMERA_UP * HY;

        float3 ray_origin = PC.CAMERA_POSITION;
        float3 ray_direction = P - ray_origin;

        /* Beware: a single direction component must not be zero.
         * This is because we often take the inverse of the direction. */
        if (ray_direction.x == 0.0) ray_direction.x = FLOAT_MIN;
        if (ray_direction.y == 0.0) ray_direction.y = FLOAT_MIN;
        if (ray_direction.z == 0.0) ray_direction.z = FLOAT_MIN;

        ray_direction = normalize(ray_direction);

        /* Path trace sample and accumulate color. */
        const float3 sample_color = evaluate_sample(rng_state, ray_origin, ray_direction);
        pixel_color += sample_color * inv_sample_count;
    }

    pixels[pixel_index] = float4(pixel_color, 1.0);
}
