#include "common.hlsl"
#include "mdl_types.hlsl"

[[vk::binding(0)]]
RWStructuredBuffer<float4> pixels;

#ifdef BVH_ENABLED
[[vk::binding(1)]]
StructuredBuffer<bvh_node> bvh_nodes;
#endif

[[vk::binding(2)]]
StructuredBuffer<face> faces;

#ifdef NEXT_EVENT_ESTIMATION
[[vk::binding(3)]]
StructuredBuffer<uint> emissive_face_indices;
#endif

[[vk::binding(4)]]
StructuredBuffer<fvertex> vertices;

#ifdef HAS_TEXTURES
[[vk::binding(5)]]
Texture2D textures[TEXTURE_COUNT];

[[vk::binding(6)]]
SamplerState tex_sampler;
#endif

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
};

// Workaround, see https://github.com/KhronosGroup/glslang/issues/1629#issuecomment-703063873
#if defined(_DXC)
[[vk::push_constant]] PushConstants PC;
#else
[[vk::push_constant]] ConstantBuffer<PushConstants> PC;
#endif

#include "mdl_interface.hlsl"

#include "intersection.hlsl"

MDL_GENERATED_CODE

struct Sample_state
{
    uint4 rng_state;
    float3 ray_origin;
    float3 ray_dir;
    float3 throughput;
    float3 radiance;
};

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
    state.radiance   = float3(0.0, 0.0, 0.0);
    state.rng_state  = rng_state;

    for (uint bounce = 1; bounce <= PC.MAX_BOUNCES; bounce++)
    {
        RayInfo ray;
        ray.origin = state.ray_origin;
        ray.tmax   = FLOAT_MAX;
        ray.dir    = state.ray_dir;

        Hit_info hit;

        bool found_hit = find_hit_closest(ray, hit);

#if AOV_ID == AOV_ID_DEBUG_BVH_STEPS
        state.radiance = float3(float(hit.bvh_steps), 0.0, 0.0);
        break;
#elif AOV_ID == AOV_ID_DEBUG_TRI_TESTS
        state.radiance = float3(0.0, 0.0, float(hit.tri_tests));
        break;
#endif

        if (!found_hit)
        {
            state.radiance += state.throughput * PC.BACKGROUND_COLOR.rgb;
            break;
        }

        /* Calculate hit point properties. */
        float3 bc = float3(1.0 - hit.bc.x - hit.bc.y, hit.bc.x, hit.bc.y);

        face f = faces[hit.face_idx];
        fvertex v_0 = vertices[f.v_0];
        fvertex v_1 = vertices[f.v_1];
        fvertex v_2 = vertices[f.v_2];

        float3 p_0 = v_0.field1.xyz;
        float3 p_1 = v_1.field1.xyz;
        float3 p_2 = v_2.field1.xyz;

        float3 n_0 = v_0.field2.xyz;
        float3 n_1 = v_1.field2.xyz;
        float3 n_2 = v_2.field2.xyz;

        float3 hit_pos = bc.x * p_0 + bc.y * p_1 + bc.z * p_2;

        float3 geom_normal = normalize(cross(p_1 - p_0, p_2 - p_0));
        if (dot(geom_normal, state.ray_dir) > 0.0) { geom_normal *= -1.0f; }

        float3 normal = normalize(bc.x * n_0 + bc.y * n_1 + bc.z * n_2);
        if (dot(normal, state.ray_dir) > 0.0) { normal *= -1.0f; }

#if AOV_ID == AOV_ID_NORMAL
        state.radiance = (normal + float3(1.0, 1.0, 1.0)) * 0.5;
        break;
#endif

        /* NEE */
#ifdef NEXT_EVENT_ESTIMATION
        {
            /* Sample light from global list. */
            float4 random4 = pcg4d_next(state.rng_state);

            uint light_idx = min(EMISSIVE_FACE_COUNT - 1, uint(random4.x * float(EMISSIVE_FACE_COUNT)));
            uint lface_index = emissive_face_indices[light_idx];

            face fl = faces[lface_index];
            float3 pl_0 = vertices[fl.v_0].field1.xyz;
            float3 pl_1 = vertices[fl.v_1].field1.xyz;
            float3 pl_2 = vertices[fl.v_2].field1.xyz;

            /* Sample point on light surface.
             * See: Ray Tracing Gems Chapter 16: Sampling Transformations Zoo 16.5.2.1 */
            float beta = 1.0 - sqrt(random4.y);
            float gamma = (1.0 - beta) * random4.z;
            float alpha = 1.0 - beta - gamma;
            float3 P = alpha * pl_0 + beta * pl_1 + gamma * pl_2;

            float3 to_light = P - hit_pos;
            float3 lgeom_normal = normalize(cross(pl_1 - pl_0, pl_2 - pl_0));
            if (dot(lgeom_normal, to_light) > 0.0) { lgeom_normal *= -1.0f; }

            float3 hit_offset = offset_ray_origin(hit_pos, geom_normal);
            float3 light_offset = offset_ray_origin(P, lgeom_normal);

            to_light = light_offset - hit_offset;
            float light_dist = length(to_light);

            ray.origin = hit_offset;
            ray.dir = (light_offset - hit_offset) / light_dist;
            ray.tmax = light_dist;
            bool is_occluded = find_hit_any(ray);

#if AOV_ID == AOV_ID_DEBUG_NEE
            /* Occlusion debug visualization. */
            state.radiance = is_occluded ? float3(1.0, 0.0, 0.0) : float3(0.0, 1.0, 0.0);
            break;
#endif
        }
#endif

        if (bounce == PC.MAX_BOUNCES)
        {
            break;
        }

        {
            /* Tangent and bitangent. */
            float3 L = abs(normal.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
            float3 tangent = normalize(cross(L, normal));
            float3 bitangent = cross(normal, tangent);

            /* EDF evaluation. */
            Shading_state_material shading_state_material;
            shading_state_material.normal = normal;
            shading_state_material.geom_normal = geom_normal;
            shading_state_material.position = hit_pos;
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
            state.radiance += state.throughput * edf_evaluate_data.edf * emission_intensity;

            if ((bsdf_sample_data.event_type & BSDF_EVENT_ABSORB) != 0)
            {
                break;
            }

            state.throughput *= bsdf_sample_data.bsdf_over_pdf;

            /* Prepare next ray. */
            bool is_transmission = ((bsdf_sample_data.event_type & BSDF_EVENT_TRANSMISSION) != 0);

            state.ray_dir = bsdf_sample_data.k2;
            state.ray_origin = offset_ray_origin(hit_pos, geom_normal * (is_transmission ? -1.0 : 1.0));
        }

        if (bounce >= PC.RR_BOUNCE_OFFSET)
        {
            if (russian_roulette(random_float, state.throughput))
            {
                break;
            }
        }
    }

    return clamp(state.radiance, float3(0.0, 0.0, 0.0), float3(PC.MAX_SAMPLE_VALUE, PC.MAX_SAMPLE_VALUE, PC.MAX_SAMPLE_VALUE));
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
