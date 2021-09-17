#include "common.hlsl"

[[vk::constant_id(0)]] const uint IMAGE_WIDTH = 1920;
[[vk::constant_id(1)]] const uint IMAGE_HEIGHT = 1080;
[[vk::constant_id(2)]] const uint SAMPLE_COUNT = 4;
[[vk::constant_id(3)]] const uint MAX_BOUNCES = 4;
[[vk::constant_id(4)]] const uint MAX_STACK_SIZE = 6;
[[vk::constant_id(5)]] const float CAMERA_ORIGIN_X = 15.0;
[[vk::constant_id(6)]] const float CAMERA_ORIGIN_Y = 15.0;
[[vk::constant_id(7)]] const float CAMERA_ORIGIN_Z = 15.0;
[[vk::constant_id(8)]] const float CAMERA_FORWARD_X = 0.0;
[[vk::constant_id(9)]] const float CAMERA_FORWARD_Y = 4.0;
[[vk::constant_id(10)]] const float CAMERA_FORWARD_Z = 3.0;
[[vk::constant_id(11)]] const float CAMERA_UP_X = 0.0;
[[vk::constant_id(12)]] const float CAMERA_UP_Y = 4.0;
[[vk::constant_id(13)]] const float CAMERA_UP_Z = 3.0;
[[vk::constant_id(14)]] const float CAMERA_VFOV = 0.872665; // radians(50.0);
[[vk::constant_id(15)]] const uint RR_BOUNCE_OFFSET = 3;
[[vk::constant_id(16)]] const float RR_INV_MIN_TERM_PROB = 1.0;

#include "bvh.hlsl"

float3 uniform_sample_hemisphere(inout uint rng_state, float3 normal)
{
    const float r1 = random_float_between_0_and_1(rng_state);
    const float r2 = random_float_between_0_and_1(rng_state);

    const float3 u = abs(normal.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
    const float3 v = normalize(cross(u, normal));
    const float3 w = cross(normal, v);

    const float phi = 2.0 * PI * r2;
    const float sin_theta = sqrt(1.0 - r1 * r1);

    const float x = sin_theta * cos(phi);
    const float y = sin_theta * sin(phi);
    const float z = r1;

    return (x * w) + (y * v) + (z * normal);
}

float3 trace_sample(inout uint rng_state, in float3 prim_ray_origin, in float3 prim_ray_dir)
{
    float3 sample_color = float3(0.0, 0.0, 0.0);
    float3 throughput = float3(1.0, 1.0, 1.0);

    float3 ray_origin = prim_ray_origin;
    float3 ray_dir = prim_ray_dir;

    for (uint bounce = 0; bounce < (MAX_BOUNCES + 1); bounce++)
    {
        hit_info hit;

        const bool found_hit = traverse_bvh(ray_origin, ray_dir, hit);

        if (!found_hit)
        {
            break;
        }

        const face f = faces[hit.face_index];
        const float3 n0 = vertices[f.v_0].field2.xyz;
        const float3 n1 = vertices[f.v_1].field2.xyz;
        const float3 n2 = vertices[f.v_2].field2.xyz;
        const float2 bc = hit.bc;

        const float3 normal = normalize(
            n0 * (1.0 - bc.x - bc.y) +
            n1 * bc.x +
            n2 * bc.y
        );

        const material m = materials[f.mat_index];

        sample_color += throughput * m.emission;

        const float PDF = 1.0 / (2.0 * PI);

        ray_dir = uniform_sample_hemisphere(rng_state, normal);

        throughput *=
            (m.albedo.rgb / PI) *
            (abs(dot(normal, ray_dir)) / PDF);

        if (bounce >= RR_BOUNCE_OFFSET)
        {
          const float p = min(max(throughput.r, max(throughput.g, throughput.b)), RR_INV_MIN_TERM_PROB);

          const float r = random_float_between_0_and_1(rng_state);

          if (r > p)
          {
            break;
          }

          throughput /= p;
        }

        ray_origin = hit.pos + normal * RAY_OFFSET_EPS;
    }

    return sample_color;
}

[numthreads(32, 32, 1)] // TODO: set using define
void CSMain(uint3 GlobalInvocationID : SV_DispatchThreadID)
{
    const uint2 pixel_pos = GlobalInvocationID.xy;

    if (pixel_pos.x >= IMAGE_WIDTH ||
        pixel_pos.y >= IMAGE_HEIGHT)
    {
        return;
    }

    const uint pixel_index = pixel_pos.x + pixel_pos.y * IMAGE_WIDTH;

    float3 camera_origin = float3(CAMERA_ORIGIN_X, CAMERA_ORIGIN_Y, CAMERA_ORIGIN_Z);
    float3 camera_forward = normalize(float3(CAMERA_FORWARD_X, CAMERA_FORWARD_Y, CAMERA_FORWARD_Z));
    float3 camera_up = normalize(float3(CAMERA_UP_X, CAMERA_UP_Y, CAMERA_UP_Z));
    const float3 camera_right = cross(camera_forward, camera_up);

    const float aspect_ratio = float(IMAGE_WIDTH) / float(IMAGE_HEIGHT);

    const float H = 1.0;
    const float W = H * aspect_ratio;
    const float d = H / (2.0 * tan(CAMERA_VFOV * 0.5));

    const float WX = W / float(IMAGE_WIDTH);
    const float HY = H / float(IMAGE_HEIGHT);

    const float3 C = camera_origin + camera_forward * d;
    const float3 L = C - camera_right * W * 0.5 - camera_up * H * 0.5;

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
            (float(pixel_pos.y) + r2) * camera_up * HY;

        float3 ray_origin = camera_origin;
        float3 ray_direction = P - camera_origin;

        /* Beware: a single direction component must not be zero.
         * This is because we often take the inverse of the direction. */
        if (ray_direction.x == 0.0) ray_direction.x = FLOAT_MIN;
        if (ray_direction.y == 0.0) ray_direction.y = FLOAT_MIN;
        if (ray_direction.z == 0.0) ray_direction.z = FLOAT_MIN;

        ray_direction = normalize(ray_direction);

        /* Path trace sample and accumulate color. */
        const float3 sample_color = trace_sample(rng_state, ray_origin, ray_direction);
        pixel_color += sample_color * inv_sample_count;
    }

    pixels[pixel_index] = float4(pixel_color, 1.0);
}
