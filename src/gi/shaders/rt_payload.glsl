struct RayPayload
{
    uvec4 rng_state;
    vec3 ray_origin;
    vec3 ray_dir;
    vec3 throughput;
    vec3 radiance;
    // Bitfield values:
    // 1xxxxxxxxxxxxxxx bool inside
    // x111111111111111 uint bounce
    uint bitfield;
};
