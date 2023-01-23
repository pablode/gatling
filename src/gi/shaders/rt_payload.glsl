struct RayPayload
{
    uvec4 rng_state;
    vec3 ray_origin;
    uint bounce;
    vec3 ray_dir;
    bool inside;
    vec3 throughput;
    float pad0;
    vec3 radiance;
    float pad1;
};
