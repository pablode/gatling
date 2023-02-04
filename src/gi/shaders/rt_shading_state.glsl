void setup_mdl_shading_state(in uint hit_face_idx, in vec2 hit_bc, in vec3 ray_dir, out State state)
{
    vec3 bc = vec3(1.0 - hit_bc.x - hit_bc.y, hit_bc.x, hit_bc.y);

    face f = faces[hit_face_idx];
    fvertex v_0 = vertices[f.v_0];
    fvertex v_1 = vertices[f.v_1];
    fvertex v_2 = vertices[f.v_2];

    // Position and geometry normal
    vec3 p_0 = v_0.field1.xyz;
    vec3 p_1 = v_1.field1.xyz;
    vec3 p_2 = v_2.field1.xyz;

    vec3 localPos = bc.x * p_0 + bc.y * p_1 + bc.z * p_2;
    vec3 pos = vec3(gl_ObjectToWorldEXT * vec4(localPos, 1.0));

    vec3 geomNormal = normalize(cross(p_1 - p_0, p_2 - p_0));
    geomNormal = normalize(vec3(geomNormal * gl_WorldToObjectEXT));

    // Shading normal
    vec3 n_0 = v_0.field2.xyz;
    vec3 n_1 = v_1.field2.xyz;
    vec3 n_2 = v_2.field2.xyz;

    vec3 localNormal = normalize(bc.x * n_0 + bc.y * n_1 + bc.z * n_2);
    vec3 normal = normalize(vec3(localNormal * gl_WorldToObjectEXT));

    // Flip normals to side of the incident ray
    if (dot(geomNormal, ray_dir) > 0.0)
    {
        geomNormal *= -1.0;
    }
    if (dot(normal, ray_dir) > 0.0)
    {
        normal *= -1.0;
    }

    // Ensure that geometry and shading normal have same sidedness
    if (dot(geomNormal, normal) < 0.0)
    {
        geomNormal = -geomNormal;
    }

    // Tangent and bitangent
    vec3 tangent, bitangent;
    orthonormal_basis(normal, tangent, bitangent);

    // UV coordinates
    vec2 uv_0 = vec2(v_0.field1.w, v_0.field2.w);
    vec2 uv_1 = vec2(v_1.field1.w, v_1.field2.w);
    vec2 uv_2 = vec2(v_2.field1.w, v_2.field2.w);
    vec2 uv = bc.x * uv_0 + bc.y * uv_1 + bc.z * uv_2;

    // State
    state.normal = normal;
    state.geom_normal = geomNormal;
    state.position = pos;
    state.tangent_u[0] = tangent;
    state.tangent_v[0] = bitangent;
    state.animation_time = 0.0;
    state.text_coords[0] = vec3(uv, 0.0);
}
