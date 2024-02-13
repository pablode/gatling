#ifndef MDL_SHADING_STATE
#define MDL_SHADING_STATE

layout(buffer_reference, std430, buffer_reference_align = 32/* largest type (see below) */) buffer IndexBuffer {
  BlasPayloadBufferPreamble preamble; // important: preamble size must match alignment
  Face data[];
};
layout(buffer_reference, std430, buffer_reference_align = 32/* largest type: vertex */) buffer VertexBuffer {
  BlasPayloadBufferPreamble preamble; // important: preamble size must match alignment
  FVertex data[];
};

void setup_mdl_shading_state(in vec2 hit_bc, out State state)
{
    BlasPayload payload = blas_payloads[gl_InstanceCustomIndexEXT];
    IndexBuffer indices = IndexBuffer(payload.bufferAddress);
    VertexBuffer vertices = VertexBuffer(payload.bufferAddress);

    Face f = indices.data[gl_PrimitiveID];
    uint vertexOffset = payload.vertexOffset;
    FVertex v_0 = vertices.data[vertexOffset + f.v_0];
    FVertex v_1 = vertices.data[vertexOffset + f.v_1];
    FVertex v_2 = vertices.data[vertexOffset + f.v_2];

    vec3 bc = vec3(1.0 - hit_bc.x - hit_bc.y, hit_bc.x, hit_bc.y);

    // Position and geometry normal
    vec3 p_0 = v_0.field1.xyz;
    vec3 p_1 = v_1.field1.xyz;
    vec3 p_2 = v_2.field1.xyz;

    vec3 localPos = bc.x * p_0 + bc.y * p_1 + bc.z * p_2;
    vec3 pos = vec3(gl_ObjectToWorldEXT * vec4(localPos, 1.0));

    vec3 geomNormal = normalize(cross(p_1 - p_0, p_2 - p_0));
    geomNormal = normalize(vec3(geomNormal * gl_WorldToObjectEXT));

    // Shading normal
    vec3 n_0 = decode_direction(floatBitsToUint(v_0.field2.x));
    vec3 n_1 = decode_direction(floatBitsToUint(v_1.field2.x));
    vec3 n_2 = decode_direction(floatBitsToUint(v_2.field2.x));

    vec3 localNormal = normalize(bc.x * n_0 + bc.y * n_1 + bc.z * n_2);
    vec3 normal = normalize(vec3(localNormal * gl_WorldToObjectEXT));

    // Flip normals to side of the incident ray
    if (dot(geomNormal, gl_WorldRayDirectionEXT) > 0.0)
    {
        geomNormal = -geomNormal;
        normal = -normal;
    }

    // Tangent and bitangent
    vec4 t_0 = vec4(decode_direction(floatBitsToUint(v_0.field2.y)), v_0.field1.w);
    vec4 t_1 = vec4(decode_direction(floatBitsToUint(v_1.field2.y)), v_1.field1.w);
    vec4 t_2 = vec4(decode_direction(floatBitsToUint(v_2.field2.y)), v_2.field1.w);

    vec3 localTangent = normalize(bc.x * t_0.xyz + bc.y * t_1.xyz + bc.z * t_2.xyz);
    vec3 tangent = normalize(vec3(gl_ObjectToWorldEXT * vec4(localTangent, 0.0)));
    // Re-orthonomalize to improve shading of surfaces with shared vertices
    // https://learnopengl.com/Advanced-Lighting/Normal-Mapping (bottom)
    tangent = normalize(tangent - dot(tangent, normal) * normal);

    float bitangentSign = bc.x * t_0.w + bc.y * t_1.w + bc.z * t_2.w;
    vec3 bitangent = cross(normal, tangent) * bitangentSign;

    // UV coordinates
    vec2 uv_0 = vec2(v_0.field2.z, v_0.field2.w);
    vec2 uv_1 = vec2(v_1.field2.z, v_1.field2.w);
    vec2 uv_2 = vec2(v_2.field2.z, v_2.field2.w);
    vec2 uv = bc.x * uv_0 + bc.y * uv_1 + bc.z * uv_2;

    // State
    state.normal = normal;
    state.geom_normal = geomNormal;
    state.position = pos;
    state.tangent_u[0] = tangent;
    state.tangent_v[0] = bitangent;
    state.animation_time = 0.0;
    state.text_coords[0] = vec3(uv, 0.0);
#ifdef SCENE_TRANSFORMS
    state.world_to_object = mat4(gl_WorldToObjectEXT);
    state.object_to_world = mat4(gl_ObjectToWorldEXT);
#endif
}

#endif
