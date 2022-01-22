/* Möller-Trumbore triangle intersection. */
bool test_face(
    in float3 ray_origin,
    in float3 ray_dir,
    in float t_max,
    in uint face_index,
    out float t,
    out float2 bc)
{
    face f = faces[face_index];
    float3 p0 = vertices[f.v_0].field1.xyz;
    float3 p1 = vertices[f.v_1].field1.xyz;
    float3 p2 = vertices[f.v_2].field1.xyz;
    float3 e1 = p1 - p0;
    float3 e2 = p2 - p0;

    float3 p = cross(ray_dir, e2);
    float det = dot(e1, p);

    if (abs(det) < TRI_EPS) {
        return false;
    }

    float inv_det = 1.0 / det;
    float3 tvec = ray_origin - p0;

    bc.x = dot(tvec, p) * inv_det;
    if (bc.x < 0.0 || bc.x > 1.0) {
        return false;
    }

    float3 q = cross(tvec, e1);
    bc.y = dot(ray_dir, q) * inv_det;
    if (bc.y < 0.0 || (bc.x + bc.y > 1.0)) {
        return false;
    }

    t = dot(e2, q) * inv_det;

    if (t <= 0.0 || t >= t_max) {
        return false;
    }

    return true;
}

uint sign_to_byte_mask4(uint a)
{
    a = a & 0x80808080;
    a = a + a - (a >> 7);
    return a;
}

uint extract_byte(uint num, uint byte_idx)
{
    return (num >> (byte_idx * 8)) & 0xFF;
}

float4 uint_unpack_float4(uint u)
{
  return float4(extract_byte(u, 0), extract_byte(u, 1), extract_byte(u, 2), extract_byte(u, 3));
}

bool bvh_find_hit_closest(in RayInfo ray, out Hit_info hit)
{
    float t_min = ray.tmin;
    float t_max = ray.tmax;
    float3 inv_dir = 1.0 / ray.dir;

    uint3 oct_inv = uint3(
        ray.dir.x >= 0.0 ? 0 : 0x4,
        ray.dir.y >= 0.0 ? 0 : 0x2,
        ray.dir.z >= 0.0 ? 0 : 0x1
    );
    uint oct_inv4 = (oct_inv.x | oct_inv.y | oct_inv.z) * 0x01010101;

    uint2 node_group = uint2(0, 0x80000000);

    uint2 stack[MAX_STACK_SIZE];
    uint stack_size = 0;

    while (true)
    {
        uint2 face_group = uint2(0, 0);

        if (node_group.y <= 0x00FFFFFF)
        {
            face_group = node_group;
            node_group = uint2(0, 0);
        }
        else
        {
            uint child_bit_idx = firstbithigh(node_group.y);
            uint slot_index = (child_bit_idx - 24) ^ (oct_inv4 & 0xFF);
            uint rel_idx = countbits(node_group.y & ~(0xFFFFFFFFu << slot_index));
            uint child_node_idx = node_group.x + rel_idx;

            node_group.y &= ~(1u << child_bit_idx);

            if (node_group.y > 0x00FFFFFF)
            {
#ifdef DEBUG_PRINTF
                if (stack_size >= MAX_STACK_SIZE)
                    printf("Error: traversal stack size exceeds maximum capacity\n");
#endif
                stack[stack_size] = node_group;
                stack_size++;
            }

            bvh_node node = bvh_nodes[child_node_idx];

            node_group.x = node.f2.x;
            face_group = uint2(node.f2.y, 0);

            uint3 node_e = uint3(extract_byte(node.f1.w, 0), extract_byte(node.f1.w, 1), extract_byte(node.f1.w, 2));
            float3 local_inv_dir = asfloat(node_e << 23) * inv_dir;
            float3 p = asfloat(node.f1.xyz);
            float3 local_orig = (p - ray.origin) * inv_dir;

            uint hitmask = 0;

            [unroll(2)]
            for (uint passIdx = 0; passIdx < 2; ++passIdx)
            {
                uint meta4 = (passIdx == 0) ? node.f2.z : node.f2.w;
                uint is_inner4 = (meta4 & (meta4 << 1)) & 0x10101010;
                uint inner_mask4 = sign_to_byte_mask4(is_inner4 << 3);
                uint bit_index4 = (meta4 ^ (oct_inv4 & inner_mask4)) & 0x1F1F1F1F;
                uint child_bits4 = (meta4 >> 5) & 0x07070707;

                bool x_lt_0 = (inv_dir.x < 0.0);
                bool y_lt_0 = (inv_dir.y < 0.0);
                bool z_lt_0 = (inv_dir.z < 0.0);

                uint q_lo_x = (passIdx == 0) ? node.f3.x : node.f3.y;
                uint q_hi_x = (passIdx == 0) ? node.f4.z : node.f4.w;
                uint q_lo_y = (passIdx == 0) ? node.f3.z : node.f3.w;
                uint q_hi_y = (passIdx == 0) ? node.f5.x : node.f5.y;
                uint q_lo_z = (passIdx == 0) ? node.f4.x : node.f4.y;
                uint q_hi_z = (passIdx == 0) ? node.f5.z : node.f5.w;

                float4 s_q_lo_x = uint_unpack_float4(x_lt_0 ? q_hi_x : q_lo_x);
                float4 s_q_hi_x = uint_unpack_float4(x_lt_0 ? q_lo_x : q_hi_x);
                float4 s_q_lo_y = uint_unpack_float4(y_lt_0 ? q_hi_y : q_lo_y);
                float4 s_q_hi_y = uint_unpack_float4(y_lt_0 ? q_lo_y : q_hi_y);
                float4 s_q_lo_z = uint_unpack_float4(z_lt_0 ? q_hi_z : q_lo_z);
                float4 s_q_hi_z = uint_unpack_float4(z_lt_0 ? q_lo_z : q_hi_z);

                float4 t_min_x = local_inv_dir.x * s_q_lo_x + local_orig.x;
                float4 t_max_x = local_inv_dir.x * s_q_hi_x + local_orig.x;
                float4 t_min_y = local_inv_dir.y * s_q_lo_y + local_orig.y;
                float4 t_max_y = local_inv_dir.y * s_q_hi_y + local_orig.y;
                float4 t_min_z = local_inv_dir.z * s_q_lo_z + local_orig.z;
                float4 t_max_z = local_inv_dir.z * s_q_hi_z + local_orig.z;

                [unroll(4)]
                for (uint child_idx = 0; child_idx < 4; ++child_idx)
                {
                    float bmin = max(max(t_min_x[child_idx], t_min_y[child_idx]), max(t_min_z[child_idx], t_min));
                    float bmax = min(min(t_max_x[child_idx], t_max_y[child_idx]), min(t_max_z[child_idx], t_max));

                    bool is_intersected = bmin <= bmax;

                    if (!is_intersected)
                    {
                        continue;
                    }

                    uint child_bits = extract_byte(child_bits4, child_idx);
                    uint bit_index = extract_byte(bit_index4, child_idx);
                    hitmask |= (child_bits << bit_index);
                }
            }

            uint node_imask = extract_byte(node.f1.w, 3);
            node_group.y = (hitmask & 0xFF000000) | node_imask;
            face_group.y = (hitmask & 0x00FFFFFF);
        }

        uint active_inv_count1 = WaveActiveCountBits(true);

        while (face_group.y != 0)
        {
            uint threshold = uint(active_inv_count1 * POSTPONE_RATIO);

            uint active_inv_count2 = WaveActiveCountBits(true);

            if (active_inv_count2 < threshold)
            {
#ifdef DEBUG_PRINTF
                if (stack_size >= MAX_STACK_SIZE)
                    printf("Error: traversal stack size exceeds maximum capacity\n");
#endif
                stack[stack_size] = face_group;
                stack_size++;
                break;
            }

            uint face_rel_index = firstbithigh(face_group.y);

            face_group.y &= ~(1u << face_rel_index);

            uint face_index = face_group.x + face_rel_index;

            float temp_t;
            float2 temp_bc;

            bool has_hit = test_face(
                ray.origin,
                ray.dir,
                t_max,
                face_index,
                temp_t,
                temp_bc
            );

            if (has_hit)
            {
                t_max = temp_t;
                hit.bc = temp_bc;
                hit.face_idx = face_index;
            }
        }

        if (node_group.y > 0x00FFFFFF)
        {
            continue;
        }

        if (stack_size > 0)
        {
            stack_size--;
            node_group = stack[stack_size];
            continue;
        }

        if (t_max != FLOAT_MAX)
        {
            hit.pos = ray.origin + ray.dir * t_max;
            return true;
        }

        return false;
    }
}
