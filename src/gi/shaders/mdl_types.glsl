#ifndef H_MDL_TYPES
#define H_MDL_TYPES

/******************************************************************************
 * Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *****************************************************************************/

struct State // Shading_state_material
{
#ifdef SCENE_TRANSFORMS
    /// A 4x4 transformation matrix transforming from world to object coordinates.
    /// It is used by the state::transform_*() methods.
    /// This field is only used if the uniform state is included.
    mat4                world_to_object;

    /// A 4x4 transformation matrix transforming from object to world coordinates.
    /// It is used by the state::transform_*() methods.
    /// This field is only used if the uniform state is included.
    mat4                object_to_world;
#endif

    /// The result of state::normal().
    /// It represents the shading normal as determined by the renderer.
    /// This field will be updated to the result of \c "geometry.normal" by BSDF init functions,
    /// if requested during code generation.
    vec3                normal;

    /// The result of state::geometry_normal().
    /// It represents the geometry normal as determined by the renderer.
    vec3                geom_normal;

    /// The result of state::position().
    /// It represents the position where the material should be evaluated.
    vec3                position;

    /// The result of state::animation_time().
    /// It represents the time of the current sample in seconds.
    float               animation_time;

    /// An array containing the results of state::texture_coordinate(i).
    /// The i-th entry represents the texture coordinates of the i-th texture space at the
    /// current position.
    vec3                text_coords[1];

    /// An array containing the results of state::texture_tangent_u(i).
    /// The i-th entry represents the texture tangent vector of the i-th texture space at the
    /// current position, which points in the direction of the projection of the tangent to the
    /// positive u axis of this texture space onto the plane defined by the original
    /// surface normal.
    vec3                tangent_u[1];

    /// An array containing the results of state::texture_tangent_v(i).
    /// The i-th entry represents the texture bitangent vector of the i-th texture space at the
    /// current position, which points in the general direction of the positive v axis of this
    /// texture space, but is orthogonal to both the original surface normal and the tangent
    /// of this texture space.
    vec3                tangent_v[1];

#if 0
    /// An offset for accesses to the read-only data segment. Will be added before
    /// calling any "mdl_read_rodata_as_*" function.
    /// The data of the read-only data segment is accessible as the first segment
    /// (index 0) returned by #mi::neuraylib::ITarget_code::get_ro_data_segment_data().
    uint                ro_data_segment_offset;

    /// The result of state::object_id().
    /// It is an application-specific identifier of the hit object as provided in a scene.
    /// It can be used to make instanced objects look different in spite of the same used material.
    /// This field is only used if the uniform state is included.
    uint                object_id;

    /// The result of state::meters_per_scene_unit().
    /// The field is only used if the \c "fold_meters_per_scene_unit" option is set to false.
    /// Otherwise, the value of the \c "meters_per_scene_unit" option will be used in the code.
    float               meters_per_scene_unit;

    /// An offset to add to any argument block read accesses.
    uint                arg_block_offset;

#if defined(RENDERER_STATE_TYPE)
    /// A user-defined structure that allows to pass renderer information; for instance about the
    /// hit-point or buffer references; to mdl run-time functions. This is especially required for
    /// the scene data access. The fields of this structure are not altered by generated code.
    RENDERER_STATE_TYPE renderer_state;
#endif
#endif
};

/// The texture wrap modes as defined by \c tex::wrap_mode in the MDL specification.
#define Tex_wrap_mode             int
#define TEX_WRAP_CLAMP            0
#define TEX_WRAP_REPEAT           1
#define TEX_WRAP_MIRRORED_REPEAT  2
#define TEX_WRAP_CLIP             3

/// The type of events created by BSDF importance sampling.
#define Bsdf_event_type         int
#define BSDF_EVENT_ABSORB       0

#define BSDF_EVENT_DIFFUSE      1
#define BSDF_EVENT_GLOSSY       (1 << 1)
#define BSDF_EVENT_SPECULAR     (1 << 2)
#define BSDF_EVENT_REFLECTION   (1 << 3)
#define BSDF_EVENT_TRANSMISSION (1 << 4)

#define BSDF_EVENT_DIFFUSE_REFLECTION    (BSDF_EVENT_DIFFUSE  | BSDF_EVENT_REFLECTION)
#define BSDF_EVENT_DIFFUSE_TRANSMISSION  (BSDF_EVENT_DIFFUSE  | BSDF_EVENT_TRANSMISSION)
#define BSDF_EVENT_GLOSSY_REFLECTION     (BSDF_EVENT_GLOSSY   | BSDF_EVENT_REFLECTION)
#define BSDF_EVENT_GLOSSY_TRANSMISSION   (BSDF_EVENT_GLOSSY   | BSDF_EVENT_TRANSMISSION)
#define BSDF_EVENT_SPECULAR_REFLECTION   (BSDF_EVENT_SPECULAR | BSDF_EVENT_REFLECTION)
#define BSDF_EVENT_SPECULAR_TRANSMISSION (BSDF_EVENT_SPECULAR | BSDF_EVENT_TRANSMISSION)

#define BSDF_EVENT_FORCE_32_BIT 0xffffffffU

#define Edf_event_type          int
#define EDF_EVENT_NONE          0

#define EDF_EVENT_EMISSION      1
#define EDF_EVENT_FORCE_32_BIT  0xffffffffU

/// MBSDFs can consist of two parts, which can be selected using this enumeration.
#define Mbsdf_part               int
#define MBSDF_DATA_REFLECTION    0
#define MBSDF_DATA_TRANSMISSION  1

/// The calling code can mark the \c x component of an IOR field in *_data with
/// \c BSDF_USE_MATERIAL_IOR, to make the BSDF functions use the MDL material's IOR
/// for this IOR field.
#define BSDF_USE_MATERIAL_IOR (-1.0f)

/// Input and output structure for BSDF sampling data.
struct Bsdf_sample_data {
    vec3 ior1;                      ///< mutual input: IOR current medium
    vec3 ior2;                      ///< mutual input: IOR other side
    vec3 k1;                        ///< mutual input: outgoing direction

    vec3 k2;                        ///< output: incoming direction
    vec4 xi;                        ///< input: pseudo-random sample numbers
    float pdf;                      ///< output: pdf (non-projected hemisphere)
    vec3 bsdf_over_pdf;             ///< output: bsdf * dot(normal, k2) / pdf
    Bsdf_event_type event_type;     ///< output: the type of event for the generated sample
    int handle;                     ///< output: handle of the sampled elemental BSDF (lobe)
};

/// Input and output structure for BSDF evaluation data.
struct Bsdf_evaluate_data {
    vec3 ior1;                      ///< mutual input: IOR current medium
    vec3 ior2;                      ///< mutual input: IOR other side
    vec3 k1;                        ///< mutual input: outgoing direction

    vec3 k2;                        ///< input: incoming direction

    vec3 bsdf_diffuse;              ///< output: (diffuse part of the) bsdf * dot(normal, k2)
    vec3 bsdf_glossy;               ///< output: (glossy part of the) bsdf * dot(normal, k2)

    float pdf;                      ///< output: pdf (non-projected hemisphere)
};

/// Input and output structure for BSDF PDF calculation data.
struct Bsdf_pdf_data {
    vec3 ior1;                      ///< mutual input: IOR current medium
    vec3 ior2;                      ///< mutual input: IOR other side
    vec3 k1;                        ///< mutual input: outgoing direction

    vec3 k2;                        ///< input: incoming direction
    float pdf;                      ///< output: pdf (non-projected hemisphere)
};

/// Input and output structure for BSDF auxiliary calculation data.
struct Bsdf_auxiliary_data {
    vec3 ior1;                      ///< mutual input: IOR current medium
    vec3 ior2;                      ///< mutual input: IOR other side
    vec3 k1;                        ///< mutual input: outgoing direction

    vec3 albedo;                    ///< output: albedo
    vec3 normal;                    ///< output: normal
};

/// Input and output structure for EDF sampling data.
struct Edf_sample_data
{
    vec4 xi;                        ///< input: pseudo-random sample numbers
    vec3 k1;                        ///< output: outgoing direction
    float pdf;                      ///< output: pdf (non-projected hemisphere)
    vec3 edf_over_pdf;              ///< output: edf * dot(normal,k1) / pdf
    Edf_event_type event_type;      ///< output: the type of event for the generated sample
    int handle;                     ///< output: handle of the sampled elemental EDF (lobe)
};

/// Input and output structure for EDF evaluation data.
struct Edf_evaluate_data
{
    vec3 k1;                        ///< input: outgoing direction

    float cos;                      ///< output: dot(normal, k1)
    vec3 edf;                       ///< output: edf
    float pdf;                      ///< output: pdf (non-projected hemisphere)
};

/// Input and output structure for EDF PDF calculation data.
struct Edf_pdf_data
{
    vec3 k1;                        ///< input: outgoing direction
    float pdf;                      ///< output: pdf (non-projected hemisphere)
};

/// Input and output structure for EDF PDF calculation data.
struct Edf_auxiliary_data
{
    vec3 k1;                        ///< input: outgoing direction
};

// Modifies state.normal with the result of "geometry.normal" of the material.
/*void Bsdf_init_function(
    inout Shading_state_material state,
    out vec4 texture_results[16],
    uint arg_block_index);
void Bsdf_sample_function(
    inout Bsdf_sample_data data,
    Shading_state_material state,
    vec4 texture_results[16],
    uint arg_block_index);
void Bsdf_evaluate_function(
    inout Bsdf_evaluate_data data,
    Shading_state_material state,
    vec4 texture_results[16],
    uint arg_block_index);
void Bsdf_pdf_function(
    inout Bsdf_evaluate_data data,
    Shading_state_material state,
    vec4 texture_results[16],
    uint arg_block_index);*/

#endif
