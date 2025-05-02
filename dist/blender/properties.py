# SPDX-FileCopyrightText: 2011-2022 Blender Foundation
#
# SPDX-License-Identifier: Apache-2.0

import bpy


class Properties(bpy.types.PropertyGroup):
    type = None

    @classmethod
    def register(cls):
        cls.type.hydra_gatling = bpy.props.PointerProperty(
            name="Hydra Gatling",
            description="Hydra Gatling properties",
            type=cls,
        )

    @classmethod
    def unregister(cls):
        del cls.type.hydra_gatling


class ViewportRenderProperties(bpy.types.PropertyGroup):
    spp: bpy.props.IntProperty(
        name="Samples per pixel",
        default=1, min=1,
    )
    max_bounces: bpy.props.IntProperty(
        name="Max bounces",
        default=7, min=1,
    )
    rr_bounce_offset: bpy.props.IntProperty(
        name="Russian roulette bounce offset",
        default=3, min=1,
    )
    rr_inv_min_term_prob: bpy.props.FloatProperty(
        name="Russian roulette inverse minimum terminate probability",
        default=0.95, min=0.0, max=1.0,
    )
    max_sample_value: bpy.props.FloatProperty(
        name="Max sample value",
        default=10.0, min=0.0,
    )
    # filter_importance_sampling: I don't think there's a need to disable it!
    # depth_of_field: currently not implemented in Hydra.
    # light_intensity_multipler: none, we don't need it!
    next_event_estimation: bpy.props.BoolProperty(
        name="Next event estimation",
        default=True,
    )
    clipping_planes: bpy.props.BoolProperty(
        name="Clipping planes",
        default=True,
    )
    medium_stack_size: bpy.props.IntProperty(
        name="Medium stack size",
        default=0, min=0,
    )
    max_volume_walk_length: bpy.props.IntProperty(
        name="Max volume walk length",
        default=7, min=1,
    )
    progressive_accumulation: bpy.props.BoolProperty(
        name="Progressive accumulation",
        default=True,
    )


class FinalRenderProperties(bpy.types.PropertyGroup):
    spp: bpy.props.IntProperty(
        name="Samples per pixel",
        default=2048, min=1,
    )
    max_bounces: bpy.props.IntProperty(
        name="Max bounces",
        default=12, min=1,
    )
    rr_bounce_offset: bpy.props.IntProperty(
        name="Russian roulette bounce offset",
        default=6, min=1,
    )
    rr_inv_min_term_prob: bpy.props.FloatProperty(
        name="Russian roulette inverse minimum terminate probability",
        default=0.975, min=0.0, max=1.0,
    )
    max_sample_value: bpy.props.FloatProperty(
        name="Max sample value",
        default=10.0, min=0.0,
    )
    next_event_estimation: bpy.props.BoolProperty(
        name="Next event estimation",
        default=True,
    )
    clipping_planes: bpy.props.BoolProperty(
        name="Clipping planes",
        default=True,
    )
    medium_stack_size: bpy.props.IntProperty(
        name="Medium stack size",
        default=0, min=0,
    )
    max_volume_walk_length: bpy.props.IntProperty(
        name="Max volume walk length",
        default=7, min=1,
    )
    progressive_accumulation: bpy.props.BoolProperty(
        name="Progressive accumulation",
        default=False,
    )


class SceneProperties(Properties):
    type = bpy.types.Scene

    viewport: bpy.props.PointerProperty(type=ViewportRenderProperties)
    final: bpy.props.PointerProperty(type=FinalRenderProperties)


register, unregister = bpy.utils.register_classes_factory((
    ViewportRenderProperties,
    FinalRenderProperties,
    SceneProperties,
))
