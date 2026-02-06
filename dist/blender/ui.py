# SPDX-FileCopyrightText: 2011-2022 Blender Foundation
#
# SPDX-License-Identifier: Apache-2.0

import bpy, os, webbrowser
from pathlib import Path

from .engine import GatlingHydraRenderEngine


class Panel(bpy.types.Panel):
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'render'
    COMPAT_ENGINES = {GatlingHydraRenderEngine.bl_idname}

    @classmethod
    def poll(cls, context):
        return context.engine in cls.COMPAT_ENGINES


#
# Quality render settings
#
class GATLING_HYDRA_RENDER_PT_quality(Panel):
    bl_label = "Quality"

    def draw(self, layout):
        pass


class GATLING_HYDRA_RENDER_PT_quality_viewport(Panel):
    bl_label = "Viewport"
    bl_parent_id = "GATLING_HYDRA_RENDER_PT_quality"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        settings = context.scene.hydra_gatling.viewport
        layout.prop(settings, 'spp')
        layout.prop(settings, 'max_bounces')
        layout.prop(settings, 'rr_bounce_offset')
        layout.prop(settings, 'rr_inv_min_term_prob')
        layout.prop(settings, 'max_sample_value')
        layout.prop(settings, 'next_event_estimation')
        layout.prop(settings, 'clipping_planes')
        layout.prop(settings, 'medium_stack_size')
        layout.prop(settings, 'max_volume_walk_length')
        layout.prop(settings, 'denoise_color_aov')


class GATLING_HYDRA_RENDER_PT_quality_render(Panel):
    bl_label = "Render"
    bl_parent_id = "GATLING_HYDRA_RENDER_PT_quality"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        settings = context.scene.hydra_gatling.final
        layout.prop(settings, 'spp')
        layout.prop(settings, 'max_bounces')
        layout.prop(settings, 'rr_bounce_offset')
        layout.prop(settings, 'rr_inv_min_term_prob')
        layout.prop(settings, 'max_sample_value')
        layout.prop(settings, 'next_event_estimation')
        layout.prop(settings, 'clipping_planes')
        layout.prop(settings, 'medium_stack_size')
        layout.prop(settings, 'max_volume_walk_length')
        layout.prop(settings, 'denoise_color_aov')


#
# Light settings
#
class GATLING_HYDRA_LIGHT_PT_light(Panel):
    bl_label = "Light"
    bl_context = 'data'

    @classmethod
    def poll(cls, context):
        return super().poll(context) and context.light

    def draw(self, context):
        layout = self.layout

        light = context.light

        layout.prop(light, "type", expand=True)

        layout.use_property_split = True
        layout.use_property_decorate = False

        main_col = layout.column()

        main_col.prop(light, "color")
        main_col.prop(light, "energy")
        main_col.separator()

        if light.type == 'POINT':
            row = main_col.row(align=True)
            row.prop(light, "shadow_soft_size", text="Radius")

        elif light.type == 'SPOT':
            col = main_col.column(align=True)
            col.prop(light, 'spot_size', slider=True)
            col.prop(light, 'spot_blend', slider=True)

            main_col.prop(light, 'show_cone')

        elif light.type == 'SUN':
            main_col.prop(light, "angle")

        elif light.type == 'AREA':
            main_col.prop(light, "shape", text="Shape")
            sub = main_col.column(align=True)

            if light.shape in {'SQUARE', 'DISK'}:
                sub.prop(light, "size")
            elif light.shape in {'RECTANGLE', 'ELLIPSE'}:
                sub.prop(light, "size", text="Size X")
                sub.prop(light, "size_y", text="Y")

            else:
                main_col.prop(light, 'size')


#
# Help panel
#
class GATLING_HYDRA_HELP_OT_open_repository(bpy.types.Operator):
    bl_idname = "gatling.open_repository"
    bl_label = "Open GitHub Repository"

    def execute(self, context):
        webbrowser.open("https://github.com/pablode/gatling")
        return {'FINISHED'}

class GATLING_HYDRA_HELP_OT_view_licenses(bpy.types.Operator):
    bl_idname = "gatling.view_licenses"
    bl_label = "View Licenses"

    def execute(self, context):
        delegate_dir = os.path.join(Path(__file__).parent, "render_delegate")
        license_path = str(os.path.join(delegate_dir, "hdGatling", "resources", 'LICENSE'))
        webbrowser.open(license_path)
        return {'FINISHED'}

class GATLING_HYDRA_PT_HELP(Panel):
    bl_label = "Help"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        row = layout.row()
        row.operator(GATLING_HYDRA_HELP_OT_open_repository.bl_idname)
        row = layout.row()
        row.operator(GATLING_HYDRA_HELP_OT_view_licenses.bl_idname)


#
# Panel registration
#
register_classes, unregister_classes = bpy.utils.register_classes_factory((
    GATLING_HYDRA_RENDER_PT_quality,
    GATLING_HYDRA_RENDER_PT_quality_viewport,
    GATLING_HYDRA_RENDER_PT_quality_render,
    GATLING_HYDRA_LIGHT_PT_light,
    GATLING_HYDRA_PT_HELP,
    GATLING_HYDRA_HELP_OT_open_repository,
    GATLING_HYDRA_HELP_OT_view_licenses
))

def get_panels():
    exclude_panels = {
        'RENDER_PT_stamp',
        'DATA_PT_light',
        'DATA_PT_spot',
        'NODE_DATA_PT_light',
        'DATA_PT_falloff_curve',
        'RENDER_PT_post_processing',
        'RENDER_PT_simplify',
        'SCENE_PT_audio',
        'RENDER_PT_freestyle'
    }
    include_eevee_panels = {
        'MATERIAL_PT_preview',
        'EEVEE_MATERIAL_PT_context_material',
        'EEVEE_MATERIAL_PT_surface',
        'EEVEE_MATERIAL_PT_volume',
        'EEVEE_MATERIAL_PT_settings',
        'EEVEE_WORLD_PT_surface'
    }

    for panel_cls in bpy.types.Panel.__subclasses__():
        if hasattr(panel_cls, 'COMPAT_ENGINES') and (
            ('BLENDER_RENDER' in panel_cls.COMPAT_ENGINES and panel_cls.__name__ not in exclude_panels) or
            ('BLENDER_EEVEE' in panel_cls.COMPAT_ENGINES and panel_cls.__name__ in include_eevee_panels)
        ):
            yield panel_cls


def register():
    register_classes()

    for panel_cls in get_panels():
        panel_cls.COMPAT_ENGINES.add(GatlingHydraRenderEngine.bl_idname)


def unregister():
    unregister_classes()

    for panel_cls in get_panels():
        if GatlingHydraRenderEngine.bl_idname in panel_cls.COMPAT_ENGINES:
            panel_cls.COMPAT_ENGINES.remove(GatlingHydraRenderEngine.bl_idname)
