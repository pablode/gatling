# SPDX-FileCopyrightText: 2011-2022 Blender Foundation
#
# SPDX-License-Identifier: Apache-2.0

import bpy

from pathlib import Path
from pxr import Plug
import os


class GatlingHydraRenderEngine(bpy.types.HydraRenderEngine):
    bl_idname = 'HYDRA_GATLING'
    bl_label = "Hydra Gatling"
    bl_info = "A cross-platform GPU path tracer with hardware ray tracing"

    bl_use_preview = True
    bl_use_materialx = True

    bl_delegate_id = 'HdGatlingRendererPlugin'

    @classmethod
    def register(cls):
        delegate_dir = os.path.join(Path(__file__).parent, "render_delegate")
        os.environ['PATH'] = delegate_dir + os.pathsep + os.environ['PATH'] # prepend DLL search dir
        Plug.Registry().RegisterPlugins(str(os.path.join(delegate_dir, "hdGatling", "resources")))

    def get_render_settings(self, engine_type):
        settings = bpy.context.scene.hydra_gatling.viewport if engine_type == 'VIEWPORT' else \
            bpy.context.scene.hydra_gatling.final
        result = {
            'spp': settings.spp,
            'max-bounces': settings.max_bounces,
            'rr-bounce-offset': settings.rr_bounce_offset,
            'rr-inv-min-term-prob': settings.rr_inv_min_term_prob,
            'max-sample-value': settings.max_sample_value,
            'next-event-estimation': settings.next_event_estimation,
            'clipping-planes': settings.clipping_planes,
            'medium-stack-size': settings.medium_stack_size,
            'max-volume-walk-length': settings.max_volume_walk_length,
            'progressive-accumulation': settings.progressive_accumulation,
            'enableInteractive': True if engine_type == 'VIEWPORT' else False
        }

        if engine_type != 'VIEWPORT':
            result |= {
                'aovToken:Combined': "color"
            }

        return result

    def update_render_passes(self, scene, render_layer):
        if render_layer.use_pass_combined:
            self.register_pass(scene, render_layer, 'Combined', 4, 'RGBA', 'COLOR')


register, unregister = bpy.utils.register_classes_factory((
    GatlingHydraRenderEngine,
))
