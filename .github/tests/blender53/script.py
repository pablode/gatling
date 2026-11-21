import bpy

# Scene settings
scene = bpy.context.scene
scene.render.engine = "HYDRA_GATLING"
scene.render.resolution_x = 64
scene.render.resolution_y = 64
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGB"
scene.render.filepath = "//test.png"

# Background (non-black for debugging)
bg = bpy.data.worlds["World"].node_tree.nodes["Background"]
bg.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
bg.inputs[1].default_value = 1.0

# Add blue cube
cube = bpy.data.objects["Cube"]
mat = bpy.data.materials.new("Blue")
mat.use_nodes = True
mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.0, 0.0, 1.0, 1.0)
mat.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 1.0
cube.data.materials.clear()
cube.data.materials.append(mat)

# Render to file and quit
bpy.ops.render.render(write_still=True)
bpy.ops.wm.quit_blender()
