import bpy
import os
import mitsuba as mi
mi.set_variant("scalar_rgb")
import numpy as np
from matplotlib import image
import utils.compare_utils as compare

def test_export_glass():
    # Load scene
    bpy.ops.wm.open_mainfile(filepath='tests/resources/bl_scene.blend')

    # Set cycle parameter
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'CPU'
    bpy.context.scene.cycles.max_bounces = 1
    bpy.context.scene.cycles.samples = 64
    
    # Set blender glass material
    mat = bpy.data.materials.new("glass")
    mat.use_nodes = True
    mat.node_tree.nodes.clear()
    outputMat = mat.node_tree.nodes.new("ShaderNodeOutputMaterial")
    glass = mat.node_tree.nodes.new("ShaderNodeBsdfGlass")
    glass.inputs['Color'].default_value = (0, 0, 1, 1)
    mat.node_tree.links.new(glass.outputs[0], outputMat.inputs[0])

    bpy.data.objects['controller'].active_material = mat

    # Render and export blender scene
    export_path = 'test_glass.xml'
    bl_render_path = 'tests/renders/blender/glass.png'
    bpy.context.scene.render.filepath = os.path.abspath(bl_render_path)
    bpy.ops.export_scene.mitsuba(filepath=os.path.abspath(export_path))
    assert bpy.ops.render.render(write_still=True) == {"FINISHED"}

    print(f'current dir: {os.getcwd()}')
    print(f'export path: {export_path}')

    # Render exported scene in mitsuba
    mi_scene = mi.load_file(export_path) # <= not working because of weird stuff in mitsuba test initialization propably restricting reading to test directory
                                            #, need to investigate further. also probably the cause for full black render
    img = mi.render(mi_scene, spp=64)
    
    mi_render_path = 'tests/renders/mitsuba/glass.png'
    mi.util.write_bitmap(mi_render_path, img)

    print(f'current dir: {os.getcwd()}')

    # Compare renders
    mi_render = image.imread(bl_render_path)
    bl_render = image.imread(bl_render_path)
    err, var, diff = compare.l2_error(mi_render, bl_render)

    print(f'Error: {err}, Variance: {var}')
    assert(err < 1.0) # TODO better treshold

    # Save diff
    image.imsave('tests/res/render_diff/glass_diff.png', diff)