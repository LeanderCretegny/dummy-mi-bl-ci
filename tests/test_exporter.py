import bpy
import os
import mitsuba as mi
mi.set_variant("cuda_ad_rgb")
import numpy as np
from matplotlib import image
import utils.compare_utils as compare
import pytest

from fixtures import *

@pytest.mark.parametrize(
    "xml_scene, blend_scene, bl_render, mi_render",
    [("scenes/glass.xml", "scenes/blender/test_scene.blend", "renders/blender/glass.png", "renders/mitsuba/glass.png")]
)
def test_export_glass(resource_resolver, xml_scene, blend_scene, bl_render, mi_render):
    # Load scene
    ref_bl_scene = resource_resolver.get_absolute_resource_path(blend_scene)
    bpy.ops.wm.open_mainfile(filepath=ref_bl_scene)

    # Set cycle parameter
    # FIXME Those can be save inside blend file, better keeping it there or not?
    # TODO write pros and cons inside notes 
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'CPU'
    bpy.context.scene.cycles.max_bounces = 2
    bpy.context.scene.cycles.samples = 64

    resolution = (1280, 720)
    bpy.context.scene.render.resolution_x = resolution[0]
    bpy.context.scene.render.resolution_y = resolution[1]
    
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
    ref_mi_scene = resource_resolver.get_absolute_resource_path(xml_scene)
    ref_bl_render = resource_resolver.get_absolute_resource_path(bl_render)
    bpy.context.scene.render.filepath = ref_bl_render
    assert bpy.ops.export_scene.mitsuba(filepath=ref_mi_scene) == {"FINISHED"}
    assert bpy.ops.render.render(write_still=True) == {"FINISHED"}

    # Render exported scene in mitsuba

    # print(f"mitsuba file resolver: {}")
    # mi_path = mi.filesystem.path(ref_mi_scene)
    # mi.FileResolver.append(arg=mi_path)

    mi_img = mi.render(mi.load_file(ref_mi_scene, resx=resolution[0], resy=resolution[1]))
    
    ref_mi_render = resource_resolver.get_absolute_resource_path(mi_render)
    mi.util.write_bitmap(ref_mi_render, mi_img)


    # Compare renders
    print(f"path to mi render: {ref_mi_render}")
    # FIXME Probably a cleaner way to do that, also need to be sure it does not alter the renders 
    bl_img = np.array(compare.convert_png(mi.Bitmap(ref_bl_render)))
    mi_img = np.array(compare.convert_png(mi.Bitmap(mi_img)))
    err, var, diff = compare.l2_error(mi_img, bl_img)

    print(f'Error: {err}, Variance: {var}')
    assert err < 1.0,  f"Error is too big (err = {err}), should be less than 1" # TODO better treshold

    # Save diff
    ref_diff = resource_resolver.get_absolute_resource_path("out/glass_diff.png")
    image.imsave(ref_diff, diff)