# Small script use for debugging, to remove in final version
import mitsuba as mi
mi.set_variant("cuda_ad_rgb") # cuda can be used to get faster renders

import plugins
mi.register_bsdf("blender_principled", lambda props: plugins.BlenderPrincipledBSDF(props))
mi.register_texture('rgb_curve', lambda props: plugins.RGBCurve(props))
mi.register_texture('brightness_contrast', lambda props: plugins.BrightnessContrast(props))
mi.register_texture('mix_color', lambda props: plugins.Mix(props))
mi.register_texture('color_ramp', lambda props: plugins.Ramp(props))
mi.register_bsdf("translucent", lambda props: plugins.TranslucentBSDF(props))
mi.register_bsdf('test', lambda props: plugins.Test(props))

import sys

import matplotlib.pyplot as plt

# import drjit as dr
# dr.set_flag(dr.JitFlag.Debug, True)

def main(scene_ref):
    scene = mi.load_file(scene_ref)
    print("scene loaded")
    # print(mi.traverse(scene))
    img = mi.render(scene)
    print("scene rendered")
    plt.imshow(img)
    plt.show()

if __name__ == "__main__":
    arg = sys.argv[1]
    main(arg)