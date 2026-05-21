import sys
import matplotlib.pyplot as plt

import mitsuba as mi
mi.set_variant("cuda_ad_rgb")

from test_bsdf import Test4
mi.register_bsdf("test_bsdf", lambda props: Test4(props))

scene_ref = sys.argv[1]
scene = mi.load_file(scene_ref)
img = mi.render(scene)
plt.imshow(img)
plt.show()