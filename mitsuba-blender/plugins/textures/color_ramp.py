from __future__ import annotations # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi
import numpy as np

color_mode_RGB = 'RGB'
color_mode_HSV = 'HSV'
color_mode_HSL = 'HSL'

inter_ease = 'EASE'
inter_card = 'CARDINAL'
inter_bspline = 'B_SPLINE'
inter_const = 'CONSTANT'
inter_lin = 'LINEAR'

hue_inter_near = 'NEAR'
hue_inter_far = 'FAR'
hue_inter_cw = 'CW'
hue_inter_cww = 'CWW'

class Ramp(mi.Texture):
    '''
    Color ramp Blender shader node
    '''
    def __init__(self, props):
        super().__init__(props)
        self.mode = props.get('color_mode', color_mode_RGB)
        self.inter = props.get('interpolation', inter_lin)
        self.hue_inter = props.get('hue_interpolation', hue_inter_near)
        temp_elements = props.get('elements')
        self.fac = props.get_texture('fac', 0.5)

        # Represent a stop with a drjit array of 5 element (rgba | pos) 
        self.elements = [self.str_list_to_float_list(e_str) for e_str in temp_elements.split(',')]
    
    #TODO add function for gradient evaluation

    def str_list_to_float_list(self, str):        
        return [float(e) for e in str.split()]

    def parameters_changed(self, keys = ...):
        pass

    def traverse(self, cb):
        cb.put('fac', self.fac, +mi.ParamFlags.Differentiable)

    def eval(self, si, active):
        return mi.UnpolarizedSpectrum(self.process(si, active)[:3])

    def eval_1(self, si, active):
        return mi.Float(self.process(si, active)[3])

    def eval_3(self, si, active):
        return mi.Color3f(self.process(si, active)[:3])

    #TODO handle alpha
    def process(self, si, active):
        fac = self.fac.eval_1(si, active)
        
        if self.mode == color_mode_RGB:
            if self.inter == inter_lin:
                right, idx = self.get_right_stop(fac)

                if idx == -1 or idx == 0:
                    return mi.Vector4f(right[:4])
                else:
                    left = self.elements[idx - 1]
                    fac = (fac - left[4]) / (right[:4] - left[:4])
                    return (1 - fac) * mi.Vector4f(left[:4]) + fac * mi.Vector4f(right[:4])
            else:
                raise NotImplementedError(f"Current implementation of Color Ramp does not support interpolation mode {self.inter}")
        else:
            raise NotImplementedError(f"Current implementation of Color Ramp does not support color mode {self.mode}")

    def get_right_stop(self, fac):
        elements = np.asarray(self.elements)
        candidate = elements[elements[:, 4] > fac]

        if candidate.size != 0: 
            return candidate[0], self.elements.index(candidate[0].tolist())
        else:
            return self.elements[-1], -1

    def resolution(self):
        return self.fac.resolution()

    def is_spatially_varying(self):
        return self.fac.is_spatially_varying()

    def to_string(self):
        return f'ColorRamp[mode={self.mode},interpolation={self.inter},hue_interpolation={self.hue_inter},factor={self.fac},elements={self.elements}]'

mi.register_texture('color_ramp', lambda props: Ramp(props))