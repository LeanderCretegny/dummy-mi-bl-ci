from __future__ import annotations # Delayed parsing of type annotations

import drjit as dr
from drjit.auto import Float, Array3f, Int
import mitsuba as mi
import numpy as np
import bpy 

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

class RampElement():
    '''
    Element of a color ramp
    '''
    def __init__(self, string_elem):
        l = string_elem.split(' ')
        self.a = mi.Float(float(l[0]))
        self.color = mi.Color3f([float(e) for e in l[1:4]])
        self.pos = mi.Float(float(l[4]))

RAMP_ELEMENT_SIZE = 5

class Ramp(mi.Texture):
    '''
    Color ramp Blender shader node
    '''
    def __init__(self, props):
        super().__init__(props)
        self.mode = props.get('color_mode', color_mode_RGB)
        self.inter = props.get('interpolation', inter_lin)
        self.hue_inter = props.get('hue_interpolation', hue_inter_near)
        # temp_elements = props.get('elements')
        self.fac = props.get_texture('fac', 0.5)

        # Represent a stop with a drjit array of 5 element (rgba | pos) 
        # self.elements = [RampElement(e_str) for e_str in props.get('elements').split('-')]
        self.parse_elements(props.get('elements')) #[self.str_list_to_float_list(e_str) for e_str in temp_elements.split(',')]

    def parse_elements(self, str):
        temp_alpha = []
        temp_col = []
        temp_pos = []
        for s in str.split('-'):
            elem = []
            for e in s.split(' '):
                elem.append(float(e))
            temp_alpha.append(elem[0])
            temp_col.append(elem[1:4])
            temp_pos.append(elem[4])
        self.alphas = Float(temp_alpha)
        self.colors = Array3f(np.asarray(temp_col).T)
        self.positions = Float(temp_pos)
    
    def get_element_color(self, idx):
        return self.colors[idx]
    
    def get_element_alpha(self, idx):
        return self.alphas[idx]
    
    def get_element_pos(self, idx):
        return self.positions[idx]
    
    def parameters_changed(self, keys = ...):
        pass

    def traverse(self, cb):
        cb.put('fac', self.fac, +mi.ParamFlags.Differentiable)

    def eval_1(self, si, active):
        _, res = self.process(si, active)
        dr.print('res: {} (should be float)', res)
        return res

    def eval_3(self, si, active):
        res, _ = self.process(si, active)
        dr.print('res: {} (should be color)', res)
        return res
    
    def eval(self, si, active = True):
        return mi.UnpolarizedSpectrum(self.eval_3(si, active))

    def process(self, si, active):
        fac = self.fac.eval_1(si, active)
        
        if self.mode == color_mode_RGB:
            if self.inter == inter_lin:
                right_idx = self.get_right_stop(fac)

                if right_idx == -1 or right_idx == 0:
                    return self.get_element_color(right_idx), self.get_element_alpha(right_idx)
                else:
                    left_idx = right_idx - 1
                    fac = (fac - self.get_element_pos(left_idx)) / (self.get_element_pos(right_idx) - self.get_element_pos(left_idx))
                    res_c = (1 - fac) * self.get_element_color(left_idx) + fac * self.get_element_color(right_idx)
                    res_a = (1 - fac) * self.get_element_alpha(left_idx) + fac * self.get_element_alpha(right_idx)
                    return res_c, res_a
            else:
                raise NotImplementedError(f"Current implementation of Color Ramp does not support interpolation mode {self.inter}")
        else:
            raise NotImplementedError(f"Current implementation of Color Ramp does not support color mode {self.mode}")

    def get_right_stop(self, fac):
        # dr.print('fac type: {}', type(fac))
        # dr.print('element type: {}', type(self.elements[0].pos))
        # candidates = [e for e in self.elements if e.pos > fac]
        pos = dr.min(dr.select(self.positions > fac, self.positions, dr.inf))
        idx = list(self.positions).index(pos)

        if pos != dr.inf: 
            return idx
        else:
            return -1

    def resolution(self):
        return self.fac.resolution()

    def is_spatially_varying(self):
        return self.fac.is_spatially_varying()

    def to_string(self):
        return f'ColorRamp[mode={self.mode},interpolation={self.inter},hue_interpolation={self.hue_inter},factor={self.fac},elements={self.elements}]'

mi.register_texture('color_ramp', lambda props: Ramp(props))