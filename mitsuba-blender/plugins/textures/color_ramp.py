from __future__ import annotations # Delayed parsing of type annotations

import drjit as dr
from drjit.auto import Float, Array4f, Int
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
        self.fac = props.get_texture('fac', 0.5)

        self.parse_elements(props.get('elements')) 

    def parse_elements(self, str):
        temp_elem = []
        temp_pos = []
        for s in str.split('-'):
            elem = []
            for e in s.split(' '):
                elem.append(float(e))
            temp_elem.append(elem[0:4])
            temp_pos.append(elem[4])
        self.elems = Array4f(np.asarray(temp_elem).T)
        self.positions = Float(temp_pos)
    
    def get_element_color(self, idx):
        return mi.Color3f(dr.gather(Array4f, self.elems, idx)[1:4])
    
    def get_element_alpha(self, idx):
        return mi.Float(dr.gather(Array4f, self.elems, idx)[0])
    
    def get_element_pos(self, idx):
        return mi.Float(dr.gather(Float, self.positions, idx))
    
    def get_element_len(self):
        return self.positions.shape[0]
    
    def parameters_changed(self, keys = ...):
        pass

    def traverse(self, cb):
        cb.put('fac', self.fac, +mi.ParamFlags.Differentiable)

    def eval_1(self, si, active):
        _, res = self.process(si, active)
        return res

    def eval_3(self, si, active):
        res, _ = self.process(si, active)
        return res
    
    def eval(self, si, active = True):
        dr.set_log_level(dr.LogLevel.Error)
        with dr.scoped_set_flag(dr.JitFlag.Debug):
            return mi.UnpolarizedSpectrum(self.eval_3(si, active))

    @dr.syntax
    def process(self, si, active):
        fac = self.fac.eval_1(si, active)
        
        res_c = mi.Color3f(0)
        res_a = mi.Float(0)
        if self.mode == color_mode_RGB:
            if self.inter == inter_lin:
                right_idx = self.get_right_stop(fac)

                if right_idx == mi.Int(0):
                    res_c = self.get_element_color(right_idx)
                    res_a = self.get_element_alpha(right_idx)
                elif right_idx == self.get_element_len():
                    res_c = self.get_element_color(right_idx - 1)
                    res_a = self.get_element_alpha(right_idx - 1)
                else:
                    left_idx = right_idx - 1
                    fac = (fac - self.get_element_pos(left_idx)) / (self.get_element_pos(right_idx) - self.get_element_pos(left_idx))
                    res_c = (1 - fac) * self.get_element_color(left_idx) + fac * self.get_element_color(right_idx)
                    res_a = (1 - fac) * self.get_element_alpha(left_idx) + fac * self.get_element_alpha(right_idx)
            else:
                raise NotImplementedError(f"Current implementation of Color Ramp does not support interpolation mode {self.inter}")
        else:
            raise NotImplementedError(f"Current implementation of Color Ramp does not support color mode {self.mode}")
        
        return res_c, res_a

    @dr.syntax
    def get_right_stop(self, fac):
        idx = mi.Int(0)
        while (idx < self.positions.shape[0]) & (self.get_element_pos(idx) < fac):
            idx += 1
        
        return idx

    def resolution(self):
        return self.fac.resolution()

    def is_spatially_varying(self):
        return self.fac.is_spatially_varying()

    def to_string(self):
        return f'ColorRamp[mode={self.mode},interpolation={self.inter},hue_interpolation={self.hue_inter},factor={self.fac},elements={self.elements}]'

mi.register_texture('color_ramp', lambda props: Ramp(props))