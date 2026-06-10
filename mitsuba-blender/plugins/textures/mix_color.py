from __future__ import annotations # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi

blend_type_mix = 'MIX'

class Mix(mi.Texture):
    '''
    Mix Blender shader node texture 
    '''
    def __init__(self, props):
        super().__init__(props)
        self.blend_type = props.get('blend_type', blend_type_mix)
        self.clamp_result = props.get('clamp_result', False)
        self.clamp_factor = props.get('clamp_factor', False)
        self.factor = props.get('factor', 0.5)
        self.a = props.get_texture('a')
        self.b = props.get_texture('b')
    
    #TODO add function for gradient evaluation

    def parameters_changed(self, keys = ...):
        pass

    def traverse(self, cb):
        cb.put('a', self.a, +mi.ParamFlags.Differentiable)
        cb.put('b', self.b, +mi.ParamFlags.Differentiable)

    def eval(self, si, active):
        val_a = self.a.eval(si, active)
        val_b = self.b.eval(si, active)
        return mi.UnpolarizedSpectrum(self.process(si, val_a, val_b, active))

    def eval_1(self, si, active):
        val_a = self.a.eval_1(si, active)
        val_b = self.b.eval_1(si, active)
        return mi.Float(self.process(si, val_a, val_b, active))

    def eval_3(self, si, active):
        val_a = self.a.eval_3(si, active)
        val_b = self.b.eval_3(si, active)
        return mi.Color3f(self.process(si, val_a, val_b, active))

    def process(self, si, val_a, val_b, active):
        if self.clamp_factor:
            fac = dr.clip(self.factor.eval_1(si, active), 0.0, 1.0)

        #TODO add remaining blend types
        if self.blend_type == blend_type_mix:
            result = (1 - fac) * val_a + fac * val_b
        else:
            raise NotImplementedError(f"Current implementation of Mix color texture does not support {self.blend_type}")

        if self.clamp_result:
            result = dr.clip(result, 0.0, 1.0)
        
        return result

    def resolution(self):
        return self.a.resolution()
    
    def mean(self):
        return (self.a.mean() + self.b.mean()) / 2
    
    def is_spatially_varying(self):
        return self.a.is_spatially_varying() or self.b.is_spatially_varying()

    def to_string(self):
        return f'Mix[blend type={self.blend_type},factor={self.factor},a={self.a},b={self.b}]' 

mi.register_texture('mix_color', lambda props: Mix(props))