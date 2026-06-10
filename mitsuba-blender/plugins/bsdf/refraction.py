import mitsuba as mi
import drjit as dr
from .common import *

class RefractionBSDF(mi.BSDF):  
    def __init__(self, props):
        mi.BSDF.__init__(self, props)

        self.color = props.get_texture('color', 1.0)
        self.roughness = props.get_texture('roughness', 0.0)
        self.ior = props.get_texture('ior', 0.0)
        #TODO add normal texture support

        # FIXME some flags might be missing
        self.m_flags = mi.BSDFFlags.GlossyTransmission | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide
        self.m_components = [self.m_flags]

    def sample(self, ctx, si, sample1, sample2, active = True):        
        a = self.roughness.eval_1(si, active)
        distr = mi.MicrofacetDistribution(mi.MicrofacetType.Beckmann, a)
        eta = self.ior.eval_1(si, active)
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        bs = mi.BSDFSample3f()

        # Ignore perfectly grazing configuration
        active &= cos_theta_i != mi.Float(0.0)

        # Sample half vector
        m = mi.warp.square_to_beckmann(sample2, a)
        # m.z = dr.select(isNegative(m.z, strictly=True), -m.z, m.z)
        b_pdf = mi.warp.square_to_beckmann_pdf(m, a)
        cos_theta_mi = dr.dot(m, si.wi)

        # Compute fresnel coefficients
        _, cos_theta_t, eta_it, eta_ti = mi.fresnel(cos_theta_mi, eta)

        bs.eta = eta_it
        bs.sampled_component = self.m_components [0]
        bs.sampled_type = +mi.BSDFFlags.GlossyTransmission

        # Transmission sampling
        bs.wo = mi.refract(si.wi, m, cos_theta_t, eta_ti)
        cos_theta_mo = dr.dot(bs.wo, m)
        weight = dr.select(ctx.mode == mi.TransportMode.Radiance, dr.square(eta_ti), mi.Float(1.0))
        weight *= self.color.eval(si, active)
        dwh_dwo = (dr.square(bs.eta) * cos_theta_mo) / dr.square(cos_theta_mi + bs.eta * cos_theta_mo)
        G = self.g(si.wi, m, a) * self.g(bs.wo, m, a)
        weight *= G * cos_theta_mi / (cos_theta_i * mi.Frame3f.cos_theta(m))

        bs.pdf = b_pdf * dr.abs(dwh_dwo)

        return bs, dr.select(active, mi.Color3f(weight), mi.Color3f(0.0))

    def eval(self, ctx, si, wo, active):
        a = self.roughness.eval_1(si, active)
        cos_o = mi.Frame3f.cos_theta(wo)
        cos_i = mi.Frame3f.cos_theta(si.wi)

        # Ignore perfectly grazing configuration
        active &= cos_i != mi.Float(0)

        # Get index of refraction
        eta = self.ior.eval_1(si, active)
        eta = dr.select(cos_i > mi.Float(0), eta, safe_divide(mi.Float(1), eta))
        inv_eta = mi.Float(1) / eta

        # Get half vector
        m = dr.normalize(si.wi + wo * eta)
        m = dr.mulsign(m, mi.Frame3f.cos_theta(m))
        cos_mi = dr.dot(si.wi, m)

        # Get value from microfacet distribution and fresnel factors
        distr = mi.MicrofacetDistribution(mi.MicrofacetType.Beckmann, a)
        D = distr.eval(m)
        G = self.g(si.wi, m, a) * self.g(wo, m, a) # distr.G(si.wi, wo, m)

        scale = dr.select(ctx.mode == mi.TransportMode.Radiance, dr.square(inv_eta), mi.Float(1.0))
        value = dr.abs(
            (scale * D * G * eta * eta * cos_mi * dr.dot(wo, m)) / 
            (cos_i * dr.square(cos_mi + eta * dr.dot(wo, m)))) 
        value *= self.color.eval(si, active)

        return dr.select(active, value, mi.Color3f(0.0))

    def pdf(self, ctx, si, wo, active):
        cos_i = mi.Frame3f.cos_theta(si.wi)
        cos_o = mi.Frame3f.cos_theta(wo)
        
        active &= cos_i != mi.Float(0)

        eta = self.ior.eval_1(si, active)
        eta = dr.select(cos_i > mi.Float(0), eta, mi.Float(0.0) / eta)

        m = dr.normalize(si.wi + wo * eta)
        m = dr.mulsign(m, mi.Frame3f.cos_theta(m))

        cos_mi = dr.dot(si.wi, m)
        cos_mo = dr.dot(wo, m)
        active &= (cos_mi * cos_i > mi.Float(0)) & (cos_mo * cos_o > mi.Float(0))

        dwh_dwo = (eta * eta * cos_mo) / dr.square(cos_mi + eta * cos_mo)

        distr = mi.MicrofacetDistribution(mi.MicrofacetType.Beckmann, self.roughness.eval_1(si, active))
        p = distr.pdf(dr.mulsign(si.wi, cos_i), m)

        return dr.select(active, p * dr.abs(dwh_dwo), mi.Float(0.0))
    
    def g(self, v, wh, a):
        xy_a2 = dr.square(a * v.x) + dr.square(a * v.y)
        tan_theta_a2 = xy_a2 / dr.square(v.z)

        a = dr.rsqrt(tan_theta_a2)
        a_sqr = dr.square(a)

        res = dr.select(a >= mi.Float(1.6), mi.Float(1.0), \
                mi.Float(3.535) * a + mi.Float(2.181) * a_sqr / (mi.Float(1.0) + mi.Float(2.276) * a + mi.Float(2.577) * a_sqr))

        # Handle perpendicular incidence (no shadowing)
        res[xy_a2 == mi.Float(0.0)] = mi.Float(1.0)

        # Ensure consistent orientation
        res[dr.dot(v, wh) * mi.Frame3f.cos_theta(v) <= mi.Float(0.0)] = mi.Float(0.0)

        return res        

    def traverse(self, cb):
        cb.put('color', self.color, mi.ParamFlags.Differentiable)
        cb.put('roughness', self.alpha_x, mi.ParamFlags.Differentiable)
        cb.put('ior', self.ior, mi.ParamFlags.Differentiable)

    def parameters_changed(self, keys):
        print("🏝️ there is nothing to do here 🏝️")

mi.register_bsdf("refraction", lambda props: RefractionBSDF(props))