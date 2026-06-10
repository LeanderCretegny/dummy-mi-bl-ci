import mitsuba as mi
import drjit as dr

def safe_div(n, d, eps = 1e-10):
    return dr.select(dr.abs(d) > eps, n / d, mi.Float(0.0))

class Test4(mi.BSDF):
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
        eta = dr.select(cos_i > mi.Float(0), eta, safe_div(mi.Float(1), eta))
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
    
class Test3(mi.BSDF):
    def __init__(self, props):
        mi.BSDF.__init__(self, props)

        self.color = props.get_texture('color', 1.0)
        self.roughness = props.get_texture('roughness', 0.0)
        self.ior = props.get_texture('ior', 0.0)
        #TODO add normal texture support

        # FIXME some flags might be missing
        self.m_flags = mi.BSDFFlags.GlossyTransmission 
        self.m_components = [self.m_flags]
        
    def eval(self, ctx, si, wo, active):

        # Init some values
        cos_wi = mi.Frame3f.cos_theta(si.wi)
        cos_wo = mi.Frame3f.cos_theta(wo)
        N = mi.Vector3f(0.0, 0.0, 1.0)
        r = self.roughness.eval_1(si, active)
        a_x = dr.clamp(r, 0.0, 1.0) 
        a_y = r
        almost_specular = (a_x * a_y) <= mi.Float(2e-10)
        ior = self.ior.eval_1(si, active)
        valid = mi.Bool(True)

        valid &= cos_wi > mi.Float(0.0)
        valid &= ~almost_specular

        H = dr.normalize(-(ior * wo + si.wi))
        cos_hi = dr.dot(H, si.wi)

        tir = dr.square(ior) - (mi.Float(1.0) - dr.square(cos_hi)) <= mi.Float(0.0)
        transmittance = dr.select(tir, mi.Color3f(0.0), self.color.eval(si, active))

        cos_nh = dr.dot(N, H)
        alpha2 = a_x * a_y
        D = self.bsdf_D(alpha2, cos_nh)
        lambdaI = self.bsdf_lambda(alpha2, cos_wi)
        lambdaO = self.bsdf_lambda(alpha2, cos_wo)
        common = D / cos_wi * (dr.square(ior) * dr.abs(cos_hi * dr.dot(H, wo)))

        return dr.select(valid, transmittance * common / (mi.Float(1.0) + lambdaI + lambdaO), mi.Color3f(0.0))
    
    def pdf(self, ctx, si, wo, active):

        cos_wi = mi.Frame3f.cos_theta(si.wi)
        N = mi.Vector3f(0.0, 0.0, 1.0)
        r = self.roughness.eval_1(si, active)
        a_x = dr.clamp(r, 0.0, 1.0) 
        a_y = r
        almost_specular = (a_x * a_y) <= mi.Float(2e-10)
        ior = self.ior.eval_1(si, active)
        valid = mi.Bool(True)

        valid &= cos_wi > mi.Float(0.0)
        valid &= ~almost_specular

        H = dr.normalize(-(ior * wo + si.wi))
        cos_hi = dr.dot(H, si.wi)

        cos_nh = dr.dot(N, H)
        alpha2 = a_x * a_y
        D = self.bsdf_D(alpha2, cos_nh)
        lambdaI = self.bsdf_lambda(alpha2, cos_wi)
        common = D / cos_wi * (dr.square(ior) * dr.abs(cos_hi * dr.dot(H, wo)))

        return dr.select(valid, common / (mi.Float(1.0) + lambdaI), mi.Float(0.0))
    
    def sample(self, ctx, si, sample1, sample2, active = True):        
        # Refraction based on fresnel and beckmann microfracet

        # Init some values
        cos_wi = mi.Frame3f.cos_theta(si.wi)
        bs = mi.BSDFSample3f()
        N = mi.Vector3f(0.0, 0.0, 1.0)
        r = self.roughness.eval_1(si, active)
        a_x = dr.clamp(r, 0.0, 1.0) 
        a_y = r
        almost_specular = (a_x * a_y) <= mi.Float(2e-10)
        ior = self.ior.eval_1(si, active)

        # Check input validity
        sample_valid = cos_wi > mi.Float(0.0)

        # Sample beckmann for Half vector (keeping blender notation) FIXME use VNDF
        H = dr.select(almost_specular, N, mi.warp.square_to_beckmann(sample2, a_x))
        cos_hi = dr.dot(H, si.wi)

        # Getting wo and transmittance value 
        # Test for total internal reflection
        tir = dr.square(ior) - (mi.Float(1.0) - dr.square(cos_hi)) <= mi.Float(0.0)
        _, cos_ho, eta_it, eta_ti = mi.fresnel(cos_hi, ior)
        bs.eta = eta_it

        # Get wo 
        bs.wo = mi.refract(si.wi, cos_ho, eta_ti)

        # Compute pdf and eval color
        eval = dr.select(tir, mi.Color3f(0.0), mi.Color3f(1.0))
        bs.pdf = mi.Float(1.0)

        # isotropic
        alpha2 = a_x * a_y
        D = self.bsdf_D(alpha2, mi.Frame3f.cos_theta(H))
        lambdaO = self.bsdf_lambda(alpha2, mi.Frame3f.cos_theta(bs.wo))
        lambdaI = self.bsdf_lambda(alpha2, cos_wi)

        common = D / cos_wi * (dr.abs(cos_hi * cos_ho) / dr.square(cos_ho + cos_hi * eta_ti))

        bs.pdf *= dr.select(almost_specular, mi.Float(1e6), common / (mi.Float(1.0) + lambdaI))
        eval *= dr.select(almost_specular, mi.Float(1e6), common / (mi.Float(1.0) + lambdaI + lambdaO))
        eval *= self.color.eval(si, active)

        bs.sampled_component = self.m_components[0]
        bs.sampled_type = +mi.BSDFFlags.GlossyTransmission
        return bs, dr.select(sample_valid, eval, mi.Color3f(0.0))
    
    def bsdf_lambda(self, alpha2, cos_theta):
        alpha_tan2 = alpha2 * dr.maximum(mi.Float(1.0) / dr.square(cos_theta) - mi.Float(1.0), mi.Float(0.0))
        return self.lambda_helper(alpha_tan2)

    def bsdf_D(self, alpha2, cos_theta):
        cos_theta2 = dr.minimum(dr.square(cos_theta), mi.Float(1.0))

        return mi.Float(1.0) / (dr.exp((1 - cos_theta2) / (cos_theta2 * alpha2)) * dr.pi * alpha2 * dr.square(cos_theta2));

    # def bsdf_aniso_lambda(self, alpha_x, alpha_y, w):
    #     alpha_tan2 = (dr.square(alpha_x * w.x) + dr.square(alpha_y * w.y)) / dr.square(w.z)
    #     return self.lambda_helper(alpha_tan2)

    # def bsdf_aniso_D(self, alpha_x, alpha_y, H):
    #     H /= mi.Vector3f(alpha_x, alpha_y, 1.0);

    #     cos_NH2 = dr.square(H.z);
    #     alpha2 = alpha_x * alpha_y;

    #     return dr.exp(-(dr.square(H.x) + dr.square(H.y)) / cos_NH2) / (dr.pi * alpha2 * dr.square(cos_NH2));

    def lambda_helper(self, alpha_tan2):
        valid = alpha_tan2 >= mi.Float(0.39)
        a = dr.select(alpha_tan2 > mi.Float(0.0), mi.Float(1.0) / dr.squaret(alpha_tan2), mi.Float(0.0))
        numerator = (mi.Float(0.396) * a - mi.Float(1.259)) * a + mi.Float(1.0)
        denom = (mi.Float(2.181) * a + mi.Float(3.535)) * a
        return dr.select(valid, numerator / denom, mi.Float(0.0))

    def traverse(self, cb):
        cb.put('color', self.color, mi.ParamFlags.Differentiable)
        cb.put('roughness', self.alpha_x, mi.ParamFlags.Differentiable)
        cb.put('ior', self.ior, mi.ParamFlags.Differentiable)

    def parameters_changed(self, keys):
        print("🏝️ there is nothing to do here 🏝️")

class Test2(mi.BSDF):
    def __init__(self, props):
        mi.BSDF.__init__(self, props)

        self.color = props.get_texture("color", 1.0)

        self.m_flags = mi.BSDFFlags.DiffuseTransmission | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide
        self.m_components = [self.m_flags]

    def eval(self, ctx, si, wo, active):
        same_hemi = mi.Frame3f.cos_theta(si.wi) * mi.Frame3f.cos_theta(wo) >= mi.Float(0.0)
        coso = dr.abs(mi.Frame3f.cos_theta(wo))
        color = self.color.eval(si, active) 
        return dr.select(same_hemi, mi.Color3f(0.0), color * dr.inv_pi * coso)

    def _pdf(self, wi, wo):
        same_hemi = mi.Frame3f.cos_theta(wi) * mi.Frame3f.cos_theta(wo) >= mi.Float(0.0)
        coso = dr.abs(mi.Frame3f.cos_theta(wo))
        return dr.select(same_hemi, 0.0, coso * dr.inv_pi)

    def pdf(self, ctx, si, wo, active):
         return self._pdf(si.wi, wo)
    
    def sample(self, ctx, si, sample1, sample2, active = True):        
        bs = mi.BSDFSample3f()
        bs.wo = mi.warp.square_to_cosine_hemisphere(sample2)
        bs.wo.z = dr.select(mi.Frame3f.cos_theta(si.wi) > mi.Float(0.0), -bs.wo.z, bs.wo.z)
        bs.pdf = self._pdf(si.wi, bs.wo)
        bs.eta = mi.Float(1.0)
        bs.sampled_component = self.m_components[0]
        bs.sampled_type = +mi.BSDFFlags.DiffuseTransmission

        return bs, self.color.eval(si, active)
    
    def traverse(self, cb):
        cb.put('color', self.color, mi.ParamFlags.Differentiable)

    def parameters_changed(self, keys):
        print("🏝️ there is nothing to do here 🏝️")

class Test(mi.BSDF):
    def __init__(self, props):
        mi.BSDF.__init__(self, props)

        # Read 'eta' and 'tint' properties from `props`
        self.eta = 1.33
        if props.has_property('eta'):
            self.eta = props['eta']

        self.tint = mi.Color3f(props['tint'])

        # Set the BSDF flags
        reflection_flags   = mi.BSDFFlags.DeltaReflection   | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide
        transmission_flags = mi.BSDFFlags.DeltaTransmission | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide
        self.m_components  = [reflection_flags, transmission_flags]
        self.m_flags = reflection_flags | transmission_flags

    def sample(self, ctx, si, sample1, sample2, active):
        # Compute Fresnel terms
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        r_i, cos_theta_t, eta_it, eta_ti = mi.fresnel(cos_theta_i, self.eta)
        t_i = dr.maximum(1.0 - r_i, 0.0)

        # Pick between reflection and transmission
        selected_r = (sample1 <= r_i) & active

        # Fill up the BSDFSample struct
        bs = mi.BSDFSample3f()
        bs.pdf = dr.select(selected_r, r_i, t_i)
        bs.sampled_component = dr.select(selected_r, mi.UInt32(0), mi.UInt32(1))
        bs.sampled_type      = dr.select(selected_r, mi.UInt32(+mi.BSDFFlags.DeltaReflection),
                                                     mi.UInt32(+mi.BSDFFlags.DeltaTransmission))
        bs.wo = dr.select(selected_r,
                          mi.reflect(si.wi),
                          mi.refract(si.wi, cos_theta_t, eta_ti))
        bs.eta = dr.select(selected_r, 1.0, eta_it)

        # For reflection, tint based on the incident angle (more tint at grazing angle)
        value_r = dr.lerp(mi.Color3f(self.tint), mi.Color3f(1.0), dr.clip(cos_theta_i, 0.0, 1.0))

        # For transmission, radiance must be scaled to account for the solid angle compression
        value_t = mi.Color3f(1.0) * dr.square(eta_ti)

        value = dr.select(selected_r, value_r, value_t)

        return (bs, value)

    def eval(self, ctx, si, wo, active):
        return 0.0

    def pdf(self, ctx, si, wo, active):
        return 0.0

    def eval_pdf(self, ctx, si, wo, active):
        return 0.0, 0.0

    def traverse(self, cb):
        cb.put('tint', self.tint, mi.ParamFlags.Differentiable)

    def parameters_changed(self, keys):
        print("🏝️ there is nothing to do here 🏝️")

    def to_string(self):
        return ('MyBSDF[\n'
                '    eta=%s,\n'
                '    tint=%s,\n'
                ']' % (self.eta, self.tint))
    
def isZero(x, eps=1e-9):
    return mi.Bool(x < eps)

def isPositive(x, strictly=False):
    return dr.select(mi.Bool(strictly), x > mi.Float(0.0), x >= mi.Float(0.0))

def isNegative(x, strictly=False):
    return ~isPositive(x, not strictly)