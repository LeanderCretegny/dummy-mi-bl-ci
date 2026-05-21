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
        self.m_flags = mi.BSDFFlags.GlossyTransmission 
        self.m_components = [self.m_flags]

    def sample(self, ctx, si, sample1, sample2, active = True):        
        alpha = self.roughness.eval_1(si, active)
        
        wh = mi.warp.square_to_beckmann(sample2, alpha)
        
        bs = mi.BSDFSample3f()
        wo = dr.normalize(-si.wi + (mi.Float(2.0) * dr.dot(si.wi, wh) * wh))
        wo.z = dr.select(wo.z > mi.Float(0.0), -wo.z, wo.z)
        bs.wo = wo

        valid = si.wi.z * wo.z < mi.Float(0.0)

        pdf = self.pdf(ctx, si, wo, active)
        valid &= pdf > mi.Float(0.0)

        bs.pdf = pdf
        bs.eta = self.ior.eval_1(si, active)
        bs.sampled_component = self.m_components[0]
        bs.sampled_type = +mi.BSDFFlags.GlossyTransmission

        color = self.eval(ctx, si, wo, active) * mi.Frame3f.cos_theta(wo) * self.color.eval(si, active)

        return bs, dr.select(valid, color / pdf, mi.Color3f(0.0))

    def eval(self, ctx, si, wo, active):
        wo.z = -wo.z
        cos_wo = mi.Frame3f.cos_theta(wo)        
        cos_wi = mi.Frame3f.cos_theta(si.wi)

        valid = cos_wo > mi.Float(0.0)
        valid &= cos_wi > mi.Float(0.0)

        wh = dr.normalize(si.wi + wo)
        cos_wh = mi.Frame3f.cos_theta(wh)

        f, _, _, _ = mi.fresnel(dr.dot(wh, si.wi), self.ior.eval_1(si, active))
        denom = mi.Float(4.0) * cos_wi * cos_wo * cos_wh 
        j = safe_div(mi.Float(1), denom)
        a = self.roughness.eval_1(si, active)
        G = self.g(si.wi, wh, a) * self.g(wo, wh, a)
        
        color = mi.warp.square_to_beckmann_pdf(wh, a) * j * G * f * self.color.eval(si, active)

        return dr.select(valid, color, mi.Color3f(0.0))

    def pdf(self, ctx, si, wo, active):
        wo.z = -wo.z

        valid = mi.Frame3f.cos_theta(wo) <= mi.Float(0.0)

        wh = dr.normalize(si.wi + wo)

        denom = mi.Float(4.0) * dr.dot(wh, wo)
        j = safe_div(mi.Float(1), denom)

        p = mi.warp.square_to_beckmann_pdf(wh, self.roughness.eval_1(si, active)) * j
        return dr.select(valid, p, mi.Float(0.0))
    
    def g(self, v, wh, a):
        x_arg = safe_div(dr.dot(v, wh), v.z)
        x = dr.select(x_arg > mi.Float(0.0), mi.Float(1.0), mi.Float(0.0))

        tanTheta = mi.Frame3f.tan_theta(v)
        b = safe_div(mi.Float(1.0), a * tanTheta)
        b2 = b * b;
        numerator = (3.535 * b) + (2.181 * b2)
        denom = 1 + (2.276 * b) + (2.577 * b2)
        right = dr.select(b < mi.Float(1.6), safe_div(numerator, denom), mi.Float(1.0))

        return x * right

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

        tir = dr.sqr(ior) - (mi.Float(1.0) - dr.sqr(cos_hi)) <= mi.Float(0.0)
        transmittance = dr.select(tir, mi.Color3f(0.0), self.color.eval(si, active))

        cos_nh = dr.dot(N, H)
        alpha2 = a_x * a_y
        D = self.bsdf_D(alpha2, cos_nh)
        lambdaI = self.bsdf_lambda(alpha2, cos_wi)
        lambdaO = self.bsdf_lambda(alpha2, cos_wo)
        common = D / cos_wi * (dr.sqr(ior) * dr.abs(cos_hi * dr.dot(H, wo)))

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
        common = D / cos_wi * (dr.sqr(ior) * dr.abs(cos_hi * dr.dot(H, wo)))

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
        tir = dr.sqr(ior) - (mi.Float(1.0) - dr.sqr(cos_hi)) <= mi.Float(0.0)
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

        common = D / cos_wi * (dr.abs(cos_hi * cos_ho) / dr.sqr(cos_ho + cos_hi * eta_ti))

        bs.pdf *= dr.select(almost_specular, mi.Float(1e6), common / (mi.Float(1.0) + lambdaI))
        eval *= dr.select(almost_specular, mi.Float(1e6), common / (mi.Float(1.0) + lambdaI + lambdaO))
        eval *= self.color.eval(si, active)

        bs.sampled_component = self.m_components[0]
        bs.sampled_type = +mi.BSDFFlags.GlossyTransmission
        return bs, dr.select(sample_valid, eval, mi.Color3f(0.0))
    
    def bsdf_lambda(self, alpha2, cos_theta):
        alpha_tan2 = alpha2 * dr.maximum(mi.Float(1.0) / dr.sqr(cos_theta) - mi.Float(1.0), mi.Float(0.0))
        return self.lambda_helper(alpha_tan2)

    def bsdf_D(self, alpha2, cos_theta):
        cos_theta2 = dr.minimum(dr.sqr(cos_theta), mi.Float(1.0))

        return mi.Float(1.0) / (dr.exp((1 - cos_theta2) / (cos_theta2 * alpha2)) * dr.pi * alpha2 * dr.sqr(cos_theta2));

    # def bsdf_aniso_lambda(self, alpha_x, alpha_y, w):
    #     alpha_tan2 = (dr.sqr(alpha_x * w.x) + dr.sqr(alpha_y * w.y)) / dr.sqr(w.z)
    #     return self.lambda_helper(alpha_tan2)

    # def bsdf_aniso_D(self, alpha_x, alpha_y, H):
    #     H /= mi.Vector3f(alpha_x, alpha_y, 1.0);

    #     cos_NH2 = dr.sqr(H.z);
    #     alpha2 = alpha_x * alpha_y;

    #     return dr.exp(-(dr.sqr(H.x) + dr.sqr(H.y)) / cos_NH2) / (dr.pi * alpha2 * dr.sqr(cos_NH2));

    def lambda_helper(self, alpha_tan2):
        valid = alpha_tan2 >= mi.Float(0.39)
        a = dr.select(alpha_tan2 > mi.Float(0.0), mi.Float(1.0) / dr.sqrt(alpha_tan2), mi.Float(0.0))
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