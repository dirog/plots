import numpy as np
from utils.entropy import h, h2
from utils.utils import InnerParams


def inner(p : InnerParams):
    two_pi_e = 2 * np.pi * np.e
    Ps = p.src.Ps; Pu = p.src.Pu; Psu = p.src.Psu
    Pap = p.aux.Pap; Pbp = p.aux.Pbp
    Pqc = p.aux.Pqc; Pqp = p.aux.Pqp
    Pwc = p.aux.Pwc; Ptx = p.aux.Ptx
    Pn1 = p.ch.Pn1; Pn2 = p.ch.Pn2
    a1 = p.a1; a2 = p.a2; g = p.g

    I_S__Ap = 1/2 * np.log2(1 + a1**2 * Ps / Pap)
    I_Y__Qc_Qp = 1/2 * np.log2(1 + (Pqc + Pqp) / (Pwc + Ptx + Pn1))
    I_U__Bp_c_S_Ap = 1/2 * np.log2(1 + a2**2 * Pu / Pbp - (a2 * Psu)**2 / (Ps * Pbp))
    I_Y__Wc_c_Qc = 1/2 * np.log2(1 + Pwc / (Pqp + Ptx + Pn1))
    I_X__Y_c_Qc_Qp_Wc = 1/2 * np.log2(1 + Ptx / Pn1)
    I_Y__Qp_c_Qc = 1/2 * np.log2(1 + Pqp / (Pwc + Ptx + Pn1))

    H_S_c_Ap = 1/2 * np.log2(two_pi_e * Ps * Pap / (p.a1**2 * Ps + Pap))
    H_Z_c_X = 1/2 * np.log2(two_pi_e * (Pn1 + Pn2))
    H_Z_c_Qc = 1/2 * np.log2(two_pi_e * (Pwc + Pqp + Ptx + Pn1 + Pn2))
    H_Z_c_Qc_Wc = 1/2 * np.log2(two_pi_e * (Pqp + Ptx + Pn1 + Pn2))

    I_Z__Qp_c_Qc = 1/2 * np.log2( 1 + Pqp / (Pwc + Ptx + Pn1 + Pn2))
    I_X__Z_c_Wc_Qc = 1/2 * np.log2( (Pqp + Ptx)*(Pn1 + Pn2 + Pqp + Ptx) / (Pn1*Pqp + Pn1*Ptx + Pn2*Pqp + Pn2*Ptx) )
    #I_X__Z_c_Wc_Qc = 1/2 * np.log2(  )

    H_U = h(p.src.Pu)
    H_S = h(p.src.Ps)
    H_SU = h2(p.src.detK)
    H_S_c_Ac = H_S

    def rateS(p : InnerParams):
        return I_S__Ap <= p.R * I_Y__Qc_Qp
    
    def rateU(p : InnerParams):
        return I_U__Bp_c_S_Ap <= p.R * (I_Y__Wc_c_Qc + I_X__Y_c_Qc_Qp_Wc)

    def distS(p : InnerParams):
        var_S_Ap = Ps - (a1 * Ps)**2 / (a1**2 * Ps + Pap)
        return p.Ds >= var_S_Ap
    
    def distU(p : InnerParams):
        #var_U_Ap_Bp = -Psu*a1*(Psu*a1*(Pbp + Ps*g**2 + 2*Psu*a2*g + Pu*a2**2)/(Pap*Pbp + Pap*Ps*g**2 + 2*Pap*Psu*a2*g + Pap*Pu*a2**2 + Pbp*Ps*a1**2 + Ps*Pu*a1**2*a2**2 - Psu**2*a1**2*a2**2) + (Psu*g + Pu*a2)*(-Ps*a1*g - Psu*a1*a2)/(Pap*Pbp + Pap*Ps*g**2 + 2*Pap*Psu*a2*g + Pap*Pu*a2**2 + Pbp*Ps*a1**2 + Ps*Pu*a1**2*a2**2 - Psu**2*a1**2*a2**2)) + Pu - (Psu*g + Pu*a2)*(Psu*a1*(-Ps*a1*g - Psu*a1*a2)/(Pap*Pbp + Pap*Ps*g**2 + 2*Pap*Psu*a2*g + Pap*Pu*a2**2 + Pbp*Ps*a1**2 + Ps*Pu*a1**2*a2**2 - Psu**2*a1**2*a2**2) + (Pap + Ps*a1**2)*(Psu*g + Pu*a2)/(Pap*Pbp + Pap*Ps*g**2 + 2*Pap*Psu*a2*g + Pap*Pu*a2**2 + Pbp*Ps*a1**2 + Ps*Pu*a1**2*a2**2 - Psu**2*a1**2*a2**2))
        var_U_Ap_Bp = Pu - (a2 * Pu)**2 / (a2**2 * Pu + Pbp)
        #var_U_Ap_Bp = Pu - a2**2 * (Ps * Pu - Psu**2) / (a2**2 * (Ps * Pu - Psu**2) + Ps * Pbp)
        return p.Du >= var_U_Ap_Bp
    
    def eqvS(p : InnerParams):
        return p.eqv.deltaS <= H_S_c_Ap + p.R * (H_Z_c_X - H_Z_c_Qc + I_Y__Qp_c_Qc)
    
    def eqvSnew(p : InnerParams):
        return p.eqv.deltaS <= H_S + p.R * (H_Z_c_X - H_Z_c_Qc + I_Z__Qp_c_Qc)
    
    def eqvU(p : InnerParams):
        return p.eqv.deltaU <= H_U - I_S__Ap - I_U__Bp_c_S_Ap + p.R * (H_Z_c_X - H_Z_c_Qc_Wc + I_Y__Qp_c_Qc + I_X__Y_c_Qc_Qp_Wc)
    
    def eqvUnew(p : InnerParams):
        return p.eqv.deltaU <= H_U + p.R * (H_Z_c_X - H_Z_c_Qc_Wc + I_X__Z_c_Wc_Qc)

    def eqvSU(p : InnerParams):
        return p.eqv.deltaSU <= H_SU - I_S__Ap - I_U__Bp_c_S_Ap + p.R * (H_Z_c_X - H_Z_c_Qc_Wc + I_Y__Qp_c_Qc + I_X__Y_c_Qc_Qp_Wc)
    
    def eqvSUnew(p : InnerParams):
        return p.eqv.deltaSU <= H_SU + p.R * (H_Z_c_X - H_Z_c_Qc_Wc + I_Z__Qp_c_Qc + I_X__Z_c_Wc_Qc)

    return rateS(p) * rateU(p) * distS(p) * eqvS(p) * distU(p) * eqvSU(p)