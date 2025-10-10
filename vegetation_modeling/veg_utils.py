import matplotlib.pyplot as plt
from constants import pi, stefan, rearth, mearth, lsol, au_m, Gsi, k_B
import numpy as np
import sys
from scipy.integrate import quad
from scipy.interpolate import interp1d

def calc_t_eq(l_star, a):
    numerator = l_star
    denominator = 16 * pi * stefan * (a ** 2)
    
    return (numerator/denominator) ** (1/4)

# def piecewise_radius_estimate(m_p):
#     return 1.02 * (m_p**0.27)

def piecewise_radius_estimate(m_p):
    if m_p < 4.37:
        return 1.02 * (m_p**0.27) * rearth
    # Intermediate-mass planets
    # H/He envelopes no longer neglible, so radius grows faster with mass than before
    if m_p < 127:
        return 0.56 * (m_p**0.67) * rearth
    # Massive planets, mass dominated by light gas.
    # Radius becomes almost constant and independent of mass
    # This gas is semi-degenerate, leading to the constant relation
    return 18.6 * (m_p ** (-0.06)) * rearth

# Equation 1 from https://lweb.cfa.harvard.edu/~lzeng/papers/Zeng2016b.pdf
def calc_cmf(m_p, r_p):
    constant_outer = 1/0.21
    constant_inner = 1.07
    rad_term = r_p/rearth
    print(rad_term)
    m_term = m_p / mearth
    print(m_term**0.27)
    exp = 0.27
    return constant_outer * (constant_inner - (rad_term/(m_term**exp)))

def calc_r_c(m_c, beta=4):
    term1 = (m_c/mearth) ** (1/beta)
    term2 = rearth
    return term1 * term2

def calc_r_B(m_c, t_eq, mu=2.2):
    m_u = 1.66053906660 * (10**(-27))
    numerator = 2 * Gsi * m_c * (mu * m_u)
    denominator = k_B * t_eq*100
    
    return numerator/denominator

def calc_r_rcb(m_c, t_eq, r_c, f, epsilon=0.03):
    const_num = 38
    mass_num = (m_c/(3*mearth)) ** (3/4)
    temp_num = (t_eq/1000) ** (-1)
    
    numerator = const_num * mass_num * temp_num
    
    const_denom = 27.9
    eps_denom = 1.5 * np.log(epsilon/0.03)
    f_denom = 2 * np.log(f/0.05)
    t_denom = 2 * np.log(t_eq/1000)
    m_denom = 2.625 * np.log(m_c/(3*mearth))
    
    denominator = const_denom - eps_denom + f_denom + t_denom - m_denom
    
    left_denom = r_c
    
    return (numerator/denominator) * r_c

# Density at R_rcb. Needed to calculate M_atm
def calc_rho_rcb(r_b, r_rcb, rho_d=(10**(-6))):
    return rho_d*np.exp(r_b/r_rcb - 1)

# Gamma = 7/5 is assumed in the paper
def calc_r_prime_b(r_b, gamma=4/3):
    return (gamma - 1)/(2 * gamma) * r_b

def calc_m_atm(rho_rcb, r_c, r_rcb, r_prime_b, gamma=4/3):
    def integrand(r):
        return (r ** 2) * (1 + r_prime_b / r - r_prime_b / r_rcb) ** (1 / (gamma - 1))
    
    integral_val, _ = quad(integrand, r_c, r_rcb)
    
    return 4 * pi * rho_rcb * integral_val

# Equation 28
def calc_f_ret_big_rcb(m_c, t_eq, r_c, r_rcb):
    #width of the atmosphere
    # Foudn below equation 20
    delta_r_a = r_rcb - r_c
    
    constant = 2.43 * (10 ** (-13))
    mass_term = (m_c/(3*mearth)) ** (3/2)
    temp_term = (t_eq/1000) ** (-1/2)
    rad_term = (delta_r_a/r_c) ** 6
    
    term1 = constant * mass_term * temp_term * rad_term
        
    exp_constant = 19
    exp_mass_term = (m_c/(3*mearth)) ** (3/4)
    exp_temp_term = (t_eq/1000) ** (-1)
    exp_rad_term = ((r_c + delta_r_a)/(2*r_c)) ** (-1)
    
    exp_term = exp_constant * exp_mass_term * exp_temp_term * exp_rad_term
    
    left_denom = 0.01
    
    return term1 * np.exp(exp_term) * left_denom

# Equation 27
def calc_f_ret_small_rcb(m_c, t_eq, r_c, r_b, r_rcb):
    constant = 1.2 * (10 ** (-13))
    mass_term = (m_c/(3*mearth)) ** (3/2)
    temp_term = (t_eq/1000) ** (-1/2)
    rad_term = (r_rcb/r_c)
    
    term1 = constant * mass_term * temp_term * rad_term
    
    exp_term = r_b/r_rcb

    left_denom = 0.01
    
    return term1 * np.exp(exp_term) * left_denom