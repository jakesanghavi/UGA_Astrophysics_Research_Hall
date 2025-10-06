import matplotlib.pyplot as plt
from constants import pi, stefan, rearth, mearth, lsol, au_m, Gsi, k_B
import numpy as np
import sys

def calc_t_eq(l_star, a):
    numerator = l_star
    denominator = 16 * pi * stefan * (a ** 2)
    
    return (numerator/denominator) ** (1/4)

def piecewise_radius_estimate(m_p):
    return 1.02 * (m_p**0.27)

# Equation 1 from https://lweb.cfa.harvard.edu/~lzeng/papers/Zeng2016b.pdf
def calc_cmf(m_p, r_p):
    constant_outer = 1/0.21
    constant_inner = 1.07
    rad_term = r_p/rearth
    m_term = m_p / mearth
    exp = 0.27
    return constant_outer * (constant_inner - (rad_term/(m_term**exp)))

def calc_r_c(m_c, beta=4):
    term1 = (m_c/mearth) ** (1/beta)
    term2 = rearth
    return term1 * term2

def calc_r_B(m_c, t_eq, mu=2.2):
    m_u = 1.66053906660 * (10**(-27))
    numerator = 2 * Gsi * m_c * (mu * m_u)
    denominator = k_B * t_eq
    
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
    
t_eqs = [255, 500, 1000, 1500, 2000]
f_rets = {str(v): [] for v in t_eqs}
range_over = np.arange(0.1, 10.1, 0.1)
colors = ['purple', 'b', 'orange', 'g', 'red']
for mass in range_over:
    for temp in t_eqs:
        # t_eq = calc_t_eq(lsol, au_m)
        t_eq = temp
        
        # This is an assumption made in the paper
        # m_c = 4 * mearth
        m_c = mass * mearth
        
        # Override their assumptions with our own
        # r_p = piecewise_radius_estimate(mass*mearth)
        # m_c = calc_cmf(mass*mearth, r_p) * mass * mearth
        r_c = calc_r_c(m_c)
        
        # This is another assumption made in the paper
        # r_rcb requires f to calculate, but f also depends on r_rcb
        r_rcb = 2 * r_c
        
        f_ret = calc_f_ret_big_rcb(m_c, t_eq, r_c, r_rcb)
        
        # r_b = calc_r_B(m_c, t_eq)
        # f_ret = calc_f_ret_small_rcb(m_c, t_eq, r_c, r_b, r_rcb)
        
        f_rets[str(temp)].append(f_ret)

for x in range(len(t_eqs)):
    plt.plot(range_over, f_rets[str(t_eqs[x])], c=colors[x], label=r"$T_{eq}$" + f"={t_eqs[x]} K")
    
min_values = []
for temp, f_ret_list in f_rets.items():
    min_values.append(min(f_ret_list))

plt.vlines(0.325, ymin=0, ymax=1, linestyles='dashed', color='k', label="Earth CMF")
plt.xlim([0.1,10])
plt.xticks([0.1] + list(range(1, 11)))
plt.ylim([np.min(min_values),1])
plt.yscale('log')
plt.legend()
# plt.xlabel("Planetary Mass (" + r'$\mathbf{M_{\oplus}}$' + ")")
plt.xlabel(r'$\mathbf{M_c/M_{\oplus}}$')
plt.ylabel(r'$\mathbf{f_{ret}}$')
# plt.title("Retained H and He fraction by Planetary Mass")
plt.title("Retained H and He fraction by Planetary Core Mass")

plt.show()
    