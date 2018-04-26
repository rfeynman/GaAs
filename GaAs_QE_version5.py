'''
Created on Feb 2018 by W. Liu
Monte Carlo methode to simulate the photoemission based on three-step model:
photoexcited, transportation and emission
|----------------------------|
|                            |
|                            |
|                            |    ^y
|                            |    |
|----------------------------|    |
------------------------------>z
electron distribution: (z,y,vz,vy,v,E) in GaAs
z direction: exponential distribution for photoexcited electrons
y direction: Gauss distribution for photoexcited electrons (depend on laser)
'''
import random
import numpy as np
from scipy import integrate
from scipy.stats import expon, maxwell
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, interp2d
import time

start_time = time.time()

# ----Define fundamental constants and general parameters----
pi = np.pi
two_pi = 2 * pi
eps0 = 8.85419e-12  # F/m, vacuum dielectric constant
eps = 12.85 * eps0  # F/m, dielectric constant for low frequency
eps_high = 10.89 * eps0  # F/m, for high frequency
kB = 1.38066e-23  # J/K, Boltzmann constant
ec = 1.60219e-19  # C
h_ = 1.05459e-34  # J*s, Planc constant
c = 2.99792e8  # m/s, light speed
m_e = 9.109e-31  # kg, electron mass
m_hh = 0.5 * m_e  # effective heavy hole mass
m_lh = 0.076 * m_e  # effective light hole mass
m_so = 0.145 * m_e  # effective split-off band mass
m_T = 0.063 * m_e  # Gamma valley effective electron mass
m_L = 0.55 * m_e  # L valley effective electron mass
m_X = 0.85 * m_e  # X valley effective electron mass
m_h = (m_hh**1.5 + m_lh**1.5 + m_so**1.5)**(2 / 3)

# ----Set material parameters----
T = 298  # K, material temperature
N_A = 1e25  # m**(-3), doping concentration
rou = 5.32e3  # kg/m**3, density of GaAs
E_T = kB * T / ec
Eg = 1.519 - 0.54 * 10**(-3) * T**2 / (T + 204)  # eV, bandgap
# Tiwari, S, Appl. Phys. Lett. 56, 6 (1990) 563-565. (experiment data)
# Eg = Eg - 2 * 10**(-11) * np.sqrt(N_A * 1e-6)
DEg = 3 * ec / 16 / pi / eps * np.sqrt(ec**2 * N_A / eps / kB / T)
Eg = Eg - DEg
Eg = float('%.2f' % Eg)
DE = 0.34  # eV, split-off energy gap
# E_B = Eg / 3  # only for NA = 10**19 cm**-3
EB_data = np.genfromtxt('GaAs_Band_Bending.csv', delimiter=',')
func0 = interp1d(EB_data[:, 0] * 1e6, EB_data[:, 1])
E_B = func0(N_A)
W_B = np.sqrt(2 * eps * E_B / ec / N_A) * 10**9  # nm
# print(Eg, E_B, W_B, DEg, E_T)
E_A = 0.01  # eV, electron affinity
thick = 1e4  # nm, thickness of GaAs active layer
surface = 0  # position of electron emission, z = 0

# ----Define simulation time, time step and total photon number----
total_time = 100e-12  # s
step_time = 1e-14  # s
Ni = 100000  # incident photon number

# ----Set the electrical field----
field_y = 0
field_z = 1e3  # V/m
E_sch = 0.001  # eV, vacuum level reduction by Schottky effect

# ----Set parameters for acoustic phonon scattering----
V_T = 7.01  # eV, acoustic deformation potential for Gamma valley
V_L = 9.2  # eV,  for L valley
V_X = 9.0  # eV, for X valley
ul = 5.24e3  # m/s, longitudial sound speed
alpha_T = 0.64  # 1/eV, nonparabolicity factor for Gamma valley
alpha_L = 0.461  # 1/eV, for L valley
alpha_X = 0.204  # 1/eV, for X valley

# ---- set parameters for piezoelectric acoustic phonon ------
K_av = 0.0252

# ----- set parameters for polar optical phonon scattering -----
E_lo = 0.03536  # eV, longitudinal optical phonon energy

# ------ set parameters for optical phonon scattering -----
num_T_valley = 1  # equivalent valleys for Gamma valley
num_L_valley = 4  # for L valley
num_X_valley = 3  # for X valley
E_L_T = 0.29  # eV, splitting energy between Gamma and L valley
E_X_T = 0.48  # eV, splitting energy between Gamma and X valley
D_T_L = 6.5e10  # eV/m, optical coupling constant for between G and L valley
D_T_X = 10e10  # eV/m, between Gamma to X valley
D_L_L = 5e10  # eV/m, between L and L valley
D_L_X = 2.9e10  # eV/m, between L and X valley
D_X_X = 7e10  # eV/m, between X and X valley
phonon_T_L = 0.0278  # eV, optical phonon energy between G and L valley
phonon_T_X = 0.0299
phonon_L_L = 0.029
phonon_L_X = 0.0293
phonon_X_X = 0.0299


def surface_reflection(hw):
    data = np.genfromtxt('GaAs_optical_constant.txt', delimiter=',')
    func = interp1d(data[:, 0], data[:, 2])
    '''
    fig, ax = plt.subplots()
    ax.plot(data[:, 0], data[:, 2])
    ax.set_xlabel('Photon energy (eV)', fontsize=16)
    ax.set_ylabel('Reflectivity (%)', fontsize=16)
    ax.set_xlim([1.25, 2.5])
    ax.set_xticks(np.arange(1.25, 2.51, 0.25))
    ax.set_ylim([0.3, 0.42])
    ax.tick_params('both', direction='in', labelsize=14)
    plt.tight_layout()
    plt.savefig('surface_reflectivity.pdf', format='pdf')
    plt.show()
    '''
    return func(hw)


def transportation_efficiency(hw):
    data = np.genfromtxt('GaAs_transportation_efficiency.csv', delimiter=',')
    func = interp1d(data[:, 0], 0.01 * data[:, 1])
    return func(hw)


def photon_to_electron(hw):
    ''' electrons in valence band aborption photon to excited to conduction
    band. Only consider these electrons in the heavy hole, light hole and
    split-off band would be excited, and can only excited into Gamma valley.
    Given photon energy, return excited electron energy. '''
    # nonparabolicity factor, 1/eV
    # alpha_T = 0.58 + (T - 77) * (0.61 - 0.58) / (300 - 77)
    Ei = random.uniform(Eg, hw)
    if Ei >= Eg + DE:
        x = random.randint(1, 6)
        if x in [1, 2, 3]:  # heavy hole
            E1 = Ei - Eg
            Gamma = 1 + m_hh / m_e + 2 * alpha_T * E1
            DE_h = (1 - np.sqrt(1 - 4 * alpha_T * E1 * (1 + alpha_T * E1) /
                                Gamma**2)) / (2 * alpha_T) * Gamma
            # DE_h = E1 / (1 + m_hh / m_T)
            E_e = E1 - DE_h
        elif x == 4:  # light hole
            E1 = Ei - Eg
            Gamma = 1 + m_lh / m_e + 2 * alpha_L * E1
            DE_h = (1 - np.sqrt(1 - 4 * alpha_L * E1 * (1 + alpha_L * E1) /
                                Gamma**2)) / (2 * alpha_T) * Gamma
            # DE_h = E1 / (1 + m_lh / m_T)
            E_e = E1 - DE_h
        elif x in [5, 6]:  # split-off band
            E1 = Ei - Eg - DE
            Gamma = 1 + m_so / m_e + 2 * alpha_X * E1
            DE_h = (1 - np.sqrt(1 - 4 * alpha_X * E1 * (1 + alpha_X * E1) /
                                Gamma**2)) / (2 * alpha_T) * Gamma
            # DE_h = E1 / (1 + m_so / m_T)
            E_e = E1 - DE_h
    elif Eg <= Ei < Eg + DE:
        x = random.randint(1, 4)
        if x in [1, 2, 3]:  # heavy hole
            E1 = Ei - Eg
            Gamma = 1 + m_hh / m_e + 2 * alpha_T * E1
            DE_h = (1 - np.sqrt(1 - 4 * alpha_T * E1 * (1 + alpha_T * E1) /
                                Gamma**2)) / (2 * alpha_T) * Gamma
            # DE_h = E1 / (1 + m_hh / m_T)
            E_e = E1 - DE_h
        elif x == 4:  # light hole
            E1 = Ei - Eg
            Gamma = 1 + m_lh / m_e + 2 * alpha_L * E1
            DE_h = (1 - np.sqrt(1 - 4 * alpha_L * E1 * (1 + alpha_L * E1) /
                                Gamma**2)) / (2 * alpha_T) * Gamma
            # DE_h = E1 / (1 + m_lh / m_T)
            E_e = E1 - DE_h
    else:
        E_e = 0.002
        E1 = 0.002
    # print(DE_h, E_e)
    return E_e


def electron_distribution(hw, types):
    '''photon decay as exponential distribution in GaAs,
    generating electrons with exponential distribution in z direction.
    Given photon energy, photon number and sample thickness.
    Return excited electron position(z,y),direction(vz,vy),velocity and energy
    '''
    energy = []
    if types == 1:  # Four bands photoexcited
        for i in range(Ni):
            energy.append(photon_to_electron(hw))
        energy = np.array(energy)
    elif types == 2:  # use density of state from reference
        DOS = np.genfromtxt('GaAs_DOS.csv', delimiter=',')
        func1 = interp1d(DOS[:, 0], DOS[:, 1])
        '''
        fig, ax = plt.subplots()
        e = np.linspace(-2.5, 2.8, 531)
        ax.plot(e, func1(e))
        #data = np.vstack([e, func1(e)]).T
        #np.savetxt('func_DOS.csv', data, delimiter=',', fmt='%.6f')
        plt.savefig('DOS.pdf', format='pdf')
        plt.show()'''
        E0 = Eg
        N = 105
        delta_E = (hw - E0) / N
        norm, err = integrate.quad(lambda e: func1(e - hw) * func1(e), E0, hw,
                                   limit=10000)
        Ei = np.linspace(E0, hw, N)
        for i in range(len(Ei)):
            num = 1.5 * Ni * func1(Ei[i]) * func1(Ei[i] - hw) * delta_E / norm
            E_num = np.empty(int(num))
            E_num.fill(Ei[i] - Eg)
            energy.extend(E_num)
        np.random.shuffle(energy)
    else:
        print('Wrong photon-to-electron type')
    absorb_data = np.genfromtxt('GaAs_optical_constant.txt', delimiter=',')
    func2 = interp1d(absorb_data[:, 0], absorb_data[:, 1])
    '''
    fig, ax = plt.subplots()
    ax.plot(absorb_data[:, 0], absorb_data[:, 1])
    ax.set_xlabel(r'Photon energy (eV)', fontsize=16)
    ax.set_ylabel(r'Absorption coefficient ($cm^{-1}$)', fontsize=16)
    ax.set_xlim([1.25, 2.5])
    ax.set_xticks(np.arange(1.25, 2.51, 0.25))
    ax.set_ylim([0, 1.4e5])
    ax.ticklabel_format(axis='y',style='sci',scilimits=(-2,2))
    ax.tick_params('both', direction='in', labelsize=14)
    plt.tight_layout()
    plt.savefig('absorption_coefficient.pdf', format='pdf')
    plt.show()
    '''
    alpha = func2(hw) * 10**-7  # 1/nm,  absorption coefficient
    # photon distribution in GaAs (exp distribution random variables)
    z_exp = expon.rvs(loc=0, scale=1 / alpha, size=Ni)
    # photon (electron) distribution in GaAs with thickness less than thick
    z_pos = [z for z in z_exp if z <= thick]
    z_pos = np.array(z_pos)
    Num = len(z_pos)
    energy = np.resize(energy, Num)
    # y axis, position set to gauss distribution
    y_pos = np.random.normal(0, 0.25e6, Num)
    x_pos = np.random.normal(0, 0.25e6, Num)
    velocity = np.sqrt(2 * np.abs(energy) * ec / m_T) * 10**9
    # Isotropic distribution, phi=2*pi*r, cos(theta)=1-2*r
    phi = two_pi * np.random.uniform(0, 1, Num)
    theta = np.arccos(1 - 2 * np.random.uniform(0, 1, Num))
    vz = velocity * np.cos(theta)
    vy = velocity * np.sin(theta) * np.sin(phi)
    vx = velocity * np.sin(theta) * np.cos(phi)
    distribution_2D = np.vstack((phi, theta, vx, vy, vz, energy, velocity, x_pos, y_pos, z_pos)).T
    return distribution_2D


def impurity_scattering(energy, types):
    '''
    types=1, scattering for Gamma valley
    types=2, scattering for L valley
    types=3, scattering for X valley
    '''
    energy = energy.clip(0.0001)
    E0 = energy
    n_i = N_A  # m**-3, impurity concentration
    T_i = T
    if types == 1:
        gamma_E = energy * (1 + alpha_T * energy)
        k_e = np.sqrt(2 * m_T * gamma_E * ec) / h_  # 1/m, wavevector
        # T_e = np.mean(energy) * ec / kB
        Bs = np.sqrt(n_i * ec**2 / eps / kB / T_i)  # 1/m
        b = 4 * k_e**2 / Bs**2
        # print(b**2 / (1 + b), np.log(1 + b) - b / (1 + b) * 2)
        # e-impurity scattering rate, (C. Jacoboni, RMP 55,645, 1983)
        Rate_ei = n_i * ec**4 * (1 + 2 * alpha_T * E0) / 32 / np.sqrt(2) / \
            pi / eps**2 / np.sqrt(m_T) / (gamma_E * ec)**1.5 * b**2 / (1 + b)
    elif types == 2:
        gamma_E = energy * (1 + alpha_L * energy)
        k_e = np.sqrt(2 * m_L * gamma_E * ec) / h_  # 1/m, wavevector
        Bs = np.sqrt(n_i * ec**2 / eps / kB / T_i)  # 1/m
        b = 4 * k_e**2 / Bs**2
        # e-impurity scattering rate, (C. Jacoboni, RMP 55,645, 1983)
        Rate_ei = n_i * ec**4 * (1 + 2 * alpha_L * E0) / 32 / np.sqrt(2) / \
            pi / eps**2 / np.sqrt(m_L) / (gamma_E * ec)**1.5 * b**2 / (1 + b)
    elif types == 3:
        gamma_E = energy * (1 + alpha_X * energy)
        k_e = np.sqrt(2 * m_X * gamma_E * ec) / h_  # 1/m, wavevector
        Bs = np.sqrt(n_i * ec**2 / eps / kB / T_i)  # 1/m
        b = 4 * k_e**2 / Bs**2
        # e-impurity scattering rate, (C. Jacoboni, RMP 55,645, 1983)
        Rate_ei = n_i * ec**4 * (1 + 2 * alpha_X * E0) / 32 / np.sqrt(2) / \
            pi / eps**2 / np.sqrt(m_X) / (gamma_E * ec)**1.5 * b**2 / (1 + b)
    else:
        print('Wrong electron impurity scattering')
    return Rate_ei


def electron_hole_scattering(energy, types):
    '''
    types=1, scattering for Gamma valley
    types=2, scattering for L valley
    types=3, scattering for X valley
    '''
    energy = energy.clip(0.0001)
    E0 = energy
    n_h = 0.05 * N_A  # m**-3, hole concentration
    T_h = T  # K, hole temperature
    beta2 = n_h * ec**2 / eps / kB / T_h
    if types == 1:
        gamma_E = energy * (1 + alpha_T * energy)
        b = 8 * m_T * gamma_E * ec / h_**2 / beta2
        Rate_eh = n_h * ec**4 / 32 / 2**0.5 / pi / eps**2 / m_T**0.5 / \
            gamma_E**1.5 / ec**1.5 * b**2 / (1 + b) * (1 + 2 * alpha_T * E0)
    elif types == 2:
        gamma_E = energy * (1 + alpha_L * energy)
        b = 8 * m_L * gamma_E * ec / h_**2 / beta2
        Rate_eh = n_h * ec**4 / 32 / 2**0.5 / pi / eps**2 / m_L**0.5 / \
            energy**1.5 / ec**1.5 * b**2 / (1 + b) * (1 + 2 * alpha_L * E0)
    elif types == 3:
        gamma_E = energy * (1 + alpha_X * energy)
        b = 8 * m_X * gamma_E * ec / h_**2 / beta2
        Rate_eh = n_h * ec**4 / 32 / 2**0.5 / pi / eps**2 / m_X**0.5 / \
            energy**1.5 / ec**1.5 * b**2 / (1 + b) * (1 + 2 * alpha_X * E0)
    else:
        print('Wrong electron-hole scattering type')
    return Rate_eh


def acoustic_phonon_scattering(energy, types):
    '''
    types=1, scattering for Gamma valley
    types=2, scattering for L valley
    types=3, scattering for X valley
    '''
    energy = energy.clip(0)
    if types == 1:
        Rate_ac = 2**0.5 * m_T**1.5 * kB * T * V_T**2 * energy**0.5 *\
            ec**2.5 / pi / h_**4 / ul**2 / rou *\
            (1 + 2 * alpha_T * energy) *\
            (1 + alpha_T * energy)**0.5
    elif types == 2:
        Rate_ac = 2**0.5 * m_L**1.5 * kB * T * V_L**2 * energy**0.5 *\
            ec**2.5 / pi / h_**4 / ul**2 / rou *\
            (1 + 2 * alpha_L * energy) *\
            (1 + alpha_L * energy)**0.5
    elif types == 3:
        Rate_ac = 2**0.5 * m_X**1.5 * kB * T * V_X**2 * energy**0.5 *\
            ec**2.5 / pi / h_**4 / ul**2 / rou *\
            (1 + 2 * alpha_X * energy) *\
            (1 + alpha_X * energy)**0.5
    else:
        print('Wrong acoustic phonon scattering type')
    return Rate_ac


def piezoelectric_phonon_scattering(energy, types):
    '''
    types=1, scattering for Gamma valley
    types=2, scattering for L valley
    types=3, scattering for X valley
    '''
    energy = energy.clip(0.001)
    n_h = 1 * N_A  # m**-3, hole concentration
    beta2 = n_h * ec**2 / eps / kB / T
    if types == 1:
        gamma_E = energy * (1 + alpha_T * energy)
        vel = np.sqrt(2 * np.abs(gamma_E) * ec / m_T)
        b = 8 * m_T * gamma_E * ec / h_**2 / beta2
        Rate_pap = ec**2 * K_av * kB * T / 4 / pi / eps / h_**2 / vel * \
            (np.log(1 + b))
    elif types == 2:
        gamma_E = energy * (1 + alpha_L * energy)
        b = 8 * m_L * gamma_E * ec / h_**2 / beta2
        vel = np.sqrt(2 * np.abs(gamma_E) * ec / m_L)
        Rate_pap = ec**2 * K_av * kB * T / 4 / pi / eps / h_**2 / vel * \
            (np.log(1 + b))
    elif types == 3:
        gamma_E = energy * (1 + alpha_X * energy)
        b = 8 * m_X * gamma_E * ec / h_**2 / beta2
        vel = np.sqrt(2 * np.abs(gamma_E) * ec / m_X)
        Rate_pap = ec**2 * K_av * kB * T / 4 / pi / eps / h_**2 / vel * \
            (np.log(1 + b))
    else:
        print('Wrong electron-hole scattering type')
    return Rate_pap


def polar_optical_scattering(energy, types):
    '''
    types=1, scattering for Gamma valley
    types=2, scattering for L valley
    types=3, scattering for X valley
    '''
    energy = energy.clip(0.001)
    eps_p = (1 / eps_high - 1 / eps)
    N_lo = kB * T / ec / E_lo
    N_i = N_A
    q0 = np.sqrt(N_i * ec**2 / eps / kB / T)
    if types == 1:
        gamma_E = energy * (1 + alpha_T * energy)
        vel = np.sqrt(2 * np.abs(energy) * ec / m_T)
        q_min_ab = np.sqrt(2 * m_T * gamma_E) / h_ * \
            (-1 + np.sqrt(1 + E_lo / energy))
        q_max_ab = np.sqrt(2 * m_T * gamma_E) / h_ * \
            (1 + np.sqrt(1 + E_lo / energy))
        C0_ab = 0.5 * q0**2 * (1 / (q_max_ab**2 + q0**2) -
                               1 / (q_min_ab**2 + q0**2)) + \
            np.log((q_max_ab**2 + q0**2) / (q_min_ab**2 + q0**2))
        Rate_ab = ec**3 * E_lo * eps_p / 4 / pi / h_**2 / vel * N_lo * C0_ab

        q_min_em = -np.sqrt(2 * m_T * gamma_E) / h_ * \
            (-1 + np.sqrt((1 - E_lo / energy).clip(0)))
        q_max_em = np.sqrt(2 * m_T * gamma_E) / h_ * \
            (1 + np.sqrt((1 - E_lo / energy).clip(0)))
        C0_em = 0.5 * q0**2 * (1 / (q_max_em**2 + q0**2) -
                               1 / (q_min_em**2 + q0**2)) + \
            np.log((q_max_em**2 + q0**2) / (q_min_em**2 + q0**2))
        Rate_em = ec**3 * E_lo * eps_p / 4 / \
            pi / h_**2 / vel * (N_lo + 1) * C0_em
    if types == 2:
        gamma_E = energy * (1 + alpha_L * energy)
        vel = np.sqrt(2 * np.abs(energy) * ec / m_L)
        q_min_ab = np.sqrt(2 * m_L * gamma_E) / h_ * \
            (-1 + np.sqrt(1 + E_lo / energy))
        q_max_ab = np.sqrt(2 * m_L * gamma_E) / h_ * \
            (1 + np.sqrt(1 + E_lo / energy))
        C0_ab = 0.5 * q0**2 * (1 / (q_max_ab**2 + q0**2) -
                               1 / (q_min_ab**2 + q0**2)) + \
            np.log((q_max_ab**2 + q0**2) / (q_min_ab**2 + q0**2))
        Rate_ab = ec**3 * E_lo * eps_p / 4 / pi / h_**2 / vel * N_lo * C0_ab

        q_min_em = -np.sqrt(2 * m_L * gamma_E) / h_ * \
            (-1 + np.sqrt((1 - E_lo / energy).clip(0)))
        q_max_em = np.sqrt(2 * m_L * gamma_E) / h_ * \
            (1 + np.sqrt((1 - E_lo / energy).clip(0)))
        C0_em = 0.5 * q0**2 * (1 / (q_max_em**2 + q0**2) -
                               1 / (q_min_em**2 + q0**2)) + \
            np.log((q_max_em**2 + q0**2) / (q_min_em**2 + q0**2))
        Rate_em = ec**3 * E_lo * eps_p / 4 / \
            pi / h_**2 / vel * (N_lo + 1) * C0_em
    if types == 3:
        gamma_E = energy * (1 + alpha_X * energy)
        vel = np.sqrt(2 * np.abs(energy) * ec / m_X)
        q_min_ab = np.sqrt(2 * m_X * gamma_E) / h_ * \
            (-1 + np.sqrt(1 + E_lo / energy))
        q_max_ab = np.sqrt(2 * m_X * gamma_E) / h_ * \
            (1 + np.sqrt(1 + E_lo / energy))
        C0_ab = 0.5 * q0**2 * (1 / (q_max_ab**2 + q0**2) -
                               1 / (q_min_ab**2 + q0**2)) + \
            np.log((q_max_ab**2 + q0**2) / (q_min_ab**2 + q0**2))
        Rate_ab = ec**3 * E_lo * eps_p / 4 / pi / h_**2 / vel * N_lo * C0_ab

        q_min_em = -np.sqrt(2 * m_X * gamma_E) / h_ * \
            (-1 + np.sqrt((1 - E_lo / energy).clip(0)))
        q_max_em = np.sqrt(2 * m_X * gamma_E) / h_ * \
            (1 + np.sqrt((1 - E_lo / energy).clip(0)))
        C0_em = 0.5 * q0**2 * (1 / (q_max_em**2 + q0**2) -
                               1 / (q_min_em**2 + q0**2)) + \
            np.log((q_max_em**2 + q0**2) / (q_min_em**2 + q0**2))
        Rate_em = ec**3 * E_lo * eps_p / 4 / \
            pi / h_**2 / vel * (N_lo + 1) * C0_em
    # print(Rate_ab, Rate_em, C0_ab, C0_em)
    return Rate_ab, Rate_em


def optical_phonon_scattering(energy, types):
    '''
    types=1, scattering for Gamma to L valley
    types=2, scattering for Gamma to X valley
    types=3, scattering for L to Gamma valley
    types=4, scattering for L to L valley
    types=5, scattering for L to X valley
    types=6, scattering for X to Gamma valley
    types=7, scattering for X to L valley
    types=8, scattering for X to X valley
    '''
    if types == 1:
        N_op = kB * T / ec / phonon_T_L
        tempEnergy = (energy + phonon_T_L - E_L_T).clip(0)
        Rate_ab = D_T_L**2 * m_T**1.5 * num_L_valley / 2**0.5 / pi / rou /\
            h_**2 / phonon_T_L * N_op * ec * \
            np.sqrt(ec * tempEnergy) *\
            (1 + 2 * alpha_L * tempEnergy) *\
            (1 + alpha_L * tempEnergy)**0.5
        tempEnergy = (energy - phonon_T_L - E_L_T).clip(0)
        Rate_em = D_T_L**2 * m_T**1.5 * num_L_valley / 2**0.5 / pi / rou /\
            h_**2 / phonon_T_L * (N_op + 1) * ec *\
            np.sqrt(ec * tempEnergy) *\
            (1 + 2 * alpha_L * tempEnergy) *\
            (1 + alpha_L * tempEnergy)**0.5
    elif types == 2:
        N_op = kB * T / ec / phonon_T_X
        tempEnergy = (energy + phonon_T_X - E_X_T).clip(0)
        Rate_ab = D_T_X**2 * m_T**1.5 * num_X_valley / 2**0.5 / pi / rou /\
            h_**2 / phonon_T_X * N_op * ec * \
            np.sqrt(ec * tempEnergy) *\
            (1 + 2 * alpha_X * tempEnergy) *\
            (1 + alpha_X * tempEnergy)**0.5
        tempEnergy = (energy - phonon_T_X - E_X_T).clip(0)
        Rate_em = D_T_X**2 * m_T**1.5 * num_X_valley / 2**0.5 / pi / rou /\
            h_**2 / phonon_T_X * (N_op + 1) * ec *\
            np.sqrt(ec * tempEnergy) *\
            (1 + 2 * alpha_X * tempEnergy) *\
            (1 + alpha_X * tempEnergy)**0.5
    elif types == 3:
        N_op = kB * T / ec / phonon_T_L
        tempEnergy = (energy + phonon_T_L + E_L_T).clip(0)
        Rate_ab = D_T_L**2 * m_L**1.5 * num_T_valley / 2**0.5 / pi / rou /\
            h_**2 / phonon_T_L * N_op * ec * \
            np.sqrt(ec * tempEnergy) *\
            (1 + 2 * alpha_T * tempEnergy) *\
            (1 + alpha_T * tempEnergy)**0.5
        tempEnergy = (energy - phonon_T_L + E_L_T).clip(0)
        Rate_em = D_T_L**2 * m_L**1.5 * num_T_valley / 2**0.5 / pi / rou /\
            h_**2 / phonon_T_L * (N_op + 1) * ec *\
            np.sqrt(ec * tempEnergy) *\
            (1 + 2 * alpha_T * tempEnergy) *\
            (1 + alpha_T * tempEnergy)**0.5
    elif types == 4:
        N_op = kB * T / ec / phonon_L_L
        tempEnergy = (energy + phonon_L_L).clip(0)
        Rate_ab = D_L_L**2 * m_L**1.5 * (num_L_valley - 1) / 2**0.5 /\
            pi / rou / h_**2 / phonon_L_L * N_op * ec * \
            np.sqrt(ec * tempEnergy) *\
            (1 + 2 * alpha_L * tempEnergy) *\
            (1 + alpha_L * tempEnergy)**0.5
        tempEnergy = (energy - phonon_L_L).clip(0)
        Rate_em = D_L_L**2 * m_L**1.5 * (num_L_valley - 1) / 2**0.5 /\
            pi / rou / h_**2 / phonon_L_L * (N_op + 1) * ec *\
            np.sqrt(ec * tempEnergy) *\
            (1 + 2 * alpha_L * tempEnergy) *\
            (1 + alpha_L * tempEnergy)**0.5
    elif types == 5:
        N_op = kB * T / ec / phonon_L_X
        tempEnergy = (energy + phonon_L_X + E_L_T - E_X_T).clip(0)
        Rate_ab = D_L_X**2 * m_L**1.5 * num_X_valley / 2**0.5 /\
            pi / rou / h_**2 / phonon_L_X * N_op * ec * \
            np.sqrt(ec * tempEnergy) *\
            (1 + 2 * alpha_X * tempEnergy) *\
            (1 + alpha_X * tempEnergy)**0.5
        tempEnergy = (energy - phonon_L_X + E_L_T - E_X_T).clip(0)
        Rate_em = D_L_X**2 * m_L**1.5 * num_X_valley / 2**0.5 /\
            pi / rou / h_**2 / phonon_L_X * (N_op + 1) * ec *\
            np.sqrt(ec * tempEnergy) *\
            (1 + 2 * alpha_X * tempEnergy) *\
            (1 + alpha_X * tempEnergy)**0.5
    elif types == 6:
        N_op = kB * T / ec / phonon_T_X
        tempEnergy = (energy + phonon_T_X + E_X_T).clip(0)
        Rate_ab = D_T_X**2 * m_X**1.5 * num_T_valley / 2**0.5 / pi / rou /\
            h_**2 / phonon_T_X * N_op * ec * \
            np.sqrt(ec * tempEnergy) *\
            (1 + 2 * alpha_T * tempEnergy) *\
            (1 + alpha_T * tempEnergy)**0.5
        tempEnergy = (energy - phonon_T_X + E_X_T).clip(0)
        Rate_em = D_T_X**2 * m_X**1.5 * num_T_valley / 2**0.5 / pi / rou /\
            h_**2 / phonon_T_L * (N_op + 1) * ec *\
            np.sqrt(ec * tempEnergy) *\
            (1 + 2 * alpha_T * tempEnergy) *\
            (1 + alpha_T * tempEnergy)**0.5
    elif types == 7:
        N_op = kB * T / ec / phonon_L_X
        tempEnergy = (energy + phonon_L_X - E_L_T + E_X_T).clip(0)
        Rate_ab = D_L_X**2 * m_X**1.5 * num_L_valley / 2**0.5 /\
            pi / rou / h_**2 / phonon_L_X * N_op * ec * \
            np.sqrt(ec * tempEnergy) *\
            (1 + 2 * alpha_L * tempEnergy) *\
            (1 + alpha_L * tempEnergy)**0.5
        tempEnergy = (energy - phonon_L_X - E_L_T + E_X_T).clip(0)
        Rate_em = D_L_X**2 * m_X**1.5 * num_L_valley / 2**0.5 /\
            pi / rou / h_**2 / phonon_L_X * (N_op + 1) * ec *\
            np.sqrt(ec * tempEnergy) *\
            (1 + 2 * alpha_L * tempEnergy) *\
            (1 + alpha_L * tempEnergy)**0.5
    elif types == 8:
        N_op = kB * T / ec / phonon_X_X
        tempEnergy = (energy + phonon_X_X).clip(0)
        Rate_ab = D_X_X**2 * m_X**1.5 * (num_X_valley - 1) / 2**0.5 /\
            pi / rou / h_**2 / phonon_X_X * N_op * ec * \
            np.sqrt(ec * tempEnergy) *\
            (1 + 2 * alpha_X * tempEnergy) *\
            (1 + alpha_X * tempEnergy)**0.5
        tempEnergy = (energy - phonon_X_X).clip(0)
        Rate_em = D_X_X**2 * m_X**1.5 * (num_X_valley - 1) / 2**0.5 /\
            pi / rou / h_**2 / phonon_X_X * (N_op + 1) * ec *\
            np.sqrt(ec * tempEnergy) *\
            (1 + 2 * alpha_X * tempEnergy) *\
            (1 + alpha_X * tempEnergy)**0.5
    else:
        print('Wrong optical phonon scattering type')
    return Rate_ab, Rate_em


def electron_impurity_transfer_energy(dist_2D, Rate, stept):
    Num = len(dist_2D)
    P = stept * Rate  # scattering probability
    P_ind = np.random.uniform(0, 1, Num) <= P
    energy_ind = dist_2D[:, 5] > 0
    happen = P_ind * energy_ind
    E_loss = np.random.uniform(0, dist_2D[:, 5]).clip(0) * happen
    # dist_2D[:, 5] = dist_2D[:, 5] - E_loss
    return dist_2D, happen


def electron_hole_transfer_energy(dist_2D, hole_energy, Rate, stept):
    Num = len(dist_2D)
    # hole_velocity = maxwell.rvs(0, np.sqrt(kB * T / m_h), Num)
    # hole_energy1 = m_h * hole_velocity**2 / 2 / ec
    P_eh = stept * Rate
    random_P_eh = np.random.uniform(0, 1, Num)
    P_eh_ind = random_P_eh <= P_eh
    energy_eh_ind = dist_2D[:, 5] > 0
    happen = P_eh_ind * energy_eh_ind
    E_h = np.mean(hole_energy)
    E_h = np.resize(hole_energy, Num)
    eh_loss = np.random.uniform(0, dist_2D[:, 5] - E_h) * happen
    dist_2D[:, 5] = dist_2D[:, 5] - eh_loss
    # hole_energy = hole_energy + np.mean(eh_loss)
    # hole_energy = np.abs(hole_energy)
    # print(np.mean(dist_2D[:, 5]), np.mean(hole_energy))
    return dist_2D, hole_energy, happen


def electron_acoustic_tranfer_energy(dist_2D, Rate, stept):
    Num = len(dist_2D)
    P_ac = stept * Rate
    # energy transfer
    # ac_energy = 2 * m_T * dist_2D[:, 4] * ul *\
    #    np.cos(np.random.uniform(0, two_pi, Num))
    # scatterred electron index
    P_ac_ind = np.random.uniform(0, 1, Num) <= P_ac
    happen = P_ac_ind
    # dist_2D[:, 5] = dist_2D[:, 5] - ac_energy * happen
    return dist_2D, happen


def electron_piezoelectric_tranfer_energy(dist_2D, Rate, stept):
    Num = len(dist_2D)
    P_ac = stept * Rate
    P_ac_ind = np.random.uniform(0, 1, Num) <= P_ac
    happen = P_ac_ind
    return dist_2D, happen


def electron_polar_transfer_energy(dist, Rate_ab, Rate_em, stept, types):
    Num = len(dist)
    P_ab = stept * Rate_ab
    P_em = stept * Rate_em
    P_ab_ind = np.random.uniform(0, 1, Num) <= P_ab
    P_em_ind = np.random.uniform(0, 1, Num) <= P_em
    Ei = dist[:, 5].clip(0.001)
    Ep_ab = (dist[:, 5] + E_lo).clip(0.0354)
    Ep_em = (dist[:, 5] - E_lo).clip(0.0001)
    happen = P_ab_ind + P_em_ind
    dist[:, 5] = dist[:, 5] + E_lo * P_ab_ind
    dist[:, 5] = dist[:, 5] - E_lo * P_em_ind

    # ----- renew the velocity -----
    '''
    renew the velocity and direction after scattering
    r is a random number uniforly distribution between 0 and 1
    for all scattering:
        phi = 2*pi*r
    for POP:
        cos(theta) = ((1+xi)-(1+2*xi)**r)/xi
        xi = 2*sqrt(gamma_E*gamma_Ep)/(sqrt(gamma_E)-sqrt(gamma_Ep))**2
    '''
    phi0 = dist[:, 0]
    theta0 = dist[:, 1]
    energy_ind = dist[:, 5] > 0
    happen = happen * energy_ind
    phi = two_pi * np.random.uniform(0, 1, Num)
    r = np.random.uniform(0, 1, Num)
    if types == 1:
        gamma_E = Ei * (1 + alpha_T * Ei)
        gamma_Ep_ab = Ep_ab * (1 + alpha_T * Ep_ab)
        gamma_Ep_em = Ep_em * (1 + alpha_T * Ep_em)
        xi_ab = 2 * np.sqrt(gamma_E * gamma_Ep_ab) / \
            (np.sqrt(gamma_E) - np.sqrt(gamma_Ep_ab))**2
        xi_em = 2 * np.sqrt(gamma_E * gamma_Ep_em) / \
            (np.sqrt(gamma_E) - np.sqrt(gamma_Ep_em))**2
        theta = np.arccos((1 + xi_ab - (1 + 2 * xi_ab)**r) / xi_ab) * \
            P_ab_ind + \
            np.arccos((1 + xi_em - (1 + 2 * xi_em)**r) / xi_em) * P_em_ind
        dist[:, 6] = dist[:, 6] * (~happen) + np.sqrt(
            2 * np.abs(dist[:, 5]) * ec / m_T) * 10**9 * happen
    elif types == 2:
        gamma_E = Ei * (1 + alpha_L * Ei)
        gamma_Ep_ab = Ep_ab * (1 + alpha_L * Ep_ab)
        gamma_Ep_em = Ep_em * (1 + alpha_L * Ep_em)
        xi_ab = 2 * np.sqrt(gamma_E * gamma_Ep_ab) / \
            (np.sqrt(gamma_E) - np.sqrt(gamma_Ep_ab))**2
        xi_em = 2 * np.sqrt(gamma_E * gamma_Ep_em) / \
            (np.sqrt(gamma_E) - np.sqrt(gamma_Ep_em))**2
        theta = np.arccos((1 + xi_ab - (1 + 2 * xi_ab)**r) / xi_ab) * \
            P_ab_ind + \
            np.arccos((1 + xi_em - (1 + 2 * xi_em)**r) / xi_em) * P_em_ind
        dist[:, 6] = dist[:, 6] * (~happen) + np.sqrt(
            2 * np.abs(dist[:, 5]) * ec / m_L) * 10**9 * happen
    elif types == 3:
        gamma_E = Ei * (1 + alpha_X * Ei)
        gamma_Ep_ab = Ep_ab * (1 + alpha_X * Ep_ab)
        gamma_Ep_em = Ep_em * (1 + alpha_X * Ep_em)
        xi_ab = 2 * np.sqrt(gamma_E * gamma_Ep_ab) / \
            (np.sqrt(gamma_E) - np.sqrt(gamma_Ep_ab))**2
        xi_em = 2 * np.sqrt(gamma_E * gamma_Ep_em) / \
            (np.sqrt(gamma_E) - np.sqrt(gamma_Ep_em))**2
        theta = np.arccos((1 + xi_ab - (1 + 2 * xi_ab)**r) / xi_ab) * \
            P_ab_ind + \
            np.arccos((1 + xi_em - (1 + 2 * xi_em)**r) / xi_em) * P_em_ind
        dist[:, 6] = dist[:, 6] * (~happen) + np.sqrt(
            2 * np.abs(dist[:, 5]) * ec / m_X) * 10**9 * happen
    else:
        print('wrong valley type: 1-Gamma, 2-L, 3-X')
        theta = np.zeros(Num)

    v_xp = dist[:, 6] * np.sin(theta) * np.cos(phi) * happen +\
        dist[:, 2] * (~happen)
    v_yp = dist[:, 6] * np.sin(theta) * np.sin(phi) * happen +\
        dist[:, 3] * (~happen)
    v_zp = dist[:, 6] * np.cos(theta) * happen + \
        dist[:, 4] * (~happen)
    dist[:, 2] = dist[:, 2] * (~happen) + happen * \
        (v_xp * np.cos(phi0) * np.cos(theta0) - \
        v_yp * np.sin(phi0) + v_zp * np.cos(phi0) * np.sin(theta0))
    dist[:, 3] = dist[:, 3] * (~happen) + happen * \
        (v_xp * np.sin(phi0) * np.cos(theta0) + \
        v_yp * np.cos(phi0) + v_zp * np.sin(phi0) * np.sin(theta0))
    dist[:, 4] = dist[:, 4] * (~happen) + \
        (-v_xp * np.sin(theta0) + v_zp * np.cos(theta0)) * happen

    zero_v_ind = dist[:, 6] == 0
    dist[zero_v_ind, 6] = 0.0001
    theta1 = np.arccos(dist[:, 4] / dist[:, 6])
    phi1 = np.arccos(dist[:, 2] / dist[:, 6] / np.sin(theta1))
    dist[:, 0] = phi1 * happen + dist[:, 0] * (~happen)
    dist[:, 1] = theta1 * happen + dist[:, 1] * (~happen)
    return dist


def electron_optical_transfer_energy(dist, Rate_ab, Rate_em,
                                     E_ab, E_em, stept, types):
    new_valley_dist = []
    new = np.array([])

    Num = len(dist)
    P_ab = stept * Rate_ab
    P_ab_ind = np.random.uniform(0, 1, Num) <= P_ab
    dist[:, 5] = dist[:, 5] + E_ab * P_ab_ind
    new = dist[P_ab_ind, :]
    dist = dist[(~P_ab_ind), :]
    new_valley_dist.extend(new.tolist())

    new = np.array([])
    Num = len(dist)
    Rate_em = Rate_em[(~P_ab_ind)]
    P_em = stept * Rate_em
    P_em_ind = np.random.uniform(0, 1, Num) <= P_em
    dist[:, 5] = dist[:, 5] + E_em * P_em_ind
    new = dist[P_em_ind, :]
    dist = dist[(~P_em_ind), :]
    new_valley_dist.extend(new.tolist())

    new_valley_dist = np.array(new_valley_dist)
    if len(new_valley_dist) > 0:
        new_valley_dist = renew_distribution(new_valley_dist, True, types)
    return dist, new_valley_dist.tolist()


def renew_coulomb_distribution(dist_2D, happen, types):
    '''renew the velocity and direction after scattering
    r is a random number uniforly distribution between 0 and 1
    for all scattering:
        phi = 2*pi*r
    for Coulomb scattering:
        cos(theta)=1-2*r/(1+4*k**2*LD**2*(1-r))
    '''
    Num = len(dist_2D)
    phi0 = dist_2D[:, 0]
    theta0 = dist_2D[:, 1]
    energy_ind = dist_2D[:, 5] > 0
    happen = happen * energy_ind
    phi = two_pi * np.random.uniform(0, 1, Num)
    r = np.random.uniform(0, 1, Num)

    energy = dist_2D[:, 5].clip(0.0001)
    n_h = 0.05 * N_A  # m**-3, hole concentration
    T_h = T  # K, hole temperature
    beta2 = n_h * ec**2 / eps / kB / T_h
    if types == 1:
        gamma_E = energy * (1 + alpha_T * energy)
        b = 8 * m_T * gamma_E * ec / h_**2 / beta2
        theta = np.arccos(1 - 2 * r / (1 + b * (1 - r))) * happen
        dist_2D[:, 6] = dist_2D[:, 6] * (~happen) + np.sqrt(
            2 * np.abs(dist_2D[:, 5]) * ec / m_T) * 10**9 * happen
    elif types == 2:
        gamma_E = energy * (1 + alpha_L * energy)
        b = 8 * m_L * gamma_E * ec / h_**2 / beta2
        theta = np.arccos(1 - 2 * r / (1 + b * (1 - r))) * happen
        dist_2D[:, 6] = dist_2D[:, 6] * (~happen) + np.sqrt(
            2 * np.abs(dist_2D[:, 5]) * ec / m_L) * 10**9 * happen
    elif types == 3:
        gamma_E = energy * (1 + alpha_X * energy)
        b = 8 * m_X * gamma_E * ec / h_**2 / beta2
        theta = np.arccos(1 - 2 * r / (1 + b * (1 - r))) * happen
        dist_2D[:, 6] = dist_2D[:, 6] * (~happen) + np.sqrt(
            2 * np.abs(dist_2D[:, 5]) * ec / m_X) * 10**9 * happen
    else:
        print('Wrong renew distribution type for coulomb scattering')

    v_xp = dist_2D[:, 6] * np.sin(theta) * np.cos(phi) * happen +\
        dist_2D[:, 2] * (~happen)
    v_yp = dist_2D[:, 6] * np.sin(theta) * np.sin(phi) * happen +\
        dist_2D[:, 3] * (~happen)
    v_zp = dist_2D[:, 6] * np.cos(theta) * happen + \
        dist_2D[:, 4] * (~happen)
    dist_2D[:, 2] = dist_2D[:, 2] * (~happen) + happen * \
        (v_xp * np.cos(phi0) * np.cos(theta0) - \
        v_yp * np.sin(phi0) + v_zp * np.cos(phi0) * np.sin(theta0))
    dist_2D[:, 3] = dist_2D[:, 3] * (~happen) + happen * \
        (v_xp * np.sin(phi0) * np.cos(theta0) + \
        v_yp * np.cos(phi0) + v_zp * np.sin(phi0) * np.sin(theta0))
    dist_2D[:, 4] = dist_2D[:, 4] * (~happen) + \
        (-v_xp * np.sin(theta0) + v_zp * np.cos(theta0)) * happen

    zero_v_ind = dist_2D[:, 6] == 0
    dist_2D[zero_v_ind, 6] = 0.0001
    theta1 = np.arccos(dist_2D[:, 4] / dist_2D[:, 6])
    phi1 = np.arccos(dist_2D[:, 2] / dist_2D[:, 6] / np.sin(theta1))
    dist_2D[:, 0] = phi1 * happen + dist_2D[:, 0] * (~happen)
    dist_2D[:, 1] = theta1 * happen + dist_2D[:, 1] * (~happen)
    return dist_2D


def renew_distribution(dist_2D, happen, types):
    '''renew the velocity and direction after scattering
    dist_2D = [z, y, vz, vy, v, E]
    r is a random number uniforly distribution between 0 and 1
    for all scattering:
        phi = 2*pi*r
    1. for isotropic scattering (acoustic and opical phonon scattering):
        cos(theta) = 1-2*r
    '''
    Num = len(dist_2D)
    energy_ind = dist_2D[:, 5] > 0
    happen = happen * energy_ind
    phi = two_pi * np.random.uniform(0, 1, Num)
    theta = np.arccos(1 - 2 * np.random.uniform(0, 1, Num))
    if types == 1:
        dist_2D[:, 6] = dist_2D[:, 6] * (~happen) + happen * \
            np.sqrt(2 * np.abs(dist_2D[:, 5]) * ec / m_T) * 10**9
    elif types == 2:
        dist_2D[:, 6] = dist_2D[:, 6] * (~happen) + happen * \
            np.sqrt(2 * np.abs(dist_2D[:, 5]) * ec / m_L) * 10**9
    elif types == 3:
        dist_2D[:, 6] = dist_2D[:, 6] * (~happen) + happen * \
            np.sqrt(2 * np.abs(dist_2D[:, 5]) * ec / m_X) * 10**9
    else:
        print('Wrong renew distribution type for isotropic scattering')
    dist_2D[:, 2] = dist_2D[:, 6] * np.sin(theta) * np.cos(phi) * happen +\
        dist_2D[:, 2] * (~happen)
    dist_2D[:, 3] = dist_2D[:, 6] * np.sin(theta) * np.sin(phi) * happen +\
        dist_2D[:, 3] * (~happen)
    dist_2D[:, 4] = dist_2D[:, 6] * np.cos(theta) * happen + \
        dist_2D[:, 4] * (~happen)
    dist_2D[:, 0] = phi * happen + dist_2D[:, 0] * (~happen)
    dist_2D[:, 1] = theta * happen + dist_2D[:, 1] * (~happen)
    return dist_2D


def electron_transport(distribution_2D, types):
    '''electron transport in the conduction banc (CB) and suffer scattering:
    1. e-phonon (e-p) scattering:
        gain or loss the energy of a phonon (ep) after each scattering
    2. e-e scattering:
        a. scattering with electrons in CB can be ignored
        b. scattering with electrons in VB when energy bigger than Eg
    3. e-h scattering:
        main scattering mechanism in p-type GaAs, loss most energy
    4. e-impurity scattering:
        non-charged scattering can be ignored
        charged scattering is considered here
    '''
    surface_2D = []
    trap_2D = []
    back_2D = []
    time_data = []
    if types == 1:
        '''
        including e-ph, e-h and e-impurity scattering
        including Gamma, L and X valley scattering
        Assume all photoexcited electrons are in Gamma valley
        '''
        dist_2D = distribution_2D  # electrons in Gamma valley
        dist_L = np.array([])  # no electrons in L valley before scattering
        dist_X = np.array([])  # no electrons in X valley before scattering
        dist_TtoL = []
        dist_TtoX = []
        dist_LtoT = []
        dist_LtoX = []
        dist_XtoT = []
        dist_XtoL = []
        t = 0
        # assuming holes are steady state of maxwell-boltzmann distribution
        # hole temperature T_h = T = 298 K
        hole_velocity = maxwell.rvs(0, np.sqrt(kB * T / m_h), len(dist_2D))
        hole_energy = m_h * hole_velocity**2 / 2 / ec
        time_data.append([t * 10**12, np.mean(dist_2D[:, 5]) * 10**3,
                          0, len(dist_2D), 0, 0])
        while t < total_time:
            # electrons in Gamma valley begin scattering
            # tempEnergy = dist_2D[:, 5].clip(0.001)
            Num = len(dist_2D)
            # e-impurity scattering rate, (Y. Nishimura, Jnp. J. Appl. Phys.)
            Rate_ei = impurity_scattering(dist_2D[:, 5], 1)
            # e-h scattering rate
            Rate_eh = electron_hole_scattering(dist_2D[:, 5], 1)
            # acounstic phonon scattering rate
            Rate_ac = acoustic_phonon_scattering(dist_2D[:, 5], 1)

            # piezoelectric scattering rate
            Rate_pie = piezoelectric_phonon_scattering(dist_2D[:, 5], 1)

            # polar optical phonon scattering rate
            Rate_pop_ab, Rate_pop_em = polar_optical_scattering(
                dist_2D[:, 5], 1)

            # optical phonon scattering rate or intervalley scattering
            Rate_TL_ab, Rate_TL_em = optical_phonon_scattering(
                dist_2D[:, 5], 1)
            Rate_TX_ab, Rate_TX_em = optical_phonon_scattering(
                dist_2D[:, 5], 2)
            max_Rate_T = np.max([np.mean(Rate_ei), np.mean(Rate_eh),
                                 np.mean(Rate_ac), np.mean(Rate_TL_ab),
                                 np.mean(Rate_pop_ab), np.mean(Rate_pop_em),
                                 np.mean(Rate_TL_em), np.mean(Rate_TX_ab),
                                 np.mean(Rate_TX_em), np.mean(Rate_pie)])
            if len(dist_L) > 0:
                Rate_ei = impurity_scattering(dist_L[:, 5], 2)
                Rate_eh = electron_hole_scattering(dist_L[:, 5], 2)
                Rate_ac = acoustic_phonon_scattering(dist_L[:, 5], 2)
                Rate_pie = piezoelectric_phonon_scattering(dist_L[:, 5], 2)
                Rate_pop_ab, Rate_pop_em = polar_optical_scattering(
                    dist_L[:, 5], 2)
                Rate_LT_ab, Rate_LT_em = optical_phonon_scattering(
                    dist_L[:, 5], 3)
                Rate_LL_ab, Rate_LL_em = optical_phonon_scattering(
                    dist_L[:, 5], 4)
                Rate_LX_ab, Rate_LX_em = optical_phonon_scattering(
                    dist_L[:, 5], 5)
                max_Rate_L = np.max([np.mean(Rate_ei), np.mean(Rate_eh),
                                     np.mean(Rate_ac), np.mean(Rate_pop_ab),
                                     np.mean(Rate_pop_em), np.mean(Rate_pie),
                                     np.mean(Rate_LT_ab), np.mean(Rate_LT_em),
                                     np.mean(Rate_LL_ab), np.mean(Rate_LL_em),
                                     np.mean(Rate_LX_ab), np.mean(Rate_LX_em)])
            else:
                max_Rate_L = 0
            if len(dist_X) > 0:
                Rate_ei = impurity_scattering(dist_X[:, 5], 3)
                Rate_eh = electron_hole_scattering(dist_X[:, 5], 3)
                Rate_ac = acoustic_phonon_scattering(dist_X[:, 5], 3)
                Rate_pie = piezoelectric_phonon_scattering(dist_X[:, 5], 3)
                Rate_pop_ab, Rate_pop_em = polar_optical_scattering(
                    dist_X[:, 5], 3)
                Rate_XT_ab, Rate_XT_em = optical_phonon_scattering(
                    dist_X[:, 5], 6)
                Rate_XL_ab, Rate_XL_em = optical_phonon_scattering(
                    dist_X[:, 5], 7)
                Rate_XX_ab, Rate_XX_em = optical_phonon_scattering(
                    dist_X[:, 5], 8)
                max_Rate_X = np.max([np.mean(Rate_ei), np.mean(Rate_eh),
                                     np.mean(Rate_ac), np.mean(Rate_pop_ab),
                                     np.mean(Rate_pop_em), np.mean(Rate_pie),
                                     np.mean(Rate_XT_ab), np.mean(Rate_XT_em),
                                     np.mean(Rate_XL_ab), np.mean(Rate_XL_em),
                                     np.mean(Rate_XX_ab), np.mean(Rate_XX_em)])
            else:
                max_Rate_X = 0

            Rate = np.max([max_Rate_T, max_Rate_L, max_Rate_X])
            stept = 1 / Rate / 25
            t += stept
            # transfer matrix after stept
            M_st = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0, 0, stept, 0, 0],
                             [0, 0, 0, 1, 0, 0, 0, 0, stept, 0],
                             [0, 0, 0, 0, 1, 0, 0, 0, 0, stept],
                             [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

            # ----------- scattering change the distribution -----------
            # ----- get the energy distribution after scattering -------

            # ------ A. scattering for electrons in Gamma valley --------
            if len(dist_2D) > 0:
                # ----- e-impurity scattering ---
                Rate_ei = impurity_scattering(dist_2D[:, 5], 1)
                dist_2D, happen_ie = electron_impurity_transfer_energy(
                    dist_2D, Rate_ei, stept)
                if np.mean(happen_ie.astype(int)) > 0:
                    dist_2D = renew_coulomb_distribution(dist_2D, happen_ie, 1)

                # ----- e-h scattering -----
                Rate_eh = electron_hole_scattering(dist_2D[:, 5], 1)
                dist_2D, hole_energy, happen_eh = \
                    electron_hole_transfer_energy(
                        dist_2D, hole_energy, Rate_eh, stept)
                if np.mean(happen_eh.astype(int)) > 0:
                    dist_2D = renew_coulomb_distribution(dist_2D, happen_eh, 1)

                # ----- e-phonon scattering -----
                # -------- 1. acoustic phonon scattering -------
                # acoustic phonon scattering probability within stept
                Rate_ac = acoustic_phonon_scattering(dist_2D[:, 5], 1)
                dist_2D, happen_ac = electron_acoustic_tranfer_energy(
                    dist_2D, Rate_ac, stept)
                if np.mean(happen_ac.astype(int)) > 0:
                    dist_2D = renew_distribution(dist_2D, happen_ac, 1)

                # ----- 2. piezoelectric scattering -------
                Rate_pie = piezoelectric_phonon_scattering(dist_2D[:, 5], 1)
                dist_2D, happen_pie = electron_piezoelectric_tranfer_energy(
                    dist_2D, Rate_pie, stept)
                if np.mean(happen_pie.astype(int)) > 0:
                    dist_2D = renew_distribution(dist_2D, happen_pie, 1)

                # ----- 3. polar opical phonon scattering --------
                Rate_pop_ab, Rate_pop_em = polar_optical_scattering(
                    dist_2D[:, 5], 1)
                dist_2D = electron_polar_transfer_energy(
                    dist_2D, Rate_pop_ab, Rate_pop_em, stept, 1)

                # ----- 4. optical phonon scattering from Gamma to L valley --
                Rate_TL_ab, Rate_TL_em = optical_phonon_scattering(dist_2D[
                                                                   :, 5], 1)
                E_TL_ab = -E_L_T + phonon_T_L
                E_TL_em = -E_L_T - phonon_T_L
                dist_2D, dist_TtoL = electron_optical_transfer_energy(
                    dist_2D, Rate_TL_ab, Rate_TL_em, E_TL_ab, E_TL_em,
                    stept, 1)

                # --- 5. optical phonon scattering from Gamma to X valley ---
                if len(dist_2D) > 0:
                    Rate_TX_ab, Rate_TX_em = \
                        optical_phonon_scattering(dist_2D[:, 5], 2)
                    E_TX_ab = -E_X_T + phonon_T_X
                    E_TX_em = -E_X_T - phonon_T_X
                    dist_2D, dist_TtoX = electron_optical_transfer_energy(
                        dist_2D, Rate_TX_ab, Rate_TX_em, E_TX_ab, E_TX_em,
                        stept, 1)
                else:
                    dist_TtoX = []
            else:
                dist_TtoL = []
                dist_TtoX = []
            # print('Gamma: ', len(dist_2D), len(dist_TtoL), len(dist_TtoX))

            # ------- B. scattering for electrons in L valley -------
            # ----- e-impurity scattering ---
            if len(dist_L) > 0:
                # ----- a. e-impurity scattering ---
                Rate_ei = impurity_scattering(dist_L[:, 5], 2)
                dist_L, happen_ie = electron_impurity_transfer_energy(
                    dist_L, Rate_ei, stept)
                if np.mean(happen_ie.astype(int)) > 0:
                    dist_L = renew_coulomb_distribution(dist_L, happen_ie, 2)

                # ----- b. e-h scattering -----
                Rate_eh = electron_hole_scattering(dist_L[:, 5], 2)
                dist_L, hole_energy, happen_eh = electron_hole_transfer_energy(
                    dist_L, hole_energy, Rate_eh, stept)
                if np.mean(happen_eh.astype(int)) > 0:
                    dist_L = renew_coulomb_distribution(dist_L, happen_eh, 2)

                # ----- c. e-phonon scattering -----
                # -------- 1. acoustic phonon scattering -------
                # acoustic phonon scattering probability within stept
                Rate_ac = acoustic_phonon_scattering(dist_L[:, 5], 2)
                dist_L, happen_ac = electron_acoustic_tranfer_energy(
                    dist_L, Rate_ac, stept)
                if np.mean(happen_ac.astype(int)) > 0:
                    dist_L = renew_distribution(dist_L, happen_ac, 2)

                # ----- 2. piezoelectric scattering -------
                Rate_pie = piezoelectric_phonon_scattering(dist_L[:, 5], 2)
                dist_L, happen_pie = electron_piezoelectric_tranfer_energy(
                    dist_L, Rate_pie, stept)
                if np.mean(happen_pie.astype(int)) > 0:
                    dist_L = renew_distribution(dist_L, happen_pie, 2)

                # ----- 3. polar opical phonon scattering --------
                Rate_pop_ab, Rate_pop_em = polar_optical_scattering(
                    dist_L[:, 5], 2)
                dist_L = electron_polar_transfer_energy(
                    dist_L, Rate_pop_ab, Rate_pop_em, stept, 2)

                # ----- 4. optical phonon scattering from L to Gamma valley ---
                Rate_LT_ab, Rate_LT_em = optical_phonon_scattering(dist_L[
                                                                   :, 5], 3)
                E_LT_ab = E_L_T + phonon_T_L
                E_LT_em = E_L_T - phonon_T_L
                dist_L, dist_LtoT = electron_optical_transfer_energy(
                    dist_L, Rate_LT_ab, Rate_LT_em, E_LT_ab, E_LT_em,
                    stept, 2)

                # -------- 5. optical phonon scattering from L to L valley ----
                if len(dist_L) > 0:
                    Rate_LL_ab, Rate_LL_em = \
                        optical_phonon_scattering(dist_L[:, 5], 4)
                    E_LL_ab = phonon_L_L
                    E_LL_em = -phonon_L_L
                    Num = len(dist_L)
                    P_LL_ab = stept * Rate_LL_ab
                    P_LL_ab_ind = np.random.uniform(0, 1, Num) <= P_LL_ab
                    happen_LL_ab = P_LL_ab_ind
                    dist_L[:, 5] = dist_L[:, 5] + E_LL_ab * happen_LL_ab

                    P_LL_em = stept * Rate_LL_em
                    P_LL_em_ind = np.random.uniform(0, 1, Num) <= P_LL_em
                    happen_LL_em = P_LL_em_ind
                    dist_L[:, 5] = dist_L[:, 5] + E_LL_em * happen_LL_em
                    happen_LL = happen_LL_ab + happen_LL_em

                    if np.mean(happen_LL.astype(int)) > 0:
                        dist_L = renew_distribution(dist_L, happen_LL, 2)

                # -------- 6. optical phonon scattering from L to X valley ----
                if len(dist_L) > 0:
                    Rate_LX_ab, Rate_LX_em =\
                        optical_phonon_scattering(dist_L[:, 5], 5)
                    E_LX_ab = -E_X_T + E_L_T + phonon_L_X
                    E_LX_em = -E_X_T + E_L_T - phonon_L_X
                    dist_L, dist_LtoX = electron_optical_transfer_energy(
                        dist_L, Rate_LX_ab, Rate_LX_em, E_LX_ab, E_LX_em,
                        stept, 2)
                else:
                    dist_LtoX = []
            else:
                dist_LtoT = []
                dist_LtoX = []
            # print('L: ', len(dist_L), len(dist_LtoT), len(dist_LtoX))

            # -------- C. scattering for electrons in X valley -------
            if len(dist_X) > 0:
                Rate_ei = impurity_scattering(dist_X[:, 5], 3)
                dist_X, happen_ie = electron_impurity_transfer_energy(
                    dist_X, Rate_ei, stept)
                if np.mean(happen_ie.astype(int)) > 0:
                    dist_X = renew_coulomb_distribution(dist_X, happen_ie, 3)

                # ----- e-h scattering -----
                Rate_eh = electron_hole_scattering(dist_X[:, 5], 3)
                dist_X, hole_energy, happen_eh = electron_hole_transfer_energy(
                    dist_X, hole_energy, Rate_eh, stept)
                if np.mean(happen_eh.astype(int)) > 0:
                    dist_X = renew_coulomb_distribution(dist_X, happen_eh, 3)

                # ----- e-phonon scattering -----
                # -------- 1. acoustic phonon scattering -------
                # acoustic phonon scattering probability within stept
                Rate_ac = acoustic_phonon_scattering(dist_X[:, 5], 3)
                dist_X, happen_ac = electron_acoustic_tranfer_energy(
                    dist_X, Rate_ac, stept)
                if np.mean(happen_ac.astype(int)) > 0:
                    dist_X = renew_distribution(dist_X, happen_ac, 3)

                # ----- 2. piezoelectric scattering -------
                Rate_pie = piezoelectric_phonon_scattering(dist_X[:, 5], 3)
                dist_X, happen_pie = electron_piezoelectric_tranfer_energy(
                    dist_X, Rate_pie, stept)
                if np.mean(happen_pie.astype(int)) > 0:
                    dist_X = renew_distribution(dist_X, happen_pie, 3)

                # ----- 3. polar opical phonon scattering --------
                Rate_pop_ab, Rate_pop_em = polar_optical_scattering(
                    dist_X[:, 5], 3)
                dist_X = electron_polar_transfer_energy(
                    dist_X, Rate_pop_ab, Rate_pop_em, stept, 2)

                # ----- 4. optical phonon scattering from X to Gamma valley ---
                Rate_XT_ab, Rate_XT_em = optical_phonon_scattering(dist_X[
                                                                   :, 5], 6)
                E_XT_ab = E_X_T + phonon_T_X
                E_XT_em = E_X_T - phonon_T_X
                dist_X, dist_XtoT = electron_optical_transfer_energy(
                    dist_X, Rate_XT_ab, Rate_XT_em, E_XT_ab, E_XT_em,
                    stept, 3)

                # -------- 5. optical phonon scattering from X to L valley ----
                if len(dist_X) > 0:
                    Rate_XL_ab, Rate_XL_em = optical_phonon_scattering(dist_X[
                        :, 5], 7)
                    E_XL_ab = E_X_T - E_L_T + phonon_L_X
                    E_XL_em = E_X_T - E_L_T - phonon_L_X
                    dist_X, dist_XtoL = electron_optical_transfer_energy(
                        dist_X, Rate_XL_ab, Rate_XL_em, E_XL_ab, E_XL_em,
                        stept, 3)

                # -------- 6. optical phonon scattering from X to X valley ----
                if len(dist_X) > 0:
                    Rate_XX_ab, Rate_XX_em = optical_phonon_scattering(dist_X[
                        :, 5], 8)
                    E_XX_ab = phonon_X_X
                    E_XX_em = -phonon_X_X
                    Num = len(dist_X)
                    P_XX_ab = stept * Rate_XX_ab
                    P_XX_ab_ind = np.random.uniform(0, 1, Num) <= P_XX_ab
                    happen_XX_ab = P_XX_ab_ind
                    dist_X[:, 5] = dist_X[:, 5] + E_XX_ab * happen_XX_ab

                    P_XX_em = stept * Rate_XX_em
                    P_XX_em_ind = np.random.uniform(0, 1, Num) <= P_XX_em
                    happen_XX_em = P_XX_em_ind
                    dist_X[:, 5] = dist_X[:, 5] + E_XX_em * happen_XX_em
                    happen_XX = happen_XX_ab + happen_XX_em

                    if np.mean(happen_XX.astype(int)) > 0:
                        dist_X = renew_distribution(dist_X, happen_XX, 3)
                else:
                    dist_XtoT = []
            else:
                dist_XtoT = []
                dist_XtoL = []
            # print('X: ', len(dist_X), len(dist_XtoT), len(dist_XtoL))

            # print('before:', np.mean(dist_2D[:, 5]))
            dist_2D = dist_2D.tolist()
            dist_2D.extend(dist_LtoT)
            dist_2D.extend(dist_XtoT)
            dist_2D = np.array(dist_2D)
            dist_L = dist_L.tolist()
            dist_L.extend(dist_TtoL)
            dist_L.extend(dist_XtoL)
            dist_L = np.array(dist_L)
            dist_X = dist_X.tolist()
            dist_X.extend(dist_TtoX)
            dist_X.extend(dist_LtoX)
            dist_X = np.array(dist_X)

            '''
            if len(dist_L) > 0 and len(dist_L) <= 10:
                # dist_2D = np.append(dist_2D, dist_L, axis=0)
                dist_L = np.array([])
            if len(dist_X) > 0 and len(dist_X) <= 10:
                # dist_2D = np.append(dist_2D, dist_X, axis=0)
                dist_X = np.array([])'''

            # ------ filtting electrons-------
            # -------- filtting electrons in Gamma valley --------
            if len(dist_2D) > 0:
                dist_2D = np.dot(dist_2D, M_st)
                bd, fd, td, dist_2D = filter(dist_2D)
                back_2D.extend(bd.tolist())
                trap_2D.extend(td.tolist())
                surface_2D.extend(fd.tolist())

            # -------- filtting electrons in L valley --------
            if len(dist_L) > 0:
                dist_L = np.dot(dist_L, M_st)
                bd, fd, td, dist_L = filter(dist_L)
                back_2D.extend(bd.tolist())
                trap_2D.extend(td.tolist())
                surface_2D.extend(fd.tolist())
                if len(dist_L) > 0:
                    energy_L = dist_L[:, 5]
                else:
                    energy_L = np.array([])
            else:
                energy_L = np.array([])

            # -------- filtting electrons in X valley --------
            if len(dist_X) > 0:
                dist_X = np.dot(dist_X, M_st)
                bd, fd, td, dist_X = filter(dist_X)
                back_2D.extend(bd.tolist())
                trap_2D.extend(td.tolist())
                surface_2D.extend(fd.tolist())
                if len(dist_X) > 0:
                    energy_X = dist_X[:, 5]
                else:
                    energy_X = np.array([])
            else:
                energy_X = np.array([])
            # print('after:', np.mean(dist_2D[:, 5]))
            total_energy = np.concatenate((dist_2D[:, 5], energy_L, energy_X), axis=0)
            time_data.append([t * 10**12, np.mean(total_energy) * 10**3,
                              len(surface_2D), len(dist_2D), len(dist_L),
                              len(dist_X)])
            # print('surface:', len(surface_2D), 'trap:', len(trap_2D),
            #      'back:', len(back_2D), 
            #      'inside:', (len(dist_2D) + len(dist_L) + len(dist_X)))
            # print(np.mean(dist_2D[:, 5]), len(dist_2D))
            if (len(dist_2D)) <= 10:
                break
        '''
        fig, ax = plt.subplots()
        ax.hist(dist_2D[:, 5], bins=100)
        plt.show()'''

        dist_2D = dist_2D.tolist()
        dist_2D.extend(back_2D)
        dist_2D.extend(trap_2D)
        dist_2D.extend(surface_2D)
        dist_2D = np.array(dist_2D)
        dist_2D[:, 5] = np.maximum(dist_2D[:, 5], 0)
        dist_2D[:, 9] = np.clip(dist_2D[:, 9], surface, thick)

    else:
        print('Wrong electron transport types')

    time_data = np.array(time_data)

    back_2D = np.array(back_2D)
    if len(back_2D) != 0:
        back_2D[:, 9] = thick

    trap_2D = np.array(trap_2D)
    if len(trap_2D) != 0:
        trap_2D[:, 5] = 0

    surface_2D = np.array(surface_2D)
    if len(surface_2D) != 0:
        surface_2D[:, 9] = surface
    '''
    fig, ax = plt.subplots()
    ax.hist(surface_2D[:, 5], bins=100)
    plt.show()'''
    return surface_2D, back_2D, trap_2D, dist_2D, time_data


def filter(dist_2D):
    ''' filter electrons
    Find these electrons diffused to surface, substrate, trapped
    and the rest electrons that will continue to diffuse
    '''
    assert thick > surface
    back = dist_2D[:, 9] >= thick
    front = dist_2D[:, 9] <= surface
    # trap = dist_2D[:, 5] <= min(E_A - E_sch, 0.0005)
    trap = dist_2D[:, 5] <= 0.0005

    back_dist = dist_2D[back, :]
    front_dist = dist_2D[front & (~trap), :]
    trap_dist = dist_2D[trap, :]
    rest_dist = dist_2D[(~back) & (~front) & (~trap), :]
    return back_dist, front_dist, trap_dist, rest_dist


def surface_electron_transmission(surface_2D, func_tp):
    surface_trap = []
    P_tp = []
    # surface_2D[:, 5] = surface_2D[:, 5] + E_B
    Num = len(surface_2D)
    theta = surface_2D[:, 1]
    E_trans = np.abs(surface_2D[:, 5] * np.sin(theta)**2)
    E_paral = np.abs(surface_2D[:, 5] * np.cos(theta)**2)
    E_t1 = 0.5 * m_T * (surface_2D[:, 2]**2 + surface_2D[:, 3]**2) / ec
    E_p1 = 0.5 * m_T * surface_2D[:, 4]**2 / ec
    print(E_trans, E_t1, E_paral, E_p1)
    # P_tp1 = func_tp(surface_2D[:, 5], 0.0)
    # print(P_tp1)
    for i in range(Num):
        P_tp.append((func_tp(E_paral[i], E_trans[i]))[0].tolist())
    P_tp = np.array(P_tp)
    # print(P_tp)
    match_ind = np.random.uniform(0, 1.0, len(surface_2D)) <= P_tp
    surface_trap.extend(surface_2D[(~match_ind), :].tolist())
    emission_2D = surface_2D[match_ind, :]
    surface_trap = np.array(surface_trap)

    return emission_2D, surface_trap


def transmission_function1(E_paral, E_trans):
    # k_trans = np.sqrt(2 * m_T * E_trans * ec) / h_
    '''
    V = np.array([0.0, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.205, 0.225, 0.25,
                  0.235, 0.218, 0.20, 0.18, 0.16, 0.14, 0.12, 0.1, 0.08,
                  0.06, 0.04, 0.02])
    # V = [0.0, 0.1]
    width = 8e-10  # m, width of barrier
    num = len(V)
    L = width / (num - 1)
    '''
    w1 = 10e-10
    w2 = 200e-10
    num = 40
    L = w1 / num
    L2 = w2 / num
    x1 = np.arange(0, w1, L)
    x2 = np.arange(w1, w2, L2)
    Q = ec / 16 / pi / eps0 * (12.85 - 1) / (12.85 + 1)
    V0 = 0.1
    F = (V0 - E_A) / w1
    V1 = V0 - F * x1
    V2 = V0 - F * w1 - Q / x2
    x = np.concatenate((x1, x2), axis=0)
    V = np.concatenate((V1, V2), axis=0)
    
    fig, ax = plt.subplots()
    ax.plot(x, V)
    plt.savefig('surface_barrier.pdf', format='pdf')
    plt.show()
    
    Tp = 1
    E_in_par = E_paral + E_trans - E_trans * m_T / m_e
    E_out_par = E_paral + E_trans - E_trans * m_T / m_e - E_A
    for i in range(num - 1):
        k1 = (np.sqrt(2 * m_T * (E_paral - V[i]).clip(0.00001) * ec) / h_)
        if E_in_par > V[i + 1]:
            k2 = np.sqrt(2 * m_T * (E_in_par - V[i + 1]) * ec) / h_
            Tr = 1 / (1 + 0.25 * ((k1**2 - k2**2) / 2 / k1 / k2)**2 *
                      np.sin(k2 * L)**2)
        elif E_in_par < V[i + 1]:
            k2 = np.sqrt(2 * m_T * (V[i + 1] - E_in_par) * ec) / h_
            Tr = 1 / (1 + 0.25 * ((k1**2 + k2**2) / 2 / k1 / k2)**2 *
                      np.sinh(k2 * L)**2)
        elif E_in_par == V[i + 1]:
            Tr = 1
        Tp = Tp * Tr
    k1 = (np.sqrt(2 * m_T * (E_in_par - V[-1]).clip(0.00001) * ec) / h_)
    if E_out_par > 0:
        k2 = np.sqrt(2 * m_e * (E_out_par) * ec) / h_
        Tr = 1 / (1 + 0.25 * ((k1**2 - k2**2) / 2 / k1 / k2)**2 *
                  np.sin(k2 * L)**2)
    elif E_out_par < 0:
        k2 = np.sqrt(2 * m_e * (-E_out_par) * ec) / h_
        Tr = 1 / (1 + 0.25 * ((k1**2 + k2**2) / 2 / k1 / k2)**2 *
                  np.sinh(k2 * L)**2)
    elif E_in_par == 0:
        Tr = 1
    Tp = Tp * Tr
    # print(Tp)
    # Tr = np.sqrt(E_out_par / E_in_par.clip(0.0001))
    # Tr = Tp * Tr
    return Tp


# transmission_function1(1, 1)


def transmission_function(E_paral, E_trans):
    
    V = np.array([0.0, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.205, 0.225, 0.25,
                  0.235, 0.218, 0.20, 0.18, 0.16, 0.14, 0.12, 0.1, 0.08,
                  0.06, 0.04, 0.02])
    V = np.array([0.28]*100)
    # V = V + E_B
    width = 8e-10  # m, width of barrier
    num = 100
    L = width / num
    '''
    width = 1.5e-10
    num = 40
    L = width / num
    x = np.arange(0, width, L) + L
    Q = 0.36 * (12.85 - 1) / (12.85 + 1)
    F = (4 - E_A) / width
    V = 0.3 - F * x'''
    '''
    w1 = 10e-10
    w2 = 200e-10
    num = 40
    L = w1 / num
    L2 = w2 / num
    x1 = np.arange(0, w1, L)
    x2 = np.arange(w1, w2, L2)
    Q = ec / 16 / pi / eps0 * (12.85 - 1) / (12.85 + 1)
    V0 = 1
    F = (V0 - E_A) / w1
    V1 = V0 - F * x1
    V2 = V0 - F * w1 - Q / x2
    x = np.concatenate((x1, x2), axis=0)
    V = np.concatenate((V1, V2), axis=0)
    '''
    Tp = np.mat([[1, 0], [0, 1]])
    E_in_par = E_paral + 0j
    E_out_par = E_paral + E_trans - E_trans * m_T / m_e + 0j
    k_in = (np.sqrt(2 * m_T * (E_in_par + 0j) * ec) / h_)
    k_out = (np.sqrt(2 * m_e * (E_out_par + 0j) * ec) / h_)
    '''
    if k_in == 0:
        P = 0
    else:
        p11 = 0.5 * (1 + m_T * k_out / k_in / m_e)  # * np.exp(1j * k_in * L)
        p12 = 0.5 * (1 - m_T * k_out / k_in / m_e)  # * np.exp(1j * k_in * L)
        p21 = 0.5 * (1 - m_T * k_out / k_in / m_e)  # * np.exp(-1j * k_in * L)
        p22 = 0.5 * (1 + m_T * k_out / k_in / m_e)  # * np.exp(-1j * k_in * L)
        P = np.mat([[p11, p12], [p21, p22]])
    Tp = Tp * P'''
    # Tp = Tp * m_T  / m_e * np.abs(k_out / k_in)
    for k in range(num - 2):
        i = k + 1
        k1 = (np.sqrt(2 * m_e * (E_out_par - V[i] + 0j) * ec) / h_)
        k2 = (np.sqrt(2 * m_e * (E_out_par - V[i + 1] + 0j) * ec) / h_)
        if k1 == 0:
            P = 0
        else:
            p11 = 0.5 * (1 + k2 / k1) * np.exp(-1j * k1 * L)
            p12 = 0.5 * (1 - k2 / k1) * np.exp(-1j * k1 * L)
            p21 = 0.5 * (1 - k2 / k1) * np.exp(1j * k1 * L)
            p22 = 0.5 * (1 + k2 / k1) * np.exp(1j * k1 * L)
            P = np.mat([[p11, p12], [p21, p22]])
        Tp = Tp * P
        # print(k1, k2)
        # print(Tp)

    k_in = (np.sqrt(2 * m_e * (E_out_par - V[-1] + 0j) * ec) / h_)
    k_out = (np.sqrt(2 * m_e * (E_out_par - E_A + 0j) * ec) / h_)
    if k_in == 0:
        P = 0
    else:
        p11 = 0.5 * (1 + k_out / k_in) * np.exp(1j * k_in * L)
        p12 = 0.5 * (1 - k_out / k_in) * np.exp(1j * k_in * L)
        p21 = 0.5 * (1 - k_out / k_in) * np.exp(-1j * k_in * L)
        p22 = 0.5 * (1 + k_out / k_in) * np.exp(-1j * k_in * L)
        P = np.mat([[p11, p12], [p21, p22]])
    Tp = Tp * P
    # print(Tp)

    if Tp[0, 0] == 0:
        Tp = 0
    else:
        Tp = 1 / np.abs(Tp[0, 0])**2
    return Tp


def transmission_probability(E_paral, E_trans):
    Tp = []
    for i in range(len(E_trans)):
        for j in range(len(E_paral)):
            Ti = transmission_function(E_paral[j], E_trans[i])
            Tp.append(Ti)
    Tp = np.array(Tp).reshape(len(E_trans), len(E_paral))
    '''
    fig, ax = plt.subplots()
    ax.plot(E_paral, Tp[0, :], E_paral, Tp[10, :], E_paral, Tp[20, :])
    ax.legend(['0', '10', '20'])
    plt.savefig('Emission_rate.pdf', format='pdf')
    plt.show()'''
    return Tp


def plot_QE(filename, data):
    exp_data = np.genfromtxt('GaAs_QE_experiment1.csv', delimiter=',')
    exp_data[:, 0] = 1240 / exp_data[:, 0]
    fig1, ax1 = plt.subplots()
    ax1.plot(data[:, 0], data[:, 1], 'o', exp_data[:, 0], exp_data[:, 1], '.')
    ax1.set_xlabel(r'Photon energy (eV)', fontsize=14)
    ax1.set_ylabel(r'QE (%)', fontsize=14)
    ax1.tick_params('both', direction='in', labelsize=12)
    ax1.legend(['Simulated', 'Experimental'])
    plt.savefig(filename + '.pdf', format='pdf')
    plt.show()


def plot_time_data(filename, time_data):
    np.savetxt(filename + '.csv', time_data, delimiter=',', fmt='%.6f')
    fig1, ax1 = plt.subplots()
    l1 = ax1.plot(time_data[:, 0], time_data[:, 1], 'bp', label='Electron energy')
    ax1.set_xlabel('Time (ps)', fontsize=14)
    ax1.set_ylabel('Energy (meV)', fontsize=14, color='b')
    ax1.tick_params('y', color='b')
    ax1.tick_params('both', direction='in', labelsize=12)

    ax2 = ax1.twinx()
    l2 = ax2.semilogy(time_data[:, 0], time_data[:, 2], 'r*', label='Surface')
    l3 = ax2.semilogy(time_data[:, 0], time_data[:, 3], 'c.', label='Gamma')
    l4 = ax2.semilogy(time_data[:, 0], time_data[:, 4], 'ys', label='L')
    l5 = ax2.semilogy(time_data[:, 0], time_data[:, 5], 'mo', label='X')
    ax2.set_ylabel('Counts (abs. units)', fontsize=14)
    ax2.tick_params('y', color='r')
    ax1.tick_params('both', direction='in', labelsize=12)
    ln = l1 + l2 + l3 + l4 + l5
    labs = [l.get_label() for l in ln]
    ax1.legend(ln, labs, loc='best', frameon=False, fontsize=12)
    fig1.tight_layout()
    plt.savefig(filename + '.pdf', format='pdf')
    plt.show()


def save_date(filename, data):
    types = 1
    # type 1 control the number of saved data
    # type 2 save all data
    if types == 0:
        file = open(filename, 'w')
        for i in range(len(data)):
            for j in range(np.size(data[0, :]) - 1):
                file.write('%.2f' % data[i, j])
                file.write(',')
            file.write('%.2f' % data[i, -1])
            file.write('\n')
        file.close()
    elif types == 1:
        np.savetxt(filename + '.csv', data, delimiter=',', fmt='%.2f')


def plot_scattering_rate(types):
    '''
    types=1, plot the scattering rate for the electrons in Gamma valley
    types=2, plot the scattering rate for the electrons in L valley
    types=3, plot the scattering rate for the electrons in X valley
    '''
    e_energy = np.linspace(0, 1.5, 50)
    if types == 1:
        # hole_velocity = maxwell.rvs(0, np.sqrt(kB * T / m_h), Ni)
        # hole_energy = m_h * hole_velocity**2 / 2 / ec
        # hole_energy = np.linspace(0, 0.1, Ni)
        # carrier_scattering(e_energy, hole_energy)
        Rate_eh = electron_hole_scattering(e_energy, 1)
        Rate_ei = impurity_scattering(e_energy, 1)
        Rate_ac = acoustic_phonon_scattering(e_energy, 1)
        Rate_pap = piezoelectric_phonon_scattering(e_energy, 1)
        Rate_pop_ab, Rate_pop_em = polar_optical_scattering(e_energy, 1)
        Rate_TL_ab, Rate_TL_em = optical_phonon_scattering(e_energy, 1)
        Rate_TX_ab, Rate_TX_em = optical_phonon_scattering(e_energy, 2)
        fig, ax = plt.subplots()
        ax.semilogy(e_energy, Rate_eh, 'o', e_energy, Rate_ei, 'v',
                    e_energy, Rate_ac, '^', e_energy, Rate_pap, '+',
                    e_energy, Rate_pop_ab, '<',
                    e_energy, Rate_pop_em, '>', e_energy, Rate_TL_ab, 's',
                    e_energy, Rate_TL_em, 'p', e_energy, Rate_TX_ab, '*',
                    e_energy, Rate_TX_em, 'h')
        ax.set_xlabel(r'Energy (eV)', fontsize=16)
        ax.set_ylabel(r'scattering rate in $\Gamma$ valley ($s^{-1}$)',
                      fontsize=16)
        ax.set_xlim([0, 1.5])
        ax.tick_params(which='both', direction='in', labelsize=14)
        leg = plt.legend(['Hole', 'Impurity', 'Deformation potential',
                          'Piezoelectric', 'POP absorb',
                          'POP emission', 'Gamma to L absorb',
                          'Gamma to L emission', 'Gamma to X absorb',
                          'Gamma to X emission'],
                         frameon=False, fontsize=10)
        plt.setp(leg.get_texts(), color='b')
        plt.tight_layout()
        plt.savefig('Gamma_valley_scattering_rate.pdf', format='pdf')
        plt.show()
    elif types == 2:
        # hole_velocity = maxwell.rvs(0, np.sqrt(kB * T / m_h), Ni)
        # hole_energy = m_h * hole_velocity**2 / 2 / ec
        # hole_energy = np.linspace(0, 0.1, Ni)
        Rate_eh = electron_hole_scattering(e_energy, 2)
        Rate_ei = impurity_scattering(e_energy, 2)
        Rate_ac = acoustic_phonon_scattering(e_energy, 2)
        Rate_pap = piezoelectric_phonon_scattering(e_energy, 2)
        Rate_pop_ab, Rate_pop_em = polar_optical_scattering(e_energy, 2)
        Rate_LT_ab, Rate_LT_em = optical_phonon_scattering(e_energy, 3)
        Rate_LL_ab, Rate_LL_em = optical_phonon_scattering(e_energy, 4)
        Rate_LX_ab, Rate_LX_em = optical_phonon_scattering(e_energy, 5)
        fig, ax = plt.subplots()
        ax.semilogy(e_energy, Rate_eh, 'o', e_energy, Rate_ei, 'v',
                    e_energy, Rate_ac, '^', e_energy, Rate_pap, '+',
                    e_energy, Rate_pop_ab, '<',
                    e_energy, Rate_pop_em, '>', e_energy, Rate_LT_ab, 's',
                    e_energy, Rate_LT_em, 'p', e_energy, Rate_LL_ab, '*',
                    e_energy, Rate_LL_em, 'h', e_energy, Rate_LX_ab, '4',
                    e_energy, Rate_LX_em, 'X')
        ax.set_xlabel(r'Energy (eV)', fontsize=16)
        ax.set_ylabel(r'scattering rate in L valley ($s^{-1}$)', fontsize=16)
        ax.set_xlim([0, 1.5])
        ax.tick_params(which='both', direction='in', labelsize=14)
        leg = plt.legend(['Hole', 'Impurity', 'Deformation potential',
                          'Piezoelectric', 'POP absorb',
                          'POP emission', 'L to Gamma absorb',
                          'L to Gamma emission', 'L to L absorb',
                          'L to L emission',
                          'L to X absorb', 'L to X emission'],
                         frameon=False, fontsize=10)
        plt.setp(leg.get_texts(), color='b')
        plt.tight_layout()
        plt.savefig('L_valley_scattering_rate.pdf', format='pdf')
        plt.show()
    elif types == 3:
        # hole_velocity = maxwell.rvs(0, np.sqrt(kB * T / m_h), Ni)
        # hole_energy = m_h * hole_velocity**2 / 2 / ec
        # hole_energy = np.linspace(0, 0.1, Ni)
        Rate_eh = electron_hole_scattering(e_energy, 3)
        Rate_ei = impurity_scattering(e_energy, 3)
        Rate_ac = acoustic_phonon_scattering(e_energy, 3)
        Rate_pap = piezoelectric_phonon_scattering(e_energy, 3)
        Rate_pop_ab, Rate_pop_em = polar_optical_scattering(e_energy, 3)
        Rate_XT_ab, Rate_XT_em = optical_phonon_scattering(e_energy, 6)
        Rate_XL_ab, Rate_XL_em = optical_phonon_scattering(e_energy, 7)
        Rate_XX_ab, Rate_XX_em = optical_phonon_scattering(e_energy, 8)
        fig, ax = plt.subplots()
        ax.semilogy(e_energy, Rate_eh, 'o', e_energy, Rate_ei, 'v',
                    e_energy, Rate_ac, '^', e_energy, Rate_pap, '+',
                    e_energy, Rate_pop_ab, '<',
                    e_energy, Rate_pop_em, '>', e_energy, Rate_XT_ab, 's',
                    e_energy, Rate_XT_em, 'p', e_energy, Rate_XL_ab, '*',
                    e_energy, Rate_XL_em, 'h', e_energy, Rate_XX_ab, '4',
                    e_energy, Rate_XX_em, 'X')
        ax.set_xlabel(r'Energy (eV)', fontsize=16)
        ax.set_ylabel(r'scattering rate in X valley ($s^{-1}$)', fontsize=16)
        ax.set_xlim([0, 1.5])
        ax.tick_params(which='both', direction='in', labelsize=14)
        leg = plt.legend(['Hole', 'Impurity', 'Deformation potential',
                          'Piezoelectric', 'POP absorb',
                          'POP emission', 'X to Gamma absorb',
                          'X to Gamma emission', 'X to L absorb',
                          'X to L emission',
                          'X to X absorb', 'X to X emission'],
                         frameon=False, fontsize=10)
        plt.setp(leg.get_texts(), color='b')
        plt.tight_layout()
        plt.savefig('X_valley_scattering_rate.pdf', format='pdf')
        plt.show()


def compare_data(filename, data):
    fig1, ax1 = plt.subplots()
    ax1.plot(data[:, 0], data[:, 3], 'o', data[:, 0],
             transportation_efficiency(data[:, 0]), '.')
    ax1.set_xlabel(r'Photon energy (eV)', fontsize=14)
    ax1.set_ylabel(r'transportation efficiency (%)', fontsize=14)
    ax1.tick_params('both', direction='in', labelsize=12)
    ax1.legend(['Simulated', 'Calculated'])
    plt.savefig(filename + '_TE.pdf', format='pdf')
    plt.show()


def plot_surface_emission_probability(E_paral, Tp, func_tp):
    P1 = []
    for i in range(len(E_paral)):
        P1.append((func_tp(E_paral[i], 0.0)[0]).tolist())
    P1 = np.array(P1)
    fig, ax = plt.subplots()
    ax.plot(E_paral, Tp[0, :], E_paral, Tp[10, :], E_paral, Tp[20, :])
    ax.set_xlabel(r'Electron longitudinal energy (eV)', fontsize=14)
    ax.set_ylabel(r'Emission probability', fontsize=14)
    ax.tick_params('both', direction='in', labelsize=12)
    ax.legend(['0', '0.2', '0.4'])
    plt.savefig('Emission_rate.pdf', format='pdf')
    plt.show()


def plot_electron_distribution(filename,dist_2D, types):
    if types == 1:
        fig, ax = plt.subplots()
        ax.hist(dist_2D[:, 5], bins=100, color='k')
        # ax.set_xlim([0, 0.6])
        # ax.set_ylim([0, 2000])
        ax.set_xlabel(r'Energy (eV)', fontsize=16)
        ax.set_ylabel(r'Counts (arb. units)', fontsize=16)
        ax.tick_params('both', direction='in', labelsize=14)
        plt.tight_layout()
        plt.savefig(filename + 'energy_distribution.pdf', format='pdf')
        plt.show()
    elif types == 2:
        fig, ax = plt.subplots()
        ax.hist(dist_2D[:, 9], bins=200, color='k')
        ax.set_xlim([0, 2000])
        ax.set_xlabel(r'Depth (nm)', fontsize=16)
        ax.set_ylabel(r'Counts (arb. units)', fontsize=16)
        ax.tick_params('both', direction='in', labelsize=14)
        plt.tight_layout()
        plt.savefig(filename + 'posization_distribution.pdf', format='pdf')
        plt.show()
    else:
        print('Wrong types')


def main(opt):
    hw_start = Eg + 0.01  # eV
    hw_end = 2.5  # eV
    hw_step = 0.05  # eV
    hw_test = 2.0  # eV
    data = []
    E_paral = np.linspace(0.0, 2.0, 100)
    E_trans = np.linspace(0.0, 2.0, 100)
    Trans_prob = transmission_probability(E_paral, E_trans)
    func_tp = interp2d(E_paral, E_trans, Trans_prob)
    if opt == 1:  # for test
        dist_2D = electron_distribution(hw_test, 2)
        print('excited electron ratio: ', len(dist_2D) / Ni)
        #plot_electron_distribution('initial', dist_2D, 1)

        surface_2D, back_2D, trap_2D, dist_2D, time_data = \
            electron_transport(dist_2D, 1)
        print('surface electron ratio: ', len(surface_2D) / Ni)
        plot_electron_distribution('surface_' + str(hw_test), surface_2D, 1)

        emiss_2D, surf_trap = surface_electron_transmission(
            surface_2D, func_tp)
        print('emission ratio:', len(emiss_2D) / len(surface_2D))
        print('QE (%): ', 100.0 * len(emiss_2D) /
              Ni * (1 - surface_reflection(hw_test)))
        '''
        emiss_2D, surf_trap = electron_emitting(surface_2D)
        print('emission ratio:', len(emiss_2D) / len(surface_2D))
        print('QE (%): ', 100.0 * len(emiss_2D) /
              Ni * (1 - surface_reflection(hw_test)))
        '''
        plot_electron_distribution('emission_' + str(hw_test), emiss_2D, 1)
        filename = 'time_evoluation_' + str(hw_test)
        plot_time_data(filename, time_data)

    elif opt == 2:
        for hw in np.arange(hw_start, hw_end, hw_step):
            dist_2D = electron_distribution(hw, 2)
            print('excited electron ratio: ', len(dist_2D) / Ni)

            surface_2D, back_2D, trap_2D, dist_2D, energy_time = \
                electron_transport(dist_2D, 1)
            print('surface electron ratio: ', len(surface_2D) / Ni)

            # emiss_2D, surf_trap = electron_emitting(surface_2D)
            emiss_2D, surf_trap = surface_electron_transmission(
                surface_2D, func_tp)

            SR = surface_reflection(hw)  # surface light reflection
            TE = len(surface_2D) / len(dist_2D)  # transportation efficiency
            SE = len(surface_2D) / Ni * (1 - SR)
            ER = len(emiss_2D) / len(surface_2D)  # surface emission rate
            QE = 100.0 * len(emiss_2D) / Ni * (1 - SR)
            print('photon energy (eV): ', hw, ', QE (%): ', QE)
            data.append([hw, QE, SR, TE, SE, ER])

        filename = 'QE_' + str(thick) + '_' + str(E_A)
        data = np.array(data)
        save_date(filename, data)
        plot_QE(filename, data)
        compare_data(filename, data)
    elif opt == 3:
        # plot_electron_distribution('', electron_distribution(hw_test, 2), 2)
        # plot_scattering_rate(1)
        plot_surface_emission_probability(E_paral, Trans_prob, func_tp)
    else:
        print('Wrong run option')

    print('run time:', time.time() - start_time, 's')


if __name__ == '__main__':
    main(1)
