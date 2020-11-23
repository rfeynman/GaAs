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
from scipy.stats import expon
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import time

start_time = time.time()
pi = np.pi
eps = 12.9 * 8.85 * 10**(-12)  # background dielectric constant
kB = 1.38 * 10**(-23)  # J/K, Boltzmann constant
ec = 1.6 * 10**(-19)  # C
h_ = 1.0546 * 10**(-34)  # J*s, Planc constant
c = 3 * 10**8  # m/s, light speed
m_e = 9.109 * 10**(-31)  # kg, electron mass
m_hh = 0.5 * m_e  # effective heavy hole mass
m_lh = 0.076 * m_e  # effective light hole mass
m_so = 0.145 * m_e  # effective split-off band mass
m_T = 0.063 * m_e  # Gamma valley effective electron mass
m_L = 0.555 * m_e  # L valley effective electron mass
m_X = 0.851 * m_e  # X valley effective electron mass
T = 300  # K, material temperature
Eg = 1.519 - 0.54 * 10**(-3) * T**2 / (T + 204)  # eV, bandgap
DE = 0.34  # eV, split-off energy gap


def photon_to_electron(hw):
    ''' electrons in valence band aborption photon to excited to conduction
    band. Only consider these electrons in the heavy hole, light hole and
    split-off band would be excited, and can only excited into Gamma valley.
    Given photon energy, return excited electron energy. '''
    # nonparabolicity factor, 1/eV
    alpha_T = 0.58 + (T - 77) * (0.61 - 0.58) / (300 - 77)
    Ei = random.uniform(Eg, hw - 0.01)
    if Ei >= Eg + DE:
        x = random.randint(1, 6)
        if x in [1, 2, 3]:  # heavy hole
            E1 = hw - Ei
            Gamma = 1 + m_hh / m_e + 2 * alpha_T * E1
            DE_h = (1 - np.sqrt(1 - 4 * alpha_T * E1 * (1 + alpha_T * E1) /
                                Gamma**2)) / (2 * alpha_T) / Gamma
            # DE_h = E1 / (1 + m_hh / m_T)
            E_e = E1 - DE_h
        elif x == 4:  # light hole
            E1 = hw - Ei
            Gamma = 1 + m_lh / m_e + 2 * alpha_T * E1
            DE_h = (1 - np.sqrt(1 - 4 * alpha_T * E1 * (1 + alpha_T * E1) /
                                Gamma**2)) / (2 * alpha_T) / Gamma
            # DE_h = E1 / (1 + m_lh / m_T)
            E_e = E1 - DE_h
        elif x in [5, 6]:  # split-off band
            E1 = hw - Ei
            Gamma = 1 + m_so / m_e + 2 * alpha_T * E1
            DE_h = (1 - np.sqrt(1 - 4 * alpha_T * E1 * (1 + alpha_T * E1) /
                                Gamma**2)) / (2 * alpha_T) / Gamma
            # DE_h = E1 / (1 + m_so / m_T)
            E_e = E1 - DE_h
    elif Eg <= Ei < Eg + DE:
        x = random.randint(1, 4)
        if x in [1, 2, 3]:  # heavy hole
            E1 = hw - Ei
            Gamma = 1 + m_hh / m_e + 2 * alpha_T * E1
            DE_h = (1 - np.sqrt(1 - 4 * alpha_T * E1 * (1 + alpha_T * E1) /
                                Gamma**2)) / (2 * alpha_T) / Gamma
            # DE_h = E1 / (1 + m_hh / m_T)
            E_e = E1 - DE_h
        elif x == 4:  # light hole
            E1 = hw - Ei
            Gamma = 1 + m_lh / m_e + 2 * alpha_T * E1
            DE_h = (1 - np.sqrt(1 - 4 * alpha_T * E1 * (1 + alpha_T * E1) /
                                Gamma**2)) / (2 * alpha_T) / Gamma
            # DE_h = E1 / (1 + m_lh / m_T)
            E_e = E1 - DE_h
    else:
        E_e = 0.02
    # print(DE_h, E_e)

    return E_e


def electron_distribution(hw, Ni, thick, types):
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
    if types == 2:  # use density of state from reference
        DOS = np.genfromtxt('GaAs_DOS.csv', delimiter=',')
        func1 = interp1d(DOS[:, 0], DOS[:, 1])
        E0 = Eg
        norm, err = integrate.quad(lambda e: func1(e - hw) * func1(e), E0, hw,
                                   limit=10000)
        Ei = np.linspace(E0, hw, int((hw - E0) / 0.001))
        for i in range(len(Ei)):
            num = 1.5 * Ni * func1(Ei[i]) * func1(Ei[i] - hw) * 0.001 / norm
            E_num = np.empty(int(num))
            E_num.fill(Ei[i] - E0)
            energy.extend(E_num)
        np.random.shuffle(energy)

    absorb_data = np.genfromtxt('absorp_coeff_GaAs.txt', delimiter=',')
    func2 = interp1d(absorb_data[:, 0], absorb_data[:, 1])
    alpha = func2(hw) * 10**-7  # 1/nm,  absorption coefficient
    # photon distribution in GaAs (exp distribution random variables)
    z_exp = expon.rvs(loc=0, scale=1 / alpha, size=Ni)
    # photon distribution in GaAs with thickness less than thick
    z_pos = [z for z in z_exp if z <= thick]
    z_pos = np.array(z_pos)
    Num = len(z_pos)
    energy = np.resize(energy, Num)
    phi = np.random.uniform(0, 2 * pi, Num)
    theta = np.random.uniform(0, 2 * pi, Num)
    y_pos = np.random.normal(0, 0.25 * 10**6, Num)
    velocity = np.sqrt(2 * np.abs(energy) * ec / m_T) * 10**9
    vz = velocity * np.cos(theta)
    vy = velocity * np.sin(theta) * np.cos(phi)
    distribution_2D = np.vstack((z_pos, y_pos, vz, vy, velocity, energy)).T

    '''
    distribution_2D = []
    for i in range(len(z_pos)):
        # initial angle between the projection on XY surface and y axis
        phi = random.uniform(0, 2 * pi)
        # initial angle between the direction and z axis
        theta = random.uniform(0, 2 * pi)
        # y_pos = random.uniform(-1 * 10**6, 1 * 10**6)
        y_pos = random.gauss(0, 0.25 * 10**6)
        velocity = np.sqrt(2 * np.abs(energy[i]) * ec / m_T) * 10**9  # nm/s
        vz = velocity * np.cos(theta)
        vy = velocity * np.sin(theta) * np.cos(phi)
        distribution_2D.append([z_pos[i], y_pos, vz, vy, velocity, energy[i]])
    distribution_2D = np.array(distribution_2D)  # ([z, y, vz, vy, v, E])'''
    '''
    plt.figure()
    plt.hist(distribution_2D[:, 5], bins=100)
    plt.show()
    '''
    return distribution_2D


def electron_transport(distribution_2D, endT, surface, thick, types):
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
    if types == 1:
        dist_2D = distribution_2D
        mfp_ep = 3  # nm, mean free path for e-p scattering
        ep = 0.027  # eV, phonon energy in GaAs
        mfp_ee = 10  # nm, mean free path for e-e scattering
        mfp_eh = 5  # nm, mean free path for e-h scattering
        mfp = np.min([mfp_ep, mfp_ee, mfp_eh])
        t = 0
        while t < endT:
            # random free path, standard normal distribution
            #  2.5 * np.random.randn(2, 4) + 3, N(3, 6.25)
            # sigma * np.random.randn() + mu, N(mu, sigma**2)
            free_path = np.abs(mfp / 3) * np.random.randn() + mfp
            # mean velocity for mean energy, nm/s
            mean_v = np.sqrt(2 * np.mean(dist_2D[:, 5]) * ec / m_T) * 10**9
            stept = free_path / mean_v / 5
            # transfer matrix after stept for electron without scattering
            M_st = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0],
                             [stept, 0, 1, 0, 0, 0], [0, stept, 0, 1, 0, 0],
                             [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
            dist_2D = np.dot(dist_2D, M_st)

            # ----------- scattering change the distribution -----------
            # ----- get the energy distribution after scattering -------
            # e-p scattering probability within stept
            P_ep = mean_v * stept / mfp_ep
            # loss energy probability for e-p scattering
            P_loss = 0.7
            # electron energy distribution after e-p scattering
            Num = len(dist_2D[:, 5])
            ep_dist = np.array([ep] * int(Num * P_ep * P_loss) +
                               [-ep] * int(Num * P_ep * (1 - P_loss)) +
                               [0] * (Num - int(Num * P_ep * P_loss) -
                                      int(Num * P_ep * (1 - P_loss))))
            np.random.shuffle(ep_dist)
            ep_dist_norm = (ep_dist / ep).astype(int)
            # if e-p scattering occur when E < ep ?????
            # dist_2D[:, 5] = dist_2D[:, 5] - ep_dist
            for i in range(len(dist_2D[:, 5])):
                if dist_2D[i, 5] > ep:
                    dist_2D[i, 5] = dist_2D[i, 5] - ep_dist[i]

            for i in range(len(dist_2D[:, 5])):
                happen = 0  # 0: scattering not happen, 1: scattering happen
                if dist_2D[i, 5] > Eg:
                    P_ee = dist_2D[i, 4] * stept / mfp_ee
                    if P_ee < 1:
                        if random.uniform(0, 1) <= P_ee:
                            happen = 1
                        else:
                            happen = 0
                    else:
                        happen = 1
                    dist_2D[i, 5] -= random.uniform(Eg, dist_2D[i, 5]) * happen
                if dist_2D[i, 5] > 0:
                    P_eh = dist_2D[i, 4] * stept / mfp_eh
                    if P_eh < 1:
                        if random.uniform(0, 1) <= P_eh:
                            happen = 1
                        else:
                            happen = 0
                    else:
                        happen = 1
                    dist_2D[i, 5] -= random.uniform(0, dist_2D[i, 5]) * happen
                elif dist_2D[i, 5] <= 0:
                    happen = 0

            # ---- renew the velocity and direction after scattering -----
                if dist_2D[i, 5] > 0:
                    dist_2D[i, 4] = np.sqrt(
                        2 * np.abs(dist_2D[i, 5]) * ec / m_T) * 10**9
                    phi = random.uniform(0, 2 * pi)
                    theta = random.uniform(0, 2 * pi)
                    if max(happen, ep_dist_norm[i]) == 1:
                        dist_2D[i, 2] = dist_2D[i, 4] * np.cos(theta)
                        dist_2D[i, 3] = dist_2D[i, 4] * np.sin(theta) *\
                            np.cos(phi)

            # ------ filter electrons-------
            bd, fd, td, dist_2D = filter(dist_2D, surface, thick, 0)

            back_2D.extend(bd.tolist())
            trap_2D.extend(td.tolist())
            surface_2D.extend(fd.tolist())

            t += stept

            if len(dist_2D) == 0:
                break

        dist_2D = dist_2D.tolist()
        dist_2D.extend(back_2D)
        dist_2D.extend(trap_2D)
        dist_2D.extend(surface_2D)
        dist_2D = np.array(dist_2D)
        dist_2D[:, 5] = np.maximum(dist_2D[:, 5], 0)
        dist_2D[:, 0] = np.clip(dist_2D[:, 0], surface, thick)

    elif types == 2:
        '''including e-p, e-h and e-impurity scattering '''
        dist_2D = distribution_2D
        mfp_ep = 3  # nm, mean free path for e-p scattering
        ep = 0.027  # eV, phonon energy in GaAs
        mfp_eh = 5  # nm, mean free path for e-h scattering
        mfp = np.min([mfp_ep, mfp_eh])
        t = 0
        while t < endT:
            # random free path, standard normal distribution
            #  2.5 * np.random.randn(2, 4) + 3, N(3, 6.25)
            # sigma * np.random.randn() + mu, N(mu, sigma**2)
            free_path = np.abs(mfp / 3) * np.random.randn() + mfp
            # mean velocity for mean energy, nm/s
            mean_v = np.sqrt(2 * np.mean(dist_2D[:, 5]) * ec / m_T) * 10**9
            stept = free_path / mean_v / 5
            t += stept
            # transfer matrix after stept for electron without scattering
            M_st = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0],
                             [stept, 0, 1, 0, 0, 0], [0, stept, 0, 1, 0, 0],
                             [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
            dist_2D = np.dot(dist_2D, M_st)

            # ----------- scattering change the distribution -----------
            # ----- get the energy distribution after scattering -------

            # --- e-p scattering ---
            # e-p scattering probability within stept
            # P_ep = mean_v * stept / mfp_ep
            P_ep = 0.2
            # loss energy probability for e-p scattering
            P_loss = 0.7
            # electron energy distribution after e-p scattering
            Num = len(dist_2D)
            ep_gain = int(Num * P_ep * P_loss)
            ep_loss = int(Num * P_ep * (1 - P_loss))
            ep_dist = np.array([ep] * ep_gain + [-ep] * ep_loss +
                               [0] * (Num - ep_gain - ep_loss))
            np.random.shuffle(ep_dist)
            P_ep_ind = (ep_dist / ep).astype(int)
            # if e-p scattering occur when E < ep ?????
            # dist_2D[:, 5] = dist_2D[:, 5] - ep_dist
            energy_ind = dist_2D[:, 5] >= ep
            # print(Num, len(P_ep_ind), len(energy_ind), len(ep_dist))
            happen_ep = P_ep_ind * energy_ind
            dist_2D[:, 5] = dist_2D[:, 5] - ep * happen_ep

            # ----- e-impurity scattering ---Y. Nishimura, Jnp. J. Appl. Phys.
            tempEnergy = dist_2D[:, 5].clip(0.001)
            k_w = np.sqrt(2 * m_T * tempEnergy * ec) / h_  # 1/m, wavevector
            n0 = 10**25  # m**-3, carrier concentration
            ni = 10**25  # m**-3, impurity concentration
            a2 = (eps * kB * T) / (4 * pi * n0 * ec**2)  # m**2
            # e-impurity scattering rate
            Rate_ei = (2 * pi * ni * ec**4 * m_T) / (eps**2 * h_**3 * k_w**3) \
                * (np.log(1 + 4 * a2 * k_w**2) -
                   (4 * a2 * k_w**2) / (1 + 4 * a2 * k_w**2))
            P_ei = stept * Rate_ei  # e-impurity scattering probability
            random_P_ei = np.random.uniform(0, 1, Num)
            P_ei_ind = random_P_ei <= P_ei
            energy_ind = dist_2D[:, 5] > 0
            happen_ie = [1] * Num * P_ei_ind * energy_ind
            ei_loss = np.random.uniform(0, tempEnergy) * happen_ie
            dist_2D[:, 5] = dist_2D[:, 5] - ei_loss

            # --- e-h scattering ---
            P_eh = dist_2D[:, 4] * stept / mfp_eh
            random_P_eh = np.random.uniform(0, 1, Num)
            P_eh_ind = random_P_eh <= P_eh
            energy_ind = dist_2D[:, 5] > 0
            happen_eh = [1] * Num * P_eh_ind * energy_ind
            eh_loss = np.random.uniform(0, tempEnergy) * happen_eh
            dist_2D[:, 5] = dist_2D[:, 5] - eh_loss
            # print(dist_2D[:, 5], len(dist_2D))
            happen = happen_ep + happen_ie + happen_eh

            # ---- renew the velocity and direction after scattering -----
            energy_ind = dist_2D[:, 5] > 0
            happen = happen * energy_ind
            phi = np.random.uniform(0, 2 * pi, Num)
            theta = np.random.uniform(0, 2 * pi, Num)
            dist_2D[:, 4] = dist_2D[:, 4] * (~energy_ind) + np.sqrt(
                2 * np.abs(dist_2D[:, 5]) * ec / m_T) * 10**9 * happen
            dist_2D[:, 2] = dist_2D[:, 4] * np.cos(theta) * happen + \
                dist_2D[:, 2] * (~energy_ind)
            dist_2D[:, 3] = dist_2D[:, 4] * np.sin(theta) * np.cos(phi) +\
                dist_2D[:, 3] * (~energy_ind)

            # ------ filter electrons-------
            bd, fd, td, dist_2D = filter(dist_2D, surface, thick, 0)

            back_2D.extend(bd.tolist())
            trap_2D.extend(td.tolist())
            surface_2D.extend(fd.tolist())

            if len(dist_2D) == 0:
                break

        dist_2D = dist_2D.tolist()
        dist_2D.extend(back_2D)
        dist_2D.extend(trap_2D)
        dist_2D.extend(surface_2D)
        dist_2D = np.array(dist_2D)
        dist_2D[:, 5] = np.maximum(dist_2D[:, 5], 0)
        dist_2D[:, 0] = np.clip(dist_2D[:, 0], surface, thick)

    else:
        print('selected wrong types')

    back_2D = np.array(back_2D)
    if len(back_2D) != 0:
        back_2D[:, 5] = 0

    trap_2D = np.array(trap_2D)
    if len(trap_2D) != 0:
        trap_2D[:, 0] = thick

    surface_2D = np.array(surface_2D)
    if len(surface_2D) != 0:
        surface_2D[:, 0] = surface

    return surface_2D, back_2D, trap_2D, dist_2D


def filter(dist_2D, surface, thick, threshold_value):
    ''' filter electrons
    Find these electrons diffused to surface, substrate, trapped
    and the rest electrons that will continue to diffuse
    '''
    assert thick > surface
    back = dist_2D[:, 0] >= thick
    front = dist_2D[:, 0] <= surface
    trap = dist_2D[:, 5] <= threshold_value

    back_dist = dist_2D[back, :]
    front_dist = dist_2D[front, :]
    trap_dist = dist_2D[trap, :]
    rest_dist = dist_2D[(~back) & (~front) & (~trap), :]
    return back_dist, front_dist, trap_dist, rest_dist


def electron_emitting(surface_2D, E_A, E_sch):
    ''' conidtions should be matched before emitting:
    1. E_out = E_e - E_A + E_sch > 0
    2. P_out > P_out_T = P_in_T
    '''
    surface_trap = []
    match_ind = surface_2D[:, 5] >= (E_A - E_sch)
    match_E = surface_2D[match_ind, :]
    surface_trap.extend(surface_2D[(~match_ind), :].tolist())
    match_E = match_E - E_A - E_sch

    phi = np.random.uniform(0, 2 * pi, (len(match_E), 1))
    match_E = np.append(match_E, phi, 1)
    match_E[:, 4] = np.sqrt(2 * match_E[:, 5] / m_T) * \
        c * 10**9 * np.cos(match_E[:, 6])
    match = np.abs(match_E[:, 4]) >= np.abs(match_E[:, 3])
    emission_2D = match_E[match, :]
    emission_2D[:, 2] = np.sqrt(emission_2D[:, 4]**2 - emission_2D[:, 3]**2)
    surface_trap.extend(match_E[(~match), :].tolist())
    surface_trap = np.array(surface_trap)
    return emission_2D, surface_trap


def plot_QE(data):
    fig1, ax1 = plt.subplots()
    ax1.plot(data[:, 0], data[:, 1])
    ax1.set_xlabel('Photon energy (eV)')
    ax1.set_ylabel('QE (%)')
    plt.savefig('QE.pdf', format='pdf')
    plt.show()


def main(opt):
    hw_start = float('%.2f' % Eg) + 0.01  # eV
    hw_end = 2.48  # eV
    hw_step = 0.02  # eV
    hw_test = 2.0  # eV
    Ni = 100000  # incident photon number
    thick = 2000  # nm, thickness of GaAs active layer
    endT = 0.01  # s
    surface = 0  # position of electron emission, z = 0
    E_A = -0.1  # eV, electron affinity
    E_sch = 0  # eV, vacuum level reduction by Schottky effect
    data = []
    if opt == 1:  # for test
        dist_2D = electron_distribution(hw_test, Ni, thick, 2)
        print('excited electron ratio: ', len(dist_2D) / Ni)

        surface_2D, back_2D, trap_2D, dist_2D = electron_transport(
            dist_2D, endT, surface, thick, 2)
        print('surface electron ratio: ', len(surface_2D) / Ni)

        emiss_2D, surf_trap = electron_emitting(surface_2D, E_A, E_sch)
        print('QE (%): ', 100.0 * len(emiss_2D) / Ni)

    elif opt == 2:
        for hw in np.arange(hw_start, hw_end, hw_step):
            dist_2D = electron_distribution(hw, Ni, thick, 2)
            print('excited electron ratio: ', len(dist_2D) / Ni)

            surface_2D, back_2D, trap_2D, dist_2D = electron_transport(
                dist_2D, endT, surface, thick, 1)
            print('surface electron ratio: ', len(surface_2D) / Ni)

            emiss_2D, surf_trap = electron_emitting(surface_2D, E_A, E_sch)

            QE = 100.0 * len(emiss_2D) / Ni
            print('photon energy (eV): ', hw, ', QE (%): ', QE)
            data.append([hw, QE])

        data = np.array(data)
        plot_QE(data)

    print('run time:', time.time() - start_time, 's')


if __name__ == '__main__':
    main(1)
