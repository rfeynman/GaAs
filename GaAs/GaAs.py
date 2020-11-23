'''
Created on Feb 5, 2015

@author: E. Wang

|----------------------------|
|                            |
|                            |
|                            |    ^y
|                            |    |
|----------------------------|    |
------------------------------>z
(z,y,vz,vy,v,ene,x)
{
The probability density function for expon is:

expon.pdf(x) = lambda * exp(- lambda*x)
for x >= 0.

The scale parameter is equal to scale = 1.0 / lambda.
rvs(loc=0, scale=1, size=1)
}
{
A special case of a chi distribution, with df = 3, loc = 0.0, and given scale = a, where a is the parameter used in the Mathworld description [R240].

The probability density function for maxwell is:

maxwell.pdf(x) = sqrt(2/pi)x**2 * exp(-x**2/2)
for x > 0.
rvs(loc=0, scale=1, size=1)    Random variates.
pdf(x, loc=0, scale=1)    Probability density function.
}
version v0: basic version, only e-p scattering
version v1: add photon energy scan function. could generate QE-Emittance, electrons loss factor
version v2, add e-h scattering function, add e-e scattering opinion, add EA scan
version v3, add e-e scattering function, ea scan for both emittance and electron loss. save figures. add green laser mark on scanned figure.
version v4, use a mfp/5 as step, reduce the step length, fix the first step too large issue
in v4plus, consider the e-p scatter may gain the energy
'''

import random
from scipy.stats import maxwell, expon
from scipy.constants import *
import matplotlib.pyplot as plt
import numpy as np

import math
from numpy import genfromtxt, dtype
from scipy.interpolate import interp1d
from scipy import integrate
import time
import os

start_time = time.time()
'''
def Max_boltE(ene,le):  ## Maxwell b distribution
    return 2(((ene/pi)**0.5)/le**1.5)*np.exp(-ene/le)
'''
ec = 1.6 * (10**(-19))
ma = 9.11 * 10**-31  # electron mess,kg
eff = 0.067
me = eff * ma  # effective electron mess,kg
bandgap = 1.42  # T = 300 K


def MBdist(n, e_photon, thick):
    # n: particle number, loct: start point(x-x0), scale: sigma
    # wl: wavelength,thick: thickness of the cathode
    assert e_photon > bandgap
    if e_photon - bandgap - 0.8 <= 0:
        scale = e_photon - bandgap
        loct = 0
    else:
        scale = 0.8
        loct = e_photon - bandgap - scale
    data = maxwell.rvs(loc=loct, scale=scale, size=n)
    data_ene = np.array(data)
    params = maxwell.fit(data, floc=0)
    data_v = np.sqrt(2 * data_ene * ec / me) * 10**9
    p2D = []
    wl = ((19.82 - 27.95 * e_photon + 11.15 * e_photon**2) * 10**-3)**-1
    pens = expon.rvs(loc=0, scale=wl, size=n)
    penss = filter(lambda x: x <= thick, pens)
    params_exp = expon.fit(pens, floc=0)
    i = 0

    for n in range(len(penss)):
        phi = random.uniform(0, 2 * math.pi)  # initial angular
        # initial y direction position
        poy = random.uniform(-1 * 10**6, 1 * 10**6)
        p2D.append([penss[i], poy, data_v[i] * math.cos(phi), data_v[i] *
                    math.sin(phi), data_v[i], data[i]])
        # p2D: (z,y,vz,vy,v,ene)
        i += 1
    p2D = np.array(p2D)
    return params, p2D, penss, params_exp


# n, the photon numbers; loct, the postion(z) of the electrons;thick, unit, nm.
def DosDist(n, e_photon, thick, data, absorb_data):

    f = interp1d(data[:, 0], data[:, 1])
    f2 = interp1d(absorb_data[:, 0], absorb_data[:, 1])

    n1 = int((e_photon - 1) / 0.01)  # change to n
    energy = np.linspace(1., e_photon, n1)
    norm, err = integrate.quad(lambda e: f(
        e - e_photon) * f(e), 1, e_photon, limit=10000)

    data_ene = []
    num_energy = []
    i = 0
    while i < n1:
        # using n instead of n1
        n3 = round(1.5 * n * f(energy[i] - e_photon) *
                   f(energy[i]) * 0.01 / norm)
        num_energy.append(n3)
        ener_array = np.empty(int(n3))
        ener_array.fill(energy[i])
        data_ene.extend(ener_array)
        i += 1
    np.random.shuffle(data_ene)

    p2D = []
    # wl=((19.82-27.95*e_photon+11.15*e_photon**2)*10**-3 )**-1
    wl = (f2(e_photon) * 10**-3)**-1
    pens = expon.rvs(loc=0, scale=wl, size=n)
    penss = list(filter(lambda x: x <= thick, pens))
    params_exp = expon.fit(pens, floc=0)

    i = 0
    for i in range(len(penss)):
        phi = random.uniform(0, 2 * math.pi)  # initial angular on 2D surface
        # initial angular on perpdiculat face
        php = random.uniform(0, 2 * math.pi)
        # initial y direction position, nm
        poy = random.uniform(-1 * 10**6, 1 * 10**6)
        v = np.sqrt(2 * np.abs((data_ene[i] - bandgap)) * ec / me) * 10**9
        p2D.append([penss[i], poy, v * math.cos(phi) * math.cos(php),
                    v * math.sin(phi) * math.cos(php), v,
                    np.abs(data_ene[i] - bandgap)])  # p2D: (z,y,vz,vy,v,ene)
        i += 1
    p2D = np.array(p2D)

    '''
    plt.subplot(211)
    plt.plot(data[:, 0], data[:, 1])
    plt.subplot(212)
    plt.hist(p2D[:, 5], bins=30)
    plt.show()
    '''
    return params_exp, p2D, penss, params_exp


def diff(stept, endT, p2Di, mfp_ps, eloss, bounde, bounda, types):

    p2D_emission = []
    p2D_trap = []
    p2D_back = []

    if types == 1:
        # non scattering transistion matrix
        tmatix_ns = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0],
                              [endT, 0, 1, 0, 0, 0], [0, endT, 0, 1, 0, 0],
                              [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
        p2De = np.dot(p2Di, tmatix_ns)
        # p2De[:,0]=np.clip(p2De[:,0],bounde,bounda)
        se, st, sb, p2De = swap(bounde, bounda, 0, p2De)
        p2D_emission.extend(se.tolist())
        p2D_trap.extend(st.tolist())
        p2D_back.extend(sb.tolist())
        p2De = p2De.tolist()

        p2De.extend(p2D_back)
        p2De.extend(p2D_trap)
        p2De.extend(p2D_emission)

        p2De = np.array(p2De)
        p2De[:, 5] = np.maximum(p2De[:, 5], 0)
        p2De[:, 0] = np.clip(p2De[:, 0], bounde, bounda)

    elif types == 2:
        tmatix_diff = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0],
                                [endT, 0, 1, 0, 0, 0], [0, endT, 0, 1, 0, 0],
                                [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
        t = 0
        for t in range(int(endT / stept)):
            q = random.uniform(0, 2 * math.pi)
            tmatix_difft = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [stept, 0, 0, 0, 0, 0], [
                                    0, stept, 0, 0, 0, 0], [0, 0, math.cos(q), math.sin(q), 1, 0], [0, 0, 0, 0, 0, 1]])
            tmatix_diff = np.dot(tmatix_diff, tmatix_difft)
            t += 1
        p2De = np.dot(p2Di, tmatix_diff)
        # p2De[:,0]=np.clip(p2De[:,0],bounde,bounda)
        se, st, sb, p2De = swap(bounde, bounda, 0, p2De)
        p2D_emission.extend(se.tolist())
        p2D_trap.extend(st.tolist())
        p2D_back.extend(sb.tolist())

        p2De = p2De.tolist()
        p2De.extend(p2D_back)
        p2De.extend(p2D_trap)
        p2De.extend(p2D_emission)

        p2De = np.array(p2De)
        print(p2De)
        p2De[:, 5] = np.maximum(p2De[:, 5], 0)

        print(p2De, bounde, bounda)

        p2De[:, 0] = np.clip(p2De[:, 0], bounde, bounda)
    elif types == 3:
        t = 0
        eemfpdata = genfromtxt('K2CsSb_ees.csv', delimiter=',')
        eemfp = interp1d(eemfpdata[:, 0], eemfpdata[:, 1])
        p2De = p2Di
        tempres = []
        while t < endT:
            ee = eloss
            # random mfp, standard normal distribution
            #  2.5 * np.random.randn(2, 4) + 3, N(3, 6.25)
            # sigma * np.random.randn() + mu, N(mu, sigma**2)
            mfp = np.abs((mfp_ps / 3) * np.random.randn() + mfp_ps)

            # 0.2*tau is the step for collision
            stept = mfp / \
                (np.sqrt(2 * np.mean(p2De[:, 5]) * ec / me) * 10**9) / 5.0
            # print np.mean(p2De[:,5]),'
            # ',np.sqrt(2*np.mean(p2De[:,5])*ec/me),'   ', mfp,'   ', stept
            tmatix_epsim = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [stept, 0, 1, 0, 0, 0], [
                                    0, stept, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
            p2De = np.dot(p2De, tmatix_epsim)

            lossoverall = 0.7  # 1

            ranlist = np.array([ee] * int(len(p2De) * 0.2 * lossoverall) + [-ee] * int(len(p2De) * 0.2 * (1 - lossoverall)) + [
                               0] * (len(p2De) - int(len(p2De) * 0.2 * lossoverall) - int(len(p2De) * 0.2 * (1 - lossoverall))))
            np.random.shuffle(ranlist)
            nor_ranlist = (ranlist / ee).astype(int)
            p2De[:, 5] = p2De[:, 5] - ranlist

            for k in range(len(p2De)):

                # start a e-e scattering   stept     ,eemfp, and e-h scattering
                if p2De[k - 1, 5] > bandgap:
                    # mfp/2 is average loss? remove later, /5 is match the
                    # tstep
                    r = float(mfp / (eemfp(p2De[k - 1, 5] - bandgap) / 10) / 5)

                    # ee-eh random.randint(0,1); if ee set to 1
                    sec1 = random.randint(0, 1)
                    # the ratio of e-e and e-h scattering
                    selc = [sec1, 1 - sec1]
                    # print int(r*100)
                    if r <= 1:
                        # ranlist=[0]*(100-int(r*100))+[1]*int(r*100)
                        if random.random() <= r:
                            happ = 1
                        else:
                            happ = 0

                        # random.uniform(bandgap,p2De[k-1,5])*happ*selc[0] #e-e
                        # scattering
                        p2De[k - 1, 5] = p2De[k - 1, 5] - \
                            random.uniform(
                                bandgap, p2De[k - 1, 5]) * happ * selc[0]
                        # e-h scattering
                        p2De[k - 1, 5] = p2De[k - 1, 5] - \
                            random.uniform(0, 1.27) * happ * selc[1]

                    else:
                        happ = 1
                        # *random.uniform(bandgap,p2De[k-1,5])*selc[0]      e-e scattering *int(mfp/(eemfp(p2De[k-1,5]-bandgap)/10))
                        p2De[k - 1, 5] = p2De[k - 1, 5] - np.minimum(int(r), int(
                            p2De[k - 1, 5] / bandgap)) * random.uniform(bandgap, p2De[k - 1, 5]) * selc[0]
                        p2De[k - 1, 5] = p2De[k - 1, 5] - np.minimum(int(r), int(
                            p2De[k - 1, 5] / bandgap)) * random.uniform(0, 1.27) * selc[1]  # e-h scattering
                else:
                    happ = 0

                if p2De[k - 1, 5] > 0:
                    p2De[k - 1, 4] = np.sqrt(2 *
                                             p2De[k - 1, 5] * ec / me) * 10**9
                    q = random.uniform(0, 2 * math.pi)
                    qp = random.uniform(0, 2 * math.pi)
                    if np.maximum(happ, nor_ranlist[k - 1]) == 1:
                        p2De[k - 1, 2] = p2De[k - 1, 4] * \
                            math.cos(q) * math.cos(qp) * \
                            np.maximum(happ, nor_ranlist[k - 1])
                        p2De[k - 1, 3] = p2De[k - 1, 4] * \
                            math.sin(q) * math.cos(qp) * \
                            np.maximum(happ, nor_ranlist[k - 1])

            se, st, sb, p2De = swap(bounde, bounda, 0, p2De, ee)

            p2D_emission.extend(se.tolist())
            p2D_trap.extend(st.tolist())
            p2D_back.extend(sb.tolist())
            ##print (t,'   ', len(p2De),'  ',len(se),'   ',len(st),'    ',len(sb))

            tempres.append([t, len(se)])

            t += stept

            if len(p2De) == 0:
                # print p2De,p2D_trap
                break

        p2De = p2De.tolist()
        p2De.extend(p2D_back)
        p2De.extend(p2D_trap)
        p2De.extend(p2D_emission)
        p2De = np.array(p2De)
        p2De[:, 5] = np.maximum(p2De[:, 5], 0)
        p2De[:, 0] = np.clip(p2De[:, 0], bounde, bounda)

    else:
        print(
            "wrong type: 1 is non scattering; 2 is brown diffusion 3 is consider scattering")

    p2D_emission = np.array(p2D_emission)

    if len(p2D_emission) != 0:

        p2D_emission[:, 0] = bounde

    p2D_back = np.array(p2D_back)

    if len(p2D_back) != 0:
        p2D_back[:, 0] = bounda

    p2D_trap = np.array(p2D_trap)
    if len(p2D_trap) != 0:
        p2D_trap[:, 2] = 0
    tempres = np.array(tempres)
    return p2D_emission, p2D_trap, p2D_back, p2De, tempres


def swap(be, ba, bb, mat, el):
    assert ba > be
    emissind = mat[:, 0] <= be
    backind = mat[:, 0] >= ba
    loweind = mat[:, 5] <= bb

    swape = mat[emissind, :]
    swape[:, 5] = swape[:, 5] + el
    swapt = mat[(~emissind) & (~backind) & loweind, :]
    swapb = mat[backind, :]
    swapr = mat[(~emissind) & (~loweind) & (~backind), :]
    # print swapt.shape
    return swape, swapt, swapb, swapr


def emission(mat, ea, sctk):

    matind = mat[:, 5] >= (ea + sctk)
    mat2 = mat[matind, :]

    mat2[:, 5] = mat2[:, 5] - ea - sctk
    phpangle = np.random.uniform(0, 2 * math.pi, (len(mat2), 1))
    mat2 = np.append(mat2, phpangle, 1)
    mat2[:, 4] = np.sqrt(2 * mat2[:, 5] * ec / me) * np.cos(mat2[:, 6]) * 10**9
    mat2ind = np.abs(mat2[:, 4]) > np.abs(mat2[:, 3])
    # print mat2ind
    emission = mat2[mat2ind, :]

    emission[:, 2] = np.sqrt(emission[:, 4]**2 - emission[:, 3]**2)
    surface_trap = len(mat) - len(emission)
    '''
    gamma=(np.mean(emission[:,5])+511000.0)/511000.0
    beta=np.sqrt(1-gamma**-2)
    Temittance= gamma*beta*np.sqrt(np.mean(emission[:,1]**2)*(10**-9)*np.mean((emission[:,3]/emission[:,2])**2)-np.mean(emission[:,1]*(10**-9)*(emission[:,3]/emission[:,2]))**2)  #sqrt(<x**2><x'**2>-<xx'>**2)
    '''
    Temittance = np.sqrt(np.mean((emission[:, 1] * 10**-9)**2)) * (np.sqrt(np.mean(
        ((emission[:, 3] * 10**-9) / np.cos(emission[:, 6]))**2))) * np.sqrt(eff) / (3 * 10**8)  # sigma_x*sqrt(<v_x**2>)/c
    # print np.mean(emission[:,1]**2),np.mean(emission[:,1]**2),'\n',np.mean((emission[:,3]/emission[:,2])**2),'\n',np.mean(emission[:,1]*(emission[:,3]/emission[:,2]))**2
    # print
    # np.sqrt(np.mean((emission[:,1]*10**-9)**2)),np.sqrt(np.mean(emission[:,3]**2))/(3*10**8)
    return emission, Temittance, surface_trap


def plot(p2D, pens, p2De, para, params_exp, thickness, tempres, petest, electron_affinity, finame):
    # print tempres
    plt.subplot(511)
    plt.xlabel('thickness[nm]')
    plt.ylabel('P')
    plt.hist(p2D[:, 5], bins=20, normed=True)
    x = np.linspace(0, 5, 80)
    if para != None:
        plt.plot(x, maxwell.pdf(x, *para), 'r', x, maxwell.cdf(x, *para), 'g')

    plt.subplot(512)
    plt.hist(pens, bins=20, normed=True)
    z = np.linspace(0, 500, 200)
    plt.xlabel('thickness[nm]')
    plt.ylabel('P')
    plt.xlim(0, 2 * thickness)
    # plt.plot(z, expon.pdf(z, *params_exp),'r',z,expon.cdf(z, *params_exp), 'g')
    plt.plot(z, expon.pdf(z, *params_exp), 'r')

    plt.subplot(513)
    plt.xlim(0, thickness + 1)
    plt.xlabel('thickness[nm]')
    plt.ylabel('size[nm]')
    plt.plot(p2D[:, 0], p2D[:, 1], 'b,')

    plt.subplot(514)
    plt.ylim(-1 * 10**6, 1 * 10**6)
    plt.xlim(0, thickness + 1)
    plt.xlabel('thickness[nm]')
    plt.ylabel('size[nm]')
    plt.plot(p2De[:, 0], p2De[:, 1], 'b,')

    plt.subplot(515)

    plt.xlabel('Time[s]')
    plt.ylabel('count[1]')
    smooth = 80

    for i in np.arange(0, len(tempres)):

        tempres[i, 1] = (np.sum(tempres[i:i + smooth, 1]) /
                         smooth).astype(float)

    plt.ylim(0, tempres[1, 1] * 1.5)
    plt.plot(tempres[:, 0], tempres[:, 1], 'r-')

    plt.savefig(finame + 'full_tk_%dpe_%.3fea_%.3f.pdf' %
                (thickness, petest, electron_affinity))
    plt.show()
    return


def emittancescanplot(wlscan, start, stop, ea, thickness, mfp, expdata, finame):
    with open(finame + 'emittance_ea_wl_scan_mfp_%.3ftk_%dbg_%.3feff_%.4f.csv' % (mfp, thickness, bandgap, eff), 'ab') as csv:
        np.savetxt(csv, wlscan, fmt='%.8e', delimiter=',')
    fig, ax1 = plt.subplots(figsize=(16, 8))
    ax2 = ax1.twinx()

    ax1.plot(wlscan[:, 0], wlscan[:, 1] * 10**6, 'g-')
    ax1.plot(2.3, 0.39, 'go')
    ax2.plot(wlscan[:, 0], wlscan[:, 2], 'b-')
    ax2.plot(expdata[:, 0], expdata[:, 1], 'ro', label='%.1f' % expdata[0, 1])
    ax1.set_xlabel('photon energy[eV]')
    # plt.xlim(1.6,6.5)
    ticks = np.arange(start - 0.2, stop + 0.2, 0.2)
    ax1.set_xticks(ticks)
    # print ticks
    ax1.set_ylabel('emittance [mrad]', color='g')

    ax2.set_ylabel('QE[%]', color='b')
    ax2.set_yscale('log')
    ax2.margins(0.02)
    plt.savefig(finame + 'emittance_ea_%.3fmfp_%.3ftk_%dbg_%.3feff_%.4f.pdf' %
                (ea, mfp, thickness, bandgap, eff))


def electronlossscanplot(wlscan, start, stop, pestep, mfp, electron_affinity, thickness, ee, eh, expdata, finame):
    with open(finame + 'ea_wl_scan_mfp_%.3fee_%deh_%d.csv' % (mfp, ee, eh), 'ab') as csv:
        np.savetxt(csv, wlscan, fmt='%.8e', delimiter=',')

    fig, ax1 = plt.subplots(figsize=(16, 8))

    ax1.plot(wlscan[:, 0], wlscan[:, 2], 'b-', label='emission')
    ax1.plot(wlscan[:, 0], wlscan[:, 3], 'r-', label='laser loss')
    ax1.plot(wlscan[:, 0], wlscan[:, 4], 'g-', label='diffuse to back')
    ax1.plot(wlscan[:, 0], wlscan[:, 5], 'y-', label='trap in crystal')
    ax1.plot(wlscan[:, 0], wlscan[:, 6], 'k-', label='trap on surface')
    ax1.plot(expdata[:, 0], expdata[:, 1], 'bo', label='%.1f' % expdata[0, 1])
    ind_annot = (2.3 - (start)) / pestep

    # print
    # bandgap1,electron_affinity,(2.3-(start))/pestep,wlscan[ind_annot,0],wlscan[0,0]
    ax1.annotate('(%s,%s)' % (wlscan[ind_annot + 1, 0], wlscan[ind_annot + 1, 2]), xy=(
        wlscan[ind_annot + 1, 0], wlscan[ind_annot + 1, 2]), textcoords='offset points')

    legend = ax1.legend(loc='best')
    ax1.set_xlabel('photon energy[eV]')
    # plt.xlim(1.6,6.5)
    ticks = np.arange(start - 0.2, stop + 0.2, 0.2)
    ax1.set_xticks(ticks)

    ax1.set_ylabel('rate[%]', color='b')
    ax1.set_yscale('log')
    ax1.margins(0.02)
    plt.savefig(finame + 'mfp_%.3fea_%.3ftk_%dee_%deh_%d.pdf' %
                (mfp, electron_affinity, thickness, ee, eh))


def main(op):

    # Density Of State (DOS) for GaAs band
    dosdata = genfromtxt('GaAs_DOS.csv', delimiter=',')
    expdata = genfromtxt('cooling_exp.csv', delimiter=',')
    # absorption coefficient from reference
    absorb_data = genfromtxt('absorp_coeff_GaAs.txt', delimiter=',')
    electron_affinity = -0.05
    thickness = 1000
    n = 100000
    # pestart=1.4
    peend = 2.5
    pestep = 0.01
    petest = 1.5
    bandgap1 = float('%.1f' % bandgap)
    schottky = 0
    tstep = 0.00001  # time step
    tend = 0.01  # total simulation time
    mfp = 3  # nm, mean free path
    emisposition = 0  # position of electron emission
    eloss = 0.027  # energy gain or loss of each e-phonon scattering
    ee = 1
    eh = 1
    finame = os.path.basename(__file__)[:-3]

    if op == 0:  # emittance vs qe with maxwell
        # partcle #, start energy, sigma, apsorbtion length, thichness
        para, p2D, pens, params_exp = MBdist(n, petest, thickness)
        # p2D: initial energy distribution, pens: initial depth distribution,
        # thickness: sample thickness, para and params_exp are fitting
        # parameters
        # diff(stept,endT,p2Di,mfp_ps,bounde,bounda,types):
        esurf, trap, back, p2De, tempres = diff(
            tstep, tend, p2D, mfp, eloss, emisposition, thickness, 3)
        # time step, time end, inital beam, mfp, emission surface, thickness, calculation type
        # esurf: reach to surface, trap: trap in body, back: get to back, p2De:
        # all the particulates information
        # beam reach to surface, electron affinity, schottky
        emiss, emittance, surtrap = emission(
            esurf, electron_affinity, schottky)
        # emiss: emit out beam, emittance:rms nor thermal emittance, surtrap: trap at surface
        # print emittance, float(len(emiss)/n)
        plot(p2D, pens, p2De, para, params_exp, thickness, tempres, finame)

        for electron_affinity in np.arange(0.1, 0.8, 0.1):
            wlscan = []
            start = bandgap1 + electron_affinity + 0.1
            for ep in np.arange(start, peend, pestep):
                emisst, emittancet, surtrapt = emission(diff(tstep, tend, MBdist(n, ep, thickness)[
                                                        1], mfp, eloss, emisposition, thickness, 3)[0], electron_affinity, schottky)
                qe = len(emisst) * 100 / float(n)
                print(ep, '    ', emittancet, '   ', qe, '%')
                wlscan.append([ep, emittancet, qe])
            wlscan = np.array(wlscan)
            emittancescanplot(wlscan, start, peend,
                              electron_affinity, thickness, mfp, expdata, finame)
        print(wlscan)

    elif op == 1:   # wavelength and ea scan with electrons go
        '''
        data_ene,p2D,pens,params_exp=DosDist(n,petest,thickness,dosdata,absorb_data)
        #p2D: initial energy distribution, pens: initial depth distribution, thickness: sample thickness, para and params_exp are fitting parameters
        esurf,trap,back,p2De,tempres=diff(tstep,tend,p2D,mfp,eloss,emisposition,thickness,3) # diff(stept,endT,p2Di,mfp_ps,bounde,bounda,types):
        # time step, time end, inital beam, mfp, emission surface, thickness, calculation type
        # esurf: reach to surface, trap: trap in body, back: get to back, p2De: all the particulates information
        emiss,emittance,surtrap=emission(esurf,electron_affinity,schottky)  # beam reach to surface, electron affinity, schottky
        #emiss: emit out beam, emittance:rms nor thermal emittance, surtrap: trap at surface
        #print emittance, float(len(emiss)/n)
        #print emiss
        #plot(p2D,pens,p2De,None,params_exp,thickness)
        #des=open('eaep_scan.dat','w')
        '''
        for electron_affinity in np.arange(0.1, 0.7, 0.1):
            wlscan = []
            start = bandgap1 + electron_affinity + 0.1
            for ep in np.arange(start, peend, pestep):
                ppd = DosDist(n, ep, thickness, dosdata, absorb_data)[1]
                es, tr, ba, p2dee, tempres = diff(
                    tstep, tend, ppd, mfp, eloss, emisposition, thickness, 3)
                emisst, emittancet, surtrapt = emission(
                    es, electron_affinity, schottky)

                qe = len(emisst) * 100 / float(n)
                labs = (float(n) - len(ppd)) * 100 / float(n)
                backd = len(ba) * 100 / float(n)
                trapin = len(tr) * 100 / float(n)
                trapsur = surtrapt * 100 / float(n)

                '''
                print ep,'  ',len(emisst)+(float(n)-len(ppd))+len(ba)+len(tr)+surtrapt
                
                
                qe=len(emisst)*100/float(len(ppd)) 
                labs=0         
                backd=len(ba)*100/float(len(ppd))
                trapin=len(tr)*100/float(len(ppd))
                trapsur=surtrapt*100/float(len(ppd))
                '''
                print(ep, '    ', emittancet, '   ', qe, '%')
                wlscan.append([ep, emittancet, qe, labs, backd,
                               trapin, trapsur, electron_affinity])

            wlscan = np.array(wlscan)
            stop = peend
            electronlossscanplot(wlscan, start, stop, pestep, mfp,
                                 electron_affinity, thickness, ee, eh, expdata, finame)
        # plt.show()
        # print wlscan

    elif op == 2:  # emittance vs qe with Dos, wavelength and ea scan
        '''
        data_ene,p2D,pens,params_exp=DosDist(n,petest,thickness,dosdata,absorb_data)
        #p2D: initial energy distribution, pens: initial depth distribution, thickness: sample thickness, para and params_exp are fitting parameters
        esurf,trap,back,p2De,tempres=diff(tstep,tend,p2D,mfp,eloss,emisposition,thickness,3) # diff(stept,endT,p2Di,mfp_ps,bounde,bounda,types):
        # time step, time end, inital beam, mfp, emission surface, thickness, calculation type
        # esurf: reach to surface, trap: trap in body, back: get to back, p2De: all the particulates information
        emiss,emittance,surtrap=emission(esurf,electron_affinity,schottky)  # beam reach to surface, electron affinity, schottky
        #emiss: emit out beam, emittance:rms nor thermal emittance, surtrap: trap at surface
        #print emittance, float(len(emiss)/n)
        #print emiss
        plot(p2D,pens,p2De,None,params_exp,thickness,tempres,petest,electron_affinity)
        '''

        for electron_affinity in np.arange(0.02, 0.1, 0.02):
            wlscan = []
            start = bandgap1 + electron_affinity + 0.1

            for ep in np.arange(start, peend, pestep):
                emisst, emittancet, surtrapt = emission(diff(tstep, tend, DosDist(n, ep, thickness, dosdata, absorb_data)[
                                                        1], mfp, eloss, emisposition, thickness, 3)[0], electron_affinity, schottky)
                qe = len(emisst) * 100 / float(n)
                print(ep, '    ', emittancet, '   ', qe, '%')
                wlscan.append([ep, emittancet, qe, electron_affinity])

            wlscan = np.array(wlscan)
            emittancescanplot(wlscan, start, peend,
                              electron_affinity, thickness, mfp, expdata, finame)

    elif op == 3:   # wavelength scan with single ea electrons go

        wlscan = []
        start = bandgap1 + electron_affinity + 0.1
        for ep in np.arange(start, peend, pestep):
            ppd = DosDist(n, ep, thickness, dosdata, absorb_data)[1]
            es, tr, ba, p2dee, tempres = diff(
                tstep, tend, ppd, mfp, eloss, emisposition, thickness, 3)
            emisst, emittancet, surtrapt = emission(
                es, electron_affinity, schottky)

            qe = len(emisst) * 100 / float(n)
            labs = (float(n) - len(ppd)) * 100 / float(n)
            backd = len(ba) * 100 / float(n)
            trapin = len(tr) * 100 / float(n)
            trapsur = surtrapt * 100 / float(n)

            '''
                print ep,'  ',len(emisst)+(float(n)-len(ppd))+len(ba)+len(tr)+surtrapt
                
                
                qe=len(emisst)*100/float(len(ppd)) 
                labs=0         
                backd=len(ba)*100/float(len(ppd))
                trapin=len(tr)*100/float(len(ppd))
                trapsur=surtrapt*100/float(len(ppd))
                '''
            print(ep, '    ', emittancet, '   ', qe, '%')
            wlscan.append([ep, emittancet, qe, labs, backd,
                           trapin, trapsur, electron_affinity])

        wlscan = np.array(wlscan)
        stop = peend
        electronlossscanplot(wlscan, start, stop, pestep, mfp,
                             electron_affinity, thickness, ee, eh, expdata, finame)
        # plt.show()
        # print wlscan

    elif op == 4:   # emittance vs qe with Dos, wavelength scan wt single ea

        wlscan = []
        start = bandgap1 + electron_affinity

        for ep in np.arange(start, peend, pestep):
            emisst, emittancet, surtrapt = emission(diff(tstep, tend, DosDist(n, ep, thickness, dosdata, absorb_data)[
                                                    1], mfp, eloss, emisposition, thickness, 3)[0], electron_affinity, schottky)
            qe = len(emisst) * 100 / float(n)
            print(ep, '    ', emittancet, '   ', qe, '%')
            wlscan.append([ep, emittancet, qe])

        wlscan = np.array(wlscan)
        emittancescanplot(wlscan, start, peend, electron_affinity,
                          thickness, mfp, expdata, finame)

    elif op == 5:  # for test

        data_ene, p2D, pens, params_exp = DosDist(
            n, petest, thickness, dosdata, absorb_data)
        # p2D: initial energy distribution, pens: initial depth distribution,
        # thickness: sample thickness, para and params_exp are fitting
        # parameters
        # diff(stept,endT,p2Di,mfp_ps,bounde,bounda,types):
        esurf, trap, back, p2De, tempres = diff(
            tstep, tend, p2D, mfp, eloss, emisposition, thickness, 3)
        # time step, time end, inital beam, mfp, emission surface, thickness, calculation type
        # esurf: reach to surface, trap: trap in body, back: get to back, p2De:
        # all the particulates information
        # beam reach to surface, electron affinity, schottky
        emiss, emittance, surtrap = emission(
            esurf, electron_affinity, schottky)
        # emiss: emit out beam, emittance:rms nor thermal emittance, surtrap: trap at surface
        # print emittance, float(len(emiss)/n)
        # print emiss

        print(petest, '    ', emittance, '   ',
              len(emiss) * 100 / float(n), '%')
        plot(p2D, pens, p2De, None, params_exp, thickness,
             tempres, petest, electron_affinity, finame)

    print("----%s sceonds----" % (time.time() - start_time))

if __name__ == '__main__':

    main(4)
