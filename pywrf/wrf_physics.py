"""PyWRF physics schemes.

Implements the WSM6 single-moment microphysics (cloud / rain / ice /
snow / graupel) and the moist-physics preparation / finalization
wrappers called by the solver. Constants come from
:mod:`pywrf.config_params`.
"""

import xarray as xr
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import sys
import os
import matplotlib.pyplot as plt
import time
import netCDF4
from pywrf.config_params import *

#torch.set_default_dtype(torch.float64)
# Device settings are managed centrally in pywrf/config_params.py - override
# the device via the PYWRF_GPU environment variable (default: GPU 0).
                      
# Prepare moisture for the microphysics step.
def moist_physics_prep_em(t_new, t_old, t0, rho, al, alb, \
                          p, p8w, p0, pb, ph, phb,        \
                          th_phy, pii, pf,                \
                          z, z_at_w, dz8w,                \
                          dt,h_diabatic,                  \
                          qv,qv_diabatic,                 \
                          qc,qc_diabatic,                 \
                          fzm, fzp,                       \
                          ids,ide, jds,jde, kds,kde,      \
                          ims,ime, jms,jme, kms,kme,      \
                          its,ite, jts,jte, kts,kte):
    i_start = its
    i_end   = min( ite,ide-1 )
    j_start = jts
    j_end   = min( jte,jde-1 )

    k_start = kts
    k_end = min( kte, kde-1 )
    
    fzm_e = fzm.unsqueeze(1).unsqueeze(2).repeat(1,nyfull,nxfull)
    fzp_e = fzp.unsqueeze(1).unsqueeze(2).repeat(1,nyfull,nxfull)
    
    z_at_w[k_start:kte, j_start:j_end, i_start:i_end] = (ph[k_start:kte, j_start:j_end, i_start:i_end] + 
            phb[k_start:kte, j_start:j_end, i_start:i_end]) / g
    dz8w[k_start:kte-1, j_start:j_end, i_start:i_end] = z_at_w[k_start+1:kte, j_start:j_end, i_start:i_end] - \
            z_at_w[k_start:kte-1, j_start:j_end, i_start:i_end]
    dz8w[kte-1, j_start:j_end, i_start:i_end] = 0.
    
    th_phy[k_start:k_end, j_start:j_end, i_start:i_end] = t_new[k_start:k_end, j_start:j_end, i_start:i_end] + t0
    h_diabatic[k_start:k_end, j_start:j_end, i_start:i_end] = th_phy[k_start:k_end, j_start:j_end, i_start:i_end] + 0.0
    qv_diabatic[k_start:k_end, j_start:j_end, i_start:i_end] = qv[k_start:k_end, j_start:j_end, i_start:i_end] + 0.0
    qc_diabatic[k_start:k_end, j_start:j_end, i_start:i_end] = qc[k_start:k_end, j_start:j_end, i_start:i_end] + 0.0
    rho[k_start:k_end, j_start:j_end, i_start:i_end] = 1. / (al[k_start:k_end, j_start:j_end, i_start:i_end] + 
            alb[k_start:k_end, j_start:j_end, i_start:i_end])
    pii[k_start:k_end, j_start:j_end, i_start:i_end] = ((p[k_start:k_end, j_start:j_end, i_start:i_end] + 
            pb[k_start:k_end, j_start:j_end, i_start:i_end]) / p0) ** rcp
    z[k_start:k_end, j_start:j_end, i_start:i_end] = 0.5 * (z_at_w[k_start:k_end, j_start:j_end, i_start:i_end] + 
            z_at_w[k_start+1:k_end+1, j_start:j_end, i_start:i_end])
    pf[k_start:k_end, j_start:j_end, i_start:i_end] = p[k_start:k_end, j_start:j_end, i_start:i_end] + \
            pb[k_start:k_end, j_start:j_end, i_start:i_end]
    
    p8w[1:k_end, j_start:j_end, i_start:i_end] = fzm_e[1:k_end, j_start:j_end, i_start:i_end] * \
            pf[1:k_end, j_start:j_end, i_start:i_end] + fzp_e[1:k_end, j_start:j_end, i_start:i_end] * \
            pf[0:k_end-1, j_start:j_end, i_start:i_end]
    
    w1 = torch.zeros((nyall,nxall)).to(device)
    w2 = torch.zeros((nyall,nxall)).to(device)
    
    #z0 = z_at_w[0, j_start:j_end, i_start:i_end]
    #z1 = z[0, j_start:j_end, i_start:i_end]
    #z2 = z[1, j_start:j_end, i_start:i_end]
    w1[j_start:j_end, i_start:i_end] = (z_at_w[0, j_start:j_end, i_start:i_end] - z[1, j_start:j_end, i_start:i_end]) / \
                                       (z[0, j_start:j_end, i_start:i_end] - z[1, j_start:j_end, i_start:i_end])
    w2[j_start:j_end, i_start:i_end] = 1. - w1[j_start:j_end, i_start:i_end]
    p8w[0, j_start:j_end, i_start:i_end] = w1[j_start:j_end, i_start:i_end] * pf[0, j_start:j_end, i_start:i_end] + \
                                           w2[j_start:j_end, i_start:i_end] * pf[1, j_start:j_end, i_start:i_end]
    
    w1[j_start:j_end, i_start:i_end] = (z_at_w[kte-1, j_start:j_end, i_start:i_end] - z[k_end-2, j_start:j_end, i_start:i_end]) / \
                                       (z[k_end-1, j_start:j_end, i_start:i_end] - z[k_end-2, j_start:j_end, i_start:i_end])
    w2[j_start:j_end, i_start:i_end] = 1. - w1[j_start:j_end, i_start:i_end]
    p8w[kde-1, j_start:j_end, i_start:i_end] = torch.exp(w1[j_start:j_end, i_start:i_end] * 
            torch.log(pf[kde-2, j_start:j_end, i_start:i_end]) + 
            w2[j_start:j_end, i_start:i_end] * torch.log(pf[kde-3, j_start:j_end, i_start:i_end]))

    return rho, th_phy, pii, pf, z, z_at_w, dz8w, p8w, h_diabatic, qv_diabatic, qc_diabatic, t_new, t_old

# Microphysics driver — dispatches to the WSM6 scheme.
def microphysics_driver(th, rho, pi_phy, p, ht,                           \
                        dz8w, p8w, dt,dx,dy,                              \
                        qv_curr,qc_curr,qr_curr,qi_curr,qs_curr,qg_curr   \
                        ,rainnc ,rainncv,snownc ,snowncv,sr               \
                        ,refl_10cm,diagflag,do_radar_ref                  \
                        ,graupelnc ,graupelncv                            \
                        # for radiation +
                        ,has_reqc                                 \
                        ,has_reqi                                 \
                        ,has_reqs                                 \
                        ,re_cloud                                 \
                        ,re_ice                                   \
                        ,re_snow                                  \
                        # for radiation -  
                        ,ids,ide, jds,jde, kds,kde \
                        ,ims,ime, jms,jme, kms,kme \
                        ,its,ite, jts,jte, kts,kte ):
    # wsm6 scheme
    th, qv_curr, qc_curr, qr_curr, qi_curr, qs_curr, qg_curr, rainnc,  \
        rainncv, snownc, snowncv, graupelnc, graupelncv, sr, refl_10cm \
        = wsm6(  th, qv_curr, qc_curr,qr_curr,qi_curr,qs_curr,qg_curr  \
                 ,rho,pi_phy,p,dz8w,dt,g,Cp,Cpv                        \
                 ,r_d,r_v,SVPT0,ep_1, ep_2, epsilon,XLS, XLV, XLF      \
                 ,rhoair0, rhowater,cliq,cice,psat                     \
                 ,rainnc ,rainncv                         \
                 ,snownc ,snowncv                         \
                 ,sr,refl_10cm                            \
                 ,diagflag                                \
                 ,do_radar_ref                            \
                 ,graupelnc, graupelncv                   \
                 ,has_reqc                                \
                 ,has_reqi                                \
                 ,has_reqs                                \
                 ,re_cloud                                \
                 ,re_ice                                  \
                 ,re_snow                                 \
                 ,ids,ide, jds,jde, kds,kde               \
                 ,ims,ime, jms,jme, kms,kme               \
                 ,its,ite, jts,jte, kts,kte)
    
    return th, qv_curr, qc_curr, qr_curr, qi_curr, qs_curr, qg_curr, rainnc, rainncv, snownc, snowncv, graupelnc, graupelncv, sr, refl_10cm

# WSM6 single-moment 6-class microphysics (cloud / rain / ice / snow / graupel).
def wsm6(th, q, qc, qr, qi, qs, qg                         \
         ,den, pii, p, delz                                \
         ,delt,g, cpd, cpv, rd, rv, t0c                    \
         ,ep1, ep2, qmin                                   \
         ,xls, xlv0, xlf0, den0, denr                      \
         ,cliq,cice,psat                                   \
         ,rain, rainncv                                    \
         ,snow, snowncv                                    \
         ,sr                                               \
         ,refl_10cm, diagflag, do_radar_ref                \
         ,graupel, graupelncv                              \
         ,has_reqc, has_reqi, has_reqs                     \
         ,re_cloud, re_ice,   re_snow                      \
         ,ids,ide, jds,jde, kds,kde                        \
         ,ims,ime, jms,jme, kms,kme                        \
         ,its,ite, jts,jte, kts,kte):
    
    qci = torch.zeros((2,nzall,nyall,nxall)).to(device)
    qrs = torch.zeros((3,nzall,nyall,nxall)).to(device)
    
    wsmtmp0 = torch.zeros((nzall,nyall,nxall)).to(device)
    wsmtmp1 = torch.zeros((nzall,nyall,nxall)).to(device)
    temp = torch.zeros((nzall,nyall,nxall)).to(device)
    qrs_tmp = torch.zeros((3,nzall,nyall,nxall)).to(device)
    work1 = torch.zeros((3,nzall,nyall,nxall)).to(device)
    workr = torch.zeros((nzall,nyall,nxall)).to(device)
    t = torch.zeros((nzall,nyall,nxall)).to(device)
    cpm = torch.zeros((nzall,nyall,nxall)).to(device)
    xl = torch.zeros((nzall,nyall,nxall)).to(device)
    delz_tmp = torch.zeros((nzall,nyall,nxall)).to(device)
    den_tmp = torch.zeros((nzall,nyall,nxall)).to(device)
    
    delqrs1 = torch.zeros((nyall,nxall)).to(device)
    delqrs2 = torch.zeros((nyall,nxall)).to(device)
    delqrs3 = torch.zeros((nyall,nxall)).to(device)
    n0sfac = torch.zeros((nzall,nyall,nxall)).to(device)
    denqci = torch.zeros((nzall,nyall,nxall)).to(device)
    delqi = torch.zeros((nyall,nxall)).to(device)
    
    kte = kte - 1
    ite = ite - 1
    jte = jte - 1
    
    t[kts:kte, jts:jte, its:ite] = th[kts:kte, jts:jte, its:ite] * pii[kts:kte, jts:jte, its:ite]
    qci[0, kts:kte, jts:jte, its:ite] = qc[kts:kte, jts:jte, its:ite] + 0.0
    qci[1, kts:kte, jts:jte, its:ite] = qi[kts:kte, jts:jte, its:ite] + 0.0
    qrs[0, kts:kte, jts:jte, its:ite] = qr[kts:kte, jts:jte, its:ite] + 0.0
    qrs[1, kts:kte, jts:jte, its:ite] = qs[kts:kte, jts:jte, its:ite] + 0.0
    qrs[2, kts:kte, jts:jte, its:ite] = qg[kts:kte, jts:jte, its:ite] + 0.0
    
    qci[qci<0.] = 0.
    qrs[qrs<0.] = 0.
    #print("in wsm6 qci4:",qci[1,2,160,12])
    # wsm6
    cpmcal = lambda x: torch.where(x >= qmin, cpv * x + (1 - x) * cpd, cpd * (1 - qmin) + cpv * qmin)
    xlcal = lambda x: xlv0-xlv1*(x-t0c)
    
    diffus = lambda x, y: 8.794e-5 * torch.exp(torch.log(x) * 1.81) / y
    viscos = lambda x, y: 1.496e-6 * (x * torch.sqrt(x)) / (x + 120.) / y
    xka = lambda x, y: 1.414e3*viscos(x,y)*y
    diffac = lambda a,b,c,d,e: d*a*a/(xka(c,d)*rv*c*c)+1./(e*diffus(c,b))
    venfac = lambda a,b,c: torch.exp(torch.log((viscos(b,c)/diffus(b,a)))*((0.3333333))) \
                     /torch.sqrt(viscos(b,c))*torch.sqrt(torch.sqrt(den0/c))
    conden = lambda a, b, c, d, e: (torch.maximum(b, torch.tensor(qmin)) - c) / (1. + d * d / (rv * e) * c / (a * a))
    
    idim = ite-its  # no need to +1
    kdim = kte-kts  # no need to +1
    
    qci[:, kts:kte, jts:jte, its:ite] = torch.maximum(qci[:, kts:kte, jts:jte, its:ite],torch.tensor(0.0))
    qrs[:, kts:kte, jts:jte, its:ite] = torch.maximum(qrs[:, kts:kte, jts:jte, its:ite],torch.tensor(0.0))
    
    cpm[kts:kte, jts:jte, its:ite] = cpmcal(q[kts:kte, jts:jte, its:ite])
    xl[kts:kte, jts:jte, its:ite] = xlcal(t[kts:kte, jts:jte, its:ite])
    
    delz_tmp[kts:kte, jts:jte, its:ite] = delz[kts:kte, jts:jte, its:ite] + 0.0
    den_tmp[kts:kte, jts:jte, its:ite] = den[kts:kte, jts:jte, its:ite] + 0.0
    
    #rainncv = torch.zeros((nyall,nxall)).to(device)
    #rain = torch.zeros((nyall,nxall)).to(device)
    tstepsnow = torch.zeros((nyall,nxall)).to(device)
    tstepgraup = torch.zeros((nyall,nxall)).to(device)
    
    rainncv[jts:jte, its:ite] = 0.0
    snowncv[jts:jte, its:ite] = 0.0
    graupelncv[jts:jte, its:ite] = 0.0
    sr[jts:jte, its:ite] = 0.0
    tstepsnow[jts:jte, its:ite] = 0.0
    tstepgraup[jts:jte, its:ite] = 0.0
       
    loops = max(delt / dtcldcr, 1)
    dtcld = 1.0 *delt / loops
    if delt <= dtcldcr:
        dtcld = delt + 0.0
    
    mstep = torch.zeros((nyall,nxall)).to(device)
    flgcld = torch.zeros((nyall,nxall)).to(device)
    denfac = torch.zeros((nzall,nyall,nxall)).to(device)
    tr = torch.zeros((nzall,nyall,nxall)).to(device)
    rh = torch.zeros((3,nzall,nyall,nxall)).to(device)
    qstmp = torch.zeros((3,nzall,nyall,nxall)).to(device)
    microtmp = torch.zeros((nzall,nyall,nxall)).to(device)
    #tvec1 = torch.zeros((nyall,nxall)).to(device)
    
    for loop in range(1,loops+1):
        mstep[jts:jte, its:ite] = 1
        flgcld[jts:jte, its:ite] = 1
        #tvec1[jts:jte, its:ite] = vrec()
        denfac[kts:kte, jts:jte, its:ite] = torch.sqrt(1.0 / den[kts:kte, jts:jte, its:ite] * den0)
        
        hsub = xls
        hvap = xlv0
        cvap = cpv
        ttp=t0c+0.01
        dldt=cvap-cliq
        xa=-dldt/rv
        xb=xa+hvap/(rv*ttp)
        dldti=cvap-cice
        xai=-dldti/rv
        xbi=xai+hsub/(rv*ttp)
        
        tr[kts:kte, jts:jte, its:ite] = ttp / t[kts:kte, jts:jte, its:ite]
        qstmp[0, kts:kte, jts:jte, its:ite] = psat * torch.exp(torch.log(tr[kts:kte, jts:jte, its:ite]) * 
                        xa) * torch.exp(xb * (1. - tr[kts:kte, jts:jte, its:ite]))
        qstmp[0, kts:kte, jts:jte, its:ite] = torch.minimum(qstmp[0,kts:kte, jts:jte, its:ite] , 
                                                         0.99 * p[kts:kte, jts:jte, its:ite])
        qstmp[0, kts:kte, jts:jte, its:ite] = ep2 * qstmp[0, kts:kte, jts:jte, its:ite] / \
              (p[kts:kte, jts:jte, its:ite] - qstmp[0, kts:kte, jts:jte, its:ite])
        qstmp[0, kts:kte, jts:jte, its:ite] = torch.maximum(qstmp[0, kts:kte, jts:jte, its:ite], torch.tensor(qmin))
        
        rh[0, kts:kte, jts:jte, its:ite] = torch.maximum(q[kts:kte, jts:jte, its:ite] / 
                                                         qstmp[0, kts:kte, jts:jte, its:ite], torch.tensor(qmin))
        
        wsmtmp0[kts:kte, jts:jte, its:ite] = psat * torch.exp(torch.log(tr[kts:kte, jts:jte, its:ite]) * 
                        xai) * torch.exp(xbi * (1. - tr[kts:kte, jts:jte, its:ite]))
        wsmtmp1[kts:kte, jts:jte, its:ite] = psat * torch.exp(torch.log(tr[kts:kte, jts:jte, its:ite]) * 
                        xa) * torch.exp(xb * (1. - tr[kts:kte, jts:jte, its:ite]))
        qstmp[1, kts:kte, jts:jte, its:ite] = torch.where(t[kts:kte, jts:jte, its:ite] < ttp, 
                        wsmtmp0[kts:kte, jts:jte, its:ite], wsmtmp1[kts:kte, jts:jte, its:ite])
        qstmp[1, kts:kte, jts:jte, its:ite] = torch.minimum(qstmp[1, kts:kte, jts:jte, its:ite], 
                                                         0.99 * p[kts:kte, jts:jte, its:ite])
        qstmp[1, kts:kte, jts:jte, its:ite] =  ep2 * qstmp[1, kts:kte, jts:jte, its:ite] / \
              (p[kts:kte, jts:jte, its:ite] - qstmp[1, kts:kte, jts:jte, its:ite])
        qstmp[1, kts:kte, jts:jte, its:ite] = torch.maximum(qstmp[1, kts:kte, jts:jte, its:ite], torch.tensor(qmin))
        rh[1, kts:kte, jts:jte, its:ite] = torch.maximum(q[kts:kte, jts:jte, its:ite] / 
                                                         qstmp[1, kts:kte, jts:jte, its:ite], torch.tensor(qmin))
        
        prevp = torch.zeros((nzall,nyall,nxall)).to(device)
        psdep = torch.zeros((nzall,nyall,nxall)).to(device)
        pgdep = torch.zeros((nzall,nyall,nxall)).to(device)
        praut = torch.zeros((nzall,nyall,nxall)).to(device)
        psaut = torch.zeros((nzall,nyall,nxall)).to(device)
        pgaut = torch.zeros((nzall,nyall,nxall)).to(device)
        pracw = torch.zeros((nzall,nyall,nxall)).to(device)
        praci = torch.zeros((nzall,nyall,nxall)).to(device)
        piacr = torch.zeros((nzall,nyall,nxall)).to(device)
        psaci = torch.zeros((nzall,nyall,nxall)).to(device)
        psacw = torch.zeros((nzall,nyall,nxall)).to(device)
        pracs = torch.zeros((nzall,nyall,nxall)).to(device)
        psacr = torch.zeros((nzall,nyall,nxall)).to(device)
        pgacw = torch.zeros((nzall,nyall,nxall)).to(device)
        paacw = torch.zeros((nzall,nyall,nxall)).to(device)
        pgaci = torch.zeros((nzall,nyall,nxall)).to(device)
        pgacr = torch.zeros((nzall,nyall,nxall)).to(device)
        pgacs = torch.zeros((nzall,nyall,nxall)).to(device)
        pigen = torch.zeros((nzall,nyall,nxall)).to(device)
        pidep = torch.zeros((nzall,nyall,nxall)).to(device)
        pcond = torch.zeros((nzall,nyall,nxall)).to(device)
        psmlt = torch.zeros((nzall,nyall,nxall)).to(device)
        pgmlt = torch.zeros((nzall,nyall,nxall)).to(device)
        pseml = torch.zeros((nzall,nyall,nxall)).to(device)
        pgeml = torch.zeros((nzall,nyall,nxall)).to(device)
        psevp = torch.zeros((nzall,nyall,nxall)).to(device)
        pgevp = torch.zeros((nzall,nyall,nxall)).to(device)
        fall = torch.zeros((3,nzall,nyall,nxall)).to(device)
        fallc = torch.zeros((nzall,nyall,nxall)).to(device)
        work2 = torch.zeros((nzall,nyall,nxall)).to(device)
        worktmp = torch.zeros((nzall,nyall,nxall)).to(device)
        coeres = torch.zeros((nzall,nyall,nxall)).to(device)
        supcol = torch.zeros((nzall,nyall,nxall)).to(device)
        supsat = torch.zeros((nzall,nyall,nxall)).to(device)
        satdt = torch.zeros((nzall,nyall,nxall)).to(device)
        ifsat = torch.zeros((nzall,nyall,nxall),dtype=torch.int).to(device)
        xni = 1.e3 * torch.ones((nzall,nyall,nxall)).to(device)
        xmi = torch.zeros((nzall,nyall,nxall)).to(device)
        work1c = torch.zeros((nzall,nyall,nxall)).to(device)
        fallsum = torch.zeros((nyall,nxall)).to(device)
        fallsum_qsi = torch.zeros((nyall,nxall)).to(device)
        fallsum_qg = torch.zeros((nyall,nxall)).to(device)
        rslope = torch.zeros((3,nzall,nyall,nxall)).to(device)
        rslopeb = torch.zeros((3,nzall,nyall,nxall)).to(device)
        rslope2 = torch.zeros((3,nzall,nyall,nxall)).to(device)
        rslope3 = torch.zeros((3,nzall,nyall,nxall)).to(device)
        qsum = torch.zeros((nzall,nyall,nxall)).to(device)
        worka = torch.zeros((nzall,nyall,nxall)).to(device)
        denqrs1 = torch.zeros((nzall,nyall,nxall)).to(device)
        denqrs2 = torch.zeros((nzall,nyall,nxall)).to(device)
        denqrs3 = torch.zeros((nzall,nyall,nxall)).to(device)
                
        # Ni: ice crystal number concentraiton
        temp[kts:kte, jts:jte, its:ite] = den[kts:kte, jts:jte, its:ite] * \
                                           torch.maximum(qci[1, kts:kte, jts:jte, its:ite], torch.tensor(qmin))
        temp[kts:kte, jts:jte, its:ite] = temp[kts:kte, jts:jte, its:ite] ** 0.75
        xni[kts:kte, jts:jte, its:ite] = torch.minimum(torch.maximum(
                     5.387e7 * temp[kts:kte, jts:jte, its:ite], torch.tensor(1.e3)), torch.tensor(1.e6))
        # compute the fallout term:
        # first, vertical terminal velosity for minor loops
        qrs_tmp[0:3, kts:kte, jts:jte, its:ite] = qrs[0:3, kts:kte, jts:jte, its:ite] + 0.0
        rslope,rslopeb,rslope2,rslope3,work1 = slope_wsm6(qrs_tmp,den_tmp,denfac,t,
                    rslope,rslopeb,rslope2,rslope3,work1,its,ite,kts,kte)
        
        workr[kts:kte, jts:jte, its:ite] = work1[0, kts:kte, jts:jte, its:ite]
        qsum[kts:kte, jts:jte, its:ite] = torch.maximum(qrs[1, kts:kte, jts:jte, its:ite] +
                    qrs[2,kts:kte, jts:jte, its:ite], torch.tensor(1.e-15))
        worka[kts:kte, jts:jte, its:ite] = torch.where(qsum[kts:kte, jts:jte, its:ite] > 1.e-15, 
                    (work1[1, kts:kte, jts:jte, its:ite] * qrs[1, kts:kte, jts:jte, its:ite] + 
                     work1[2, kts:kte, jts:jte, its:ite] * qrs[2, kts:kte, jts:jte, its:ite]) / 
                     qsum[kts:kte, jts:jte, its:ite], 0.)
        denqrs1[kts:kte, jts:jte, its:ite] = den[kts:kte, jts:jte, its:ite] * \
                    qrs[0, kts:kte, jts:jte, its:ite]
        denqrs2[kts:kte, jts:jte, its:ite] = den[kts:kte, jts:jte, its:ite] * \
                    qrs[1, kts:kte, jts:jte, its:ite]
        denqrs3[kts:kte, jts:jte, its:ite] = den[kts:kte, jts:jte, its:ite] * \
                    qrs[2, kts:kte, jts:jte, its:ite]
        workr[kts:kte, jts:jte, its:ite] = torch.where(qrs[0, kts:kte, jts:jte, its:ite] <= 0.0,
                    0.0, workr[kts:kte, jts:jte, its:ite])
        delqrs1, denqrs1 = nislfv_rain_plm(kdim,den_tmp,denfac,t,delz_tmp,workr,denqrs1,delqrs1,dtcld,1,1)
        
        delqrs2, delqrs3, denqrs2, denqrs3 = nislfv_rain_plm6(kdim,den_tmp,denfac,t,delz_tmp,worka,denqrs2,denqrs3,delqrs2,delqrs3,dtcld,1,1)
        
        qrs[0, kts:kte, jts:jte, its:ite] = torch.maximum(denqrs1[kts:kte, jts:jte, its:ite] / 
                    den[kts:kte, jts:jte, its:ite], torch.tensor(0))
        
        qrs[1, kts:kte, jts:jte, its:ite] = torch.maximum(denqrs2[kts:kte, jts:jte, its:ite] / 
                    den[kts:kte, jts:jte, its:ite], torch.tensor(0))
        
        qrs[2, kts:kte, jts:jte, its:ite] = torch.maximum(denqrs3[kts:kte, jts:jte, its:ite] / 
                    den[kts:kte, jts:jte, its:ite], torch.tensor(0))
        
        fall[0, kts:kte, jts:jte, its:ite] = denqrs1[kts:kte, jts:jte, its:ite] * workr[kts:kte, jts:jte, its:ite] / \
                    delz[kts:kte, jts:jte, its:ite]
        fall[1, kts:kte, jts:jte, its:ite] = denqrs2[kts:kte, jts:jte, its:ite] * worka[kts:kte, jts:jte, its:ite] / \
                    delz[kts:kte, jts:jte, its:ite]
        fall[2, kts:kte, jts:jte, its:ite] = denqrs3[kts:kte, jts:jte, its:ite] * worka[kts:kte, jts:jte, its:ite] / \
                    delz[kts:kte, jts:jte, its:ite]
        fall[0, 0, jts:jte, its:ite] = delqrs1[jts:jte, its:ite] / delz[0, jts:jte, its:ite] / dtcld
        fall[1, 0, jts:jte, its:ite] = delqrs2[jts:jte, its:ite] / delz[0, jts:jte, its:ite] / dtcld
        fall[2, 0, jts:jte, its:ite] = delqrs3[jts:jte, its:ite] / delz[0, jts:jte, its:ite] / dtcld
        qrs_tmp[0:3, kts:kte, jts:jte, its:ite] = qrs[0:3, kts:kte, jts:jte, its:ite] + 0.0
        
        rslope,rslopeb,rslope2,rslope3,work1 = slope_wsm6(qrs_tmp,den_tmp,denfac,t,
                    rslope,rslopeb,rslope2,rslope3,work1,its,ite,kts,kte)
        
        mstep_e = mstep.repeat(nzall,1,1)
        #for k in range(kte-1,kts-1,-1):
        xlf = xlf0 + 0.
        supcol[kts:kte, jts:jte, its:ite] = t0c - t[kts:kte, jts:jte, its:ite]
        n0sfac[kts:kte, jts:jte, its:ite] = torch.maximum(torch.minimum(torch.exp
                (alpha * supcol[kts:kte, jts:jte, its:ite]), torch.tensor(n0smax/n0s)), torch.tensor(1))
        # psmlt: melting of snow
        condition = t[kts:kte, jts:jte, its:ite] > t0c
        worktmp[kts:kte, jts:jte, its:ite] = venfac(p[kts:kte, jts:jte, its:ite], t[kts:kte, jts:jte, its:ite], 
                                            den[kts:kte, jts:jte, its:ite])
        work2[kts:kte, jts:jte, its:ite] = torch.where(condition, worktmp[kts:kte, jts:jte, its:ite], 
                                                 work2[kts:kte, jts:jte, its:ite])
        
        condition2 = qrs[1, kts:kte, jts:jte, its:ite] > 0.
        condition2 = condition2 & condition
        coeres[kts:kte, jts:jte, its:ite] = torch.where(condition2, rslope2[1, kts:kte, jts:jte, its:ite] * 
                (rslope[1, kts:kte, jts:jte, its:ite] * rslopeb[1, kts:kte, jts:jte, its:ite]) ** 0.5, 
                coeres[kts:kte, jts:jte, its:ite])
        psmlt[kts:kte, jts:jte, its:ite] = torch.where(condition2, xka(t[kts:kte, jts:jte, its:ite], 
                den[kts:kte, jts:jte, its:ite]) / xlf * (t0c - t[kts:kte, jts:jte, its:ite]) * pi / 2. * 
                n0sfac[kts:kte, jts:jte, its:ite] * (precs1 * rslope2[1, kts:kte, jts:jte, its:ite] + 
                precs2 * work2[kts:kte, jts:jte, its:ite] * coeres[kts:kte, jts:jte, its:ite]) / 
                den[kts:kte, jts:jte, its:ite], psmlt[kts:kte, jts:jte, its:ite])
        psmlt[kts:kte, jts:jte, its:ite] = torch.where(condition2, torch.minimum(torch.maximum(
                psmlt[kts:kte, jts:jte, its:ite] * dtcld / mstep_e[kts:kte, jts:jte, its:ite], 
                -qrs[1, kts:kte, jts:jte, its:ite] / mstep_e[kts:kte, jts:jte, its:ite]), torch.tensor(0)), 
                psmlt[kts:kte, jts:jte, its:ite])
        qrs[1, kts:kte, jts:jte, its:ite] = torch.where(condition2, qrs[1, kts:kte, jts:jte, its:ite] + 
                psmlt[kts:kte, jts:jte, its:ite], qrs[1, kts:kte, jts:jte, its:ite])
        
        qrs[0, kts:kte, jts:jte, its:ite] = torch.where(condition2, qrs[0, kts:kte, jts:jte, its:ite] - 
                psmlt[kts:kte, jts:jte, its:ite], qrs[0, kts:kte, jts:jte, its:ite])
        
        t[kts:kte, jts:jte, its:ite] = torch.where(condition2, t[kts:kte, jts:jte, its:ite] + 
                xlf / cpm[kts:kte, jts:jte, its:ite] * psmlt[kts:kte, jts:jte, its:ite], t[kts:kte, jts:jte, its:ite])
        
        # pgmlt: melting of graupel
        condition2 = qrs[2, kts:kte, jts:jte, its:ite] > 0.
        condition2 = condition2 & condition
        coeres[:,:,:] = 0.
        coeres[kts:kte, jts:jte, its:ite] = torch.where(condition2, rslope2[2, kts:kte, jts:jte, its:ite] * 
                (rslope[2, kts:kte, jts:jte, its:ite] * rslopeb[2, kts:kte, jts:jte, its:ite]) ** 0.5, 
                coeres[kts:kte, jts:jte, its:ite])
        pgmlt[kts:kte, jts:jte, its:ite] = torch.where(condition2, xka(t[kts:kte, jts:jte, its:ite], 
                den[kts:kte, jts:jte, its:ite]) / xlf * (t0c - t[kts:kte, jts:jte, its:ite]) * 
                (precg1 * rslope2[2, kts:kte, jts:jte, its:ite] + 
                 precg2 * work2[kts:kte, jts:jte, its:ite] * coeres[kts:kte, jts:jte, its:ite]) / 
                den[kts:kte, jts:jte, its:ite], pgmlt[kts:kte, jts:jte, its:ite])
        pgmlt[kts:kte, jts:jte, its:ite] = torch.where(condition2, torch.minimum(torch.maximum(
                pgmlt[kts:kte, jts:jte, its:ite] * dtcld / mstep_e[kts:kte, jts:jte, its:ite], 
                -qrs[2, kts:kte, jts:jte, its:ite] / mstep_e[kts:kte, jts:jte, its:ite]), torch.tensor(0)), 
                pgmlt[kts:kte, jts:jte, its:ite])
        
        qrs[2, kts:kte, jts:jte, its:ite] = torch.where(condition2, qrs[2, kts:kte, jts:jte, its:ite] + 
                pgmlt[kts:kte, jts:jte, its:ite], qrs[2, kts:kte, jts:jte, its:ite])
        
        qrs[0, kts:kte, jts:jte, its:ite] = torch.where(condition2, qrs[0, kts:kte, jts:jte, its:ite] - 
                pgmlt[kts:kte, jts:jte, its:ite], qrs[0, kts:kte, jts:jte, its:ite])
        
        t[kts:kte, jts:jte, its:ite] = torch.where(condition2, t[kts:kte, jts:jte, its:ite] + 
                xlf / cpm[kts:kte, jts:jte, its:ite] * pgmlt[kts:kte, jts:jte, its:ite], t[kts:kte, jts:jte, its:ite])
        
        # Vice: fallout of ice crystal
        condition = qci[1, kts:kte, jts:jte, its:ite] <= 0.
        worktmp[kts:kte, jts:jte, its:ite] = torch.maximum(torch.minimum(dicon * (den[kts:kte, jts:jte, its:ite] * 
                qci[1, kts:kte, jts:jte, its:ite] / xni[kts:kte, jts:jte, its:ite]) ** 0.5, torch.tensor(dimax)), torch.tensor(1.e-25))
        worktmp[kts:kte, jts:jte, its:ite] = 1.49e4*torch.exp(torch.log(worktmp[kts:kte, jts:jte, its:ite]) * 1.31)
        work1c[kts:kte, jts:jte, its:ite] = torch.where(condition, torch.tensor(0), worktmp[kts:kte, jts:jte, its:ite])
        denqci[kts:kte, jts:jte, its:ite] = den[kts:kte, jts:jte, its:ite] * qci[1, kts:kte, jts:jte, its:ite]
        
        delqi, denqci = nislfv_rain_plm(kdim,den_tmp,denfac,t,delz_tmp,work1c,denqci,delqi,dtcld,1,0)
        #print("in wsm6 qci4",denfac[2,160,12],t[2,160,12],delz_tmp[2,160,12],qci[1,:,160,12],delqi[160,12],xni[:,160,12],den[:,160,12])
        qci[1, kts:kte, jts:jte, its:ite] = torch.maximum(denqci[kts:kte, jts:jte, its:ite] / 
                                                          den[kts:kte, jts:jte, its:ite], torch.tensor(0))
        #print("in wsm6 qci3",qci[1,2,160,12],denqci[2,160,12],den[2,160,12])
        fallc[0, jts:jte, its:ite] = delqi[jts:jte, its:ite] / delz[0, jts:jte, its:ite] / dtcld
        # rain 
        fallsum[jts:jte, its:ite] = fall[0, kts, jts:jte, its:ite] + fall[1, kts, jts:jte, its:ite] + \
                                    fall[2, kts, jts:jte, its:ite] + fallc[kts, jts:jte, its:ite]
        fallsum_qsi[jts:jte, its:ite] = fall[1, kts, jts:jte, its:ite] + fallc[kts, jts:jte, its:ite]
        fallsum_qg[jts:jte, its:ite] = fall[2, kts, jts:jte, its:ite]
        
        xlf3 = torch.zeros((nzall,nyall,nxall)).to(device)
        pfrzdtc = torch.zeros((nzall,nyall,nxall)).to(device)
        pfrzdtr = torch.zeros((nzall,nyall,nxall)).to(device)
        eacrs = torch.zeros((nzall,nyall,nxall)).to(device)
        diameter = torch.zeros((nzall,nyall,nxall)).to(device)
        vt2i = torch.zeros((nzall,nyall,nxall)).to(device)
        vt2r = torch.zeros((nzall,nyall,nxall)).to(device)
        vt2s = torch.zeros((nzall,nyall,nxall)).to(device)
        vt2g = torch.zeros((nzall,nyall,nxall)).to(device)
        vt2ave = torch.zeros((nzall,nyall,nxall)).to(device)
        acrfac = torch.zeros((nzall,nyall,nxall)).to(device)
        egi = torch.zeros((nzall,nyall,nxall)).to(device)
        supice = torch.zeros((nzall,nyall,nxall)).to(device)
        roqi0 = torch.zeros((nzall,nyall,nxall)).to(device)
        delta2 = torch.zeros((nzall,nyall,nxall)).to(device)
        delta3 = torch.zeros((nzall,nyall,nxall)).to(device)
        value = torch.zeros((nzall,nyall,nxall)).to(device)
        source = torch.zeros((nzall,nyall,nxall)).to(device)
        factor = torch.zeros((nzall,nyall,nxall)).to(device)
        xlwork2 = torch.zeros((nzall,nyall,nxall)).to(device)
        tr = torch.zeros((nzall,nyall,nxall)).to(device)
        
        condition = fallsum[jts:jte, its:ite] > 0.
        rainncv[jts:jte, its:ite] = torch.where(condition, fallsum[jts:jte, its:ite] * delz[kts, jts:jte, its:ite] / \
                                    denr * dtcld * 1000. + rainncv[jts:jte, its:ite], rainncv[jts:jte, its:ite])
        rain[jts:jte, its:ite] = torch.where(condition, fallsum[jts:jte, its:ite] * delz[kts, jts:jte, its:ite] / \
                                 denr * dtcld * 1000. + rain[jts:jte, its:ite], rain[jts:jte, its:ite])
        
        condition = fallsum_qsi[jts:jte, its:ite] > 0.
        tstepsnow[jts:jte, its:ite] = torch.where(condition, fallsum_qsi[jts:jte, its:ite] * delz[kts, jts:jte, its:ite] / \
                                      denr * dtcld *1000. + tstepsnow[jts:jte, its:ite], tstepsnow[jts:jte, its:ite])
        snowncv[jts:jte, its:ite] = torch.where(condition, fallsum_qsi[jts:jte, its:ite] * delz[kts, jts:jte, its:ite] / \
                                    denr * dtcld *1000. + snowncv[jts:jte, its:ite], snowncv[jts:jte, its:ite])
        snow[jts:jte, its:ite] = torch.where(condition, fallsum_qsi[jts:jte, its:ite] * delz[kts, jts:jte, its:ite] / \
                                      denr * dtcld *1000. + snow[jts:jte, its:ite], snow[jts:jte, its:ite])
        
        condition = fallsum_qg[jts:jte, its:ite] > 0.
        tstepgraup[jts:jte, its:ite] = torch.where(condition, fallsum_qg[jts:jte, its:ite] * delz[kts, jts:jte, its:ite] / \
                                      denr * dtcld *1000. + tstepgraup[jts:jte, its:ite], tstepgraup[jts:jte, its:ite])
        graupelncv[jts:jte, its:ite] = torch.where(condition, fallsum_qg[jts:jte, its:ite] * delz[kts, jts:jte, its:ite] / \
                                    denr * dtcld *1000. + graupelncv[jts:jte, its:ite], graupelncv[jts:jte, its:ite])
        graupel[jts:jte, its:ite] = torch.where(condition, fallsum_qg[jts:jte, its:ite] * delz[kts, jts:jte, its:ite] / \
                                      denr * dtcld *1000. + graupel[jts:jte, its:ite], graupel[jts:jte, its:ite])
        
        condition = fallsum[jts:jte, its:ite] > 0.
        sr[jts:jte, its:ite] = torch.where(condition, (snowncv[jts:jte, its:ite] + graupelncv[jts:jte, its:ite]) / 
                                           (rainncv[jts:jte, its:ite] + 1.e-12), sr[jts:jte, its:ite])
        # pimlt
        supcol[kts:kte, jts:jte, its:ite] = t0c - t[kts:kte, jts:jte, its:ite]
        xlf3[kts:kte, jts:jte, its:ite] = xls - xl[kts:kte, jts:jte, its:ite]
        condition = supcol[kts:kte, jts:jte, its:ite] < 0.
        condition1 = qci[1, kts:kte, jts:jte, its:ite] > 0.
        condition = condition & condition1
        qci[0, kts:kte, jts:jte, its:ite] = torch.where(condition, qci[0, kts:kte, jts:jte, its:ite] + 
                            qci[1, kts:kte, jts:jte, its:ite], qci[0, kts:kte, jts:jte, its:ite])
        
        t[kts:kte, jts:jte, its:ite] = torch.where(condition, t[kts:kte, jts:jte, its:ite] - xlf3[kts:kte, jts:jte, its:ite] /
                            cpm[kts:kte, jts:jte, its:ite] * qci[1, kts:kte, jts:jte, its:ite], t[kts:kte, jts:jte, its:ite])
        
        qci[1, kts:kte, jts:jte, its:ite] = torch.where(condition, torch.tensor(0), qci[1, kts:kte, jts:jte, its:ite])
        # pihmf
        condition = supcol[kts:kte, jts:jte, its:ite] > 40.
        condition1 = qci[0, kts:kte, jts:jte, its:ite] > 0.
        condition = condition & condition1
        #print("in wsm6 qci2:",qci[1,2,160,12],qci[0,2,160,12])
        qci[1, kts:kte, jts:jte, its:ite] = torch.where(condition, qci[1, kts:kte, jts:jte, its:ite] + 
                            qci[0,kts:kte, jts:jte, its:ite], qci[1, kts:kte, jts:jte, its:ite])
        
        t[kts:kte, jts:jte, its:ite] = torch.where(condition, t[kts:kte, jts:jte, its:ite] + xlf3[kts:kte, jts:jte, its:ite] /
                            cpm[kts:kte, jts:jte, its:ite] * qci[0, kts:kte, jts:jte, its:ite], t[kts:kte, jts:jte, its:ite])
        
        qci[0, kts:kte, jts:jte, its:ite] = torch.where(condition, torch.tensor(0), qci[0, kts:kte, jts:jte, its:ite])
        
        # pihtf
        condition = supcol[kts:kte, jts:jte, its:ite] > 0.
        condition1 = qci[0, kts:kte, jts:jte, its:ite] > qmin
        condition = condition & condition1
        pfrzdtc[kts:kte, jts:jte, its:ite] = torch.minimum(pfrz1 * (torch.exp(pfrz2 * 
                            torch.minimum(supcol[kts:kte, jts:jte, its:ite], torch.tensor(50.))) - 1.) * 
                            den[kts:kte, jts:jte, its:ite] / denr / xncr * qci[0, kts:kte, jts:jte, its:ite] * 
                            qci[0, kts:kte, jts:jte, its:ite] * dtcld, qci[0, kts:kte, jts:jte, its:ite])
        #print("in wsm6 qci1:",qci[1,2,160,12],pfrzdtc[2,160,12])
        qci[1, kts:kte, jts:jte, its:ite] = torch.where(condition, qci[1, kts:kte, jts:jte, its:ite] + 
                            pfrzdtc[kts:kte, jts:jte, its:ite], qci[1, kts:kte, jts:jte, its:ite])
        t[kts:kte, jts:jte, its:ite] = torch.where(condition, t[kts:kte, jts:jte, its:ite] + xlf3[kts:kte, jts:jte, its:ite] / 
                            cpm[kts:kte, jts:jte, its:ite] * pfrzdtc[kts:kte, jts:jte, its:ite], t[kts:kte, jts:jte, its:ite])
        
        qci[0, kts:kte, jts:jte, its:ite] = torch.where(condition, qci[0, kts:kte, jts:jte, its:ite] - 
                            pfrzdtc[kts:kte, jts:jte, its:ite], qci[0, kts:kte, jts:jte, its:ite])
        
        # pgfrz
        condition = supcol[kts:kte, jts:jte, its:ite] > 0.
        condition1 = qrs[0, kts:kte, jts:jte, its:ite] > 0.
        condition = condition & condition1
        pfrzdtr[kts:kte, jts:jte, its:ite] = torch.minimum(20. * pi * pi * pfrz1 * n0r * denr / den[kts:kte, jts:jte, its:ite] *
                            (torch.exp(pfrz2 * torch.minimum(supcol[kts:kte, jts:jte, its:ite], torch.tensor(50.))) - 1.) * 
                            rslope3[0, kts:kte, jts:jte, its:ite] * rslope3[0, kts:kte, jts:jte, its:ite] * 
                            rslope[0, kts:kte, jts:jte, its:ite] * dtcld, qrs[0, kts:kte, jts:jte, its:ite])
        qrs[2, kts:kte, jts:jte, its:ite] = torch.where(condition, qrs[2, kts:kte, jts:jte, its:ite] + 
                            pfrzdtr[kts:kte, jts:jte, its:ite], qrs[2, kts:kte, jts:jte, its:ite])
        
        t[kts:kte, jts:jte, its:ite] = torch.where(condition, t[kts:kte, jts:jte, its:ite] + xlf3[kts:kte, jts:jte, its:ite] /
                            cpm[kts:kte, jts:jte, its:ite] * pfrzdtr[kts:kte, jts:jte, its:ite], t[kts:kte, jts:jte, its:ite])
        
        qrs[0, kts:kte, jts:jte, its:ite] = torch.where(condition, qrs[0, kts:kte, jts:jte, its:ite] - 
                            pfrzdtr[kts:kte, jts:jte, its:ite], qrs[0, kts:kte, jts:jte, its:ite])
        
        # update the slope parameters
        qrs_tmp[0:3, kts:kte, jts:jte, its:ite] = qrs[0:3, kts:kte, jts:jte, its:ite] + 0.0
        
        rslope,rslopeb,rslope2,rslope3,work1 = slope_wsm6(qrs_tmp,den_tmp,denfac,t,
                    rslope,rslopeb,rslope2,rslope3,work1,its,ite,kts,kte)
        
        work1[0, kts:kte, jts:jte, its:ite] = diffac(xl[kts:kte, jts:jte, its:ite], p[kts:kte, jts:jte, its:ite], 
                                                     t[kts:kte, jts:jte, its:ite], den[kts:kte, jts:jte, its:ite], 
                                                     qstmp[0, kts:kte, jts:jte, its:ite])
        work1[1, kts:kte, jts:jte, its:ite] = diffac(xls, p[kts:kte, jts:jte, its:ite], 
                                                     t[kts:kte, jts:jte, its:ite], den[kts:kte, jts:jte, its:ite], 
                                                     qstmp[1, kts:kte, jts:jte, its:ite])
        work2[kts:kte, jts:jte, its:ite] = venfac(p[kts:kte, jts:jte, its:ite], t[kts:kte, jts:jte, its:ite],
                                                  den[kts:kte, jts:jte, its:ite])
        
        # warm rain processes
        supsat[kts:kte, jts:jte, its:ite] = torch.maximum(q[kts:kte, jts:jte, its:ite], torch.tensor(qmin)) - qstmp[0, kts:kte, jts:jte, its:ite]
        
        # praut: auto conversion from cloud to rain
        condition = qci[0, kts:kte, jts:jte, its:ite] > qc0
        praut[kts:kte, jts:jte, its:ite] = qck1 * qci[0, kts:kte, jts:jte, its:ite] ** (7./3.)
        
        praut[kts:kte, jts:jte, its:ite] = torch.where(condition, torch.minimum(praut[kts:kte, jts:jte, its:ite], 
                    qci[0, kts:kte, jts:jte, its:ite] / dtcld), torch.tensor(0))
        
        # pracw: accretion of cloud water by rain
        condition = qrs[0, kts:kte, jts:jte, its:ite] > qcrmin
        condition1 = qci[0, kts:kte, jts:jte, its:ite] > qmin
        condition = condition & condition1
        pracw[kts:kte, jts:jte, its:ite] = torch.where(condition, torch.minimum(pacrr * rslope3[0, kts:kte, jts:jte, its:ite] * 
                    rslopeb[0, kts:kte, jts:jte, its:ite] * qci[0, kts:kte, jts:jte, its:ite] * 
                    denfac[kts:kte, jts:jte, its:ite], qci[0, kts:kte, jts:jte, its:ite] / dtcld), pracw[kts:kte, jts:jte, its:ite])
        # prevp: evaporation/condensation rate of rain
        condition = qrs[0, kts:kte, jts:jte, its:ite] > 0.
        coeres[kts:kte, jts:jte, its:ite] = rslope2[0, kts:kte, jts:jte, its:ite] * (
                    rslope[0, kts:kte, jts:jte, its:ite] * rslopeb[0, kts:kte, jts:jte, its:ite]) ** 0.5
        prevp[kts:kte, jts:jte, its:ite] = torch.where(condition, (rh[0, kts:kte, jts:jte, its:ite] - 1.) * (precr1 * rslope2[0, kts:kte, jts:jte, its:ite] +
                    precr2 * work2[kts:kte, jts:jte, its:ite] * coeres[kts:kte, jts:jte, its:ite]) / work1[0, kts:kte, jts:jte, its:ite], prevp[kts:kte, jts:jte, its:ite])
        condition2 = prevp[kts:kte, jts:jte, its:ite] < 0.
        condition2 = condition2 & condition
        condition3 = prevp[kts:kte, jts:jte, its:ite] >= 0.
        condition3 = condition3 & condition
        
        prevp[kts:kte, jts:jte, its:ite] = torch.where(condition2, torch.maximum(prevp[kts:kte, jts:jte, its:ite], 
                    -qrs[0, kts:kte, jts:jte, its:ite] / dtcld), prevp[kts:kte, jts:jte, its:ite])
        
        prevp[kts:kte, jts:jte, its:ite] = torch.where(condition2, torch.maximum(prevp[kts:kte, jts:jte, its:ite], 
                    supsat[kts:kte, jts:jte, its:ite] / dtcld /2.), prevp[kts:kte, jts:jte, its:ite])
        
        prevp[kts:kte, jts:jte, its:ite] = torch.where(condition3, torch.minimum(prevp[kts:kte, jts:jte, its:ite], 
                    supsat[kts:kte, jts:jte, its:ite] / dtcld /2.), prevp[kts:kte, jts:jte, its:ite])
        
        # cold rain processes
        supcol[kts:kte, jts:jte, its:ite] = t0c - t[kts:kte, jts:jte, its:ite]
        n0sfac[kts:kte, jts:jte, its:ite] = torch.maximum(torch.minimum(torch.exp(alpha * supcol[kts:kte, jts:jte, its:ite]), torch.tensor(n0smax/n0s)), torch.tensor(1.))
        
        supsat[kts:kte, jts:jte, its:ite] = torch.maximum(q[kts:kte, jts:jte, its:ite], torch.tensor(qmin)) - qstmp[1, kts:kte, jts:jte, its:ite]
        
        satdt[kts:kte, jts:jte, its:ite] = supsat[kts:kte, jts:jte, its:ite] / dtcld
        ifsat[kts:kte, jts:jte, its:ite] = 0
        
        # Ni ice crystal number concentraiton
        worktmp[kts:kte, jts:jte, its:ite] = den[kts:kte, jts:jte, its:ite] * torch.maximum(qci[1, kts:kte, jts:jte, its:ite], torch.tensor(qmin))
        worktmp[kts:kte, jts:jte, its:ite] = worktmp[kts:kte, jts:jte, its:ite] ** 0.75
        xni[kts:kte, jts:jte, its:ite] = torch.minimum(torch.maximum(5.38e7 * worktmp[kts:kte, jts:jte, its:ite], torch.tensor(1.e3)), torch.tensor(1.e6))
        eacrs[kts:kte, jts:jte, its:ite] = torch.exp(0.07 * (-supcol[kts:kte, jts:jte, its:ite]))
        
        xmi[kts:kte, jts:jte, its:ite] = den[kts:kte, jts:jte, its:ite] * qci[1, kts:kte, jts:jte, its:ite] / \
                                         xni[kts:kte, jts:jte, its:ite]
        diameter[kts:kte, jts:jte, its:ite] = torch.minimum(dicon * xmi[kts:kte, jts:jte, its:ite] ** 0.5, torch.tensor(dimax))
        vt2i[kts:kte, jts:jte, its:ite] = 1.49e4 * diameter[kts:kte, jts:jte, its:ite] ** 1.31
        vt2r[kts:kte, jts:jte, its:ite] = pvtr * rslopeb[0, kts:kte, jts:jte, its:ite] * denfac[kts:kte, jts:jte, its:ite]
        vt2s[kts:kte, jts:jte, its:ite] = pvts * rslopeb[1, kts:kte, jts:jte, its:ite] * denfac[kts:kte, jts:jte, its:ite]
        vt2g[kts:kte, jts:jte, its:ite] = pvtg * rslopeb[2, kts:kte, jts:jte, its:ite] * denfac[kts:kte, jts:jte, its:ite]
        qsum[kts:kte, jts:jte, its:ite] = torch.maximum((qrs[1, kts:kte, jts:jte, its:ite] + qrs[2, kts:kte, jts:jte, its:ite]), torch.tensor(1.e-15))
        
        condition = qsum[kts:kte, jts:jte, its:ite] > 1.e-15
        vt2ave[kts:kte, jts:jte, its:ite] = (vt2s[kts:kte, jts:jte, its:ite] * qrs[1, kts:kte, jts:jte, its:ite] + 
                vt2g[kts:kte, jts:jte, its:ite] * qrs[2, kts:kte, jts:jte, its:ite]) / qsum[kts:kte, jts:jte, its:ite]
        vt2ave[kts:kte, jts:jte, its:ite] = torch.where(condition, vt2ave[kts:kte, jts:jte, its:ite], torch.tensor(0))
        
        condition = supcol[kts:kte, jts:jte, its:ite] > 0.
        condition2 = qci[1, kts:kte, jts:jte, its:ite] > qmin
        condition = condition & condition2
        condition1 = qrs[0, kts:kte, jts:jte, its:ite] > qcrmin
        condition1 = condition & condition1
        # praci: Accretion of cloud ice by rain
        acrfac[kts:kte, jts:jte, its:ite] = 2. * rslope3[0, kts:kte, jts:jte, its:ite] + 2. * diameter[kts:kte, jts:jte, its:ite] * \
                rslope2[0, kts:kte, jts:jte, its:ite] + diameter[kts:kte, jts:jte, its:ite] ** 2 * rslope[0, kts:kte, jts:jte, its:ite]
        praci[kts:kte, jts:jte, its:ite] = pi * qci[1, kts:kte, jts:jte, its:ite] * n0r * torch.abs(vt2r[kts:kte, jts:jte, its:ite] - 
                                           vt2i[kts:kte, jts:jte, its:ite]) * acrfac[kts:kte, jts:jte, its:ite]
        praci[kts:kte, jts:jte, its:ite] = torch.minimum(praci[kts:kte, jts:jte, its:ite], qci[1, kts:kte, jts:jte, its:ite] / dtcld)
        praci[kts:kte, jts:jte, its:ite] = torch.where(condition1, praci[kts:kte, jts:jte, its:ite], torch.tensor(0))
        # piacr: Accretion of rain by cloud ice
        piacr[kts:kte, jts:jte, its:ite] = pi ** 2 * avtr * n0r * denr * xni[kts:kte, jts:jte, its:ite] * \
                denfac[kts:kte, jts:jte, its:ite] * g6pbr * rslope3[0, kts:kte, jts:jte, its:ite] ** 2 * \
                rslopeb[0, kts:kte, jts:jte, its:ite] / 24. / den[kts:kte, jts:jte, its:ite]
        piacr[kts:kte, jts:jte, its:ite] = torch.minimum(piacr[kts:kte, jts:jte, its:ite], qrs[0, kts:kte, jts:jte, its:ite] / dtcld)
        piacr[kts:kte, jts:jte, its:ite] = torch.where(condition1, piacr[kts:kte, jts:jte, its:ite], torch.tensor(0))
        # psaci: Accretion of cloud ice by snow
        condition1 = qrs[1, kts:kte, jts:jte, its:ite] > qcrmin
        condition1 = condition1 & condition
        acrfac[kts:kte, jts:jte, its:ite] = 2. * rslope3[1, kts:kte, jts:jte, its:ite] + 2. * diameter[kts:kte, jts:jte, its:ite] * \
                rslope2[1, kts:kte, jts:jte, its:ite] + diameter[kts:kte, jts:jte, its:ite] ** 2 * rslope[1, kts:kte, jts:jte, its:ite]
        psaci[kts:kte, jts:jte, its:ite] = pi * qci[1, kts:kte, jts:jte, its:ite] * eacrs[kts:kte, jts:jte, its:ite] * n0s * \
                n0sfac[kts:kte, jts:jte, its:ite] * torch.abs(vt2ave[kts:kte, jts:jte, its:ite] - vt2i[kts:kte, jts:jte, its:ite]) * \
                acrfac[kts:kte, jts:jte, its:ite] / 4.
        psaci[kts:kte, jts:jte, its:ite] = torch.minimum(psaci[kts:kte, jts:jte, its:ite], qci[1, kts:kte, jts:jte, its:ite] / dtcld)
        psaci[kts:kte, jts:jte, its:ite] = torch.where(condition1, psaci[kts:kte, jts:jte, its:ite], torch.tensor(0))
        # pgaci: Accretion of cloud ice by graupel
        condition1 = qrs[2, kts:kte, jts:jte, its:ite] > qcrmin
        condition1 = condition1 & condition
        egi[kts:kte, jts:jte, its:ite] = torch.exp( -0.07 * supcol[kts:kte, jts:jte, its:ite])
        acrfac[kts:kte, jts:jte, its:ite] = 2. * rslope3[2, kts:kte, jts:jte, its:ite] + 2. * diameter[kts:kte, jts:jte, its:ite] * rslope2[2, kts:kte, jts:jte, its:ite] + \
                diameter[kts:kte, jts:jte, its:ite] **2 * rslope[2, kts:kte, jts:jte, its:ite]
        pgaci[kts:kte, jts:jte, its:ite] = pi * egi[kts:kte, jts:jte, its:ite] * qci[1, kts:kte, jts:jte, its:ite] * n0g * \
                torch.abs(vt2ave[kts:kte, jts:jte, its:ite] - vt2i[kts:kte, jts:jte, its:ite]) * acrfac[kts:kte, jts:jte, its:ite] / 4.
        pgaci[kts:kte, jts:jte, its:ite] = torch.min(pgaci[kts:kte, jts:jte, its:ite], qci[1, kts:kte, jts:jte, its:ite] / dtcld)
        pgaci[kts:kte, jts:jte, its:ite] = torch.where(condition1, pgaci[kts:kte, jts:jte, its:ite], torch.tensor(0))
        
        # psacw: Accretion of cloud water by snow
        condition = qrs[1, kts:kte, jts:jte, its:ite] > qcrmin
        condition1 = qci[0, kts:kte, jts:jte, its:ite] > qmin
        condition = condition & condition1
        psacw[kts:kte, jts:jte, its:ite] = torch.minimum(pacrc * n0sfac[kts:kte, jts:jte, its:ite] * rslope3[1, kts:kte, jts:jte, its:ite] * 
                              rslopeb[1, kts:kte, jts:jte, its:ite] * qci[0, kts:kte, jts:jte, its:ite] * 
                              denfac[kts:kte, jts:jte, its:ite], qci[0, kts:kte, jts:jte, its:ite] / dtcld)
        psacw[kts:kte, jts:jte, its:ite] = torch.where(condition, psacw[kts:kte, jts:jte, its:ite], torch.tensor(0))
        # pgacw: Accretion of cloud water by graupel
        condition = qrs[2, kts:kte, jts:jte, its:ite] > qcrmin 
        condition1 = qci[0, kts:kte, jts:jte, its:ite] > qmin
        condition = condition & condition1
        pgacw[kts:kte, jts:jte, its:ite] = torch.minimum(pacrc * rslope3[2, kts:kte, jts:jte, its:ite] * 
                              rslopeb[2, kts:kte, jts:jte, its:ite] * qci[0, kts:kte, jts:jte, its:ite] * 
                              denfac[kts:kte, jts:jte, its:ite], qci[0, kts:kte, jts:jte, its:ite] / dtcld)
        pgacw[kts:kte, jts:jte, its:ite] = torch.where(condition, pgacw[kts:kte, jts:jte, its:ite], torch.tensor(0))
        
        # paacw: Accretion of cloud water by averaged snow/graupel
        condition = qsum[kts:kte, jts:jte, its:ite] > 1.e-15
        paacw[kts:kte, jts:jte, its:ite] = (qrs[1, kts:kte, jts:jte, its:ite] * psacw[kts:kte, jts:jte, its:ite] + 
                                            qrs[2, kts:kte, jts:jte, its:ite] * pgacw[kts:kte, jts:jte, its:ite] ) / \
                                            qsum[kts:kte, jts:jte, its:ite]
        paacw[kts:kte, jts:jte, its:ite] = torch.where(condition, paacw[kts:kte, jts:jte, its:ite], torch.tensor(0))
        
        # pracs: Accretion of snow by rain
        condition = qrs[1, kts:kte, jts:jte, its:ite] > qcrmin 
        condition2 = qrs[0, kts:kte, jts:jte, its:ite] > qcrmin
        condition = condition & condition2
        condition1 = supcol[kts:kte, jts:jte, its:ite] > 0 
        condition1 = condition1 & condition
        acrfac[kts:kte, jts:jte, its:ite] = 5. * rslope3[1, kts:kte, jts:jte, its:ite] ** 2 * \
                rslope[0, kts:kte, jts:jte, its:ite] + 2. * rslope3[1, kts:kte, jts:jte, its:ite] * rslope2[1, kts:kte, jts:jte, its:ite] * \
                rslope2[0, kts:kte, jts:jte, its:ite] + 0.5 * rslope2[1, kts:kte, jts:jte, its:ite] ** 2 * rslope3[0, kts:kte, jts:jte, its:ite]
        pracs[kts:kte, jts:jte, its:ite] = pi ** 2 * n0r * n0s * n0sfac[kts:kte, jts:jte, its:ite] * \
                torch.abs(vt2r[kts:kte, jts:jte, its:ite] - vt2ave[kts:kte, jts:jte, its:ite]) * (dens / den[kts:kte, jts:jte, its:ite]) * \
                acrfac[kts:kte, jts:jte, its:ite]
        pracs[kts:kte, jts:jte, its:ite] = torch.minimum(pracs[kts:kte, jts:jte, its:ite], qrs[1, kts:kte, jts:jte, its:ite] / dtcld)
        pracs[kts:kte, jts:jte, its:ite] = torch.where(condition1, pracs[kts:kte, jts:jte, its:ite], torch.tensor(0))
        
        # psacr: Accretion of rain by snow
        acrfac[kts:kte, jts:jte, its:ite] = 5. * rslope3[0, kts:kte, jts:jte, its:ite] ** 2 * \
                rslope[1, kts:kte, jts:jte, its:ite] + 2. * rslope3[0, kts:kte, jts:jte, its:ite] * rslope2[0, kts:kte, jts:jte, its:ite] * \
                rslope2[1, kts:kte, jts:jte, its:ite] + 0.5 * rslope2[0, kts:kte, jts:jte, its:ite] ** 2 * rslope3[1, kts:kte, jts:jte, its:ite]
        psacr[kts:kte, jts:jte, its:ite] = pi ** 2 * n0r * n0s * n0sfac[kts:kte, jts:jte, its:ite] * \
                torch.abs(vt2r[kts:kte, jts:jte, its:ite] - vt2ave[kts:kte, jts:jte, its:ite]) * (denr / den[kts:kte, jts:jte, its:ite]) * \
                acrfac[kts:kte, jts:jte, its:ite]
        psacr[kts:kte, jts:jte, its:ite] = torch.minimum(psacr[kts:kte, jts:jte, its:ite], qrs[0, kts:kte, jts:jte, its:ite] / dtcld)
        psacr[kts:kte, jts:jte, its:ite] = torch.where(condition, psacr[kts:kte, jts:jte, its:ite], torch.tensor(0))
        # pgacr: Accretion of rain by graupel
        condition = qrs[2, kts:kte, jts:jte, its:ite] > qcrmin 
        condition1 = qrs[0, kts:kte, jts:jte, its:ite] > qcrmin
        condition = condition & condition1
        acrfac[kts:kte, jts:jte, its:ite] = 5. * rslope3[0, kts:kte, jts:jte, its:ite] ** 2 * \
                rslope[2, kts:kte, jts:jte, its:ite] + 2. * rslope3[0, kts:kte, jts:jte, its:ite] * rslope2[0, kts:kte, jts:jte, its:ite] * \
                rslope2[2, kts:kte, jts:jte, its:ite] + 0.5 * rslope2[0, kts:kte, jts:jte, its:ite] ** 2 * rslope3[2, kts:kte, jts:jte, its:ite]
        pgacr[kts:kte, jts:jte, its:ite] = pi ** 2 * n0r * n0g * torch.abs(vt2r[kts:kte, jts:jte, its:ite] - vt2ave[kts:kte, jts:jte, its:ite]) * (
                denr / den[kts:kte, jts:jte, its:ite]) * acrfac[kts:kte, jts:jte, its:ite]
        pgacr[kts:kte, jts:jte, its:ite] = torch.minimum(pgacr[kts:kte, jts:jte, its:ite], qrs[0, kts:kte, jts:jte, its:ite] / dtcld)
        pgacr[kts:kte, jts:jte, its:ite] = torch.where(condition, pgacr[kts:kte, jts:jte, its:ite], torch.tensor(0))
        # pgacs: Accretion of snow by graupel
        condition = qrs[2, kts:kte, jts:jte, its:ite] > qcrmin 
        condition1 = qrs[1, kts:kte, jts:jte, its:ite] > qcrmin
        condition = condition & condition1
        pgacs[kts:kte, jts:jte, its:ite] =  torch.where(condition, torch.tensor(0), pgacs[kts:kte, jts:jte, its:ite])
        condition = supcol[kts:kte, jts:jte, its:ite] <= 0.
        # pseml: Enhanced melting of snow by accretion of water
        condition1 = qrs[1, kts:kte, jts:jte, its:ite] > 0. 
        condition1 = condition & condition1
        pseml[kts:kte, jts:jte, its:ite] = torch.minimum(torch.maximum(cliq * supcol[kts:kte, jts:jte, its:ite] * 
                (paacw[kts:kte, jts:jte, its:ite] + psacr[kts:kte, jts:jte, its:ite]) / xlf0, -qrs[1, kts:kte, jts:jte, its:ite] /
                dtcld), torch.tensor(0.))
        pseml[kts:kte, jts:jte, its:ite] = torch.where(condition1, pseml[kts:kte, jts:jte, its:ite], torch.tensor(0))
        # pgeml: Enhanced melting of graupel by accretion of water
        condition1 = qrs[2, kts:kte, jts:jte, its:ite] > 0.
        condition1 = condition & condition1
        pgeml[kts:kte, jts:jte, its:ite] = torch.minimum(torch.maximum(cliq * supcol[kts:kte, jts:jte, its:ite] * 
                (paacw[kts:kte, jts:jte, its:ite] + psacr[kts:kte, jts:jte, its:ite]) / xlf0, -qrs[2, kts:kte, jts:jte, its:ite] /
                dtcld), torch.tensor(0.))
        pgeml[kts:kte, jts:jte, its:ite] = torch.where(condition1, pgeml[kts:kte, jts:jte, its:ite], torch.tensor(0))
        
        condition = supcol[kts:kte, jts:jte, its:ite] > 0.
        # pidep: Deposition/Sublimation rate of ice
        condition1 = qci[1, kts:kte, jts:jte, its:ite] > 0. 
        condition2 = ifsat[kts:kte, jts:jte, its:ite] != 1 
        condition1 = condition1 & condition2 & condition
        pidep[kts:kte, jts:jte, its:ite] = torch.where(condition1, 4. * diameter[kts:kte, jts:jte, its:ite] * xni[kts:kte, jts:jte, its:ite] * \
                (rh[1, kts:kte, jts:jte, its:ite] - 1.) / work1[1, kts:kte, jts:jte, its:ite], torch.tensor(0))
        supice[kts:kte, jts:jte, its:ite] = torch.where(condition1, satdt[kts:kte, jts:jte, its:ite] - prevp[kts:kte, jts:jte, its:ite], torch.tensor(0))
        condition2 = pidep[kts:kte, jts:jte, its:ite] < 0. 
        condition2 = condition2 & condition1
        condition3 = pidep[kts:kte, jts:jte, its:ite] >= 0.
        condition3 = condition3 & condition1
        pidep[kts:kte, jts:jte, its:ite] = torch.where(condition2, torch.maximum(torch.maximum(torch.maximum(
                pidep[kts:kte, jts:jte, its:ite], satdt[kts:kte, jts:jte, its:ite] / 2), supice[kts:kte, jts:jte, its:ite]), 
                -qci[1, kts:kte, jts:jte, its:ite] / dtcld), pidep[kts:kte, jts:jte, its:ite])
        pidep[kts:kte, jts:jte, its:ite] = torch.where(condition3, torch.minimum(torch.minimum(
                pidep[kts:kte, jts:jte, its:ite], satdt[kts:kte, jts:jte, its:ite] / 2), supice[kts:kte, jts:jte, its:ite]), pidep[kts:kte, jts:jte, its:ite])
        condition2 = torch.abs(prevp[kts:kte, jts:jte, its:ite] + pidep[kts:kte, jts:jte, its:ite]) >= \
                torch.abs(satdt[kts:kte, jts:jte, its:ite]) 
        condition2 = condition2 & condition1
        ifsat[kts:kte, jts:jte, its:ite] = torch.where(condition2, torch.tensor(1), ifsat[kts:kte, jts:jte, its:ite])
        # psdep: deposition/sublimation rate of snow
        condition1 = qrs[1, kts:kte, jts:jte, its:ite] > 0. 
        condition2 = ifsat[kts:kte, jts:jte, its:ite] != 1 
        condition1 = condition1 & condition2 & condition
        
        coeres[kts:kte, jts:jte, its:ite] = rslope2[1, kts:kte, jts:jte, its:ite] * (rslope[1, kts:kte, jts:jte, its:ite] * 
                                            rslopeb[1, kts:kte, jts:jte, its:ite]) ** 0.5
        psdep[kts:kte, jts:jte, its:ite] = (rh[1, kts:kte, jts:jte, its:ite] - 1.) * n0sfac[kts:kte, jts:jte, its:ite] * (
                precs1 * rslope2[1, kts:kte, jts:jte, its:ite] + precs2 * work2[kts:kte, jts:jte, its:ite] * 
                coeres[kts:kte, jts:jte, its:ite]) / work1[1, kts:kte, jts:jte, its:ite]
        
        psdep[kts:kte, jts:jte, its:ite] = torch.where(condition1, psdep[kts:kte, jts:jte, its:ite], torch.tensor(0))
        supice[kts:kte, jts:jte, its:ite] = torch.where(condition1, satdt[kts:kte, jts:jte, its:ite] - prevp[kts:kte, jts:jte, its:ite] - \
                                            pidep[kts:kte, jts:jte, its:ite], torch.tensor(0))
        
        condition2 = psdep[kts:kte, jts:jte, its:ite] < 0. 
        condition2 = condition2 & condition1
        condition3 = psdep[kts:kte, jts:jte, its:ite] >= 0.
        condition3 = condition3 & condition1

        psdep[kts:kte, jts:jte, its:ite] = torch.where(condition2, torch.maximum(torch.maximum(torch.maximum(
                psdep[kts:kte, jts:jte, its:ite], -qrs[1, kts:kte, jts:jte, its:ite] / dtcld), satdt[kts:kte, jts:jte, its:ite] / 2), 
                supice[kts:kte, jts:jte, its:ite]), psdep[kts:kte, jts:jte, its:ite])
        psdep[kts:kte, jts:jte, its:ite] = torch.where(condition3, torch.minimum(torch.minimum(
                psdep[kts:kte, jts:jte, its:ite], satdt[kts:kte, jts:jte, its:ite] / 2), supice[kts:kte, jts:jte, its:ite]), psdep[kts:kte, jts:jte, its:ite])
        
        condition2 = torch.abs(prevp[kts:kte, jts:jte, its:ite] + pidep[kts:kte, jts:jte, its:ite] + psdep[kts:kte, jts:jte, its:ite]) >= \
                torch.abs(satdt[kts:kte, jts:jte, its:ite]) 
        condition2 = condition2 & condition1
        ifsat[kts:kte, jts:jte, its:ite] = torch.where(condition2, torch.tensor(1), ifsat[kts:kte, jts:jte, its:ite])
        # pgdep deposition/sublimation rate of graupel
        condition1 = qrs[2, kts:kte, jts:jte, its:ite] > 0. 
        condition2 = ifsat[kts:kte, jts:jte, its:ite] != 1 
        condition1 = condition1 & condition2 & condition
        
        coeres[kts:kte, jts:jte, its:ite] = rslope2[2, kts:kte, jts:jte, its:ite] * (rslope[2, kts:kte, jts:jte, its:ite] * 
                                            rslopeb[2, kts:kte, jts:jte, its:ite]) ** 0.5
        pgdep[kts:kte, jts:jte, its:ite] = (rh[1, kts:kte, jts:jte, its:ite] - 1.) * (
                precg1 * rslope2[2, kts:kte, jts:jte, its:ite] + precg2 * work2[kts:kte, jts:jte, its:ite] * 
                coeres[kts:kte, jts:jte, its:ite]) / work1[1, kts:kte, jts:jte, its:ite]
        
        pgdep[kts:kte, jts:jte, its:ite] = torch.where(condition1, pgdep[kts:kte, jts:jte, its:ite], torch.tensor(0))
        
        supice[kts:kte, jts:jte, its:ite] = torch.where(condition1, satdt[kts:kte, jts:jte, its:ite] - prevp[kts:kte, jts:jte, its:ite] - \
                pidep[kts:kte, jts:jte, its:ite] - psdep[kts:kte, jts:jte, its:ite], torch.tensor(0))
        condition2 = pgdep[kts:kte, jts:jte, its:ite] < 0. 
        condition2 = condition2 & condition1
        condition3 = pgdep[kts:kte, jts:jte, its:ite] >= 0.
        condition3 = condition3 & condition1
        pgdep[kts:kte, jts:jte, its:ite] = torch.where(condition2, torch.maximum(torch.maximum(torch.maximum(
                pgdep[kts:kte, jts:jte, its:ite], -qrs[2, kts:kte, jts:jte, its:ite] / dtcld), satdt[kts:kte, jts:jte, its:ite] / 2), 
                supice[kts:kte, jts:jte, its:ite]), pgdep[kts:kte, jts:jte, its:ite])
        
        pgdep[kts:kte, jts:jte, its:ite] = torch.where(condition3, torch.minimum(torch.minimum(
                pgdep[kts:kte, jts:jte, its:ite], satdt[kts:kte, jts:jte, its:ite] / 2), supice[kts:kte, jts:jte, its:ite]), pgdep[kts:kte, jts:jte, its:ite])
        
        condition2 = torch.abs(prevp[kts:kte, jts:jte, its:ite] + pidep[kts:kte, jts:jte, its:ite] + 
                               psdep[kts:kte, jts:jte, its:ite] + pgdep[kts:kte, jts:jte, its:ite]) >= \
                     torch.abs(satdt[kts:kte, jts:jte, its:ite]) 
        condition2 = condition2 & condition1
        ifsat[kts:kte, jts:jte, its:ite] = torch.where(condition2, torch.tensor(1), ifsat[kts:kte, jts:jte, its:ite])
        # pigen: generation(nucleation) of ice from vapor
        condition1 = supsat[kts:kte, jts:jte, its:ite] > 0. 
        condition2 = ifsat[kts:kte, jts:jte, its:ite] != 1
        condition1 = condition1 & condition2 & condition
        supice[kts:kte, jts:jte, its:ite] = satdt[kts:kte, jts:jte, its:ite] - prevp[kts:kte, jts:jte, its:ite] - \
                pidep[kts:kte, jts:jte, its:ite] - psdep[kts:kte, jts:jte, its:ite] - pgdep[kts:kte, jts:jte, its:ite]
        roqi0[kts:kte, jts:jte, its:ite] = 4.92e-11 * (1.e3 * torch.exp(0.1 * supcol[kts:kte, jts:jte, its:ite])) ** 1.33
        pigen[kts:kte, jts:jte, its:ite] = torch.maximum(torch.tensor(0.), (roqi0[kts:kte, jts:jte, its:ite] / den[kts:kte, jts:jte, its:ite] -
                                                              torch.maximum(qci[1, kts:kte, jts:jte, its:ite], torch.tensor(0.))) / dtcld)
        pigen[kts:kte, jts:jte, its:ite] = torch.minimum(torch.minimum(pigen[kts:kte, jts:jte, its:ite], 
                                           satdt[kts:kte, jts:jte, its:ite]), supice[kts:kte, jts:jte, its:ite])
        #print("in wsm6 pigen2:",pigen[2,160,12],roqi0[2,160,12],satdt[2,160,12],supice[2,160,12],den[2,160,12],qci[1,2,160,12])
        pigen[kts:kte, jts:jte, its:ite] = torch.where(condition1, pigen[kts:kte, jts:jte, its:ite], torch.tensor(0.))
        # psaut: conversion(aggregation) of ice to snow
        condition1 = qci[1, kts:kte, jts:jte, its:ite] > 0. 
        condition1 = condition1 & condition
        psaut[kts:kte, jts:jte, its:ite] = torch.maximum(torch.tensor(0.), (qci[1, kts:kte, jts:jte, its:ite] - roqimax / 
                                                              den[kts:kte, jts:jte, its:ite]) / dtcld)
        psaut[kts:kte, jts:jte, its:ite] = torch.where(condition1, psaut[kts:kte, jts:jte, its:ite], torch.tensor(0.))
        
        # pgaut: conversion(aggregation) of snow to graupel
        condition1 = qrs[1, kts:kte, jts:jte, its:ite] > 0. 
        condition1 = condition1 & condition
        pgaut[kts:kte, jts:jte, its:ite] = torch.minimum(torch.maximum(torch.tensor(0.), 1.e-3 * torch.exp(
                -0.09 * supcol[kts:kte, jts:jte, its:ite]) * (qrs[1, kts:kte, jts:jte, its:ite] - qs0)), 
                qrs[1, kts:kte, jts:jte, its:ite] / dtcld)
        pgaut[kts:kte, jts:jte, its:ite] = torch.where(condition1, pgaut[kts:kte, jts:jte, its:ite], torch.tensor(0.))
        # psevp: Evaporation of melting snow
        condition = supcol[kts:kte, jts:jte, its:ite] < 0.
        condition1 = qrs[1, kts:kte, jts:jte, its:ite] > 0. 
        condition2 = rh[0, kts:kte, jts:jte, its:ite] < 1. 
        condition1 = condition1 & condition2 & condition
        coeres[kts:kte, jts:jte, its:ite] = rslope2[1, kts:kte, jts:jte, its:ite] * (rslope[1, kts:kte, jts:jte, its:ite] * 
                                            rslopeb[1, kts:kte, jts:jte, its:ite]) ** 0.5
        psevp[kts:kte, jts:jte, its:ite] = (rh[0, kts:kte, jts:jte, its:ite] - 1.) * n0sfac[kts:kte, jts:jte, its:ite] * (
                precs1 * rslope2[1, kts:kte, jts:jte, its:ite] + precs2 * work2[kts:kte, jts:jte, its:ite] * 
                coeres[kts:kte, jts:jte, its:ite]) / work1[0, kts:kte, jts:jte, its:ite]
        psevp[kts:kte, jts:jte, its:ite] = torch.minimum(torch.maximum(psevp[kts:kte, jts:jte, its:ite], 
                                           -qrs[1, kts:kte, jts:jte, its:ite] / dtcld), torch.tensor(0.))
        psevp[kts:kte, jts:jte, its:ite] = torch.where(condition1, psevp[kts:kte, jts:jte, its:ite], torch.tensor(0.))
        # pgevp: Evaporation of melting graupel
        condition1 = qrs[2, kts:kte, jts:jte, its:ite] > 0. 
        condition2 = rh[0, kts:kte, jts:jte, its:ite] < 1. 
        condition1 = condition1 & condition2 & condition
        coeres[kts:kte, jts:jte, its:ite] = rslope2[2, kts:kte, jts:jte, its:ite] * (rslope[2, kts:kte, jts:jte, its:ite] * 
                                            rslopeb[2, kts:kte, jts:jte, its:ite]) ** 0.5
        pgevp[kts:kte, jts:jte, its:ite] = (rh[0, kts:kte, jts:jte, its:ite] - 1.) * (
                precg1 * rslope2[2, kts:kte, jts:jte, its:ite] + precg2 * work2[kts:kte, jts:jte, its:ite] * 
                coeres[kts:kte, jts:jte, its:ite]) / work1[0, kts:kte, jts:jte, its:ite]
        pgevp[kts:kte, jts:jte, its:ite] = torch.minimum(torch.maximum(pgevp[kts:kte, jts:jte, its:ite], 
                                           -qrs[2, kts:kte, jts:jte, its:ite] / dtcld), torch.tensor(0.))
        pgevp[kts:kte, jts:jte, its:ite] = torch.where(condition1, pgevp[kts:kte, jts:jte, its:ite], torch.tensor(0.))
        
        # check mass conservation and feedback to the large scale
        condition = qrs[0, kts:kte, jts:jte, its:ite] < 1.e-4 
        condition1 = qrs[1, kts:kte, jts:jte, its:ite] < 1.e-4
        condition = condition1 & condition
        delta2[kts:kte, jts:jte, its:ite] = torch.where(condition, torch.tensor(1.), torch.tensor(0.))
        condition = qrs[0, kts:kte, jts:jte, its:ite] < 1.e-4
        delta3[kts:kte, jts:jte, its:ite] = torch.where(condition, torch.tensor(1.), torch.tensor(0.))
        
        condition = t[kts:kte, jts:jte, its:ite] <= t0c
        # cloud water
        value[kts:kte, jts:jte, its:ite] = torch.maximum(torch.tensor(qmin), qci[0, kts:kte, jts:jte, its:ite])
        source[kts:kte, jts:jte, its:ite] = (praut[kts:kte, jts:jte, its:ite] + pracw[kts:kte, jts:jte, its:ite] + 
                  paacw[kts:kte, jts:jte, its:ite] + paacw[kts:kte, jts:jte, its:ite]) * dtcld
        condition1 = source[kts:kte, jts:jte, its:ite] > value[kts:kte, jts:jte, its:ite] 
        condition1 = condition1 & condition
        factor[kts:kte, jts:jte, its:ite] = value[kts:kte, jts:jte, its:ite] / source[kts:kte, jts:jte, its:ite]
        praut[kts:kte, jts:jte, its:ite] = torch.where(condition1, praut[kts:kte, jts:jte, its:ite] * factor[kts:kte, jts:jte, its:ite], praut[kts:kte, jts:jte, its:ite])
        
        pracw[kts:kte, jts:jte, its:ite] = torch.where(condition1, pracw[kts:kte, jts:jte, its:ite] * factor[kts:kte, jts:jte, its:ite], pracw[kts:kte, jts:jte, its:ite])
        paacw[kts:kte, jts:jte, its:ite] = torch.where(condition1, paacw[kts:kte, jts:jte, its:ite] * factor[kts:kte, jts:jte, its:ite], paacw[kts:kte, jts:jte, its:ite])
        
        # cloud ice
        value[kts:kte, jts:jte, its:ite] = torch.maximum(torch.tensor(qmin), qci[1, kts:kte, jts:jte, its:ite])
        source[kts:kte, jts:jte, its:ite] = (psaut[kts:kte, jts:jte, its:ite] - pigen[kts:kte, jts:jte, its:ite] - 
                                             pidep[kts:kte, jts:jte, its:ite] + praci[kts:kte, jts:jte, its:ite] + 
                                             psaci[kts:kte, jts:jte, its:ite] + pgaci[kts:kte, jts:jte, its:ite]) * dtcld
        condition1 = source[kts:kte, jts:jte, its:ite] > value[kts:kte, jts:jte, its:ite]
        condition1 = condition1 & condition
        
        factor[kts:kte, jts:jte, its:ite] = value[kts:kte, jts:jte, its:ite] / source[kts:kte, jts:jte, its:ite]
        psaut[kts:kte, jts:jte, its:ite] = torch.where(condition1, psaut[kts:kte, jts:jte, its:ite] * factor[kts:kte, jts:jte, its:ite], psaut[kts:kte, jts:jte, its:ite])
        #print("in wsm6 pigen1:",pigen[2,160,12],factor[2,160,12])
        pigen[kts:kte, jts:jte, its:ite] = torch.where(condition1, pigen[kts:kte, jts:jte, its:ite] * factor[kts:kte, jts:jte, its:ite], pigen[kts:kte, jts:jte, its:ite])
        pidep[kts:kte, jts:jte, its:ite] = torch.where(condition1, pidep[kts:kte, jts:jte, its:ite] * factor[kts:kte, jts:jte, its:ite], pidep[kts:kte, jts:jte, its:ite])
        praci[kts:kte, jts:jte, its:ite] = torch.where(condition1, praci[kts:kte, jts:jte, its:ite] * factor[kts:kte, jts:jte, its:ite], praci[kts:kte, jts:jte, its:ite])
        psaci[kts:kte, jts:jte, its:ite] = torch.where(condition1, psaci[kts:kte, jts:jte, its:ite] * factor[kts:kte, jts:jte, its:ite], psaci[kts:kte, jts:jte, its:ite])
        pgaci[kts:kte, jts:jte, its:ite] = torch.where(condition1, pgaci[kts:kte, jts:jte, its:ite] * factor[kts:kte, jts:jte, its:ite], pgaci[kts:kte, jts:jte, its:ite])
        
        # rain
        value[kts:kte, jts:jte, its:ite] = torch.maximum(torch.tensor(qmin), qrs[0, kts:kte, jts:jte, its:ite])
        source[kts:kte, jts:jte, its:ite] = (-praut[kts:kte, jts:jte, its:ite] - prevp[kts:kte, jts:jte, its:ite] - 
                                             pracw[kts:kte, jts:jte, its:ite] + piacr[kts:kte, jts:jte, its:ite] + 
                                             psacr[kts:kte, jts:jte, its:ite] + pgacr[kts:kte, jts:jte, its:ite]) * dtcld
        condition1 = source[kts:kte, jts:jte, its:ite] > value[kts:kte, jts:jte, its:ite]
        condition1 = condition1 & condition
        factor[kts:kte, jts:jte, its:ite] = value[kts:kte, jts:jte, its:ite] / source[kts:kte, jts:jte, its:ite]
        praut[kts:kte, jts:jte, its:ite] = torch.where(condition1, praut[kts:kte, jts:jte, its:ite] * factor[kts:kte, jts:jte, its:ite], praut[kts:kte, jts:jte, its:ite])
        
        prevp[kts:kte, jts:jte, its:ite] = torch.where(condition1, prevp[kts:kte, jts:jte, its:ite] * factor[kts:kte, jts:jte, its:ite], prevp[kts:kte, jts:jte, its:ite])
        
        pracw[kts:kte, jts:jte, its:ite] = torch.where(condition1, pracw[kts:kte, jts:jte, its:ite] * factor[kts:kte, jts:jte, its:ite], pracw[kts:kte, jts:jte, its:ite])
        piacr[kts:kte, jts:jte, its:ite] = torch.where(condition1, piacr[kts:kte, jts:jte, its:ite] * factor[kts:kte, jts:jte, its:ite], piacr[kts:kte, jts:jte, its:ite])
        psacr[kts:kte, jts:jte, its:ite] = torch.where(condition1, psacr[kts:kte, jts:jte, its:ite] * factor[kts:kte, jts:jte, its:ite], psacr[kts:kte, jts:jte, its:ite])
        pgacr[kts:kte, jts:jte, its:ite] = torch.where(condition1, pgacr[kts:kte, jts:jte, its:ite] * factor[kts:kte, jts:jte, its:ite], pgacr[kts:kte, jts:jte, its:ite])
        # snow
        value[kts:kte, jts:jte, its:ite] = torch.maximum(torch.tensor(qmin), qrs[1, kts:kte, jts:jte, its:ite])
        source[kts:kte, jts:jte, its:ite] = -(psdep[kts:kte, jts:jte, its:ite] + psaut[kts:kte, jts:jte, its:ite] - 
                                              pgaut[kts:kte, jts:jte, its:ite] + paacw[kts:kte, jts:jte, its:ite] + 
                                              piacr[kts:kte, jts:jte, its:ite] * delta3[kts:kte, jts:jte, its:ite] + 
                                              praci[kts:kte, jts:jte, its:ite] * delta3[kts:kte, jts:jte, its:ite] - 
                                              pracs[kts:kte, jts:jte, its:ite] * (1. - delta2[kts:kte, jts:jte, its:ite]) + 
                                              psacr[kts:kte, jts:jte, its:ite] * delta2[kts:kte, jts:jte, its:ite] + 
                                              psaci[kts:kte, jts:jte, its:ite] - pgacs[kts:kte, jts:jte, its:ite]) * dtcld
        condition1 = source[kts:kte, jts:jte, its:ite] > value[kts:kte, jts:jte, its:ite]
        condition1 = condition1 & condition
        factor[kts:kte, jts:jte, its:ite] = value[kts:kte, jts:jte, its:ite] / source[kts:kte, jts:jte, its:ite]
        psdep[kts:kte, jts:jte, its:ite] = torch.where(condition1, psdep[kts:kte, jts:jte, its:ite] * factor[kts:kte, jts:jte, its:ite], psdep[kts:kte, jts:jte, its:ite])
        psaut[kts:kte, jts:jte, its:ite] = torch.where(condition1, psaut[kts:kte, jts:jte, its:ite] * factor[kts:kte, jts:jte, its:ite], psaut[kts:kte, jts:jte, its:ite])
        pgaut[kts:kte, jts:jte, its:ite] = torch.where(condition1, pgaut[kts:kte, jts:jte, its:ite] * factor[kts:kte, jts:jte, its:ite], pgaut[kts:kte, jts:jte, its:ite])
        paacw[kts:kte, jts:jte, its:ite] = torch.where(condition1, paacw[kts:kte, jts:jte, its:ite] * factor[kts:kte, jts:jte, its:ite], paacw[kts:kte, jts:jte, its:ite])
        piacr[kts:kte, jts:jte, its:ite] = torch.where(condition1, piacr[kts:kte, jts:jte, its:ite] * factor[kts:kte, jts:jte, its:ite], piacr[kts:kte, jts:jte, its:ite])
        praci[kts:kte, jts:jte, its:ite] = torch.where(condition1, praci[kts:kte, jts:jte, its:ite] * factor[kts:kte, jts:jte, its:ite], praci[kts:kte, jts:jte, its:ite])
        psaci[kts:kte, jts:jte, its:ite] = torch.where(condition1, psaci[kts:kte, jts:jte, its:ite] * factor[kts:kte, jts:jte, its:ite], psaci[kts:kte, jts:jte, its:ite])
        pracs[kts:kte, jts:jte, its:ite] = torch.where(condition1, pracs[kts:kte, jts:jte, its:ite] * factor[kts:kte, jts:jte, its:ite], pracs[kts:kte, jts:jte, its:ite])
        psacr[kts:kte, jts:jte, its:ite] = torch.where(condition1, psacr[kts:kte, jts:jte, its:ite] * factor[kts:kte, jts:jte, its:ite], psacr[kts:kte, jts:jte, its:ite])
        pgacs[kts:kte, jts:jte, its:ite] = torch.where(condition1, pgacs[kts:kte, jts:jte, its:ite] * factor[kts:kte, jts:jte, its:ite], pgacs[kts:kte, jts:jte, its:ite])
        # graupel
        value[kts:kte, jts:jte, its:ite] = torch.maximum(torch.tensor(qmin), qrs[2, kts:kte, jts:jte, its:ite])
        source[kts:kte, jts:jte, its:ite] = -(pgdep[kts:kte, jts:jte, its:ite] + pgaut[kts:kte, jts:jte, its:ite] + 
                                              piacr[kts:kte, jts:jte, its:ite] * (1. - delta3[kts:kte, jts:jte, its:ite]) + 
                                              praci[kts:kte, jts:jte, its:ite] * (1. - delta3[kts:kte, jts:jte, its:ite]) + 
                                              psacr[kts:kte, jts:jte, its:ite] * (1. - delta2[kts:kte, jts:jte, its:ite]) + 
                                              pracs[kts:kte, jts:jte, its:ite] * (1. - delta2[kts:kte, jts:jte, its:ite]) + 
                                              pgaci[kts:kte, jts:jte, its:ite] + paacw[kts:kte, jts:jte, its:ite] + 
                                              pgacr[kts:kte, jts:jte, its:ite] + pgacs[kts:kte, jts:jte, its:ite]) * dtcld
        condition1 = source[kts:kte, jts:jte, its:ite] > value[kts:kte, jts:jte, its:ite]
        condition1 = condition1 & condition
        
        factor[kts:kte, jts:jte, its:ite] = value[kts:kte, jts:jte, its:ite] / source[kts:kte, jts:jte, its:ite]
        pgdep[kts:kte, jts:jte, its:ite] = torch.where(condition1, pgdep[kts:kte, jts:jte, its:ite] * factor[kts:kte, jts:jte, its:ite], pgdep[kts:kte, jts:jte, its:ite])
        pgaut[kts:kte, jts:jte, its:ite] = torch.where(condition1, pgaut[kts:kte, jts:jte, its:ite] * factor[kts:kte, jts:jte, its:ite], pgaut[kts:kte, jts:jte, its:ite])
        piacr[kts:kte, jts:jte, its:ite] = torch.where(condition1, piacr[kts:kte, jts:jte, its:ite] * factor[kts:kte, jts:jte, its:ite], piacr[kts:kte, jts:jte, its:ite])
        praci[kts:kte, jts:jte, its:ite] = torch.where(condition1, praci[kts:kte, jts:jte, its:ite] * factor[kts:kte, jts:jte, its:ite], praci[kts:kte, jts:jte, its:ite])
        psacr[kts:kte, jts:jte, its:ite] = torch.where(condition1, psacr[kts:kte, jts:jte, its:ite] * factor[kts:kte, jts:jte, its:ite], psacr[kts:kte, jts:jte, its:ite])
        pracs[kts:kte, jts:jte, its:ite] = torch.where(condition1, pracs[kts:kte, jts:jte, its:ite] * factor[kts:kte, jts:jte, its:ite], pracs[kts:kte, jts:jte, its:ite])
        paacw[kts:kte, jts:jte, its:ite] = torch.where(condition1, paacw[kts:kte, jts:jte, its:ite] * factor[kts:kte, jts:jte, its:ite], paacw[kts:kte, jts:jte, its:ite])
        pgaci[kts:kte, jts:jte, its:ite] = torch.where(condition1, pgaci[kts:kte, jts:jte, its:ite] * factor[kts:kte, jts:jte, its:ite], pgaci[kts:kte, jts:jte, its:ite])
        pgacr[kts:kte, jts:jte, its:ite] = torch.where(condition1, pgacr[kts:kte, jts:jte, its:ite] * factor[kts:kte, jts:jte, its:ite], pgacr[kts:kte, jts:jte, its:ite])
        pgacs[kts:kte, jts:jte, its:ite] = torch.where(condition1, pgacs[kts:kte, jts:jte, its:ite] * factor[kts:kte, jts:jte, its:ite], pgacs[kts:kte, jts:jte, its:ite])
        
        work2[kts:kte, jts:jte, its:ite] = torch.where(condition, -(prevp[kts:kte, jts:jte, its:ite] + psdep[kts:kte, jts:jte, its:ite] + 
                            pgdep[kts:kte, jts:jte, its:ite] + pigen[kts:kte, jts:jte, its:ite] + pidep[kts:kte, jts:jte, its:ite]), 
                            work2[kts:kte, jts:jte, its:ite])
        
        # update
        
        q[kts:kte, jts:jte, its:ite] = torch.where(condition, q[kts:kte, jts:jte, its:ite] + 
                 work2[kts:kte, jts:jte, its:ite] * dtcld, q[kts:kte, jts:jte, its:ite])
        #print("in wsm q update 1:", q[2,160,12], prevp[2,160,12], psdep[2,160,12], pgdep[2,160,12], pigen[2,160,12], pidep[2,160,12])
        #注意paacw加了两次？
        worktmp[kts:kte, jts:jte, its:ite] = torch.maximum(qci[0, kts:kte, jts:jte, its:ite] - (praut[kts:kte, jts:jte, its:ite] + 
                 pracw[kts:kte, jts:jte, its:ite] + paacw[kts:kte, jts:jte, its:ite] + paacw[kts:kte, jts:jte, its:ite]) * dtcld, torch.tensor(0.))
        qci[0, kts:kte, jts:jte, its:ite] = torch.where(condition, worktmp[kts:kte, jts:jte, its:ite], qci[0, kts:kte, jts:jte, its:ite])
        
        worktmp[kts:kte, jts:jte, its:ite] = torch.maximum(qrs[0, kts:kte, jts:jte, its:ite] + (praut[kts:kte, jts:jte, its:ite] + 
                 pracw[kts:kte, jts:jte, its:ite] + prevp[kts:kte, jts:jte, its:ite] - piacr[kts:kte, jts:jte, its:ite] - 
                 pgacr[kts:kte, jts:jte, its:ite] - psacr[kts:kte, jts:jte, its:ite]) * dtcld, torch.tensor(0.))
        qrs[0, kts:kte, jts:jte, its:ite] = torch.where(condition, worktmp[kts:kte, jts:jte, its:ite], qrs[0, kts:kte, jts:jte, its:ite])
        
        worktmp[kts:kte, jts:jte, its:ite] = torch.maximum(qci[1, kts:kte, jts:jte, its:ite] - (psaut[kts:kte, jts:jte, its:ite] + 
                 praci[kts:kte, jts:jte, its:ite] + psaci[kts:kte, jts:jte, its:ite] + pgaci[kts:kte, jts:jte, its:ite] - 
                 pigen[kts:kte, jts:jte, its:ite] - pidep[kts:kte, jts:jte, its:ite]) * dtcld, torch.tensor(0.))
        qci[1, kts:kte, jts:jte, its:ite] = torch.where(condition, worktmp[kts:kte, jts:jte, its:ite], qci[1, kts:kte, jts:jte, its:ite])
        worktmp[kts:kte, jts:jte, its:ite] = torch.maximum(qrs[1, kts:kte, jts:jte, its:ite] + (psdep[kts:kte, jts:jte, its:ite] + 
                 psaut[kts:kte, jts:jte, its:ite] + paacw[kts:kte, jts:jte, its:ite] - pgaut[kts:kte, jts:jte, its:ite] + 
                 piacr[kts:kte, jts:jte, its:ite] * delta3[kts:kte, jts:jte, its:ite] + praci[kts:kte, jts:jte, its:ite] * 
                 delta3[kts:kte, jts:jte, its:ite] + psaci[kts:kte, jts:jte, its:ite] - pgacs[kts:kte, jts:jte, its:ite] - 
                 pracs[kts:kte, jts:jte, its:ite] * (1. - delta2[kts:kte, jts:jte, its:ite]) + psacr[kts:kte, jts:jte, its:ite] * 
                 delta2[kts:kte, jts:jte, its:ite]) * dtcld, torch.tensor(0.))
        
        qrs[1, kts:kte, jts:jte, its:ite] = torch.where(condition, worktmp[kts:kte, jts:jte, its:ite], qrs[1, kts:kte, jts:jte, its:ite])
        
        worktmp[kts:kte, jts:jte, its:ite] = torch.maximum(qrs[2, kts:kte, jts:jte, its:ite] + (pgdep[kts:kte, jts:jte, its:ite] + 
                 pgaut[kts:kte, jts:jte, its:ite] + piacr[kts:kte, jts:jte, its:ite] * (1. - delta3[kts:kte, jts:jte, its:ite]) + 
                 praci[kts:kte, jts:jte, its:ite] * (1. - delta3[kts:kte, jts:jte, its:ite]) + psacr[kts:kte, jts:jte, its:ite] * (1. - 
                 delta2[kts:kte, jts:jte, its:ite]) + pracs[kts:kte, jts:jte, its:ite] * (1. - delta2[kts:kte, jts:jte, its:ite]) + 
                 pgaci[kts:kte, jts:jte, its:ite] + paacw[kts:kte, jts:jte, its:ite] + pgacr[kts:kte, jts:jte, its:ite] + 
                 pgacs[kts:kte, jts:jte, its:ite]) * dtcld, torch.tensor(0.))
        qrs[2, kts:kte, jts:jte, its:ite] = torch.where(condition, worktmp[kts:kte, jts:jte, its:ite], qrs[2, kts:kte, jts:jte, its:ite])
        
        xlf3[kts:kte, jts:jte, its:ite] = xls - xl[kts:kte, jts:jte, its:ite]
        xlwork2[kts:kte, jts:jte, its:ite] = -xls * (psdep[kts:kte, jts:jte, its:ite] + pgdep[kts:kte, jts:jte, its:ite] + 
                 pidep[kts:kte, jts:jte, its:ite] + pigen[kts:kte, jts:jte, its:ite]) - xl[kts:kte, jts:jte, its:ite] * \
                 prevp[kts:kte, jts:jte, its:ite] - xlf3[kts:kte, jts:jte, its:ite] * (piacr[kts:kte, jts:jte, its:ite] + 
                 paacw[kts:kte, jts:jte, its:ite] + paacw[kts:kte, jts:jte, its:ite] + pgacr[kts:kte, jts:jte, its:ite] + 
                 psacr[kts:kte, jts:jte, its:ite])
        
        t[kts:kte, jts:jte, its:ite] = torch.where(condition, t[kts:kte, jts:jte, its:ite] - xlwork2[kts:kte, jts:jte, its:ite] / 
                 cpm[kts:kte, jts:jte, its:ite] * dtcld, t[kts:kte, jts:jte, its:ite])
        
        condition = t[kts:kte, jts:jte, its:ite] > t0c
        # cloud water
        value[kts:kte, jts:jte, its:ite] = torch.maximum(torch.tensor(qmin), qci[0, kts:kte, jts:jte, its:ite])
        source[kts:kte, jts:jte, its:ite] = (praut[kts:kte, jts:jte, its:ite] + pracw[kts:kte, jts:jte, its:ite] + 
                  paacw[kts:kte, jts:jte, its:ite] + paacw[kts:kte, jts:jte, its:ite]) * dtcld
        condition1 = source[kts:kte, jts:jte, its:ite] > value[kts:kte, jts:jte, its:ite]
        condition1 = condition1 & condition
        factor[kts:kte, jts:jte, its:ite] = value[kts:kte, jts:jte, its:ite] / source[kts:kte, jts:jte, its:ite]
        praut[kts:kte, jts:jte, its:ite] = torch.where(condition1, praut[kts:kte, jts:jte, its:ite] * factor[kts:kte, jts:jte, its:ite], praut[kts:kte, jts:jte, its:ite])
        pracw[kts:kte, jts:jte, its:ite] = torch.where(condition1, pracw[kts:kte, jts:jte, its:ite] * factor[kts:kte, jts:jte, its:ite], pracw[kts:kte, jts:jte, its:ite])
        paacw[kts:kte, jts:jte, its:ite] = torch.where(condition1, paacw[kts:kte, jts:jte, its:ite] * factor[kts:kte, jts:jte, its:ite], paacw[kts:kte, jts:jte, its:ite])
        # rain
        value[kts:kte, jts:jte, its:ite] = torch.maximum(torch.tensor(qmin), qrs[0, kts:kte, jts:jte, its:ite])
        source[kts:kte, jts:jte, its:ite] = (-paacw[kts:kte, jts:jte, its:ite] - praut[kts:kte, jts:jte, its:ite] + 
                                             pseml[kts:kte, jts:jte, its:ite] + pgeml[kts:kte, jts:jte, its:ite] - 
                                             pracw[kts:kte, jts:jte, its:ite] - paacw[kts:kte, jts:jte, its:ite] -
                                             prevp[kts:kte, jts:jte, its:ite]) * dtcld
        condition1 = source[kts:kte, jts:jte, its:ite] > value[kts:kte, jts:jte, its:ite]
        condition1 = condition1 & condition
        factor[kts:kte, jts:jte, its:ite] = value[kts:kte, jts:jte, its:ite] / source[kts:kte, jts:jte, its:ite]
        praut[kts:kte, jts:jte, its:ite] = torch.where(condition1, praut[kts:kte, jts:jte, its:ite] * factor[kts:kte, jts:jte, its:ite], praut[kts:kte, jts:jte, its:ite])
        prevp[kts:kte, jts:jte, its:ite] = torch.where(condition1, prevp[kts:kte, jts:jte, its:ite] * factor[kts:kte, jts:jte, its:ite], prevp[kts:kte, jts:jte, its:ite])
        pracw[kts:kte, jts:jte, its:ite] = torch.where(condition1, pracw[kts:kte, jts:jte, its:ite] * factor[kts:kte, jts:jte, its:ite], pracw[kts:kte, jts:jte, its:ite])
        paacw[kts:kte, jts:jte, its:ite] = torch.where(condition1, paacw[kts:kte, jts:jte, its:ite] * factor[kts:kte, jts:jte, its:ite], paacw[kts:kte, jts:jte, its:ite])
        pseml[kts:kte, jts:jte, its:ite] = torch.where(condition1, pseml[kts:kte, jts:jte, its:ite] * factor[kts:kte, jts:jte, its:ite], pseml[kts:kte, jts:jte, its:ite])
        pgeml[kts:kte, jts:jte, its:ite] = torch.where(condition1, pgeml[kts:kte, jts:jte, its:ite] * factor[kts:kte, jts:jte, its:ite], pgeml[kts:kte, jts:jte, its:ite])
        # snow
        value[kts:kte, jts:jte, its:ite] = torch.maximum(torch.tensor(qcrmin), qrs[1, kts:kte, jts:jte, its:ite])
        source[kts:kte, jts:jte, its:ite] = (pgacs[kts:kte, jts:jte, its:ite] - pseml[kts:kte, jts:jte, its:ite] - 
                                             psevp[kts:kte, jts:jte, its:ite]) * dtcld
        condition1 = source[kts:kte, jts:jte, its:ite] > value[kts:kte, jts:jte, its:ite]
        condition1 = condition1 & condition
        factor[kts:kte, jts:jte, its:ite] = value[kts:kte, jts:jte, its:ite] / source[kts:kte, jts:jte, its:ite]
        pgacs[kts:kte, jts:jte, its:ite] = torch.where(condition1, pgacs[kts:kte, jts:jte, its:ite] * factor[kts:kte, jts:jte, its:ite], pgacs[kts:kte, jts:jte, its:ite])
        psevp[kts:kte, jts:jte, its:ite] = torch.where(condition1, psevp[kts:kte, jts:jte, its:ite] * factor[kts:kte, jts:jte, its:ite], psevp[kts:kte, jts:jte, its:ite])
        pseml[kts:kte, jts:jte, its:ite] = torch.where(condition1, pseml[kts:kte, jts:jte, its:ite] * factor[kts:kte, jts:jte, its:ite], pseml[kts:kte, jts:jte, its:ite])
        # graupel
        value[kts:kte, jts:jte, its:ite] = torch.maximum(torch.tensor(qcrmin), qrs[2, kts:kte, jts:jte, its:ite])
        source[kts:kte, jts:jte, its:ite] = -(pgacs[kts:kte, jts:jte, its:ite] + pgevp[kts:kte, jts:jte, its:ite] + 
                                              pgeml[kts:kte, jts:jte, its:ite]) * dtcld
        condition1 = source[kts:kte, jts:jte, its:ite] > value[kts:kte, jts:jte, its:ite]
        condition1 = condition1 & condition
        factor[kts:kte, jts:jte, its:ite] = value[kts:kte, jts:jte, its:ite] / source[kts:kte, jts:jte, its:ite]
        pgacs[kts:kte, jts:jte, its:ite] = torch.where(condition1, pgacs[kts:kte, jts:jte, its:ite] * factor[kts:kte, jts:jte, its:ite], pgacs[kts:kte, jts:jte, its:ite])
        pgevp[kts:kte, jts:jte, its:ite] = torch.where(condition1, pgevp[kts:kte, jts:jte, its:ite] * factor[kts:kte, jts:jte, its:ite], pgevp[kts:kte, jts:jte, its:ite])
        pgeml[kts:kte, jts:jte, its:ite] = torch.where(condition1, pgeml[kts:kte, jts:jte, its:ite] * factor[kts:kte, jts:jte, its:ite], pgeml[kts:kte, jts:jte, its:ite])
        
        work2[kts:kte, jts:jte, its:ite] = torch.where(condition, -(prevp[kts:kte, jts:jte, its:ite] + psevp[kts:kte, jts:jte, its:ite] + 
                            pgevp[kts:kte, jts:jte, its:ite]), work2[kts:kte, jts:jte, its:ite])
        # update
        q[kts:kte, jts:jte, its:ite] = torch.where(condition, q[kts:kte, jts:jte, its:ite] + 
                 work2[kts:kte, jts:jte, its:ite] * dtcld, q[kts:kte, jts:jte, its:ite])
        #print("in wsm q update 2:", q[2,160,12], prevp[2,160,12], psevp[2,160,12], pgevp[2,160,12])
        worktmp[kts:kte, jts:jte, its:ite] = torch.maximum(qci[0, kts:kte, jts:jte, its:ite] - (praut[kts:kte, jts:jte, its:ite] + 
                 pracw[kts:kte, jts:jte, its:ite] + paacw[kts:kte, jts:jte, its:ite] + paacw[kts:kte, jts:jte, its:ite]) * dtcld, torch.tensor(0.))
        qci[0, kts:kte, jts:jte, its:ite] = torch.where(condition, worktmp[kts:kte, jts:jte, its:ite], qci[0, kts:kte, jts:jte, its:ite])
        
        worktmp[kts:kte, jts:jte, its:ite] = torch.maximum(qrs[0, kts:kte, jts:jte, its:ite] + (praut[kts:kte, jts:jte, its:ite] + 
                 pracw[kts:kte, jts:jte, its:ite] + prevp[kts:kte, jts:jte, its:ite] + paacw[kts:kte, jts:jte, its:ite] + 
                 paacw[kts:kte, jts:jte, its:ite] - pseml[kts:kte, jts:jte, its:ite] - pgeml[kts:kte, jts:jte, its:ite]) * dtcld, torch.tensor(0.))
        
        qrs[0, kts:kte, jts:jte, its:ite] = torch.where(condition, worktmp[kts:kte, jts:jte, its:ite], qrs[0, kts:kte, jts:jte, its:ite])
        
        worktmp[kts:kte, jts:jte, its:ite] = torch.maximum(qrs[1, kts:kte, jts:jte, its:ite] + (psevp[kts:kte, jts:jte, its:ite] - 
                 pgacs[kts:kte, jts:jte, its:ite] + pseml[kts:kte, jts:jte, its:ite]) * dtcld, torch.tensor(0.))
        qrs[1, kts:kte, jts:jte, its:ite] = torch.where(condition, worktmp[kts:kte, jts:jte, its:ite], qrs[1, kts:kte, jts:jte, its:ite])
        
        worktmp[kts:kte, jts:jte, its:ite] = torch.maximum(qrs[2, kts:kte, jts:jte, its:ite] + (pgacs[kts:kte, jts:jte, its:ite] + 
                 pgevp[kts:kte, jts:jte, its:ite] + pgeml[kts:kte, jts:jte, its:ite]) * dtcld, torch.tensor(0.))
        qrs[2, kts:kte, jts:jte, its:ite] = torch.where(condition, worktmp[kts:kte, jts:jte, its:ite], qrs[2, kts:kte, jts:jte, its:ite])
        
        xlf3[kts:kte, jts:jte, its:ite] = xls - xl[kts:kte, jts:jte, its:ite]
        xlwork2[kts:kte, jts:jte, its:ite] = - xl[kts:kte, jts:jte, its:ite] * (prevp[kts:kte, jts:jte, its:ite] + 
                 psevp[kts:kte, jts:jte, its:ite] + pgevp[kts:kte, jts:jte, its:ite]) - xlf3[kts:kte, jts:jte, its:ite] * (
                 pseml[kts:kte, jts:jte, its:ite] + pgeml[kts:kte, jts:jte, its:ite])
        
        t[kts:kte, jts:jte, its:ite] = torch.where(condition, t[kts:kte, jts:jte, its:ite] - xlwork2[kts:kte, jts:jte, its:ite] / 
                 cpm[kts:kte, jts:jte, its:ite] * dtcld, t[kts:kte, jts:jte, its:ite])
        
        # Inline expansion for fpvs
        hsub = xls
        hvap = xlv0
        cvap = cpv
        ttp=t0c+0.01
        dldt=cvap-cliq
        xa=-dldt/rv
        xb=xa+hvap/(rv*ttp)
        dldti=cvap-cice
        xai=-dldti/rv
        xbi=xai+hsub/(rv*ttp)
        
        tr[kts:kte, jts:jte, its:ite] = ttp / t[kts:kte, jts:jte, its:ite]
        qstmp[0, kts:kte, jts:jte, its:ite] = psat * torch.exp(torch.log(tr[kts:kte, jts:jte, its:ite]) * 
                        xa) * torch.exp(xb * (1. - tr[kts:kte, jts:jte, its:ite]))
        qstmp[0, kts:kte, jts:jte, its:ite] = torch.minimum(qstmp[0,kts:kte, jts:jte, its:ite] , 
                                                         0.99 * p[kts:kte, jts:jte, its:ite])
        qstmp[0, kts:kte, jts:jte, its:ite] = ep2 * qstmp[0, kts:kte, jts:jte, its:ite] / \
              (p[kts:kte, jts:jte, its:ite] - qstmp[0, kts:kte, jts:jte, its:ite])
        qstmp[0, kts:kte, jts:jte, its:ite] = torch.maximum(qstmp[0, kts:kte, jts:jte, its:ite], torch.tensor(qmin))
        #rh[0, kts:kte, jts:jte, its:ite] = torch.maximum(q[kts:kte, jts:jte, its:ite] / 
        #                                                 qstmp[0, kts:kte, jts:jte, its:ite], qmin)
        wsmtmp0[kts:kte, jts:jte, its:ite] = psat * torch.exp(torch.log(tr[kts:kte, jts:jte, its:ite]) * 
                        xai) * torch.exp(xbi * (1. - tr[kts:kte, jts:jte, its:ite]))
        wsmtmp1[kts:kte, jts:jte, its:ite] = psat * torch.exp(torch.log(tr[kts:kte, jts:jte, its:ite]) * 
                        xa) * torch.exp(xb * (1. - tr[kts:kte, jts:jte, its:ite]))
        qstmp[1, kts:kte, jts:jte, its:ite] = torch.where(t[kts:kte, jts:jte, its:ite] < ttp, 
                        wsmtmp0[kts:kte, jts:jte, its:ite], wsmtmp1[kts:kte, jts:jte, its:ite])
        qstmp[1, kts:kte, jts:jte, its:ite] = torch.minimum(qstmp[1, kts:kte, jts:jte, its:ite], 
                                                         0.99 * p[kts:kte, jts:jte, its:ite])
        qstmp[1, kts:kte, jts:jte, its:ite] =  ep2 * qstmp[1, kts:kte, jts:jte, its:ite] / \
              (p[kts:kte, jts:jte, its:ite] - qstmp[1, kts:kte, jts:jte, its:ite])
        qstmp[1, kts:kte, jts:jte, its:ite] = torch.maximum(qstmp[1, kts:kte, jts:jte, its:ite], torch.tensor(qmin))
        
        # pcond
        work1[0, kts:kte, jts:jte, its:ite] = conden(t[kts:kte, jts:jte, its:ite], 
                        q[kts:kte, jts:jte, its:ite], qstmp[0, kts:kte, jts:jte, its:ite], 
                        xl[kts:kte, jts:jte, its:ite], cpm[kts:kte, jts:jte, its:ite])
        work2[kts:kte, jts:jte, its:ite] = qci[0, kts:kte, jts:jte, its:ite] + work1[0, kts:kte, jts:jte, its:ite]
        pcond[kts:kte, jts:jte, its:ite] = torch.minimum(torch.maximum(work1[0, kts:kte, jts:jte, its:ite] / dtcld, torch.tensor(0.)), 
                                                         torch.maximum(q[kts:kte, jts:jte, its:ite], torch.tensor(0.)) / dtcld)
        
        condition = qci[0, kts:kte, jts:jte, its:ite] > 0. 
        condition1 = work1[0, kts:kte, jts:jte, its:ite] < 0.
        condition = condition1 & condition
        
        pcond[kts:kte, jts:jte, its:ite] = torch.where(condition, torch.maximum(work1[0, kts:kte, jts:jte, its:ite], 
                        -qci[0, kts:kte, jts:jte, its:ite]) / dtcld, pcond[kts:kte, jts:jte, its:ite])
        
        q[kts:kte, jts:jte, its:ite] = q[kts:kte, jts:jte, its:ite] - pcond[kts:kte, jts:jte, its:ite] * dtcld
        #print("in wsm q update 3:", q[2,160,12],pcond[2,160,12])
        qci[0, kts:kte, jts:jte, its:ite] = torch.maximum(qci[0, kts:kte, jts:jte, its:ite] + pcond[kts:kte, jts:jte, its:ite] * dtcld, torch.tensor(0.))
        
        t[kts:kte, jts:jte, its:ite] = t[kts:kte, jts:jte, its:ite] + pcond[kts:kte, jts:jte, its:ite] * xl[kts:kte, jts:jte, its:ite] / \
                        cpm[kts:kte, jts:jte, its:ite] * dtcld
        
        # padding for small values
        condition = qci[0, kts:kte, jts:jte, its:ite] <= qmin
        qci[0, kts:kte, jts:jte, its:ite] = torch.where(condition, torch.tensor(0.), qci[0, kts:kte, jts:jte, its:ite])
        
        condition = qci[1, kts:kte, jts:jte, its:ite] <= qmin
        qci[1, kts:kte, jts:jte, its:ite] = torch.where(condition, torch.tensor(0.), qci[1, kts:kte, jts:jte, its:ite])
        
    # end of former wsm62D subroutine
    
    th[kts:kte, jts:jte, its:ite] = t[kts:kte, jts:jte, its:ite] / pii[kts:kte, jts:jte, its:ite]
    qc[kts:kte, jts:jte, its:ite] = qci[0, kts:kte, jts:jte, its:ite] + 0.0
    qi[kts:kte, jts:jte, its:ite] = qci[1, kts:kte, jts:jte, its:ite] + 0.0
    qr[kts:kte, jts:jte, its:ite] = qrs[0, kts:kte, jts:jte, its:ite] + 0.0
    qs[kts:kte, jts:jte, its:ite] = qrs[1, kts:kte, jts:jte, its:ite] + 0.0
    qg[kts:kte, jts:jte, its:ite] = qrs[2, kts:kte, jts:jte, its:ite] + 0.0
    
    return th, q, qc, qr, qi, qs, qg, rain, rainncv, snow, snowncv, graupel, graupelncv, sr, refl_10cm

def nislfv_rain_plm(km,denl,denfacl,tkl,dzl,wwl,rql,precip,dt,id,iter):
    
    allold = torch.zeros((nyall,nxall)).to(device)
    wd = torch.zeros((nzall,nyall,nxall)).to(device)
    tmp = torch.zeros((nzall,nyall,nxall)).to(device)
    tmp1 = torch.zeros((nzall,nyall,nxall)).to(device)
    tmp2 = torch.zeros((nzall,nyall,nxall)).to(device)
    tmp3 = torch.zeros((nzall,nyall,nxall)).to(device)
    wa = torch.zeros((nzall,nyall,nxall)).to(device)
    was = torch.zeros((nzall,nyall,nxall)).to(device)
    qr = torch.zeros((nzall,nyall,nxall)).to(device)
    decfl = torch.zeros((nzall,nyall,nxall)).to(device)
    dip = torch.zeros((nzall,nyall,nxall)).to(device)
    dim = torch.zeros((nzall,nyall,nxall)).to(device)
    
    za = torch.zeros((nzall+1,nyall,nxall)).to(device)
    zi = torch.zeros((nzall+1,nyall,nxall)).to(device)
    wi = torch.zeros((nzall+1,nyall,nxall)).to(device)
    dza = torch.zeros((nzall+1,nyall,nxall)).to(device)
    qa = torch.zeros((nzall+1,nyall,nxall)).to(device)
    qmi = torch.zeros((nzall+1,nyall,nxall)).to(device)
    qpi = torch.zeros((nzall+1,nyall,nxall)).to(device)
    
    qn = torch.zeros((nzall,nyall,nxall)).to(device)
    dzi = torch.zeros((nzall+1,nyall,nxall)).to(device)
    dz = torch.zeros((nzall,nyall,nxall)).to(device)
    qq = torch.zeros((nzall,nyall,nxall)).to(device)
    ww = torch.zeros((nzall,nyall,nxall)).to(device)
    den = torch.zeros((nzall,nyall,nxall)).to(device)
    denfac = torch.zeros((nzall,nyall,nxall)).to(device)
    tk = torch.zeros((nzall,nyall,nxall)).to(device)
    
    dz[kts:kte, jts:jte, its:ite] = dzl[kts:kte, jts:jte, its:ite] + 0.0
    qq[kts:kte, jts:jte, its:ite] = rql[kts:kte, jts:jte, its:ite] + 0.0
    ww[kts:kte, jts:jte, its:ite] = wwl[kts:kte, jts:jte, its:ite] + 0.0
    den[kts:kte, jts:jte, its:ite] = denl[kts:kte, jts:jte, its:ite] + 0.0
    denfac[kts:kte, jts:jte, its:ite] = denfacl[kts:kte, jts:jte, its:ite] + 0.0
    tk[kts:kte, jts:jte, its:ite] = tkl[kts:kte, jts:jte, its:ite] + 0.0
    
    allold[jts:jte, its:ite] = qq[kts:kte, jts:jte, its:ite].sum(dim = 0)
    zi[0, jts:jte, its:ite] = 0.0
    
    for k in range(0,km):
        zi[k+1, jts:jte, its:ite] = zi[k, jts:jte, its:ite] + dz[k, jts:jte, its:ite]
    wd[0:km, jts:jte, its:ite] = ww[0:km, jts:jte, its:ite] + 0.0
    
    n=1
    if iter == 0:
        n = 0
    while n<= iter+1:
        # 2nd order interpolation
        wi[0, jts:jte, its:ite] = ww[0, jts:jte, its:ite] + 0.0
        wi[km, jts:jte, its:ite] = ww[km-1, jts:jte, its:ite] + 0.0
        for k in range(1,km):
            wi[k ,jts:jte, its:ite] = (ww[k, jts:jte, its:ite] * dz[k-1, jts:jte, its:ite] + 
                                        ww[k-1, jts:jte, its:ite] * dz[k, jts:jte, its:ite]) / \
                                       (dz[k-1, jts:jte, its:ite] + dz[k, jts:jte, its:ite])
        # 3rd order interpolation
        fa1 = 9./16.
        fa2 = 1./16.
        wi[0, jts:jte, its:ite] = ww[0, jts:jte, its:ite] + 0.0
        wi[1, jts:jte, its:ite] = 0.5 * (ww[1, jts:jte, its:ite] + ww[0, jts:jte, its:ite])
        for k in range(2, km-1):
            wi[k, jts:jte, its:ite] = fa1 * (ww[k, jts:jte, its:ite] + ww[k-1, jts:jte, its:ite]) - \
                                       fa2 * (ww[k+1, jts:jte, its:ite] + ww[k-2, jts:jte, its:ite])
        wi[km-1, jts:jte, its:ite] = 0.5 * (ww[km-1, jts:jte, its:ite] + ww[km-2, jts:jte, its:ite])
        wi[km, jts:jte, its:ite] = ww[km-1, jts:jte, its:ite] + 0.0
        
        wi[1:km, jts:jte, its:ite] = torch.where(ww[1:km, jts:jte, its:ite] == 0.0, 
                ww[0:km-1, jts:jte, its:ite], wi[1:km, jts:jte, its:ite])
        # diffusivity of wi
        #decfl[0:km, jts:jte, its:ite] = (wi[1:km+1, jts:jte, its:ite] - wi[0:km, jts:jte, its:ite]) * \
        #        dt / dz[0:km, jts:jte, its:ite]
        for k in range(km-1,-1,-1):
            decfl[k, jts:jte, its:ite] = (wi[k+1, jts:jte, its:ite] - wi[k, jts:jte, its:ite]) * \
                dt / dz[k, jts:jte, its:ite]
            wi[k, jts:jte, its:ite] = torch.where(decfl[k, jts:jte, its:ite] > 0.05, 
                wi[k+1, jts:jte, its:ite] - 0.05 * dz[k, jts:jte, its:ite] /dt, wi[k, jts:jte, its:ite])
        
        #
        za[0:km+1, jts:jte, its:ite] = zi[0:km+1, jts:jte, its:ite] - wi[0:km+1, jts:jte, its:ite] * dt
        dza[0:km, jts:jte, its:ite] = za[1:km+1, jts:jte, its:ite] - za[0:km, jts:jte, its:ite]
        dza[km, jts:jte, its:ite] = zi[km, jts:jte, its:ite] - za[km, jts:jte, its:ite]
        #
        qa[0:km, jts:jte, its:ite] = qq[0:km, jts:jte, its:ite] * dz[0:km, jts:jte, its:ite] / \
                dza[0:km, jts:jte, its:ite]
        #print("in nislfv2:",qq[2,160,12],dza[2,160,12])
        qr[0:km, jts:jte, its:ite] = qa[0:km, jts:jte, its:ite] / den[0:km, jts:jte, its:ite]
        qa[km, jts:jte, its:ite] = 0.0
        
        if n <= iter:
            tmp, tmp1, tmp2, tmp3, wa = slope_rain(qr, den, denfac, tk, tmp, tmp1, tmp2, tmp3, wa, its, ite, kts, km)
            if n >= 2:
                wa[0:km, jts:jte, its:ite] = 0.5 * (wa[0:km, jts:jte, its:ite] + was[0:km, jts:jte, its:ite])
            ww[kts:kte, jts:jte, its:ite] = 0.5 * (wd[kts:kte, jts:jte, its:ite] + wa[kts:kte, jts:jte, its:ite])
            was[kts:kte, jts:jte, its:ite] = wa[kts:kte, jts:jte, its:ite] + 0.0
        n = n+1
        if iter == 0:
            n = 2
    # estimate values at arrival cell interface with monotone
    dip[1:km, jts:jte, its:ite] = (qa[2:km+1, jts:jte, its:ite] - qa[1:km, jts:jte, its:ite]) / \
            (dza[2:km+1, jts:jte, its:ite] + dza[1:km, jts:jte, its:ite])
    dim[1:km, jts:jte, its:ite] = (qa[1:km, jts:jte, its:ite] - qa[0:km-1, jts:jte, its:ite]) / \
            (dza[0:km-1, jts:jte, its:ite] + dza[1:km, jts:jte, its:ite])
    qpi[1:km, jts:jte, its:ite] = torch.where(dip[1:km, jts:jte, its:ite] * 
            dim[1:km, jts:jte, its:ite] <= 0.0, qa[1:km, jts:jte, its:ite], 
            qa[1:km, jts:jte, its:ite] + 0.5 * (dip[1:km, jts:jte, its:ite] + 
            dim[1:km, jts:jte, its:ite]) * dza[1:km, jts:jte, its:ite])
    qmi[1:km, jts:jte, its:ite] = torch.where(dip[1:km, jts:jte, its:ite] * 
            dim[1:km, jts:jte, its:ite] <= 0.0, qa[1:km, jts:jte, its:ite], 
            2.0 * qa[1:km, jts:jte, its:ite] - qpi[1:km, jts:jte, its:ite])
    condition1 = qpi[1:km, jts:jte, its:ite] < 0.0
    condition2 = qmi[1:km, jts:jte, its:ite] < 0.0
    condition = condition1 | condition2
    qpi[1:km, jts:jte, its:ite] = torch.where(condition, qa[1:km, jts:jte, its:ite], qpi[1:km, jts:jte, its:ite])
    qmi[1:km, jts:jte, its:ite] = torch.where(condition, qa[1:km, jts:jte, its:ite], qmi[1:km, jts:jte, its:ite])
    qpi[0, jts:jte, its:ite] = qa[0, jts:jte, its:ite] + 0.0
    qmi[0, jts:jte, its:ite] = qa[0, jts:jte, its:ite] + 0.0
    qpi[km, jts:jte, its:ite] = qa[km, jts:jte, its:ite] + 0.0
    qmi[km, jts:jte, its:ite] = qa[km, jts:jte, its:ite] + 0.0
    
    # interpolation to regular point
    qn[:,:,:] = 0.
    dzi[0:km, jts:jte, its:ite] = zi[1:km+1, jts:jte, its:ite] - zi[0:km, jts:jte, its:ite]
    dzi[km, jts:jte, its:ite] = zi[km, jts:jte, its:ite] - zi[km-1, jts:jte, its:ite]
    
    # interpolation based on torch broadcasting
    zi_low = zi[0:km,:,:]
    za_low = za[0:km,:,:]
    
    zi_high = torch.cat([zi[1:km,:,:],zi[km-1:km,:,:]+3800.],dim=0) # 随意设最高层厚度3800
    za_high = torch.cat([za[1:km,:,:],za[km-1:km,:,:]+3800.],dim=0) # 随意设最高层厚度3800
    
    min_high = torch.minimum(zi_high.unsqueeze(1), za_high.unsqueeze(0))  # 重叠区间的上限
    max_low = torch.maximum(zi_low.unsqueeze(1), za_low.unsqueeze(0))      # 重叠区间的下限
    overlap = torch.maximum(torch.tensor(0.0), min_high - max_low)  # 重叠长度（非负）
    
    qa[km-1,:,:] = 0.0
    qa_by_overlap = qa[0:km,:,:].unsqueeze(0) * overlap
    qq_new = qa_by_overlap.sum(dim=1)
    #print("in nislfv rain",qq_new[2,160,12],dz[2,160,12],qa[2,160,12],overlap.shape, zi[:,160,12], za[:,160,12])
    qq_new = qq_new[0:km,:,:] / dz[0:km,:,:]
    qn[0:km, jts:jte, its:ite] = qq_new[0:km,jts:jte,its:ite] + 0.0
    
    del min_high
    del max_low
    del overlap
    del qa_by_overlap
    del qq_new
    
    for k in range(0,km):
        condition1 = za[k, jts:jte, its:ite] < 0.0 
        condition2 = za[k+1, jts:jte, its:ite] < 0.0
        condition1 = condition1 & condition2
        precip[jts:jte, its:ite] = torch.where(condition1, precip[jts:jte, its:ite] + 
                qa[k, jts:jte, its:ite] * dza[k, jts:jte, its:ite], precip[jts:jte, its:ite])
        condition3 = za[k, jts:jte, its:ite] < 0.0 
        condition4 = za[k+1, jts:jte, its:ite] > 0.0
        condition3 = condition3 & condition4
        precip[jts:jte, its:ite] = torch.where(condition3, precip[jts:jte, its:ite] + 
                qa[k, jts:jte, its:ite] * (0.0 - za[k, jts:jte, its:ite]), precip[jts:jte, its:ite])
    condition = allold[jts:jte, its:ite] > 0.
    precip[jts:jte, its:ite] = torch.where(condition, precip[jts:jte, its:ite], 0.)
    condition_e = condition.repeat(nzall-1,1,1)
    rql[0:km, jts:jte, its:ite] = torch.where(condition_e, qn[0:km, jts:jte, its:ite], 0.)
    return precip, rql

def nislfv_rain_plm6(km,denl,denfacl,tkl,dzl,wwl,rql,rql2, precip1, precip2,dt,id,iter):
    
    allold = torch.zeros((nyall,nxall)).to(device)
    wd = torch.zeros((nzall,nyall,nxall)).to(device)
    tmp = torch.zeros((nzall,nyall,nxall)).to(device)
    tmp1 = torch.zeros((nzall,nyall,nxall)).to(device)
    tmp2 = torch.zeros((nzall,nyall,nxall)).to(device)
    tmp3 = torch.zeros((nzall,nyall,nxall)).to(device)
    wa = torch.zeros((nzall,nyall,nxall)).to(device)
    was = torch.zeros((nzall,nyall,nxall)).to(device)
    qr = torch.zeros((nzall,nyall,nxall)).to(device)
    decfl = torch.zeros((nzall,nyall,nxall)).to(device)
    dip = torch.zeros((nzall,nyall,nxall)).to(device)
    dim = torch.zeros((nzall,nyall,nxall)).to(device)
    
    za = torch.zeros((nzall+1,nyall,nxall)).to(device)
    zi = torch.zeros((nzall+1,nyall,nxall)).to(device)
    wi = torch.zeros((nzall+1,nyall,nxall)).to(device)
    dza = torch.zeros((nzall+1,nyall,nxall)).to(device)
    qa = torch.zeros((nzall+1,nyall,nxall)).to(device)
    qmi = torch.zeros((nzall+1,nyall,nxall)).to(device)
    qpi = torch.zeros((nzall+1,nyall,nxall)).to(device)
    
    qa2 = torch.zeros((nzall+1,nyall,nxall)).to(device)
    wa2 = torch.zeros((nzall,nyall,nxall)).to(device)
    qr2 = torch.zeros((nzall,nyall,nxall)).to(device)
    
    qn = torch.zeros((nzall,nyall,nxall)).to(device)
    dzi = torch.zeros((nzall+1,nyall,nxall)).to(device)
    
    precip = torch.zeros((nyall,nxall)).to(device)
    precip1 = torch.zeros((nyall,nxall)).to(device)
    precip2 = torch.zeros((nyall,nxall)).to(device)
    
    dz = torch.zeros((nzall,nyall,nxall)).to(device)
    qq = torch.zeros((nzall,nyall,nxall)).to(device)
    qq2 = torch.zeros((nzall,nyall,nxall)).to(device)
    ww = torch.zeros((nzall,nyall,nxall)).to(device)
    den = torch.zeros((nzall,nyall,nxall)).to(device)
    denfac = torch.zeros((nzall,nyall,nxall)).to(device)
    tk = torch.zeros((nzall,nyall,nxall)).to(device)
    
    dz[kts:kte, jts:jte, its:ite] = dzl[kts:kte, jts:jte, its:ite] + 0.0
    qq[kts:kte, jts:jte, its:ite] = rql[kts:kte, jts:jte, its:ite] + 0.0
    qq2[kts:kte, jts:jte, its:ite] = rql2[kts:kte, jts:jte, its:ite] + 0.0
    ww[kts:kte, jts:jte, its:ite] = wwl[kts:kte, jts:jte, its:ite] + 0.0
    den[kts:kte, jts:jte, its:ite] = denl[kts:kte, jts:jte, its:ite] + 0.0
    denfac[kts:kte, jts:jte, its:ite] = denfacl[kts:kte, jts:jte, its:ite] + 0.0
    tk[kts:kte, jts:jte, its:ite] = tkl[kts:kte, jts:jte, its:ite] + 0.0
    
    allold[jts:jte, its:ite] = qq[kts:kte, jts:jte, its:ite].sum(dim = 0) + \
                               qq2[kts:kte, jts:jte, its:ite].sum(dim = 0)
    zi[0, jts:jte, its:ite] = 0.0
    
    for k in range(0,km):
        zi[k+1, jts:jte, its:ite] = zi[k, jts:jte, its:ite] + dz[k, jts:jte, its:ite]
    wd[0:km, jts:jte, its:ite] = ww[0:km, jts:jte, its:ite] + 0.0
    
    n=1
    if iter == 0:
        n = 0
    while n<= iter+1:
        # 2nd order interpolation
        wi[0, jts:jte, its:ite] = ww[0, jts:jte, its:ite] + 0.0
        wi[km, jts:jte, its:ite] = ww[km-1, jts:jte, its:ite] + 0.0
        for k in range(1,km):
            wi[k ,jts:jte, its:ite] = (ww[k, jts:jte, its:ite] * dz[k-1, jts:jte, its:ite] + 
                                        ww[k-1, jts:jte, its:ite] * dz[k, jts:jte, its:ite]) / \
                                       (dz[k-1, jts:jte, its:ite] + dz[k, jts:jte, its:ite])
        # 3rd order interpolation
        fa1 = 9./16.
        fa2 = 1./16.
        wi[0, jts:jte, its:ite] = ww[0, jts:jte, its:ite] + 0.0
        wi[1, jts:jte, its:ite] = 0.5 * (ww[1, jts:jte, its:ite] + ww[0, jts:jte, its:ite])
        for k in range(2, km-1):
            wi[k, jts:jte, its:ite] = fa1 * (ww[k, jts:jte, its:ite] + ww[k-1, jts:jte, its:ite]) - \
                                       fa2 * (ww[k+1, jts:jte, its:ite] + ww[k-2, jts:jte, its:ite])
        wi[km-1, jts:jte, its:ite] = 0.5 * (ww[km-1, jts:jte, its:ite] + ww[km-2, jts:jte, its:ite])
        wi[km, jts:jte, its:ite] = ww[km-1, jts:jte, its:ite] + 0.0
        
        wi[1:km, jts:jte, its:ite] = torch.where(ww[1:km, jts:jte, its:ite] == 0.0, 
                ww[0:km-1, jts:jte, its:ite], wi[1:km, jts:jte, its:ite])
        # diffusivity of wi
        #decfl[0:km, jts:jte, its:ite] = (wi[1:km+1, jts:jte, its:ite] - wi[0:km, jts:jte, its:ite]) * \
        #        dt / dz[0:km, jts:jte, its:ite]
        for k in range(km-1,-1,-1):
            decfl[k, jts:jte, its:ite] = (wi[k+1, jts:jte, its:ite] - wi[k, jts:jte, its:ite]) * \
                dt / dz[k, jts:jte, its:ite]
            wi[k, jts:jte, its:ite] = torch.where(decfl[k, jts:jte, its:ite] > 0.05, 
                wi[k+1, jts:jte, its:ite] - 0.05 * dzl[k, jts:jte, its:ite] / dt, wi[k, jts:jte, its:ite])
        #
        za[0:km+1, jts:jte, its:ite] = zi[0:km+1, jts:jte, its:ite] - wi[0:km+1, jts:jte, its:ite] * dt
        dza[0:km, jts:jte, its:ite] = za[1:km+1, jts:jte, its:ite] - za[0:km, jts:jte, its:ite]
        dza[km, jts:jte, its:ite] = zi[km, jts:jte, its:ite] - za[km, jts:jte, its:ite]
        #
        qa[0:km, jts:jte, its:ite] = qq[0:km, jts:jte, its:ite] * dz[0:km, jts:jte, its:ite] / \
                                     dza[0:km, jts:jte, its:ite]
        #qa[0:km, jts:jte, its:ite] = qq[0:km, jts:jte, its:ite] + 0.0
        qr[0:km, jts:jte, its:ite] = qa[0:km, jts:jte, its:ite] / den[0:km, jts:jte, its:ite]
        qa2[0:km, jts:jte, its:ite] = qq2[0:km, jts:jte, its:ite] * dz[0:km, jts:jte, its:ite] / \
                dza[0:km, jts:jte, its:ite]
        #qa2[0:km, jts:jte, its:ite] = qq2[0:km, jts:jte, its:ite] + 0.0
        qr2[0:km, jts:jte, its:ite] = qa2[0:km, jts:jte, its:ite] / den[0:km, jts:jte, its:ite]
        qa[km, jts:jte, its:ite] = 0.0
        qa2[km, jts:jte, its:ite] = 0.0
        
        if n <= iter:
            tmp, tmp1, tmp2, tmp3, wa = slope_snow(qr, den, denfac, tk, tmp, tmp1, tmp2, tmp3, wa, its, ite, kts, km)
            tmp, tmp1, tmp2, tmp3, wa2 = slope_graup(qr2, den, denfac, tk, tmp, tmp1, tmp2, tmp3, wa2, its, ite, kts, km)
            tmp[kts:kte, jts:jte, its:ite] = torch.maximum((qr[kts:kte, jts:jte, its:ite] + 
                                                            qr2[kts:kte, jts:jte, its:ite]), torch.tensor(1.e-15))
            condition = tmp[kts:kte, jts:jte, its:ite] > 1.e-15
            wa[kts:kte, jts:jte, its:ite] = torch.where(condition, 
                        (wa[kts:kte, jts:jte, its:ite] * qr[kts:kte, jts:jte, its:ite] + 
                         wa2[kts:kte, jts:jte, its:ite] * qr2[kts:kte, jts:jte, its:ite]) / 
                        tmp[kts:kte, jts:jte, its:ite], 0.)
            if n >= 2:
                wa[0:km, jts:jte, its:ite] = 0.5 * (wa[0:km, jts:jte, its:ite] + was[0:km, jts:jte, its:ite])
            ww[kts:kte, jts:jte, its:ite] = 0.5 * (wd[kts:kte, jts:jte, its:ite] + wa[kts:kte, jts:jte, its:ite])
            was[kts:kte, jts:jte, its:ite] = wa[kts:kte, jts:jte, its:ite] + 0.0
        n = n+1
        if iter == 0:
            n = 2
    
    # estimate values at arrival cell interface with monotone
    for ist in range(0,2):
        if ist == 1:
            qa[:,:,:] = qa2[:,:,:] + 0.0
        precip[:,:] = 0.
    
        dip[1:km, jts:jte, its:ite] = (qa[2:km+1, jts:jte, its:ite] - qa[1:km, jts:jte, its:ite]) / \
                (dza[2:km+1, jts:jte, its:ite] + dza[1:km, jts:jte, its:ite])
        dim[1:km, jts:jte, its:ite] = (qa[1:km, jts:jte, its:ite] - qa[0:km-1, jts:jte, its:ite]) / \
                (dza[0:km-1, jts:jte, its:ite] + dza[1:km, jts:jte, its:ite])
        qpi[1:km, jts:jte, its:ite] = torch.where(dip[1:km, jts:jte, its:ite] * 
                dim[1:km, jts:jte, its:ite] <= 0.0, qa[1:km, jts:jte, its:ite], 
                qa[1:km, jts:jte, its:ite] + 0.5 * (dip[1:km, jts:jte, its:ite] + 
                dim[1:km, jts:jte, its:ite]) * dza[1:km, jts:jte, its:ite])
        qmi[1:km, jts:jte, its:ite] = torch.where(dip[1:km, jts:jte, its:ite] * 
                dim[1:km, jts:jte, its:ite] <= 0.0, qa[1:km, jts:jte, its:ite], 
                2.0 * qa[1:km, jts:jte, its:ite] - qpi[1:km, jts:jte, its:ite])
        condition = qpi[1:km, jts:jte, its:ite] < 0.0 
        condition1 = qmi[1:km, jts:jte, its:ite] < 0.0
        condition = condition | condition1
        qpi[1:km, jts:jte, its:ite] = torch.where(condition, qa[1:km, jts:jte, its:ite], qpi[1:km, jts:jte, its:ite])
        qmi[1:km, jts:jte, its:ite] = torch.where(condition, qa[1:km, jts:jte, its:ite], qmi[1:km, jts:jte, its:ite])
        qpi[0, jts:jte, its:ite] = qa[0, jts:jte, its:ite] + 0.0
        qmi[0, jts:jte, its:ite] = qa[0, jts:jte, its:ite] + 0.0
        qpi[km, jts:jte, its:ite] = qa[km, jts:jte, its:ite] + 0.0
        qmi[km, jts:jte, its:ite] = qa[km, jts:jte, its:ite] + 0.0
        
        # interpolation to regular point
        qn[:,:,:] = 0.
        #if zi >= za :...
        dzi[0:km, jts:jte, its:ite] = zi[1:km+1, jts:jte, its:ite] - zi[0:km, jts:jte, its:ite]
        dzi[km, jts:jte, its:ite] = zi[km, jts:jte, its:ite] - zi[km-1, jts:jte, its:ite]
        
        # interpolation based on torch broadcasting
        zi_low = zi[0:km,:,:]
        za_low = za[0:km,:,:]
        
        zi_high = torch.cat([zi[1:km,:,:],zi[km-1:km,:,:]+3800.],dim=0) # 随意设最高层厚度3800
        za_high = torch.cat([za[1:km,:,:],za[km-1:km,:,:]+3800.],dim=0) # 随意设最高层厚度3800
        #print(zi_high.shape,zi.shape,km)
        
        min_high = torch.minimum(zi_high.unsqueeze(1), za_high.unsqueeze(0))  # 重叠区间的上限
        max_low = torch.maximum(zi_low.unsqueeze(1), za_low.unsqueeze(0))      # 重叠区间的下限
        overlap = torch.maximum(torch.tensor(0.0), min_high - max_low)  # 重叠长度（非负
        
        qa[km-1,:,:] = 0.0
        qa_by_overlap = qa[0:km,:,:].unsqueeze(0) * overlap
        qq_new = qa_by_overlap.sum(dim=1)
        qq_new = qq_new[0:km,:,:] / dz[0:km,:,:]
        
        qn[0:km, jts:jte, its:ite] = qq_new[0:km,jts:jte,its:ite] + 0.0
        
        del min_high
        del max_low
        del overlap
        del qa_by_overlap
        del qq_new
        
        for k in range(0,km):
            condition1 = za[k, jts:jte, its:ite] < 0.0 
            condition3 = za[k+1, jts:jte, its:ite] < 0.0
            condition1 = condition1 & condition3
            precip[jts:jte, its:ite] = torch.where(condition1, precip[jts:jte, its:ite] + 
                    qa[k, jts:jte, its:ite] * dza[k, jts:jte, its:ite], precip[jts:jte, its:ite])
            condition2 = za[k, jts:jte, its:ite] < 0.0
            condition4 = za[k+1, jts:jte, its:ite] > 0.0
            condition2 = condition2 & condition4
            precip[jts:jte, its:ite] = torch.where(condition2, precip[jts:jte, its:ite] + 
                    qa[k, jts:jte, its:ite] * (0.0 - za[k, jts:jte, its:ite]), precip[jts:jte, its:ite])
        condition = allold[jts:jte, its:ite] > 0.
        condition_e = condition.repeat(nzall-1,1,1)
        if ist == 0:
            precip1[jts:jte, its:ite] = torch.where(condition, precip[jts:jte, its:ite], 0.)
            rql[0:km, jts:jte, its:ite] = torch.where(condition_e, qn[0:km, jts:jte, its:ite], 0.)
        if ist == 1:
            precip2[jts:jte, its:ite] = torch.where(condition, precip[jts:jte, its:ite], 0.)
            rql2[0:km, jts:jte, its:ite] = torch.where(condition_e, qn[0:km, jts:jte, its:ite], 0.)
    return precip1, precip2, rql, rql2

def nislfv_rain_plm_oldver(km,denl,denfacl,tkl,dzl,wwl,rql,precip,dt,id,iter):
    
    allold = torch.zeros((nyall,nxall)).to(device)
    wd = torch.zeros((nzall,nyall,nxall)).to(device)
    tmp = torch.zeros((nzall,nyall,nxall)).to(device)
    tmp1 = torch.zeros((nzall,nyall,nxall)).to(device)
    tmp2 = torch.zeros((nzall,nyall,nxall)).to(device)
    tmp3 = torch.zeros((nzall,nyall,nxall)).to(device)
    wa = torch.zeros((nzall,nyall,nxall)).to(device)
    was = torch.zeros((nzall,nyall,nxall)).to(device)
    qr = torch.zeros((nzall,nyall,nxall)).to(device)
    decfl = torch.zeros((nzall,nyall,nxall)).to(device)
    dip = torch.zeros((nzall,nyall,nxall)).to(device)
    dim = torch.zeros((nzall,nyall,nxall)).to(device)
    
    za = torch.zeros((nzall+1,nyall,nxall)).to(device)
    zi = torch.zeros((nzall+1,nyall,nxall)).to(device)
    wi = torch.zeros((nzall+1,nyall,nxall)).to(device)
    dza = torch.zeros((nzall+1,nyall,nxall)).to(device)
    qa = torch.zeros((nzall+1,nyall,nxall)).to(device)
    qmi = torch.zeros((nzall+1,nyall,nxall)).to(device)
    qpi = torch.zeros((nzall+1,nyall,nxall)).to(device)
    
    qn = torch.zeros((nzall,nyall,nxall)).to(device)
    dzi = torch.zeros((nzall+1,nyall,nxall)).to(device)
    dz = torch.zeros((nzall,nyall,nxall)).to(device)
    qq = torch.zeros((nzall,nyall,nxall)).to(device)
    ww = torch.zeros((nzall,nyall,nxall)).to(device)
    den = torch.zeros((nzall,nyall,nxall)).to(device)
    denfac = torch.zeros((nzall,nyall,nxall)).to(device)
    tk = torch.zeros((nzall,nyall,nxall)).to(device)
    
    dz[kts:kte, jts:jte, its:ite] = dzl[kts:kte, jts:jte, its:ite] + 0.0
    qq[kts:kte, jts:jte, its:ite] = rql[kts:kte, jts:jte, its:ite] + 0.0
    ww[kts:kte, jts:jte, its:ite] = wwl[kts:kte, jts:jte, its:ite] + 0.0
    den[kts:kte, jts:jte, its:ite] = denl[kts:kte, jts:jte, its:ite] + 0.0
    denfac[kts:kte, jts:jte, its:ite] = denfacl[kts:kte, jts:jte, its:ite] + 0.0
    tk[kts:kte, jts:jte, its:ite] = tkl[kts:kte, jts:jte, its:ite] + 0.0
    
    allold[jts:jte, its:ite] = qq[kts:kte, jts:jte, its:ite].sum(dim = 0)
    zi[0, jts:jte, its:ite] = 0.0
    
    for k in range(0,km):
        zi[k+1, jts:jte, its:ite] = zi[k, jts:jte, its:ite] + dz[k, jts:jte, its:ite]
    wd[0:km, jts:jte, its:ite] = ww[0:km, jts:jte, its:ite] + 0.0
    
    n=1
    if iter == 0:
        n = 0
    while n<= iter+1:
        # 2nd order interpolation
        wi[0, jts:jte, its:ite] = ww[0, jts:jte, its:ite] + 0.0
        wi[km, jts:jte, its:ite] = ww[km-1, jts:jte, its:ite] + 0.0
        for k in range(1,km):
            wi[k ,jts:jte, its:ite] = (ww[k, jts:jte, its:ite] * dz[k-1, jts:jte, its:ite] + 
                                        ww[k-1, jts:jte, its:ite] * dz[k, jts:jte, its:ite]) / \
                                       (dz[k-1, jts:jte, its:ite] + dz[k, jts:jte, its:ite])
        # 3rd order interpolation
        fa1 = 9./16.
        fa2 = 1./16.
        wi[0, jts:jte, its:ite] = ww[0, jts:jte, its:ite] + 0.0
        wi[1, jts:jte, its:ite] = 0.5 * (ww[1, jts:jte, its:ite] + ww[0, jts:jte, its:ite])
        for k in range(2, km-1):
            wi[k, jts:jte, its:ite] = fa1 * (ww[k, jts:jte, its:ite] + ww[k-1, jts:jte, its:ite]) - \
                                       fa2 * (ww[k+1, jts:jte, its:ite] + ww[k-2, jts:jte, its:ite])
        wi[km-1, jts:jte, its:ite] = 0.5 * (ww[km-1, jts:jte, its:ite] + ww[km-2, jts:jte, its:ite])
        wi[km, jts:jte, its:ite] = ww[km-1, jts:jte, its:ite] + 0.0
        
        wi[1:km, jts:jte, its:ite] = torch.where(ww[1:km, jts:jte, its:ite] == 0.0, 
                ww[0:km-1, jts:jte, its:ite], wi[1:km, jts:jte, its:ite])
        # diffusivity of wi
        #decfl[0:km, jts:jte, its:ite] = (wi[1:km+1, jts:jte, its:ite] - wi[0:km, jts:jte, its:ite]) * \
        #        dt / dz[0:km, jts:jte, its:ite]
        for k in range(km-1,-1,-1):
            decfl[k, jts:jte, its:ite] = (wi[k+1, jts:jte, its:ite] - wi[k, jts:jte, its:ite]) * \
                dt / dz[k, jts:jte, its:ite]
            wi[k, jts:jte, its:ite] = torch.where(decfl[k, jts:jte, its:ite] > 0.05, 
                wi[k+1, jts:jte, its:ite] - 0.05 * dz[k, jts:jte, its:ite] /dt, wi[k, jts:jte, its:ite])
        
        #
        za[0:km+1, jts:jte, its:ite] = zi[0:km+1, jts:jte, its:ite] - wi[0:km+1, jts:jte, its:ite] * dt
        dza[0:km, jts:jte, its:ite] = za[1:km+1, jts:jte, its:ite] - za[0:km, jts:jte, its:ite]
        dza[km, jts:jte, its:ite] = zi[km, jts:jte, its:ite] - za[km, jts:jte, its:ite]
        #
        qa[0:km, jts:jte, its:ite] = qq[0:km, jts:jte, its:ite] * dz[0:km, jts:jte, its:ite] / \
                dza[0:km, jts:jte, its:ite]
        
        qr[0:km, jts:jte, its:ite] = qa[0:km, jts:jte, its:ite] / den[0:km, jts:jte, its:ite]
        qa[km, jts:jte, its:ite] = 0.0
        
        if n <= iter:
            tmp, tmp1, tmp2, tmp3, wa = slope_rain(qr, den, denfac, tk, tmp, tmp1, tmp2, tmp3, wa, its, ite, kts, km)
            if n >= 2:
                wa[0:km, jts:jte, its:ite] = 0.5 * (wa[0:km, jts:jte, its:ite] + was[0:km, jts:jte, its:ite])
            ww[kts:kte, jts:jte, its:ite] = 0.5 * (wd[kts:kte, jts:jte, its:ite] + wa[kts:kte, jts:jte, its:ite])
            was[kts:kte, jts:jte, its:ite] = wa[kts:kte, jts:jte, its:ite] + 0.0
        n = n+1
        if iter == 0:
            n = 2
    # estimate values at arrival cell interface with monotone
    dip[1:km, jts:jte, its:ite] = (qa[2:km+1, jts:jte, its:ite] - qa[1:km, jts:jte, its:ite]) / \
            (dza[2:km+1, jts:jte, its:ite] + dza[1:km, jts:jte, its:ite])
    dim[1:km, jts:jte, its:ite] = (qa[1:km, jts:jte, its:ite] - qa[0:km-1, jts:jte, its:ite]) / \
            (dza[0:km-1, jts:jte, its:ite] + dza[1:km, jts:jte, its:ite])
    qpi[1:km, jts:jte, its:ite] = torch.where(dip[1:km, jts:jte, its:ite] * 
            dim[1:km, jts:jte, its:ite] <= 0.0, qa[1:km, jts:jte, its:ite], 
            qa[1:km, jts:jte, its:ite] + 0.5 * (dip[1:km, jts:jte, its:ite] + 
            dim[1:km, jts:jte, its:ite]) * dza[1:km, jts:jte, its:ite])
    qmi[1:km, jts:jte, its:ite] = torch.where(dip[1:km, jts:jte, its:ite] * 
            dim[1:km, jts:jte, its:ite] <= 0.0, qa[1:km, jts:jte, its:ite], 
            2.0 * qa[1:km, jts:jte, its:ite] - qpi[1:km, jts:jte, its:ite])
    condition1 = qpi[1:km, jts:jte, its:ite] < 0.0
    condition2 = qmi[1:km, jts:jte, its:ite] < 0.0
    condition = condition1 | condition2
    qpi[1:km, jts:jte, its:ite] = torch.where(condition, qa[1:km, jts:jte, its:ite], qpi[1:km, jts:jte, its:ite])
    qmi[1:km, jts:jte, its:ite] = torch.where(condition, qa[1:km, jts:jte, its:ite], qmi[1:km, jts:jte, its:ite])
    qpi[0, jts:jte, its:ite] = qa[0, jts:jte, its:ite] + 0.0
    qmi[0, jts:jte, its:ite] = qa[0, jts:jte, its:ite] + 0.0
    qpi[km, jts:jte, its:ite] = qa[km, jts:jte, its:ite] + 0.0
    qmi[km, jts:jte, its:ite] = qa[km, jts:jte, its:ite] + 0.0
    
    # interpolation to regular point
    qn[:,:,:] = 0.
    dzi[0:km, jts:jte, its:ite] = zi[1:km+1, jts:jte, its:ite] - zi[0:km, jts:jte, its:ite]
    dzi[km, jts:jte, its:ite] = zi[km, jts:jte, its:ite] - zi[km-1, jts:jte, its:ite]
    
    # linear interpolation
    for k in range(0,km):   #km
        condition0 = zi[:, jts:jte, its:ite] < za[k, jts:jte, its:ite]
        condition1 = condition0.clone()
        condition1[:,:,:] = False
        condition1[0:km,:,:] = condition0[1:km+1,:,:]
        qntmp1 = qn[0:km, jts:jte, its:ite] * 0.0
        qntmp2 = qn[0:km, jts:jte, its:ite] * 0.0
        condition1 = ~ condition1
        condition2 = condition0 & condition1
        condition2 = condition2[0:km,:,:]
        
        condition3 = condition0.clone()
        condition3[:,:,:] = False
        condition3[1:km+1,:,:] = condition0[0:km,:,:]
        condition4 = ~ condition0
        condition4 = condition4 & condition3
        condition4 = condition4[0:km,:,:]
        qa_e = qa[k,jts:jte, its:ite].repeat(km,1,1)
        za_e = za[k, jts:jte, its:ite].repeat(km,1,1)
        dza_e = dza[k, jts:jte, its:ite].repeat(km,1,1)
        zip1 = zi + 0.0
        zim1 = zi + 0.0
        zip1[0:km, jts:jte, its:ite] = zi[1:km+1, jts:jte, its:ite]
        zim1[1:km+1, jts:jte, its:ite] = zi[0:km, jts:jte, its:ite]
        
        wgt_qn1 = torch.minimum((zip1[0:km,jts:jte,its:ite] - za_e[0:km,:,:])/(zip1[0:km,jts:jte,its:ite] - zi[0:km,jts:jte,its:ite]) * dza_e/dzi[0:km,jts:jte,its:ite],torch.tensor(1.))
        qntmp1 = torch.where(condition2, qa_e * wgt_qn1, qntmp1) 
        wgt_qn2 = torch.minimum((za_e[0:km,:,:] - zim1[0:km,jts:jte,its:ite])/(zi[0:km,jts:jte,its:ite] - zim1[0:km,jts:jte,its:ite]) * dza_e / dzi[0:km,jts:jte,its:ite], torch.tensor(1.))
        qntmp2 = torch.where(condition4, qa_e * wgt_qn2, qntmp2)
        
        # fall below surface situation
        condition0 = zi[0, jts:jte, its:ite] >= za[k, jts:jte, its:ite]
        condition1 = zi[0, jts:jte, its:ite] <= za[k, jts:jte, its:ite] + dza[k, jts:jte, its:ite]
        condition1 = condition0 & condition1
        qntmp1[0,:,:] = torch.where(condition1, qa[k, jts:jte, its:ite] * ((za[k, jts:jte, its:ite] + 
                   dza[k, jts:jte, its:ite] - zi[0, jts:jte, its:ite])/dza[k, jts:jte, its:ite]), qntmp1[0,:,:])
        
        qn[0:km, jts:jte, its:ite] = qn[0:km, jts:jte, its:ite] + qntmp1
        qn[0:km, jts:jte, its:ite] = qn[0:km, jts:jte, its:ite] + qntmp2
    
    for k in range(0,km):
        condition1 = za[k, jts:jte, its:ite] < 0.0 
        condition2 = za[k+1, jts:jte, its:ite] < 0.0
        condition1 = condition1 & condition2
        precip[jts:jte, its:ite] = torch.where(condition1, precip[jts:jte, its:ite] + 
                qa[k, jts:jte, its:ite] * dza[k, jts:jte, its:ite], precip[jts:jte, its:ite])
        condition3 = za[k, jts:jte, its:ite] < 0.0 
        condition4 = za[k+1, jts:jte, its:ite] > 0.0
        condition3 = condition3 & condition4
        precip[jts:jte, its:ite] = torch.where(condition3, precip[jts:jte, its:ite] + 
                qa[k, jts:jte, its:ite] * (0.0 - za[k, jts:jte, its:ite]), precip[jts:jte, its:ite])
    condition = allold[jts:jte, its:ite] > 0.
    precip[jts:jte, its:ite] = torch.where(condition, precip[jts:jte, its:ite], 0.)
    condition_e = condition.repeat(nzall-1,1,1)
    rql[0:km, jts:jte, its:ite] = torch.where(condition_e, qn[0:km, jts:jte, its:ite], 0.)
    return precip, rql

def nislfv_rain_plm6_oldver(km,denl,denfacl,tkl,dzl,wwl,rql,rql2, precip1, precip2,dt,id,iter):
    
    allold = torch.zeros((nyall,nxall)).to(device)
    wd = torch.zeros((nzall,nyall,nxall)).to(device)
    tmp = torch.zeros((nzall,nyall,nxall)).to(device)
    tmp1 = torch.zeros((nzall,nyall,nxall)).to(device)
    tmp2 = torch.zeros((nzall,nyall,nxall)).to(device)
    tmp3 = torch.zeros((nzall,nyall,nxall)).to(device)
    wa = torch.zeros((nzall,nyall,nxall)).to(device)
    was = torch.zeros((nzall,nyall,nxall)).to(device)
    qr = torch.zeros((nzall,nyall,nxall)).to(device)
    decfl = torch.zeros((nzall,nyall,nxall)).to(device)
    dip = torch.zeros((nzall,nyall,nxall)).to(device)
    dim = torch.zeros((nzall,nyall,nxall)).to(device)
    
    za = torch.zeros((nzall+1,nyall,nxall)).to(device)
    zi = torch.zeros((nzall+1,nyall,nxall)).to(device)
    wi = torch.zeros((nzall+1,nyall,nxall)).to(device)
    dza = torch.zeros((nzall+1,nyall,nxall)).to(device)
    qa = torch.zeros((nzall+1,nyall,nxall)).to(device)
    qmi = torch.zeros((nzall+1,nyall,nxall)).to(device)
    qpi = torch.zeros((nzall+1,nyall,nxall)).to(device)
    
    qa2 = torch.zeros((nzall+1,nyall,nxall)).to(device)
    wa2 = torch.zeros((nzall,nyall,nxall)).to(device)
    qr2 = torch.zeros((nzall,nyall,nxall)).to(device)
    
    qn = torch.zeros((nzall,nyall,nxall)).to(device)
    dzi = torch.zeros((nzall+1,nyall,nxall)).to(device)
    
    precip = torch.zeros((nyall,nxall)).to(device)
    precip1 = torch.zeros((nyall,nxall)).to(device)
    precip2 = torch.zeros((nyall,nxall)).to(device)
    
    dz = torch.zeros((nzall,nyall,nxall)).to(device)
    qq = torch.zeros((nzall,nyall,nxall)).to(device)
    qq2 = torch.zeros((nzall,nyall,nxall)).to(device)
    ww = torch.zeros((nzall,nyall,nxall)).to(device)
    den = torch.zeros((nzall,nyall,nxall)).to(device)
    denfac = torch.zeros((nzall,nyall,nxall)).to(device)
    tk = torch.zeros((nzall,nyall,nxall)).to(device)
    
    dz[kts:kte, jts:jte, its:ite] = dzl[kts:kte, jts:jte, its:ite] + 0.0
    qq[kts:kte, jts:jte, its:ite] = rql[kts:kte, jts:jte, its:ite] + 0.0
    qq2[kts:kte, jts:jte, its:ite] = rql2[kts:kte, jts:jte, its:ite] + 0.0
    ww[kts:kte, jts:jte, its:ite] = wwl[kts:kte, jts:jte, its:ite] + 0.0
    den[kts:kte, jts:jte, its:ite] = denl[kts:kte, jts:jte, its:ite] + 0.0
    denfac[kts:kte, jts:jte, its:ite] = denfacl[kts:kte, jts:jte, its:ite] + 0.0
    tk[kts:kte, jts:jte, its:ite] = tkl[kts:kte, jts:jte, its:ite] + 0.0
    
    allold[jts:jte, its:ite] = qq[kts:kte, jts:jte, its:ite].sum(dim = 0) + \
                               qq2[kts:kte, jts:jte, its:ite].sum(dim = 0)
    zi[0, jts:jte, its:ite] = 0.0
    
    for k in range(0,km):
        zi[k+1, jts:jte, its:ite] = zi[k, jts:jte, its:ite] + dz[k, jts:jte, its:ite]
    wd[0:km, jts:jte, its:ite] = ww[0:km, jts:jte, its:ite] + 0.0
    
    n=1
    if iter == 0:
        n = 0
    while n<= iter+1:
        # 2nd order interpolation
        wi[0, jts:jte, its:ite] = ww[0, jts:jte, its:ite] + 0.0
        wi[km, jts:jte, its:ite] = ww[km-1, jts:jte, its:ite] + 0.0
        for k in range(1,km):
            wi[k ,jts:jte, its:ite] = (ww[k, jts:jte, its:ite] * dz[k-1, jts:jte, its:ite] + 
                                        ww[k-1, jts:jte, its:ite] * dz[k, jts:jte, its:ite]) / \
                                       (dz[k-1, jts:jte, its:ite] + dz[k, jts:jte, its:ite])
        # 3rd order interpolation
        fa1 = 9./16.
        fa2 = 1./16.
        wi[0, jts:jte, its:ite] = ww[0, jts:jte, its:ite] + 0.0
        wi[1, jts:jte, its:ite] = 0.5 * (ww[1, jts:jte, its:ite] + ww[0, jts:jte, its:ite])
        for k in range(2, km-1):
            wi[k, jts:jte, its:ite] = fa1 * (ww[k, jts:jte, its:ite] + ww[k-1, jts:jte, its:ite]) - \
                                       fa2 * (ww[k+1, jts:jte, its:ite] + ww[k-2, jts:jte, its:ite])
        wi[km-1, jts:jte, its:ite] = 0.5 * (ww[km-1, jts:jte, its:ite] + ww[km-2, jts:jte, its:ite])
        wi[km, jts:jte, its:ite] = ww[km-1, jts:jte, its:ite] + 0.0
        
        wi[1:km, jts:jte, its:ite] = torch.where(ww[1:km, jts:jte, its:ite] == 0.0, 
                ww[0:km-1, jts:jte, its:ite], wi[1:km, jts:jte, its:ite])
        # diffusivity of wi
        #decfl[0:km, jts:jte, its:ite] = (wi[1:km+1, jts:jte, its:ite] - wi[0:km, jts:jte, its:ite]) * \
        #        dt / dz[0:km, jts:jte, its:ite]
        for k in range(km-1,-1,-1):
            decfl[k, jts:jte, its:ite] = (wi[k+1, jts:jte, its:ite] - wi[k, jts:jte, its:ite]) * \
                dt / dz[k, jts:jte, its:ite]
            wi[k, jts:jte, its:ite] = torch.where(decfl[k, jts:jte, its:ite] > 0.05, 
                wi[k+1, jts:jte, its:ite] - 0.05 * dzl[k, jts:jte, its:ite] / dt, wi[k, jts:jte, its:ite])
        #
        za[0:km+1, jts:jte, its:ite] = zi[0:km+1, jts:jte, its:ite] - wi[0:km+1, jts:jte, its:ite] * dt
        dza[0:km, jts:jte, its:ite] = za[1:km+1, jts:jte, its:ite] - za[0:km, jts:jte, its:ite]
        dza[km, jts:jte, its:ite] = zi[km, jts:jte, its:ite] - za[km, jts:jte, its:ite]
        #
        qa[0:km, jts:jte, its:ite] = qq[0:km, jts:jte, its:ite] * dz[0:km, jts:jte, its:ite] / \
                                     dza[0:km, jts:jte, its:ite]
        #qa[0:km, jts:jte, its:ite] = qq[0:km, jts:jte, its:ite] + 0.0
        qr[0:km, jts:jte, its:ite] = qa[0:km, jts:jte, its:ite] / den[0:km, jts:jte, its:ite]
        qa2[0:km, jts:jte, its:ite] = qq2[0:km, jts:jte, its:ite] * dz[0:km, jts:jte, its:ite] / \
                dza[0:km, jts:jte, its:ite]
        #qa2[0:km, jts:jte, its:ite] = qq2[0:km, jts:jte, its:ite] + 0.0
        qr2[0:km, jts:jte, its:ite] = qa2[0:km, jts:jte, its:ite] / den[0:km, jts:jte, its:ite]
        qa[km, jts:jte, its:ite] = 0.0
        qa2[km, jts:jte, its:ite] = 0.0
        
        if n <= iter:
            tmp, tmp1, tmp2, tmp3, wa = slope_snow(qr, den, denfac, tk, tmp, tmp1, tmp2, tmp3, wa, its, ite, kts, km)
            tmp, tmp1, tmp2, tmp3, wa2 = slope_graup(qr2, den, denfac, tk, tmp, tmp1, tmp2, tmp3, wa2, its, ite, kts, km)
            tmp[kts:kte, jts:jte, its:ite] = torch.maximum((qr[kts:kte, jts:jte, its:ite] + 
                                                            qr2[kts:kte, jts:jte, its:ite]), torch.tensor(1.e-15))
            condition = tmp[kts:kte, jts:jte, its:ite] > 1.e-15
            wa[kts:kte, jts:jte, its:ite] = torch.where(condition, 
                        (wa[kts:kte, jts:jte, its:ite] * qr[kts:kte, jts:jte, its:ite] + 
                         wa2[kts:kte, jts:jte, its:ite] * qr2[kts:kte, jts:jte, its:ite]) / 
                        tmp[kts:kte, jts:jte, its:ite], 0.)
            if n >= 2:
                wa[0:km, jts:jte, its:ite] = 0.5 * (wa[0:km, jts:jte, its:ite] + was[0:km, jts:jte, its:ite])
            ww[kts:kte, jts:jte, its:ite] = 0.5 * (wd[kts:kte, jts:jte, its:ite] + wa[kts:kte, jts:jte, its:ite])
            was[kts:kte, jts:jte, its:ite] = wa[kts:kte, jts:jte, its:ite] + 0.0
        n = n+1
        if iter == 0:
            n = 2
    
    # estimate values at arrival cell interface with monotone
    for ist in range(0,2):
        if ist == 1:
            qa[:,:,:] = qa2[:,:,:] + 0.0
        precip[:,:] = 0.
    
        dip[1:km, jts:jte, its:ite] = (qa[2:km+1, jts:jte, its:ite] - qa[1:km, jts:jte, its:ite]) / \
                (dza[2:km+1, jts:jte, its:ite] + dza[1:km, jts:jte, its:ite])
        dim[1:km, jts:jte, its:ite] = (qa[1:km, jts:jte, its:ite] - qa[0:km-1, jts:jte, its:ite]) / \
                (dza[0:km-1, jts:jte, its:ite] + dza[1:km, jts:jte, its:ite])
        qpi[1:km, jts:jte, its:ite] = torch.where(dip[1:km, jts:jte, its:ite] * 
                dim[1:km, jts:jte, its:ite] <= 0.0, qa[1:km, jts:jte, its:ite], 
                qa[1:km, jts:jte, its:ite] + 0.5 * (dip[1:km, jts:jte, its:ite] + 
                dim[1:km, jts:jte, its:ite]) * dza[1:km, jts:jte, its:ite])
        qmi[1:km, jts:jte, its:ite] = torch.where(dip[1:km, jts:jte, its:ite] * 
                dim[1:km, jts:jte, its:ite] <= 0.0, qa[1:km, jts:jte, its:ite], 
                2.0 * qa[1:km, jts:jte, its:ite] - qpi[1:km, jts:jte, its:ite])
        condition = qpi[1:km, jts:jte, its:ite] < 0.0 
        condition1 = qmi[1:km, jts:jte, its:ite] < 0.0
        condition = condition | condition1
        qpi[1:km, jts:jte, its:ite] = torch.where(condition, qa[1:km, jts:jte, its:ite], qpi[1:km, jts:jte, its:ite])
        qmi[1:km, jts:jte, its:ite] = torch.where(condition, qa[1:km, jts:jte, its:ite], qmi[1:km, jts:jte, its:ite])
        qpi[0, jts:jte, its:ite] = qa[0, jts:jte, its:ite] + 0.0
        qmi[0, jts:jte, its:ite] = qa[0, jts:jte, its:ite] + 0.0
        qpi[km, jts:jte, its:ite] = qa[km, jts:jte, its:ite] + 0.0
        qmi[km, jts:jte, its:ite] = qa[km, jts:jte, its:ite] + 0.0
        
        # interpolation to regular point
        qn[:,:,:] = 0.
        #if zi >= za :...
        dzi[0:km, jts:jte, its:ite] = zi[1:km+1, jts:jte, its:ite] - zi[0:km, jts:jte, its:ite]
        dzi[km, jts:jte, its:ite] = zi[km, jts:jte, its:ite] - zi[km-1, jts:jte, its:ite]
        
        for k in range(0,km):
            condition0 = zi[:, jts:jte, its:ite] < za[k, jts:jte, its:ite]
            condition1 = condition0.clone()
            condition1[:,:,:] = False
            condition1[0:km,:,:] = condition0[1:km+1,:,:]
            qntmp1 = qn[0:km, jts:jte, its:ite] * 0.0
            qntmp2 = qn[0:km, jts:jte, its:ite] * 0.0
            condition1 = ~ condition1
            condition2 = condition0 & condition1
            condition2 = condition2[0:km,:,:]
            
            condition3 = condition0.clone()
            condition3[:,:,:] = False
            condition3[1:km+1,:,:] = condition0[0:km,:,:]
            condition4 = ~ condition0
            condition4 = condition4 & condition3
            condition4 = condition4[0:km,:,:]
            qa_e = qa[k,jts:jte, its:ite].repeat(km,1,1)
            za_e = za[k, jts:jte, its:ite].repeat(km,1,1)
            dza_e = dza[k, jts:jte, its:ite].repeat(km,1,1)
            zip1 = zi + 0.0
            zim1 = zi + 0.0
            zip1[0:km, jts:jte, its:ite] = zi[1:km+1, jts:jte, its:ite]
            zim1[1:km+1, jts:jte, its:ite] = zi[0:km, jts:jte, its:ite]
            
            wgt_qn1 = torch.minimum((zip1[0:km,jts:jte,its:ite] - za_e[0:km,:,:])/(zip1[0:km,jts:jte,its:ite] - zi[0:km,jts:jte,its:ite]) * dza_e/dzi[0:km,jts:jte,its:ite],torch.tensor(1.))
            qntmp1 = torch.where(condition2, qa_e * wgt_qn1, qntmp1) 
            wgt_qn2 = torch.minimum((za_e[0:km,:,:] - zim1[0:km,jts:jte,its:ite])/(zi[0:km,jts:jte,its:ite] - zim1[0:km,jts:jte,its:ite]) * dza_e/dzi[0:km,jts:jte,its:ite], torch.tensor(1.))
            qntmp2 = torch.where(condition4, qa_e * wgt_qn2, qntmp2)
            
            # fall below surface situation
            condition0 = zi[0, jts:jte, its:ite] >= za[k, jts:jte, its:ite]
            condition1 = zi[0, jts:jte, its:ite] <= za[k, jts:jte, its:ite] + dza[k, jts:jte, its:ite]
            condition1 = condition0 & condition1
            qntmp1[0,:,:] = torch.where(condition1, qa[k, jts:jte, its:ite] * ((za[k, jts:jte, its:ite] +
                        dza[k, jts:jte, its:ite] - zi[0, jts:jte, its:ite])/dza[k, jts:jte, its:ite]), qntmp1[0,:,:]) 
            
            qn[0:km, jts:jte, its:ite] = qn[0:km, jts:jte, its:ite] + qntmp1
            qn[0:km, jts:jte, its:ite] = qn[0:km, jts:jte, its:ite] + qntmp2

        for k in range(0,km):
            condition1 = za[k, jts:jte, its:ite] < 0.0 
            condition3 = za[k+1, jts:jte, its:ite] < 0.0
            condition1 = condition1 & condition3
            precip[jts:jte, its:ite] = torch.where(condition1, precip[jts:jte, its:ite] + 
                    qa[k, jts:jte, its:ite] * dza[k, jts:jte, its:ite], precip[jts:jte, its:ite])
            condition2 = za[k, jts:jte, its:ite] < 0.0
            condition4 = za[k+1, jts:jte, its:ite] > 0.0
            condition2 = condition2 & condition4
            precip[jts:jte, its:ite] = torch.where(condition2, precip[jts:jte, its:ite] + 
                    qa[k, jts:jte, its:ite] * (0.0 - za[k, jts:jte, its:ite]), precip[jts:jte, its:ite])
        condition = allold[jts:jte, its:ite] > 0.
        condition_e = condition.repeat(nzall-1,1,1)
        if ist == 0:
            precip1[jts:jte, its:ite] = torch.where(condition, precip[jts:jte, its:ite], 0.)
            rql[0:km, jts:jte, its:ite] = torch.where(condition_e, qn[0:km, jts:jte, its:ite], 0.)
        if ist == 1:
            precip2[jts:jte, its:ite] = torch.where(condition, precip[jts:jte, its:ite], 0.)
            rql2[0:km, jts:jte, its:ite] = torch.where(condition_e, qn[0:km, jts:jte, its:ite], 0.)
    return precip1, precip2, rql, rql2
    
def slope_rain(qrs,den,denfac,t,rslope,rslopeb,rslope2,rslope3,   \
                            vt,its,ite,kts,kte):
    lamdar = lambda x, y: (pidn0r / (x * y)) ** 0.25
    rslope[kts:kte, jts:jte, its:ite] = torch.where(qrs[kts:kte, jts:jte, its:ite] <= qcrmin, 
                rslopermax, 1./lamdar(qrs[kts:kte, jts:jte, its:ite], den[kts:kte, jts:jte, its:ite]))
    rslopeb[kts:kte, jts:jte, its:ite] = torch.where(qrs[kts:kte, jts:jte, its:ite] <= qcrmin, 
                rsloperbmax, rslope[kts:kte, jts:jte, its:ite] ** bvtr)
    rslope2[kts:kte, jts:jte, its:ite] = torch.where(qrs[kts:kte, jts:jte, its:ite] <= qcrmin, 
                rsloper2max, rslope[kts:kte, jts:jte, its:ite] ** 2)
    rslope3[kts:kte, jts:jte, its:ite] = torch.where(qrs[kts:kte, jts:jte, its:ite] <= qcrmin, 
                rsloper3max, rslope[kts:kte, jts:jte, its:ite] ** 3)
    vt[kts:kte, jts:jte, its:ite] = pvtr * rslopeb[kts:kte, jts:jte, its:ite] * \
                denfac[kts:kte, jts:jte, its:ite]
    vt[kts:kte, jts:jte, its:ite] = torch.where(qrs[kts:kte, jts:jte, its:ite] <= 0.0, 
                0.0, vt[kts:kte, jts:jte, its:ite])
    return rslope, rslopeb, rslope2, rslope3, vt

def slope_snow(qrs,den,denfac,t,rslope,rslopeb,rslope2,rslope3,   \
                            vt,its,ite,kts,kte):
    lamdas = lambda x, y, z: (pidn0s * z  /  (x * y)) ** 0.25
    supcol = torch.zeros((nzall,nyall,nxall)).to(device)
    n0sfac = torch.zeros((nzall,nyall,nxall)).to(device)
    
    supcol[kts:kte, jts:jte, its:ite] = t0c - t[kts:kte, jts:jte, its:ite]
    n0sfac[kts:kte, jts:jte, its:ite] = torch.maximum(torch.minimum(torch.exp
            (alpha * supcol[kts:kte, jts:jte, its:ite]), torch.tensor(n0smax/n0s)), torch.tensor(1.))
    
    rslope[kts:kte, jts:jte, its:ite] = torch.where(qrs[kts:kte, jts:jte, its:ite] <= qcrmin, 
                rslopermax, 1./lamdas(qrs[kts:kte, jts:jte, its:ite], den[kts:kte, jts:jte, its:ite], 
                                      n0sfac[kts:kte, jts:jte, its:ite]))
    rslopeb[kts:kte, jts:jte, its:ite] = torch.where(qrs[kts:kte, jts:jte, its:ite] <= qcrmin, 
                rsloperbmax, rslope[kts:kte, jts:jte, its:ite] ** bvts)
    rslope2[kts:kte, jts:jte, its:ite] = torch.where(qrs[kts:kte, jts:jte, its:ite] <= qcrmin, 
                rsloper2max, rslope[kts:kte, jts:jte, its:ite] ** 2)
    rslope3[kts:kte, jts:jte, its:ite] = torch.where(qrs[kts:kte, jts:jte, its:ite] <= qcrmin, 
                rsloper3max, rslope[kts:kte, jts:jte, its:ite] ** 3)
    vt[kts:kte, jts:jte, its:ite] = pvts * rslopeb[kts:kte, jts:jte, its:ite] * \
                denfac[kts:kte, jts:jte, its:ite]
    vt[kts:kte, jts:jte, its:ite] = torch.where(qrs[kts:kte, jts:jte, its:ite] <= 0.0, 
                0.0, vt[kts:kte, jts:jte, its:ite])
    return rslope, rslopeb, rslope2, rslope3, vt

def slope_graup(qrs,den,denfac,t,rslope,rslopeb,rslope2,rslope3,   \
                            vt,its,ite,kts,kte):
    lamdag = lambda x, y: (pidn0g / (x * y)) ** 0.25
    
    rslope[kts:kte, jts:jte, its:ite] = torch.where(qrs[kts:kte, jts:jte, its:ite] <= qcrmin, 
                rslopermax, 1./lamdag(qrs[kts:kte, jts:jte, its:ite], den[kts:kte, jts:jte, its:ite]))
    rslopeb[kts:kte, jts:jte, its:ite] = torch.where(qrs[kts:kte, jts:jte, its:ite] <= qcrmin, 
                rsloperbmax, rslope[kts:kte, jts:jte, its:ite] ** bvtg)
    rslope2[kts:kte, jts:jte, its:ite] = torch.where(qrs[kts:kte, jts:jte, its:ite] <= qcrmin, 
                rsloper2max, rslope[kts:kte, jts:jte, its:ite] ** 2)
    rslope3[kts:kte, jts:jte, its:ite] = torch.where(qrs[kts:kte, jts:jte, its:ite] <= qcrmin, 
                rsloper3max, rslope[kts:kte, jts:jte, its:ite] ** 3)
    vt[kts:kte, jts:jte, its:ite] = pvtg * rslopeb[kts:kte, jts:jte, its:ite] * \
                denfac[kts:kte, jts:jte, its:ite]
    vt[kts:kte, jts:jte, its:ite] = torch.where(qrs[kts:kte, jts:jte, its:ite] <= 0.0, 
                0.0, vt[kts:kte, jts:jte, its:ite])
    return rslope, rslopeb, rslope2, rslope3, vt

def slope_wsm6(qrs,den,denfac,t,rslope,rslopeb,rslope2,rslope3,   \
                            vt,its,ite,kts,kte):
    lamdar = lambda x, y: (pidn0r / (x * y)) ** 0.25
    lamdas = lambda x, y, z: (pidn0s * z / (x * y)) ** 0.25
    lamdag = lambda x, y: (pidn0g / (x * y)) ** 0.25
    
    #mpcol = torch.zeros((nzall,nyall,nxall)).to(device)
    supcol = torch.zeros((nzall,nyall,nxall)).to(device)
    n0sfac = torch.zeros((nzall,nyall,nxall)).to(device)
    
    supcol[kts:kte, jts:jte, its:ite] = t0c - t[kts:kte, jts:jte, its:ite]
    n0sfac[kts:kte, jts:jte, its:ite] = torch.maximum(torch.minimum(torch.exp
            (alpha * supcol[kts:kte, jts:jte, its:ite]), torch.tensor(n0smax/n0s)), torch.tensor(1.))
    
    rslope[0, kts:kte, jts:jte, its:ite] = torch.where(qrs[0, kts:kte, jts:jte, its:ite] <= qcrmin, 
                rslopermax, 1./lamdar(qrs[0, kts:kte, jts:jte, its:ite], den[kts:kte, jts:jte, its:ite]))
    rslopeb[0, kts:kte, jts:jte, its:ite] = torch.where(qrs[0, kts:kte, jts:jte, its:ite] <= qcrmin, 
                rsloperbmax, rslope[0, kts:kte, jts:jte, its:ite] ** bvtr)
    rslope2[0, kts:kte, jts:jte, its:ite] = torch.where(qrs[0, kts:kte, jts:jte, its:ite] <= qcrmin, 
                rsloper2max, rslope[0, kts:kte, jts:jte, its:ite] ** 2)
    rslope3[0, kts:kte, jts:jte, its:ite] = torch.where(qrs[0, kts:kte, jts:jte, its:ite] <= qcrmin, 
                rsloper3max, rslope[0, kts:kte, jts:jte, its:ite] ** 3)
    
    rslope[1, kts:kte, jts:jte, its:ite] = torch.where(qrs[1, kts:kte, jts:jte, its:ite] <= qcrmin, 
                rslopermax, 1./lamdas(qrs[1, kts:kte, jts:jte, its:ite], den[kts:kte, jts:jte, its:ite],
                n0sfac[kts:kte, jts:jte, its:ite]))
    rslopeb[1, kts:kte, jts:jte, its:ite] = torch.where(qrs[1, kts:kte, jts:jte, its:ite] <= qcrmin, 
                rsloperbmax, rslope[1, kts:kte, jts:jte, its:ite] ** bvts)
    rslope2[1, kts:kte, jts:jte, its:ite] = torch.where(qrs[1, kts:kte, jts:jte, its:ite] <= qcrmin, 
                rsloper2max, rslope[1, kts:kte, jts:jte, its:ite] ** 2)
    rslope3[1, kts:kte, jts:jte, its:ite] = torch.where(qrs[1, kts:kte, jts:jte, its:ite] <= qcrmin, 
                rsloper3max, rslope[1, kts:kte, jts:jte, its:ite] ** 3)
    
    rslope[2, kts:kte, jts:jte, its:ite] = torch.where(qrs[2, kts:kte, jts:jte, its:ite] <= qcrmin, 
                rslopermax, 1./lamdag(qrs[2, kts:kte, jts:jte, its:ite], den[kts:kte, jts:jte, its:ite]))
    rslopeb[2, kts:kte, jts:jte, its:ite] = torch.where(qrs[2, kts:kte, jts:jte, its:ite] <= qcrmin, 
                rsloperbmax, rslope[2, kts:kte, jts:jte, its:ite] ** bvtg)
    rslope2[2, kts:kte, jts:jte, its:ite] = torch.where(qrs[2, kts:kte, jts:jte, its:ite] <= qcrmin, 
                rsloper2max, rslope[2, kts:kte, jts:jte, its:ite] ** 2)
    rslope3[2, kts:kte, jts:jte, its:ite] = torch.where(qrs[2, kts:kte, jts:jte, its:ite] <= qcrmin, 
                rsloper3max, rslope[2, kts:kte, jts:jte, its:ite] ** 3)
    
    vt[0, kts:kte, jts:jte, its:ite] = pvtr * rslopeb[0, kts:kte, jts:jte, its:ite] * \
                denfac[kts:kte, jts:jte, its:ite]
    vt[1, kts:kte, jts:jte, its:ite] = pvts * rslopeb[1, kts:kte, jts:jte, its:ite] * \
                denfac[kts:kte, jts:jte, its:ite]
    vt[2, kts:kte, jts:jte, its:ite] = pvtg * rslopeb[2, kts:kte, jts:jte, its:ite] * \
                denfac[kts:kte, jts:jte, its:ite]
    vt[0, kts:kte, jts:jte, its:ite] = torch.where(qrs[0, kts:kte, jts:jte, its:ite] <= 0.0, 
                0.0, vt[0, kts:kte, jts:jte, its:ite])
    vt[1, kts:kte, jts:jte, its:ite] = torch.where(qrs[1, kts:kte, jts:jte, its:ite] <= 0.0, 
                0.0, vt[1, kts:kte, jts:jte, its:ite])
    vt[2, kts:kte, jts:jte, its:ite] = torch.where(qrs[2, kts:kte, jts:jte, its:ite] <= 0.0, 
                0.0, vt[2, kts:kte, jts:jte, its:ite])
    
    return rslope, rslopeb, rslope2, rslope3, vt

# Finish the moisture update after microphysics.
def moist_physics_finish_em(t_new, t_old, t0, mut,     \
                            th_phy, h_diabatic, dt,    \
                            qv,qv_diabatic,            \
                            qc,qc_diabatic,            \
                            ids,ide, jds,jde, kds,kde, \
                            ims,ime, jms,jme, kms,kme, \
                            its,ite, jts,jte, kts,kte):
    i_start = its
    i_end   = min( ite,ide-1 )
    j_start = jts
    j_end   = min( jte,jde-1 )
    k_start = kts
    k_end = min( kte, kde-1 )
    
    mpten = torch.zeros((nzall,nyall,nxall)).to(device)
    qvten = torch.zeros((nzall,nyall,nxall)).to(device)
    qcten = torch.zeros((nzall,nyall,nxall)).to(device)
    
    mpten[k_start:k_end, j_start:j_end, i_start:i_end] = th_phy[k_start:k_end, j_start:j_end, i_start:i_end] - h_diabatic[k_start:k_end, j_start:j_end, i_start:i_end]
    qvten[k_start:k_end, j_start:j_end, i_start:i_end] = qv[k_start:k_end, j_start:j_end, i_start:i_end] - qv_diabatic[k_start:k_end, j_start:j_end, i_start:i_end]
    qcten[k_start:k_end, j_start:j_end, i_start:i_end] = qc[k_start:k_end, j_start:j_end, i_start:i_end] - qc_diabatic[k_start:k_end, j_start:j_end, i_start:i_end]
    
    t_new[k_start:k_end, j_start:j_end, i_start:i_end] = t_new[k_start:k_end, j_start:j_end, i_start:i_end] + mpten[k_start:k_end, j_start:j_end, i_start:i_end]
    h_diabatic[k_start:k_end, j_start:j_end, i_start:i_end] = mpten[k_start:k_end, j_start:j_end, i_start:i_end] / dt
    qv_diabatic[k_start:k_end, j_start:j_end, i_start:i_end] = qvten[k_start:k_end, j_start:j_end, i_start:i_end] / dt
    qc_diabatic[k_start:k_end, j_start:j_end, i_start:i_end] = qcten[k_start:k_end, j_start:j_end, i_start:i_end] / dt
    
    #h_diabatic[k_start:k_end, j_start:j_end, i_start:i_end] = 0.
    #qv_diabatic[k_start:k_end, j_start:j_end, i_start:i_end] = 0.
    #qc_diabatic[k_start:k_end, j_start:j_end, i_start:i_end] = 0.
   
    return t_new, t_old, th_phy, h_diabatic, qv_diabatic, qc_diabatic
