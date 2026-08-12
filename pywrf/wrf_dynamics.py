"""PyWRF dynamical core — a PyTorch port of the WRF ARW dynamics.

RK3 time integration, momentum / heat / moisture advection,
pressure-gradient and buoyancy forces, diffusion, the small (acoustic)
time steps, and the physics-scratch preparation used by the solver.

All run parameters and physical constants come from
:mod:`pywrf.config_params` (imported via ``from ... import *``); this
module is driven by :class:`pywrf.solver.WrfSolver`.
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

# rk_step_prep functions
def calculate_full(mut,mub,mu,
                   ids, ide, jds, jde, kds, kde,
                   ims, ime, jms, jme, kms, kme,
                   its, ite, jts, jte, kts, kte):
    itf = min(ite,ide-1)
    jtf = min(jte,jde-1)
    ktf = min(kte,kde-1)
    mut[jts:jtf,its:itf]=mub[jts:jtf,its:itf]+mu[jts:jtf,its:itf] #+1?
    return mut

# Corner masses for u / v momentum.
def calc_mu_uv(mu, mub, muu, muv,\
        ids, ide, jds, jde, kds, kde,\
        ims, ime, jms, jme, kms, kme,\
         its, ite, jts, jte, kts, kte):
    itf=ite
    jtf=min(jte,jde-1)
    muu[jts:jtf,its+1:itf-1]=0.5*(mu[jts:jtf,its+1:itf-1]+mu[jts:jtf,its:itf-2] + \
            mub[jts:jtf,its+1:itf-1]+mub[jts:jtf,its:itf-2])
    muu[jts:jtf,its] = mu[jts:jtf,its] + mub[jts:jtf,its]
    muu[jts:jtf,ite-1] = mu[jts:jtf,ite-2] + mub[jts:jtf,ite-2] 
    #mu[jts:jtf,ite-3] + mub[jts:jtf,ite-3])
    #print("calc mu uv",mu[309,603],mub[309,603],mu[309,602],mub[309,602])
    itf=min(ite,ide-1)
    jtf=jte
    muv[jts+1:jtf-1,its:itf]=0.5*(mu[jts+1:jtf-1,its:itf]+mu[jts:jtf-2,its:itf]+\
            mub[jts+1:jtf-1,its:itf]+mub[jts:jtf-2,its:itf])
    muv[jts,its:itf] = mu[jts,its:itf] + mub[jts,its:itf]
    muv[jte-1,its:itf] = mu[jte-2,its:itf] + mub[jte-2,its:itf] 
    #mu[jte-3,its:itf] + mub[jte-3,its:itf])
    #print("calc muuv in",jte,muv[604,601],mub[603,601])
    return muu,muv

# Couple u / v / w momentum from wind and mass.
def couple_momentum(muu, ru, u, msfu,\
        muv, rv, v, msfv, msfvx_inv,\
        mut, rw, w, msft,\
        c1h, c2h, c1f, c2f,\
        ids, ide, jds, jde, kds, kde,\
        ims, ime, jms, jme, kms, kme,\
        its, ite, jts, jte, kts, kte):
    ktf=min(kte,kde-1)
    itf=ite
    jtf=min(jte,jde-1)
    muu_e = muu.repeat(kte-kts,1,1)
    muv_e = muv.repeat(kte-kts,1,1)
    mut_e = mut.repeat(kte-kts,1,1)
    msfu_e = msfu.repeat(kte-kts,1,1)
    msfvx_inv_e = msfvx_inv.repeat(kte-kts,1,1)
    msft_e = msft.repeat(kte-kts,1,1)
    ru[kts:ktf,jts:jtf,its:itf]=u[kts:ktf,jts:jtf,its:itf]*muu_e[kts:ktf,jts:jtf,its:itf]/msfu_e[kts:ktf,jts:jtf,its:itf]
    itf=min(ite,ide-1)
    jtf=jte
    rv[kts:ktf,jts:jtf,its:itf]=v[kts:ktf,jts:jtf,its:itf]*muv_e[kts:ktf,jts:jtf,its:itf]*msfvx_inv_e[kts:ktf,jts:jtf,its:itf]
    jtf=min(jte,jde-1)
    rw[kts:kte,jts:jtf,its:itf]=w[kts:kte,jts:jtf,its:itf]*mut_e[kts:kte,jts:jtf,its:itf]/msft_e[kts:kte,jts:jtf,its:itf]
    return ru,rv,rw

# Coupled vertical velocity ww from c1h / c2h.
def calc_ww_cp(u, v, mu, mub, muu, muv, c1h, c2h, ww,\
        rdx, rdy, msftx, msfty,\
        msfux, msfuy, msfvx, msfvx_inv,\
        msfvy, dnw,\
        ids, ide, jds, jde, kds, kde,\
        ims, ime, jms, jme, kms, kme,\
        its, ite, jts, jte, kts, kte):
    jtf=min(jte,jde-1)
    ktf=min(kte,kde-1)
    itf=min(ite,ide-1)
    
    muu_e = muu.repeat(kte-kts,1,1)
    muv_e = muv.repeat(kte-kts,1,1)
    dmdt = torch.zeros((nyfull,nxfull)).to(device)
    divv = torch.zeros((nzfull,nyfull,nxfull)).to(device)
    msftx_e = msftx.repeat(kte-kts,1,1)
    msfty_e = msfty.repeat(kte-kts,1,1)
    msfuy_e = msfuy.repeat(kte-kts,1,1)
    msfvx_inv_e = msfvx_inv.repeat(kte-kts,1,1)
    dnw_e = dnw.unsqueeze(1).unsqueeze(2).repeat(1,nyfull,nxfull)
    divv[kts:ktf,jts:jtf,its:itf] = msftx_e[kts:ktf,jts:jtf,its:itf]*dnw_e[kts:ktf,jts:jtf,its:itf]*(rdx*(muu_e[kts:ktf,jts:jtf,its+1:itf+1]*u[kts:ktf,jts:jtf,its+1:itf+1]/msfuy_e[kts:ktf,jts:jtf,its+1:itf+1] \
                        -muu_e[kts:ktf,jts:jtf,its:itf]*u[kts:ktf,jts:jtf,its:itf]/msfuy_e[kts:ktf,jts:jtf,its:itf]) \
                        +rdy*(muv_e[kts:ktf,jts+1:jtf+1,its:itf]*v[kts:ktf,jts+1:jtf+1,its:itf]*msfvx_inv_e[kts:ktf,jts+1:jtf+1,its:itf] \
                        -muv_e[kts:ktf,jts:jtf,its:itf]*v[kts:ktf,jts:jtf,its:itf]*msfvx_inv_e[kts:ktf,jts:jtf,its:itf]))
    dmdt = divv.sum(dim=0)
    
    for k in range(ktf-1):
        ww[k+1,jts:jtf,its:itf] = ww[k,jts:jtf,its:itf] - dnw[k]*c1h[k]*dmdt[jts:jtf,its:itf] - divv[k,jts:jtf,its:itf]    # hybrid coord?
        
    return ww

# Coupled moisture q * w for advection.
def calc_cq(moist, cqu, cqv, cqw, n_moist,\
            ids, ide, jds, jde, kds, kde,\
            ims, ime, jms, jme, kms, kme,\
            its, ite, jts, jte, kts, kte):
    ktf=min(kte,kde-1)
    itf=ite
    jtf=min(jte,jde-1)
    cqu[kts:ktf,jts:jtf,its:itf] = 1.0/(1.0 + 0.5*(moist[:,kts:ktf,jts:jtf,its:itf].sum(dim=0)+moist[:,kts:ktf,jts:jtf,its-1:itf-1].sum(dim=0)))
    itf=min(ite,ide-1)
    jtf=jte
    cqv[kts:ktf,jts:jtf,its:itf] = 1.0/(1.0 + 0.5*(moist[:,kts:ktf,jts:jtf,its:itf].sum(dim=0)+moist[:,kts:ktf,jts-1:jtf-1,its:itf].sum(dim=0)))
    itf=min(ite,ide-1)
    jtf=min(jte,jde-1)
    cqw[kts+1:ktf,jts:jtf,its:itf] = 0.5*(moist[:,kts+1:ktf,jts:jtf,its:itf].sum(dim=0)+moist[:,kts:ktf-1,jts:jtf,its:itf].sum(dim=0))
    
    return cqu,cqv,cqw

# Inverse density (full + base).
def calc_alt(alt, al, alb,\
             ids, ide, jds, jde, kds, kde,\
             ims, ime, jms, jme, kms, kme,\
             its, ite, jts, jte, kts, kte):
    itf=min(ite,ide-1)
    jtf=min(jte,jde-1)
    ktf=min(kte,kde-1)
    alt[kts:ktf,jts:jtf,its:itf] = al[kts:ktf,jts:jtf,its:itf] + alb[kts:ktf,jts:jtf,its:itf]
    return alt

# Geopotential tendency from coupled mass.
def calc_php(php, ph, phb,\
             ids, ide, jds, jde, kds, kde,\
             ims, ime, jms, jme, kms, kme,\
             its, ite, jts, jte, kts, kte):
    itf=min(ite,ide-1)
    jtf=min(jte,jde-1)
    ktf=min(kte,kde-1)
    php[kts:ktf,jts:jtf,its:itf] = 0.5*(phb[kts:ktf,jts:jtf,its:itf]+phb[kts+1:ktf+1,jts:jtf,its:itf]+ph[kts:ktf,jts:jtf,its:itf]+ph[kts+1:ktf+1,jts:jtf,its:itf])
    
    return php

########## end rk_step_prep functions ##########

# boundary functions
def set_physical_bc2d(dat, variable_in,   \
                      ids,ide, jds,jde,   \
                      ims,ime, jms,jme,   \
                      ips,ipe, jps,jpe,   \
                      its,ite, jts,jte):
    variable = variable_in.lower()
    if variable == 'u' or variable == 'v' or variable == 'w' or variable == 't' or \
       variable == 'x' or variable == 'y' or variable == 'r' or variable == 'p':
        open_bc_copy = True
    istag = -1
    jstag = -1
    
    if variable == 'u' or variable == 'x':
        istag = 0
    if variable == 'v' or variable == 'y':
        jstag = 0
    if variable == 'd': 
        istag = 0
        jstag = 0
    if variable == 'e':
        istag = 0
    if variable == 'f':
        jstag = 0
    istart = max(ids, its-1)
    iend = min(ite+1, ide+istag)
    if open_bc_copy:
        dat[jds:jde+jstag, ids-1] = dat[jds:jde+jstag, ids]
        dat[jds:jde+jstag, ids-2] = dat[jds:jde+jstag, ids]
        dat[jds:jde+jstag, ids-3] = dat[jds:jde+jstag, ids]
        if variable != 'u' and variable != 'x':
            dat[jds:jde+jstag, ide-1] = dat[jds:jde+jstag, ide-2]
            dat[jds:jde+jstag, ide] = dat[jds:jde+jstag, ide-2]
            dat[jds:jde+jstag, ide+1] = dat[jds:jde+jstag, ide-2]
        else:
            dat[jds:jde+jstag, ide] = dat[jds:jde+jstag, ide-1]
            dat[jds:jde+jstag, ide+1] = dat[jds:jde+jstag, ide-1]
            dat[jds:jde+jstag, ide+2] = dat[jds:jde+jstag, ide-1]
        dat[jds-1, istart:iend] = dat[jds, istart:iend]
        dat[jds-2, istart:iend] = dat[jds, istart:iend]
        dat[jds-3, istart:iend] = dat[jds, istart:iend]
        if variable != 'v' and variable != 'y':
            dat[jde-1, istart:iend] = dat[jde-2, istart:iend]
            dat[jde, istart:iend] = dat[jde-2, istart:iend]
            dat[jde+1, istart:iend] = dat[jde-2, istart:iend]
        else:
            dat[jde, istart:iend] = dat[jde-1, istart:iend]
            dat[jde+1, istart:iend] = dat[jde-1, istart:iend]
            dat[jde+2, istart:iend] = dat[jde-1, istart:iend]
    return dat


# Lateral physical boundary conditions for a 3-D field.
def set_physical_bc3d(dat, variable_in,        \
                      ids,ide, jds,jde, kds,kde,  \
                      ims,ime, jms,jme, kms,kme,  \
                      ips,ipe, jps,jpe, kps,kpe,  \
                      its,ite, jts,jte, kts,kte ):
    variable = variable_in
    #open_bc_copy = False
    if variable == 'U':
        variable = 'u'
    if variable == 'V':
        variable = 'v'
    if variable == 'M':
        variable = 'm'
    if variable == 'H':
        variable = 'h'
    
    if variable == 'u' or variable == 'v' or variable == 'w' or variable == 't' or \
       variable == 'd' or variable == 'e' or variable == 'x' or variable == 'y' or \
       variable == 'f' or variable == 'r' or variable == 'p':
        open_bc_copy = True
    istag = -1
    jstag = -1
    k_end = min(kde-1,kte)
    
    if variable == 'u' or variable == 'x':
        istag = 0
    if variable == 'v' or variable == 'y':
        jstag = 0
    if variable == 'd' or variable == 'xy': 
        istag = 0
        jstag = 0
    if variable == 'e':
        istag = 0
        k_end = min(kde,kte)
    if variable == 'f':
        jstag = 0
        k_end = min(kde,kte)
    if variable == 'w':
        k_end = min(kde,kte)
    if open_bc_copy:
        dat[kts:k_end, jts-4:jde+jstag+4, ids-1] = dat[kts:k_end, jts-4:jde+jstag+4, ids]
        dat[kts:k_end, jts-4:jde+jstag+4, ids-2] = dat[kts:k_end, jts-4:jde+jstag+4, ids]
        dat[kts:k_end, jts-4:jde+jstag+4, ids-3] = dat[kts:k_end, jts-4:jde+jstag+4, ids]
        if variable != 'u' and variable != 'x':
            dat[kts:k_end, jts-4:jde+jstag+4, ide-1] = dat[kts:k_end, jts-4:jde+jstag+4, ide-2] 
            dat[kts:k_end, jts-4:jde+jstag+4, ide] = dat[kts:k_end, jts-4:jde+jstag+4, ide-2]
            dat[kts:k_end, jts-4:jde+jstag+4, ide+1] = dat[kts:k_end, jts-4:jde+jstag+4, ide-2]
        else:
            dat[kts:k_end, jts-5:jde+jstag+4, ide] = dat[kts:k_end, jts-5:jde+jstag+4, ide-1]
            dat[kts:k_end, jts-5:jde+jstag+4, ide+1] = dat[kts:k_end, jts-5:jde+jstag+4, ide-1]
            dat[kts:k_end, jts-5:jde+jstag+4, ide+2] = dat[kts:k_end, jts-5:jde+jstag+4, ide-1]
        dat[kts:k_end, jds-1, ids:ide+istag] = dat[kts:k_end, jds, ids:ide+istag]
        dat[kts:k_end, jds-2, ids:ide+istag] = dat[kts:k_end, jds, ids:ide+istag]
        dat[kts:k_end, jds-3, ids:ide+istag] = dat[kts:k_end, jds, ids:ide+istag]
        if variable != 'v' and variable != 'y':
            dat[kts:k_end, jde-1, ids:ide+istag] = dat[kts:k_end, jde-2, ids:ide+istag]
            dat[kts:k_end, jde, ids:ide+istag] = dat[kts:k_end, jde-2, ids:ide+istag]
            dat[kts:k_end, jde+1, ids:ide+istag] = dat[kts:k_end, jde-2, ids:ide+istag]
        else:
            dat[kts:k_end, jde, ids:ide+istag] = dat[kts:k_end, jde-1, ids:ide+istag]
            dat[kts:k_end, jde+1, ids:ide+istag] = dat[kts:k_end, jde-1, ids:ide+istag]
            dat[kts:k_end, jde+2, ids:ide+istag] = dat[kts:k_end, jde-1, ids:ide+istag]
    return dat

# first rk step functions
def first_rk_step_part1 (   grid , config_flags              \
                         , moist , moist_tend               \
                         , chem  , chem_tend                \
                         , tracer, tracer_tend              \
                         , scalar , scalar_tend             \
                         , fdda3d, fdda2d                   \
                         , aerod                            \
                         , ru_tendf, rv_tendf               \
                         , rw_tendf, t_tendf                \
                         , ph_tendf, mu_tendf               \
                         , tke_tend                         \
                         , adapt_step_flag , curr_secs      \
                         , psim , psih , wspd , gz1oz0 , chklowq \
                         , cu_act_flag , hol , th_phy       \
                         , pi_phy , p_phy , t_phy           \
                         , dz8w , p8w , t8w                 \
                         , ids, ide, jds, jde, kds, kde     \
                         , ims, ime, jms, jme, kms, kme     \
                         , ips, ipe, jps, jpe, kps, kpe     \
                         , imsx,imex,jmsx,jmex,kmsx,kmex    \
                         , ipsx,ipex,jpsx,jpex,kpsx,kpex    \
                         , imsy,imey,jmsy,jmey,kmsy,kmey    \
                         , ipsy,ipey,jpsy,jpey,kpsy,kpey    \
                         , k_start , k_end                  \
                         , f_flux ):
    

    return

# Constants t0/rcp/p1000mb/g/p_top come from pywrf/config_params via the star
# import above (identical values) — no local redefinition needed.

# Fill physics scratch arrays (th_phy, p_phy, rho, dz8w, ...).
def phy_prep(mut, muu, muv,                               \
             c1h, c2h, c1f, c2f,                          \
             u, v, p, pb, alt, ph,                        \
             phb, t, moist, n_moist,                      \
             rho, th_phy, p_phy , pi_phy ,                \
             u_phy, v_phy, p8w, t_phy, t8w,               \
             z, z_at_w, dz8w,                             \
             p_hyd, p_hyd_w, dnw,                         \
             fzm, fzp, znw, p_top,                        \
             ids, ide, jds, jde, kds, kde,                \
             ims, ime, jms, jme, kms, kme,                \
             its, ite, jts, jte, kts, kte                ):
    c1 = c1h
    c2 = c2h
             
    i_start = its
    i_end   = min( ite,ide-1 )
    j_start = jts
    j_end   = min( jte,jde-1 )

    k_start = kts
    k_end = min( kte, kde-1 )
    
    mut_e = mut.repeat(kte-kts,1,1)
    
    q_tot = torch.zeros((nzall,nyall,nxall)).to(device)
    
    th_phy[k_start:k_end,j_start:j_end,i_start:i_end] = t[k_start:k_end,j_start:j_end,i_start:i_end] + t0
    p_phy[k_start:k_end,j_start:j_end,i_start:i_end] = p[k_start:k_end,j_start:j_end,i_start:i_end] \
         + pb[k_start:k_end,j_start:j_end,i_start:i_end]
    pi_phy = (p_phy/p1000mb)**rcp
    t_phy = th_phy * pi_phy
    rho[k_start:k_end,j_start:j_end,i_start:i_end] = 1./alt[k_start:k_end,j_start:j_end,i_start:i_end] \
         * (1+moist[P_QV,k_start:k_end,j_start:j_end,i_start:i_end])
    u_phy[k_start:k_end,j_start:j_end,i_start:i_end] = 0.5 * \
        (u[k_start:k_end,j_start:j_end,i_start:i_end] + \
         u[k_start:k_end,j_start:j_end,i_start+1:i_end+1])
    v_phy[k_start:k_end,j_start:j_end,i_start:i_end] = 0.5 * \
        (v[k_start:k_end,j_start:j_end,i_start:i_end] + \
         v[k_start:k_end,j_start+1:j_end+1,i_start:i_end])
    
    z_at_w[k_start:k_end,j_start:j_end,i_start:i_end] = \
        (phb[k_start:k_end,j_start:j_end,i_start:i_end] + \
         ph[k_start:k_end,j_start:j_end,i_start:i_end])/g
    
    dz8w[k_start:kte-1,j_start:j_end,i_start:i_end] = \
        z_at_w[k_start+1:kte,j_start:j_end,i_start:i_end] - \
        z_at_w[k_start:kte-1,j_start:j_end,i_start:i_end]
    dz8w[kte-1,j_start:j_end,i_start:i_end] = 0.0
    
    z[k_start:k_end,j_start:j_end,i_start:i_end] = 0.5 * \
        (z_at_w[k_start:k_end,j_start:j_end,i_start:i_end] + \
         z_at_w[k_start+1:k_end+1,j_start:j_end,i_start:i_end])
    
    fzm_e = fzm.unsqueeze(1).unsqueeze(2).repeat(1,nyfull,nxfull)
    fzp_e = fzp.unsqueeze(1).unsqueeze(2).repeat(1,nyfull,nxfull)
    p8w[k_start+1:k_end,j_start:j_end,i_start:i_end] = fzm_e[k_start+1:k_end,j_start:j_end,i_start:i_end] \
        * p_phy[k_start+1:k_end,j_start:j_end,i_start:i_end] \
        + fzp_e[k_start+1:k_end,j_start:j_end,i_start:i_end] \
        * p_phy[k_start:k_end-1,j_start:j_end,i_start:i_end]
    t8w[k_start+1:k_end,j_start:j_end,i_start:i_end] = fzm_e[k_start+1:k_end,j_start:j_end,i_start:i_end] \
        * t_phy[k_start+1:k_end,j_start:j_end,i_start:i_end] \
        + fzp_e[k_start+1:k_end,j_start:j_end,i_start:i_end] \
        * t_phy[k_start:k_end-1,j_start:j_end,i_start:i_end]
        
    #bottom
    z0 = z_at_w[0,j_start:j_end,i_start:i_end]
    z1 = z[0,j_start:j_end,i_start:i_end]
    z2 = z[1,j_start:j_end,i_start:i_end]
    w1 = (z0 - z2)/(z1 - z2)
    w2 = 1. - w1
    p8w[0,j_start:j_end,i_start:i_end] = w1 * p_phy[0,j_start:j_end,i_start:i_end] + \
        w2 * p_phy[1,j_start:j_end,i_start:i_end]
    t8w[0,j_start:j_end,i_start:i_end] = w1 * t_phy[0,j_start:j_end,i_start:i_end] + \
        w2 * t_phy[1,j_start:j_end,i_start:i_end]
    #top
    z0 = z_at_w[kte-1,j_start:j_end,i_start:i_end]
    z1 = z[k_end-1,j_start:j_end,i_start:i_end]
    z2 = z[k_end-2,j_start:j_end,i_start:i_end]
    w1 = (z0 - z2)/(z1 - z2)
    w2 = 1. - w1
    p8w[kde-1,j_start:j_end,i_start:i_end] = torch.exp(
        w1 * torch.log(p_phy[kde-2,j_start:j_end,i_start:i_end]) + 
        w2 * torch.log(p_phy[kde-3,j_start:j_end,i_start:i_end]) )
    t8w[kde-1,j_start:j_end,i_start:i_end] = w1 * t_phy[kde-2,j_start:j_end,i_start:i_end] + \
        w2 * t_phy[kde-3,j_start:j_end,i_start:i_end]
    
    p_hyd_w[kte-1,j_start:j_end,i_start:i_end] = p_top
    
    q_tot[k_start:k_end-1,j_start:j_end,i_start:i_end] = \
        moist[:,k_start:k_end-1,j_start:j_end,i_start:i_end].sum(dim=0)
    for k in range(kte-2 , k_start-1, -1):
        p_hyd_w[k, j_start:j_end,i_start:i_end] = p_hyd_w[k+1, j_start:j_end,i_start:i_end] - \
            (1. + q_tot[k, j_start:j_end,i_start:i_end]) * mut_e[k, j_start:j_end,i_start:i_end] * dnw[k]
    p_hyd[k_start:k_end, j_start:j_end,i_start:i_end] = 0.5 * (
        p_hyd_w[k_start:k_end, j_start:j_end,i_start:i_end] + p_hyd_w[k_start+1:k_end+1, j_start:j_end,i_start:i_end])
    
    return th_phy, p_phy, pi_phy, t_phy, rho, u_phy, v_phy, z_at_w, dz8w, z, p8w, t8w, p_hyd_w, p_hyd

def pbl_driver():
    
    return

def first_rk_step_part2(grid, config_flags         \
                        , moist , moist_tend            \
                        , chem  , chem_tend             \
                        , tracer, tracer_tend           \
                        , scalar , scalar_tend          \
                        , fdda3d, fdda2d                \
                        , ru_tendf, rv_tendf            \
                        , rw_tendf, t_tendf             \
                        , ph_tendf, mu_tendf            \
                        , tke_tend                      \
                        , adapt_step_flag , curr_secs   \
                        , psim , psih , wspd , gz1oz0   \
                        , chklowq                        \
                        , cu_act_flag , hol , th_phy     \
                        , pi_phy , p_phy , t_phy    \
                        , dz8w , p8w , t8w               \
                        , nba_mij, num_nba_mij           \
                        , nba_rij, num_nba_rij           \
                        , ids, ide, jds, jde, kds, kde   \
                        , ims, ime, jms, jme, kms, kme   \
                        , ips, ipe, jps, jpe, kps, kpe   \
                        , imsx, imex, jmsx, jmex, kmsx, kmex    \
                        , ipsx, ipex, jpsx, jpex, kpsx, kpex    \
                        , imsy, imey, jmsy, jmey, kmsy, kmey    \
                        , ipsy, ipey, jpsy, jpey, kpsy, kpey    \
                        , k_start , k_end ):
    rk_step = 1
    
    
    return

# Physics tendencies (theta, u / v, moisture, ...).
def calculate_phy_tend(c1,c2,                     \
                     mut,muu,muv,pi3d,                         \
                     RUBLTEN,RVBLTEN,RTHBLTEN,                 \
                     RQVBLTEN,RQCBLTEN,RQIBLTEN,               \
                     scalar, scalar_tend, num_scalar,          \
                     ids,ide, jds,jde, kds,kde,                \
                     ims,ime, jms,jme, kms,kme,                \
                     its,ite, jts,jte, kts,kte):
    itf=min(ite,ide-1)
    jtf=min(jte,jde-1)
    ktf=min(kte,kde-1)
    itsu=max(its,ids+1)
    jtsv=max(jts,jds+1)
    
    # radiation
    
    # cumulus
    
    # shallow cumulus
    
    # pbl
    mut_e = mut.repeat(nzall,1,1)
    RUBLTEN[kts:ktf,jts:jtf,its:itf] = mut_e[kts:ktf,jts:jtf,its:itf] * \
                                       RUBLTEN[kts:ktf,jts:jtf,its:itf]
    RVBLTEN[kts:ktf,jts:jtf,its:itf] = mut_e[kts:ktf,jts:jtf,its:itf] * \
                                       RVBLTEN[kts:ktf,jts:jtf,its:itf]
    RTHBLTEN[kts:ktf,jts:jtf,its:itf] = mut_e[kts:ktf,jts:jtf,its:itf] * \
                                       RTHBLTEN[kts:ktf,jts:jtf,its:itf]
    RQVBLTEN[kts:ktf,jts:jtf,its:itf] = mut_e[kts:ktf,jts:jtf,its:itf] * \
                                       RQVBLTEN[kts:ktf,jts:jtf,its:itf]
    RQCBLTEN[kts:ktf,jts:jtf,its:itf] = mut_e[kts:ktf,jts:jtf,its:itf] * \
                                       RQCBLTEN[kts:ktf,jts:jtf,its:itf]
    RQIBLTEN[kts:ktf,jts:jtf,its:itf] = mut_e[kts:ktf,jts:jtf,its:itf] * \
                                       RQIBLTEN[kts:ktf,jts:jtf,its:itf]
    # scalar_tend
    for im in range(3):
        scalar_tend[im,kts:ktf,jts:jtf,its:itf] = mut_e[kts:ktf,jts:jtf,its:itf] * \
                                       scalar_tend[im,kts:ktf,jts:jtf,its:itf]
    
    return RUBLTEN,RVBLTEN,RTHBLTEN,RQVBLTEN,RQCBLTEN,RQIBLTEN,scalar_tend

# Vertical metric terms rdz / rdzw and map-scale derivatives.
def compute_diff_metrics(ph, phb, z, rdz, rdzw,  \
                         zx, zy, rdx, rdy,                     \
                         ids, ide, jds, jde, kds, kde,         \
                         ims, ime, jms, jme, kms, kme,         \
                         its, ite, jts, jte, kts, kte):
    ktf = min( kte, kde-1 )     
    j_start = jts-1             
    j_end   = jte               
    
    i_start = its
    i_end = ide-1
    
    z_at_w = torch.zeros((nzall,nyall,nxall)).to(device)
    
    z_at_w[kts:kte,j_start:j_end,i_start:i_end] = (ph[kts:kte,j_start:j_end,i_start:i_end] + 
                                                  phb[kts:kte,j_start:j_end,i_start:i_end]) / g
    rdzw[kts:ktf,j_start:j_end,i_start:i_end] = 1.0 / (z_at_w[kts+1:ktf+1,j_start:j_end,i_start:i_end] - \
                                                z_at_w[kts:ktf,j_start:j_end,i_start:i_end])
    rdz[kts+1:ktf,j_start:j_end,i_start:i_end] = 2.0 / (z_at_w[kts+2:ktf+1,j_start:j_end,i_start:i_end] - \
                                               z_at_w[kts:ktf-1,j_start:j_end,i_start:i_end])
    rdz[0,j_start:j_end,i_start:i_end] = 2.0 / (z_at_w[1,j_start:j_end,i_start:i_end] - \
                                               z_at_w[0,j_start:j_end,i_start:i_end])   ###  2.0 注意，错误???
    
    i_start = its
    i_end   = min( ite, ide-1 )
    j_start = jts
    j_end   = min( jte, jde-1 )
    zx[kts:kte,j_start:j_end,ids+1:i_end] = rdx * (phb[kts:kte,j_start:j_end,ids+1:i_end] -
                                               phb[kts:kte,j_start:j_end,ids:i_end-1]) / g \
                                      + rdx * (ph[kts:kte,j_start:j_end,ids+1:i_end] -
                                               ph[kts:kte,j_start:j_end,ids:i_end-1]) / g
    zy[kts:kte,jds+1:j_end,i_start:i_end] = rdy * (phb[kts:kte,jds+1:j_end,i_start:i_end] -
                                               phb[kts:kte,jds:j_end-1,i_start:i_end]) / g \
                                      + rdy * (ph[kts:kte,jds+1:j_end,i_start:i_end] -
                                               ph[kts:kte,jds:j_end-1,i_start:i_end]) / g
    zx[0:ktf,j_start:j_end,ide-1] = 0.0
    zx[0:ktf,j_start:j_end,ids-1] = 0.0
    zy[0:ktf,jde-1,i_start:i_end] = 0.0
    zy[0:ktf,jds-1,i_start:i_end] = 0.0
    
    z[0:ktf, j_start:j_end, i_start:i_end] = 0.5 * (ph[0:ktf, j_start:j_end, i_start:i_end] +
                                                    phb[0:ktf, j_start:j_end, i_start:i_end] +
                                                    ph[1:ktf+1, j_start:j_end, i_start:i_end] +
                                                    phb[1:ktf+1, j_start:j_end, i_start:i_end])
    
    return rdzw, rdz, zx, zy, z

# Horizontal deformation and divergence for diffusion.
def cal_deform_and_div(u, v, w, div,       \
                       defor11, defor22, defor33,        \
                       defor12, defor13, defor23,        \
                       u_base, v_base, msfux, msfuy,     \
                       msfvx, msfvy, msftx, msfty,       \
                       rdx, rdy, dn, dnw, rdz, rdzw,     \
                       fnm, fnp, cf1, cf2, cf3, zx, zy,  \
                       ids, ide, jds, jde, kds, kde,     \
                       ims, ime, jms, jme, kms, kme,     \
                       its, ite, jts, jte, kts, kte      ):
    ktes1   = kte-2
    ktes2   = kte-3
    
    cft2    = - 0.5 * dnw[ktes1] / dn[ktes1]
    cft1    = 1.0 - cft2
    
    ktf = kde-1
    
    i_start = its
    i_end   = min( ite, ide-1 )
    j_start = jts
    j_end   = min( jte, jde-1 )
    
    mm = torch.zeros((nyall,nxall)).to(device)
    hat = torch.zeros((nzall,nyall,nxall)).to(device)
    hatavg = torch.zeros((nzall,nyall,nxall)).to(device)
    tmp1 = torch.zeros((nzall,nyall,nxall)).to(device)
    
    mm[j_start:j_end,i_start:i_end] = msftx[j_start:j_end,i_start:i_end] * msfty[j_start:j_end,i_start:i_end]
    mm_e = mm.repeat(nzall,1,1)
    
    msfuy_e = msfuy.repeat(nzall,1,1)
    msfty_e = msfty.repeat(nzall,1,1)
    msftx_e = msftx.repeat(nzall,1,1)
    
    hat[kts:ktf,j_start:j_end,i_start:i_end+1] = u[kts:ktf,j_start:j_end,i_start:i_end+1] / \
                                                 msfuy_e[kts:ktf,j_start:j_end,i_start:i_end+1]
    
    fnm_e = fnm.unsqueeze(1).unsqueeze(2).repeat(1,nyall,nxall)
    fnp_e = fnp.unsqueeze(1).unsqueeze(2).repeat(1,nyall,nxall)
    hatavg[kts+1:ktf,j_start:j_end,i_start:i_end] = 0.5 * (fnm_e[kts+1:ktf,j_start:j_end,i_start:i_end] 
          * (hat[kts+1:ktf,j_start:j_end,i_start:i_end] + hat[kts+1:ktf,j_start:j_end,i_start+1:i_end+1])
          + fnp_e[kts+1:ktf,j_start:j_end,i_start:i_end]
          * (hat[kts:ktf-1,j_start:j_end,i_start:i_end] + hat[kts:ktf-1,j_start:j_end,i_start+1:i_end+1]))
    hatavg[0,j_start:j_end,i_start:i_end] = 0.5 * (cf1 * hat[0,j_start:j_end,i_start:i_end] +
                                                   cf2 * hat[1,j_start:j_end,i_start:i_end] +
                                                   cf3 * hat[2,j_start:j_end,i_start:i_end] +
                                                   cf1 * hat[0,j_start:j_end,i_start+1:i_end+1] +
                                                   cf2 * hat[1,j_start:j_end,i_start+1:i_end+1] +
                                                   cf3 * hat[2,j_start:j_end,i_start+1:i_end+1])
    hatavg[kte-1,j_start:j_end,i_start:i_end] = 0.5 * (cft1 * (hat[ktes1,j_start:j_end,i_start:i_end] +
                                                             hat[ktes1,j_start:j_end,i_start+1:i_end+1]) +
                                                     cft2 * (hat[ktes2,j_start:j_end,i_start:i_end] +
                                                             hat[ktes2,j_start:j_end,i_start+1:i_end+1]))
    
    tmp1[kts:ktf,j_start:j_end,i_start:i_end] =  (hatavg[kts+1:ktf+1,j_start:j_end,i_start:i_end]- 
                                                 hatavg[kts:ktf,j_start:j_end,i_start:i_end]) * (
                                                 0.25*(zx[kts:ktf,j_start:j_end,i_start:i_end] + 
                                                 zx[kts:ktf,j_start:j_end,i_start+1:i_end+1] +
                                                 zx[kts+1:ktf+1,j_start:j_end,i_start:i_end] +
                                                 zx[kts+1:ktf+1,j_start:j_end,i_start+1:i_end+1]) * 
                                                 rdzw[kts:ktf,j_start:j_end,i_start:i_end])
    
    tmp1[kts:ktf,j_start:j_end,i_start:i_end] =  mm_e[kts:ktf,j_start:j_end,i_start:i_end] * (rdx * 
                                                 (hat[kts:ktf,j_start:j_end,i_start+1:i_end+1] - 
                                                 hat[kts:ktf,j_start:j_end,i_start:i_end]) - tmp1[kts:ktf,j_start:j_end,i_start:i_end])
    
    defor11[kts:ktf,j_start:j_end,i_start:i_end] = 2.0 * tmp1[kts:ktf,j_start:j_end,i_start:i_end]
    
    div[kts:ktf,j_start:j_end,i_start:i_end] = tmp1[kts:ktf,j_start:j_end,i_start:i_end] + 0.0
    
    msfvx_e = msfvx.repeat(nzall,1,1)
    hat[kts:ktf,j_start:j_end+1,i_start:i_end] = v[kts:ktf,j_start:j_end+1,i_start:i_end] / \
                                                 msfvx_e[kts:ktf,j_start:j_end+1,i_start:i_end]
    hatavg[kts+1:ktf,j_start:j_end,i_start:i_end] = 0.5 * (fnm_e[kts+1:ktf,j_start:j_end,i_start:i_end] 
          * (hat[kts+1:ktf,j_start:j_end,i_start:i_end] + hat[kts+1:ktf,j_start+1:j_end+1,i_start:i_end])
          + fnp_e[kts+1:ktf,j_start:j_end,i_start:i_end]
          * (hat[kts:ktf-1,j_start:j_end,i_start:i_end] + hat[kts:ktf-1,j_start+1:j_end+1,i_start:i_end]))
    hatavg[0,j_start:j_end,i_start:i_end] = 0.5 * (cf1 * hat[0,j_start:j_end,i_start:i_end] +
                                                   cf2 * hat[1,j_start:j_end,i_start:i_end] +
                                                   cf3 * hat[2,j_start:j_end,i_start:i_end] +
                                                   cf1 * hat[0,j_start+1:j_end+1,i_start:i_end] +
                                                   cf2 * hat[1,j_start+1:j_end+1,i_start:i_end] +
                                                   cf3 * hat[2,j_start+1:j_end+1,i_start:i_end])
    hatavg[kte-1,j_start:j_end,i_start:i_end] = 0.5 * (cft1 * (hat[ktes1,j_start:j_end,i_start:i_end] +
                                                             hat[ktes1,j_start+1:j_end+1,i_start:i_end]) +
                                                     cft2 * (hat[ktes2,j_start:j_end,i_start:i_end] +
                                                             hat[ktes2,j_start+1:j_end+1,i_start:i_end]))
    tmp1[kts:ktf,j_start:j_end,i_start:i_end] =  mm_e[kts:ktf,j_start:j_end,i_start:i_end] * (rdy * 
                                                 (hat[kts:ktf,j_start+1:j_end+1,i_start:i_end] - 
                                                 hat[kts:ktf,j_start:j_end,i_start:i_end]) - 
                                                 (hatavg[kts+1:ktf+1,j_start:j_end,i_start:i_end]- 
                                                 hatavg[kts:ktf,j_start:j_end,i_start:i_end]) * (
                                                 0.25*(zy[kts:ktf,j_start:j_end,i_start:i_end] + 
                                                 zy[kts:ktf,j_start+1:j_end+1,i_start:i_end] +
                                                 zy[kts+1:ktf+1,j_start:j_end,i_start:i_end] +
                                                 zy[kts+1:ktf+1,j_start+1:j_end+1,i_start:i_end]) * 
                                                 rdzw[kts:ktf,j_start:j_end,i_start:i_end]))
    defor22[kts:ktf,j_start:j_end,i_start:i_end] = 2.0 * tmp1[kts:ktf,j_start:j_end,i_start:i_end]
    
    div[kts:ktf,j_start:j_end,i_start:i_end] = tmp1[kts:ktf,j_start:j_end,i_start:i_end] + \
                                               div[kts:ktf,j_start:j_end,i_start:i_end]
    
    tmp1[kts:ktf,j_start:j_end,i_start:i_end] = (w[kts+1:ktf+1,j_start:j_end,i_start:i_end] - \
                                                w[kts:ktf,j_start:j_end,i_start:i_end]) * \
                                                rdzw[kts:ktf,j_start:j_end,i_start:i_end]
    defor33[kts:ktf,j_start:j_end,i_start:i_end] = 2.0 * tmp1[kts:ktf,j_start:j_end,i_start:i_end]
    div[kts:ktf,j_start:j_end,i_start:i_end] = tmp1[kts:ktf,j_start:j_end,i_start:i_end] + \
                                               div[kts:ktf,j_start:j_end,i_start:i_end]
    
    i_start = its + 1
    i_end = ite - 1
    j_start = jts + 1
    j_end = jte - 1
    mm[j_start:j_end,i_start:i_end] = 0.25 * (msfux[j_start-1:j_end-1,i_start:i_end] + \
                                              msfux[j_start:j_end,i_start:i_end]) * \
                                             (msfvy[j_start:j_end,i_start-1:i_end-1] +\
                                              msfvy[j_start:j_end,i_start:i_end])
    mm_e = mm.repeat(nzall,1,1)
    msfux_e = msfux.repeat(nzall,1,1)
    hat[kts:ktf,j_start-1:j_end,i_start:i_end] = u[kts:ktf,j_start-1:j_end,i_start:i_end] \
                                                 / msfux_e[kts:ktf,j_start-1:j_end,i_start:i_end]
    
    hatavg[kts+1:ktf,j_start:j_end,i_start:i_end] = 0.5 * (fnm_e[kts+1:ktf,j_start:j_end,i_start:i_end] 
          * (hat[kts+1:ktf,j_start-1:j_end-1,i_start:i_end] + hat[kts+1:ktf,j_start:j_end,i_start:i_end])
          + fnp_e[kts+1:ktf,j_start:j_end,i_start:i_end]
          * (hat[kts:ktf-1,j_start-1:j_end-1,i_start:i_end] + hat[kts:ktf-1,j_start:j_end,i_start:i_end]))
    hatavg[0,j_start:j_end,i_start:i_end] = 0.5 * (cf1 * hat[0,j_start-1:j_end-1,i_start:i_end] +
                                                   cf2 * hat[1,j_start-1:j_end-1,i_start:i_end] +
                                                   cf3 * hat[2,j_start-1:j_end-1,i_start:i_end] +
                                                   cf1 * hat[0,j_start:j_end,i_start:i_end] +
                                                   cf2 * hat[1,j_start:j_end,i_start:i_end] +
                                                   cf3 * hat[2,j_start:j_end,i_start:i_end])
    hatavg[kte-1,j_start:j_end,i_start:i_end] = 0.5 * (cft1 * (hat[ktes1,j_start-1:j_end-1,i_start:i_end] +
                                                             hat[ktes1,j_start:j_end,i_start:i_end]) +
                                                     cft2 * (hat[ktes2,j_start-1:j_end-1,i_start:i_end] +
                                                             hat[ktes2,j_start:j_end,i_start:i_end]))
    
    tmp1[kts:ktf,j_start:j_end,i_start:i_end] =  (hatavg[kts+1:ktf+1,j_start:j_end,i_start:i_end]- 
                                                 hatavg[kts:ktf,j_start:j_end,i_start:i_end]) * (
                                                 0.25*(zy[kts:ktf,j_start:j_end,i_start-1:i_end-1] + 
                                                 zy[kts:ktf,j_start:j_end,i_start:i_end] +
                                                 zy[kts+1:ktf+1,j_start:j_end,i_start-1:i_end-1] +
                                                 zy[kts+1:ktf+1,j_start:j_end,i_start:i_end]) * 
                                                 0.25 * (rdzw[kts:ktf,j_start:j_end,i_start:i_end] + 
                                                 rdzw[kts:ktf,j_start:j_end,i_start-1:i_end-1] +
                                                 rdzw[kts:ktf,j_start-1:j_end-1,i_start-1:i_end-1] +
                                                 rdzw[kts:ktf,j_start-1:j_end-1,i_start:i_end]))
    
    defor12[kts:ktf,j_start:j_end,i_start:i_end] = mm_e[kts:ktf,j_start:j_end,i_start:i_end ] * \
                                                   (rdy * (hat[kts:ktf,j_start:j_end,i_start:i_end] - \
                                                          hat[kts:ktf,j_start-1:j_end-1,i_start:i_end]) - \
                                                    tmp1[kts:ktf,j_start:j_end,i_start:i_end])
    
    msfvy_e = msfvy.repeat(ktf-kts,1,1)
    hat[kts:ktf,j_start:j_end,i_start-1:i_end] = v[kts:ktf,j_start:j_end,i_start-1:i_end] \
                                                 / msfvy_e[kts:ktf,j_start:j_end,i_start-1:i_end]
    hatavg[kts+1:ktf,j_start:j_end,i_start:i_end] = 0.5 * (fnm_e[kts+1:ktf,j_start:j_end,i_start:i_end] 
          * (hat[kts+1:ktf,j_start:j_end,i_start-1:i_end-1] + hat[kts+1:ktf,j_start:j_end,i_start:i_end])
          + fnp_e[kts+1:ktf,j_start:j_end,i_start:i_end]
          * (hat[kts:ktf-1,j_start:j_end,i_start-1:i_end-1] + hat[kts:ktf-1,j_start:j_end,i_start:i_end]))
    hatavg[0,j_start:j_end,i_start:i_end] = 0.5 * (cf1 * hat[0,j_start:j_end,i_start-1:i_end-1] +
                                                   cf2 * hat[1,j_start:j_end,i_start-1:i_end-1] +
                                                   cf3 * hat[2,j_start:j_end,i_start-1:i_end-1] +
                                                   cf1 * hat[0,j_start:j_end,i_start:i_end] +
                                                   cf2 * hat[1,j_start:j_end,i_start:i_end] +
                                                   cf3 * hat[2,j_start:j_end,i_start:i_end])
    hatavg[kte-1,j_start:j_end,i_start:i_end] = 0.5 * (cft1 * (hat[ktes1,j_start:j_end,i_start-1:i_end-1] +
                                                             hat[ktes1,j_start:j_end,i_start:i_end]) +
                                                     cft2 * (hat[ktes2,j_start:j_end,i_start-1:i_end-1] +
                                                             hat[ktes2,j_start:j_end,i_start:i_end]))
    tmp1[kts:ktf,j_start:j_end,i_start:i_end] =  (hatavg[kts+1:ktf+1,j_start:j_end,i_start:i_end]- 
                                                 hatavg[kts:ktf,j_start:j_end,i_start:i_end]) * (
                                                 0.25*(zx[kts:ktf,j_start-1:j_end-1,i_start:i_end] + 
                                                 zx[kts:ktf,j_start:j_end,i_start:i_end] +
                                                 zx[kts+1:ktf+1,j_start-1:j_end-1,i_start:i_end] +
                                                 zx[kts+1:ktf+1,j_start:j_end,i_start:i_end]) * 
                                                 0.25 * (rdzw[kts:ktf,j_start:j_end,i_start:i_end] + 
                                                 rdzw[kts:ktf,j_start-1:j_end-1,i_start:i_end] +
                                                 rdzw[kts:ktf,j_start-1:j_end-1,i_start-1:i_end-1] +
                                                 rdzw[kts:ktf,j_start:j_end,i_start-1:i_end-1]))
    
    # suppose sfs_opt = 0
    defor12[kts:ktf,j_start:j_end,i_start:i_end] = defor12[kts:ktf,j_start:j_end,i_start:i_end] + \
                                                   mm_e[kts:ktf,j_start:j_end,i_start:i_end] * ( rdx *
                                                   (hat[kts:ktf,j_start:j_end,i_start:i_end] - 
                                                    hat[kts:ktf,j_start:j_end,i_start-1:i_end-1]) - 
                                                   tmp1[kts:ktf,j_start:j_end,i_start:i_end])
    defor12[kts:kte,jts:jte, ids] = defor12[kts:kte,jts:jte, ids+1]
    defor12[kts:kte,jts:jte, ide-1] = defor12[kts:kte,jts:jte, ide-2]
    defor12[kts:kte,jds, its:ite] = defor12[kts:kte,jds+1, its:ite]
    defor12[kts:kte,jde-1, its:ite] = defor12[kts:kte,jde-2, its:ite]
      
    i_start = its
    i_end = ide - 1
    j_start = jts
    j_end = jde - 1
    
    mm[j_start:j_end,i_start:i_end] = msfux[j_start:j_end,i_start:i_end] \
                                    * msfuy[j_start:j_end,i_start:i_end]
    mm_e = mm.repeat(nzall,1,1)
    hat[kts:kte,j_start:j_end,i_start:i_end] = w[kts:kte,j_start:j_end,i_start:i_end] \
                                             / msfty_e[kts:kte,j_start:j_end,i_start:i_end]
    hat[kts:kte,j_start:j_end,i_start-1] = w[kts:kte,j_start:j_end,i_start-1] \
                                         / msfty_e[kts:kte,j_start:j_end,i_start-1]
    hat[kts:kte,j_start-1,i_start:i_end] = w[kts:kte,j_start-1,i_start:i_end] \
                                         / msfty_e[kts:kte,j_start-1,i_start:i_end]
    hatavg[kts:ktf,j_start:j_end,i_start:i_end] = 0.25 * (hat[kts:ktf,j_start:j_end,i_start:i_end] +
                                                          hat[kts+1:ktf+1,j_start:j_end,i_start:i_end] +
                                                          hat[kts:ktf,j_start:j_end,i_start-1:i_end-1] +
                                                          hat[kts+1:ktf+1,j_start:j_end,i_start-1:i_end-1])
    tmp1[kts+1:ktf,j_start:j_end,i_start:i_end] = (hatavg[kts+1:ktf,j_start:j_end,i_start:i_end] -
                                                   hatavg[kts:ktf-1,j_start:j_end,i_start:i_end]) * \
                                                   zx[kts+1:ktf,j_start:j_end,i_start:i_end] * \
                                                0.5 * (rdz[kts+1:ktf,j_start:j_end,i_start:i_end] +
                                                       rdz[kts+1:ktf,j_start:j_end,i_start-1:i_end-1])
    defor13[kts+1:ktf,j_start:j_end,i_start:i_end] = mm_e[kts+1:ktf,j_start:j_end,i_start:i_end] * (
                                                     rdx * (hat[kts+1:ktf,j_start:j_end,i_start:i_end] -
                                                            hat[kts+1:ktf,j_start:j_end,i_start-1:i_end-1]) 
                                                     - tmp1[kts+1:ktf,j_start:j_end,i_start:i_end])
    defor13[kts,j_start:j_end,i_start:i_end] = 0.
    defor13[ktf,j_start:j_end,i_start:i_end] = 0.
    
    tmp1[kts+1:ktf,j_start:j_end,i_start:i_end] = (u[kts+1:ktf,j_start:j_end,i_start:i_end] -
                                                   u[kts:ktf-1,j_start:j_end,i_start:i_end]) * \
                                            0.5 * (rdz[kts+1:ktf,j_start:j_end,i_start:i_end]+
                                                   rdz[kts+1:ktf,j_start:j_end,i_start-1:i_end-1])
    defor13[kts+1:ktf,j_start:j_end,i_start:i_end] = defor13[kts+1:ktf,j_start:j_end,i_start:i_end] + \
                                                     tmp1[kts+1:ktf,j_start:j_end,i_start:i_end]
                                                     
    i_start = its
    i_end = ide - 1
    j_start = jts
    j_end = jde - 1
    
    mm[j_start:j_end,i_start:i_end] = msfvx[j_start:j_end,i_start:i_end] \
                                    * msfvy[j_start:j_end,i_start:i_end]
    mm_e = mm.repeat(nzall,1,1)
    hat[kts:kte,j_start:j_end,i_start:i_end] = w[kts:kte,j_start:j_end,i_start:i_end] \
                                             / msftx_e[kts:kte,j_start:j_end,i_start:i_end]
    hat[kts:kte,j_start:j_end,i_start-1] = w[kts:kte,j_start:j_end,i_start-1] \
                                         / msftx_e[kts:kte,j_start:j_end,i_start-1]
    hat[kts:kte,j_start-1,i_start:i_end] = w[kts:kte,j_start-1,i_start:i_end] \
                                         / msftx_e[kts:kte,j_start-1,i_start:i_end]
    hatavg[kts:ktf,j_start:j_end,i_start:i_end] = 0.25 * (hat[kts:ktf,j_start:j_end,i_start:i_end] +
                                                          hat[kts+1:ktf+1,j_start:j_end,i_start:i_end] +
                                                          hat[kts:ktf,j_start-1:j_end-1,i_start:i_end] +
                                                          hat[kts+1:ktf+1,j_start-1:j_end-1,i_start:i_end])
    tmp1[kts+1:ktf,j_start:j_end,i_start:i_end] = (hatavg[kts+1:ktf,j_start:j_end,i_start:i_end] -
                                                   hatavg[kts:ktf-1,j_start:j_end,i_start:i_end]) * \
                                                   zy[kts+1:ktf,j_start:j_end,i_start:i_end] * \
                                                0.5 * (rdz[kts+1:ktf,j_start:j_end,i_start:i_end] +
                                                       rdz[kts+1:ktf,j_start-1:j_end-1,i_start:i_end])
    defor23[kts+1:ktf,j_start:j_end,i_start:i_end] = mm_e[kts+1:ktf,j_start:j_end,i_start:i_end] * (
                                                     rdy * (hat[kts+1:ktf,j_start:j_end,i_start:i_end] -
                                                            hat[kts+1:ktf,j_start-1:j_end-1,i_start:i_end])
                                                     - tmp1[kts+1:ktf,j_start:j_end,i_start:i_end])
    defor23[kts,j_start:j_end,i_start:i_end] = 0.
    defor23[ktf,j_start:j_end,i_start:i_end] = 0.
    
    tmp1[kts+1:ktf,j_start:j_end,i_start:i_end] = (v[kts+1:ktf,j_start:j_end,i_start:i_end] -
                                                   v[kts:ktf-1,j_start:j_end,i_start:i_end]) * \
                                            0.5 * (rdz[kts+1:ktf,j_start:j_end,i_start:i_end]+
                                                   rdz[kts+1:ktf,j_start-1:j_end-1,i_start:i_end])
    defor23[kts+1:ktf,j_start:j_end,i_start:i_end] = defor23[kts+1:ktf,j_start:j_end,i_start:i_end] + \
                                                     tmp1[kts+1:ktf,j_start:j_end,i_start:i_end]
    defor13[kts:kte,jts:jte, ids] = defor13[kts:kte,jts:jte, ids+1]
    defor13[kts:kte,jts:jte, ide-1] = defor13[kts:kte,jts:jte, ide-2]
    defor13[kts:kte,jds, its:ite] = defor13[kts:kte,jds+1, its:ite]
    defor13[kts:kte,jde-1, its:ite] = defor13[kts:kte,jde-2, its:ite]
    defor23[kts:kte,jts:jte, ids] = defor23[kts:kte,jts:jte, ids+1]
    defor23[kts:kte,jts:jte, ide-1] = defor23[kts:kte,jts:jte, ide-2]
    defor23[kts:kte,jds, its:ite] = defor23[kts:kte,jds+1, its:ite]
    defor23[kts:kte,jde-1, its:ite] = defor23[kts:kte,jde-2, its:ite]
    return defor11,defor12,defor13,defor22,defor23,defor33,div

# Turbulent diffusivities xkmh / xkmv / xkhh / xkhv.
def calculate_km_kh(dt,                        \
                    dampcoef, zdamp, damp_opt,               \
                    xkmh, xkmv, xkhh, xkhv,                  \
                    BN2, khdif, kvdif, div,                  \
                    defor11, defor22, defor33,               \
                    defor12, defor13, defor23,               \
                    tke, p8w, t8w, theta, t, p, moist,       \
                    dn, dnw, dx, dy, rdz, rdzw, isotropic,   \
                    n_moist, cf1, cf2, cf3, warm_rain,       \
                    mix_upper_bound,                         \
                    msftx, msfty,                            \
                    zx, zy,                                  \
                    ids, ide, jds, jde, kds, kde,            \
                    ims, ime, jms, jme, kms, kme,            \
                    its, ite, jts, jte, kts, kte             ):
    ktf     = min( kte, kde-1 )
    i_start = its
    i_end   = min( ite, ide-1 )
    j_start = jts
    j_end   = min( jte, jde-1 )
    
    # calculate (B)N2
    ktes1 = kte - 2
    ktes2 = kte - 3
    qc_cr   = 0.00001
    
    dnw_e = dnw.unsqueeze(1).unsqueeze(2).repeat(1,nyall,nxall)
    dn_e = dn.unsqueeze(1).unsqueeze(2).repeat(1,nyall,nxall)
    
    tmp1sfc = torch.zeros((nyall,nxall)).to(device)
    tmp1top = torch.zeros((nyall,nxall)).to(device)
    qctmp = torch.zeros((nzall,nyall,nxall)).to(device)
    tmp1 = torch.zeros((nzall,nyall,nxall)).to(device)
    es = torch.zeros((nzall,nyall,nxall)).to(device)
    qvs = torch.zeros((nzall,nyall,nxall)).to(device)
    tmpdz = torch.zeros((nzall,nyall,nxall)).to(device)
    xlvqv = torch.zeros((nzall,nyall,nxall)).to(device)
    coefa = torch.zeros((nzall,nyall,nxall)).to(device)
    thetaep1 = torch.zeros((nzall,nyall,nxall)).to(device)
    thetaem1 = torch.zeros((nzall,nyall,nxall)).to(device)
    BN2_1 = torch.zeros((nzall,nyall,nxall)).to(device)
    BN2_0 = torch.zeros((nzall,nyall,nxall)).to(device)
    BN2 = torch.zeros((nzall,nyall,nxall)).to(device)
    thetasfc = torch.zeros((nyall,nxall)).to(device)
    thetaesfc = torch.zeros((nyall,nxall)).to(device)
    qvsfc = torch.zeros((nyall,nxall)).to(device)
    
    def2 = torch.zeros((nzall,nyall,nxall)).to(device)
    tmp = torch.zeros((nzall,nyall,nxall)).to(device)
    mlen_h = torch.zeros((nyall,nxall)).to(device)
    
    qctmp[kts:ktf,j_start:j_end,i_start:i_end] = moist[P_QC,kts:ktf,j_start:j_end,i_start:i_end] + 0.0
    tmp1[kts:ktf,j_start:j_end,i_start:i_end] = moist[P_QV,kts:ktf,j_start:j_end,i_start:i_end] + \
            moist[P_QC,kts:ktf,j_start:j_end,i_start:i_end] + moist[P_QI,kts:ktf,j_start:j_end,i_start:i_end]
    tmp1sfc[j_start:j_end,i_start:i_end] = cf1 * (moist[P_QV,0,j_start:j_end,i_start:i_end] + moist[P_QC,0,j_start:j_end,i_start:i_end] + moist[P_QI,0,j_start:j_end,i_start:i_end]) \
            + cf2 * (moist[P_QV,1,j_start:j_end,i_start:i_end] + moist[P_QC,1,j_start:j_end,i_start:i_end] + moist[P_QI,1,j_start:j_end,i_start:i_end]) \
            + cf3 * (moist[P_QV,2,j_start:j_end,i_start:i_end] + moist[P_QC,2,j_start:j_end,i_start:i_end] + moist[P_QI,2,j_start:j_end,i_start:i_end])
    tmp1top[j_start:j_end,i_start:i_end] = moist[P_QV,ktes1,j_start:j_end,i_start:i_end] + (moist[P_QV,ktes1,j_start:j_end,i_start:i_end] - moist[P_QV,ktes2,j_start:j_end,i_start:i_end]) * \
              0.5 * dnw_e[ktes1,j_start:j_end,i_start:i_end]/dn_e[ktes1,j_start:j_end,i_start:i_end]  \
            + moist[P_QC,ktes1,j_start:j_end,i_start:i_end] + (moist[P_QC,ktes1,j_start:j_end,i_start:i_end] - moist[P_QC,ktes2,j_start:j_end,i_start:i_end]) * \
              0.5 * dnw_e[ktes1,j_start:j_end,i_start:i_end]/dn_e[ktes1,j_start:j_end,i_start:i_end]  \
            + moist[P_QI,ktes1,j_start:j_end,i_start:i_end] + (moist[P_QI,ktes1,j_start:j_end,i_start:i_end] - moist[P_QI,ktes2,j_start:j_end,i_start:i_end]) * \
              0.5 * dnw_e[ktes1,j_start:j_end,i_start:i_end]/dn_e[ktes1,j_start:j_end,i_start:i_end]  
    
    es[kts:ktf,j_start:j_end,i_start:i_end] = 1000.0 * SVP1 * torch.exp(SVP2 * (t[kts:ktf,j_start:j_end,i_start:i_end] - SVPT0) / \
                                   (t[kts:ktf,j_start:j_end,i_start:i_end] - SVP3))
    qvs[kts:ktf,j_start:j_end,i_start:i_end] =  EP_2 * es[kts:ktf,j_start:j_end,i_start:i_end] / \
                (p[kts:ktf,j_start:j_end,i_start:i_end] - es[kts:ktf,j_start:j_end,i_start:i_end])
    
    tmpdz[kts+1:ktf-1,j_start:j_end,i_start:i_end] = 1.0/rdz[kts+1:ktf-1,j_start:j_end,i_start:i_end] + \
                                                     1.0/rdz[kts+2:ktf,j_start:j_end,i_start:i_end]
    xlvqv[kts+1:ktf-1, j_start:j_end, i_start:i_end] = XLV * \
                              moist[P_QV, kts+1:ktf-1, j_start:j_end, i_start:i_end]
    coefa[kts+1:ktf-1, j_start:j_end, i_start:i_end] = (1.0 + xlvqv[kts+1:ktf-1, j_start:j_end, i_start:i_end]/ \
                              r_d / t[kts+1:ktf-1, j_start:j_end, i_start:i_end] ) / \
                              (1.0 + XLV * xlvqv[kts+1:ktf-1, j_start:j_end, i_start:i_end]/ Cp / r_v / \
                               t[kts+1:ktf-1, j_start:j_end, i_start:i_end]/ \
                               t[kts+1:ktf-1, j_start:j_end, i_start:i_end])/ \
                               theta[kts+1:ktf-1, j_start:j_end, i_start:i_end]
    thetaep1[kts+1:ktf-1, j_start:j_end, i_start:i_end] = theta[kts+2:ktf, j_start:j_end, i_start:i_end] * (1.0 + XLV * 
                    qvs[kts+2:ktf, j_start:j_end, i_start:i_end]) / Cp / t[kts+2:ktf, j_start:j_end, i_start:i_end]
    thetaem1[kts+1:ktf-1, j_start:j_end, i_start:i_end] = theta[kts:ktf-2, j_start:j_end, i_start:i_end] * (1.0 + XLV * 
                    qvs[kts:ktf-2, j_start:j_end, i_start:i_end]) / Cp / t[kts:ktf-2, j_start:j_end, i_start:i_end]
    BN2_1[kts+1:ktf-1, j_start:j_end, i_start:i_end] = g * (coefa[kts+1:ktf-1, j_start:j_end, i_start:i_end] * 
                                                     (thetaep1[kts+1:ktf-1, j_start:j_end, i_start:i_end] \
                                                      - thetaem1[kts+1:ktf-1, j_start:j_end, i_start:i_end] \
                                                      ) / tmpdz[kts+1:ktf-1, j_start:j_end, i_start:i_end] - 
                                                     (tmp1[kts+2:ktf, j_start:j_end, i_start:i_end] - 
                                                      tmp1[kts:ktf-2, j_start:j_end, i_start:i_end])/ \
                                                      tmpdz[kts+1:ktf-1, j_start:j_end, i_start:i_end] )
    BN2_0[kts+1:ktf-1, j_start:j_end, i_start:i_end] = g*((theta[kts+2:ktf, j_start:j_end, i_start:i_end] - 
                                                           theta[kts:ktf-2, j_start:j_end, i_start:i_end]) / \
                                                           theta[kts+1:ktf-1, j_start:j_end, i_start:i_end] / \
                                                           tmpdz[kts+1:ktf-1, j_start:j_end, i_start:i_end] \
                                                          + 1.61 * (moist[P_QV,kts+2:ktf, j_start:j_end, i_start:i_end] -
                                                               moist[P_QV,kts:ktf-2, j_start:j_end, i_start:i_end]) / \
                                                           tmpdz[kts+1:ktf-1, j_start:j_end, i_start:i_end] \
                                                          - (tmp1[kts+2:ktf, j_start:j_end, i_start:i_end] -
                                                            tmp1[kts:ktf-2, j_start:j_end, i_start:i_end]) / \
                                                           tmpdz[kts+1:ktf-1, j_start:j_end, i_start:i_end])
    condition = torch.logical_or(moist[P_QV,:,:,:] >= qvs[:,:,:], qctmp[:,:,:] >= torch.tensor(qc_cr))
    BN2 = torch.where(condition, BN2_1, BN2_0)
    
    tmpdz[kts,j_start:j_end,i_start:i_end] = 1.0/rdz[kts+1,j_start:j_end,i_start:i_end] + \
                                             0.5/rdzw[kts,j_start:j_end,i_start:i_end]
    thetasfc[j_start:j_end, i_start:i_end] = t8w[kts, j_start:j_end, i_start:i_end] / ((
                                             p8w[kts, j_start:j_end, i_start:i_end] / p1000mb)**(r_d / Cp))
    qvsfc[j_start:j_end, i_start:i_end] = cf1 * qvs[0,j_start:j_end, i_start:i_end] + \
                                          cf2 * qvs[1,j_start:j_end, i_start:i_end] + \
                                          cf3 * qvs[2,j_start:j_end, i_start:i_end]
    xlvqv[kts,j_start:j_end, i_start:i_end] = XLV * moist[P_QV, kts, j_start:j_end, i_start:i_end]
    coefa[kts,j_start:j_end, i_start:i_end] = (1.0 + xlvqv[kts, j_start:j_end, i_start:i_end]/ \
                              r_d / t[kts, j_start:j_end, i_start:i_end] ) / \
                              (1.0 + XLV * xlvqv[kts, j_start:j_end, i_start:i_end]/ Cp / r_v / \
                               t[kts, j_start:j_end, i_start:i_end]/ \
                               t[kts, j_start:j_end, i_start:i_end])/ \
                               theta[kts, j_start:j_end, i_start:i_end]
    thetaep1[kts, j_start:j_end, i_start:i_end] = theta[kts+1, j_start:j_end, i_start:i_end] * (1.0 + XLV * 
                    qvs[kts+1, j_start:j_end, i_start:i_end]) / Cp / t[kts+1, j_start:j_end, i_start:i_end]
    thetaesfc[j_start:j_end,i_start:i_end] = thetasfc[j_start:j_end,i_start:i_end] * (1.0 + \
                    XLV * qvsfc[j_start:j_end,i_start:i_end] / Cp / t8w[kts,j_start:j_end,i_start:i_end])
    BN2_1[kts,j_start:j_end,i_start:i_end] = g * (coefa[kts,j_start:j_end, i_start:i_end] * (
                    thetaep1[kts,j_start:j_end, i_start:i_end] - thetaesfc[j_start:j_end, i_start:i_end])/
                    tmpdz[kts,j_start:j_end, i_start:i_end] - (tmp1[kts+1,j_start:j_end, i_start:i_end] - 
                         tmp1sfc[j_start:j_end, i_start:i_end])/tmpdz[kts,j_start:j_end, i_start:i_end])
    
    qvsfc[j_start:j_end, i_start:i_end] = cf1 * moist[P_QV, 0,j_start:j_end, i_start:i_end] + \
                                          cf2 * moist[P_QV, 1,j_start:j_end, i_start:i_end] + \
                                          cf3 * moist[P_QV, 2,j_start:j_end, i_start:i_end]
    tmpdz[kts,j_start:j_end,i_start:i_end] = 1./rdzw[kts,j_start:j_end,i_start:i_end]
    BN2_0[kts,j_start:j_end,i_start:i_end] = g * ((theta[kts+1,j_start:j_end,i_start:i_end] - 
                    theta[kts,j_start:j_end,i_start:i_end])/ theta[kts,j_start:j_end,i_start:i_end]/
                    tmpdz[kts,j_start:j_end,i_start:i_end] +
                    1.61 * (moist[P_QV,kts+1,j_start:j_end,i_start:i_end] - qvsfc[j_start:j_end,i_start:i_end])/
                    tmpdz[kts,j_start:j_end,i_start:i_end] -
                    (tmp1[kts+1,j_start:j_end,i_start:i_end] - tmp1sfc[j_start:j_end,i_start:i_end])/
                    tmpdz[kts,j_start:j_end,i_start:i_end])
    condition = torch.logical_or(moist[P_QV,kts,:,:] >= qvs[kts,:,:], qctmp[kts,:,:] >= qc_cr)
    BN2[kts,:,:] = torch.where(condition, BN2_1[kts,:,:], BN2_0[kts,:,:])
    
    BN2[ktf-1,:,:] = BN2[ktf-2,:,:]
        
    ### smag2d_km
    ktf = min(kte,kde-1)
    i_start = its+1
    i_end   = min(ite,ide-2)
    j_start = jts+1
    j_end   = min(jte,jde-2)
    
    pr = prandtl
        
    def2[kts:ktf,j_start:j_end,i_start:i_end] = 0.25 * ( 
        (defor11[kts:ktf,j_start:j_end,i_start:i_end] - defor22[kts:ktf,j_start:j_end,i_start:i_end]) 
      * (defor11[kts:ktf,j_start:j_end,i_start:i_end] - defor22[kts:ktf,j_start:j_end,i_start:i_end]))
    #print("in calc xkmh 0:", defor11[20,443,5],defor22[20,443,580])
    tmp[kts:ktf,j_start:j_end,i_start:i_end] = 0.25 * ( defor12[kts:ktf,j_start:j_end,i_start:i_end] +
                                                        defor12[kts:ktf,j_start+1:j_end+1,i_start:i_end] +
                                                        defor12[kts:ktf,j_start:j_end,i_start+1:i_end+1] +
                                                        defor12[kts:ktf,j_start+1:j_end+1,i_start+1:i_end+1])
    
    def2[kts:ktf,j_start:j_end,i_start:i_end] = def2[kts:ktf,j_start:j_end,i_start:i_end] + \
        tmp[kts:ktf,j_start:j_end,i_start:i_end] * tmp[kts:ktf,j_start:j_end,i_start:i_end]
    
    mlen_h[j_start:j_end,i_start:i_end] = (dx/msftx[j_start:j_end,i_start:i_end] * 
                                           dy/msfty[j_start:j_end,i_start:i_end]) ** 0.5
    tmp[kts:ktf,j_start:j_end,i_start:i_end] = def2[kts:ktf,j_start:j_end,i_start:i_end] ** 0.5
    mlen_h_e = mlen_h.repeat(nzall,1,1)
    
    xkmh[kts:ktf,j_start:j_end,i_start:i_end] = c_s * c_s * mlen_h_e[kts:ktf,j_start:j_end,i_start:i_end] \
                                          * mlen_h_e[kts:ktf,j_start:j_end,i_start:i_end] \
                                          * tmp[kts:ktf,j_start:j_end,i_start:i_end]
    
    condition = xkmh < 10. *  mlen_h_e
    xkmh = torch.where(condition, xkmh, 10.*mlen_h_e)
    
    xkmv[:,:,:] = 0.
    xkhh[kts:ktf,j_start:j_end,i_start:i_end] = xkmh[kts:ktf,j_start:j_end,i_start:i_end] / pr
    xkhv[:,:,:] = 0.
    
    return BN2,xkmh,xkmv,xkhh,xkhv

# Fold physics tendencies into the accumulators.
def update_phy_ten(rph_tendf,rt_tendf,ru_tendf,rv_tendf,moist_tendf, \
                      scalar_tendf,mu_tendf,                         \
                      RTHBLTEN,RUBLTEN,RVBLTEN,                      \
                      RQVBLTEN,RQCBLTEN,RQIBLTEN,                    \
                      n_moist,n_scalar,rk_step,adv_moist_cond,       \
                      ids, ide, jds, jde, kds, kde,                  \
                      ims, ime, jms, jme, kms, kme,                  \
                      its, ite, jts, jte, kts, kte):
    
    ## only do update for bl_pbl_physics
    
    ## phy_bl_ten
    ## suppose YSU scheme
    rt_tendf = add_a2a(rt_tendf, RTHBLTEN,                \
                   ids,ide, jds, jde, kds, kde,             \
                   ims, ime, jms, jme, kms, kme,            \
                   its, ite, jts, jte, kts, kte             )
    ru_tendf = add_a2c_u(ru_tendf, RUBLTEN,                \
                   ids,ide, jds, jde, kds, kde,             \
                   ims, ime, jms, jme, kms, kme,            \
                   its, ite, jts, jte, kts, kte             )
    rv_tendf = add_a2c_v(rv_tendf, RVBLTEN,                \
                   ids,ide, jds, jde, kds, kde,             \
                   ims, ime, jms, jme, kms, kme,            \
                   its, ite, jts, jte, kts, kte             )
    moist_tendf[P_QV,:,:,:] = add_a2a(moist_tendf[P_QV,:,:,:], RQVBLTEN,    \
                   ids,ide, jds, jde, kds, kde,             \
                   ims, ime, jms, jme, kms, kme,            \
                   its, ite, jts, jte, kts, kte             )
    moist_tendf[P_QC,:,:,:] = add_a2a(moist_tendf[P_QC,:,:,:], RQCBLTEN,    \
                   ids,ide, jds, jde, kds, kde,             \
                   ims, ime, jms, jme, kms, kme,            \
                   its, ite, jts, jte, kts, kte             )
    moist_tendf[P_QI,:,:,:] = add_a2a(moist_tendf[P_QI,:,:,:], RQIBLTEN,    \
                   ids,ide, jds, jde, kds, kde,             \
                   ims, ime, jms, jme, kms, kme,            \
                   its, ite, jts, jte, kts, kte             )
    return rt_tendf, ru_tendf, rv_tendf, moist_tendf

def add_a2a(lvar,rvar,                  \
            ids,ide, jds, jde, kds, kde,             \
            ims, ime, jms, jme, kms, kme,            \
            its, ite, jts, jte, kts, kte             ):
    i_start = its
    i_end   = min(ite,ide-1)
    j_start = jts
    j_end   = min(jte,jde-1)
    ktf = min(kte,kde-1)
    lvar[kts:ktf,j_start:j_end,i_start:i_end] = lvar[kts:ktf,j_start:j_end,i_start:i_end] + \
                 rvar[kts:ktf,j_start:j_end,i_start:i_end]
    return lvar

def add_a2c_u(lvar,rvar,                  \
            ids,ide, jds, jde, kds, kde,             \
            ims, ime, jms, jme, kms, kme,            \
            its, ite, jts, jte, kts, kte             ):
    ktf=min(kte,kde-1)
    i_start = its
    i_end   = ite
    j_start = jts
    j_end   = min(jte,jde-1)
    lvar[kts:ktf,j_start:j_end,i_start:i_end] = lvar[kts:ktf,j_start:j_end,i_start:i_end] + \
              0.5 * rvar[kts:ktf,j_start:j_end,i_start:i_end] + \
              0.5 * rvar[kts:ktf,j_start:j_end,i_start-1:i_end-1]
    return lvar

def add_a2c_v(lvar,rvar,                  \
            ids,ide, jds, jde, kds, kde,             \
            ims, ime, jms, jme, kms, kme,            \
            its, ite, jts, jte, kts, kte             ):
    ktf=min(kte,kde-1)
    i_start = its
    i_end   = min(ite,ide-1)
    j_start = jts
    j_end   = jte
    lvar[kts:ktf,j_start:j_end,i_start:i_end] = lvar[kts:ktf,j_start:j_end,i_start:i_end] + \
              0.5 * rvar[kts:ktf,j_start:j_end,i_start:i_end] + \
              0.5 * rvar[kts:ktf,j_start-1:j_end-1,i_start:i_end]
    return lvar

# Wrapper for one RK advection tendency step (stub in this port).
def rk_tendency(rk_step,                                         \
                ru_tend, rv_tend, rw_tend, ph_tend, t_tend,      \
                ru_tendf, rv_tendf, rw_tendf, ph_tendf, t_tendf, \
                mu_tend, u_save, v_save, w_save, ph_save,        \
                t_save, mu_save, RTHFTEN,                        \
                ru, rv, rw, ww,                                  \
                u, v, w, t, ph,                                  \
                u_old, v_old, w_old, t_old, ph_old,              \
                h_diabatic, phb,t_init,                          \
                mu, mut, muu, muv, mub, c1h, c2h, c1f, c2f,      \
                al, alt, p, pb, php, cqu, cqv, cqw,              \
                u_base, v_base, t_base, qv_base, z_base,         \
                msfux, msfuy, msfvx, msfvx_inv,                  \
                msfvy, msftx, msfty,                             \
                clat, f, e, sina, cosa,                          \
                fnm, fnp, rdn, rdnw,                             \
                dt, rdx, rdy, khdif, kvdif, xkmhd, xkhh,         \
                diff_6th_opt, diff_6th_factor,                   \
                adv_opt,                                         \
                dampcoef,zdamp,damp_opt,rad_nudge,               \
                cf1, cf2, cf3, cfn, cfn1, n_moist,               \
                non_hydrostatic, top_lid,                        \
                u_frame, v_frame,                                \
                ids, ide, jds, jde, kds, kde,                    \
                ims, ime, jms, jme, kms, kme,                    \
                its, ite, jts, jte, kts, kte,                    \
                max_vert_cfl, max_horiz_cfl):
    
    #ru_tend = advect_u()
    
    return

# Advection tendency for u.
def advect_u(u, u_old, tendency,            \
             ru, rv, rom,                   \
             c1, c2,                        \
             mut, time_step,                \
             msfux, msfuy, msfvx, msfvy,    \
             msftx, msfty,                  \
             fzm, fzp,                      \
             rdx, rdy, rdzw,                \
             ids, ide, jds, jde, kds, kde,  \
             ims, ime, jms, jme, kms, kme,  \
             its, ite, jts, jte, kts, kte):
    ktf=min(kte,kde-1)

    # horz_order=5
    # y advection
    i_start = its
    i_end   = ite
    
    i_start = max(ids+1,its)
    i_end   = min(ide-1,ite)
    
    j_start = jts
    j_end   = min(jte,jde-1)
    
    j_start_f = j_start
    j_end_f   = j_end+1
    
    j_start = max(jts,jds+1)
    j_start_f = jds+3
    
    j_end = min(jte,jde-2)
    j_end_f = jde-3
    
    jp1 = 1
    jp0 = 0
    
    fqy = torch.zeros((2,nzall,nyall,nxall))
    vel = torch.zeros((nzall,nyall,nxall))
    mrdy = torch.zeros((nyall,nxall))
    fqx = torch.zeros((nzall,nyall,nxall))
    mrdx = torch.zeros((nyall,nxall))
    vflux = torch.zeros((nzall,nyall,nxall))
    
    rdzw_e = rdzw.unsqueeze(1).unsqueeze(2).repeat(1,nyall,nxall)
    
    vel[kts:ktf, j_start_f:j_end_f, i_start:i_end] = 0.5*(rv[kts:ktf, j_start_f:j_end_f, i_start:i_end] + \
                                                      rv[kts:ktf, j_start_f:j_end_f, i_start-1:i_end-1])
    vel_modified8 = vel.clone()
    fqy[jp1,kts:ktf, j_start_f:j_end_f, i_start:i_end] = vel_modified8[kts:ktf, j_start_f:j_end_f, i_start:i_end] * \
              flux5_u(u[kts:ktf, j_start_f-3:j_end_f-3, i_start:i_end],u[kts:ktf, j_start_f-2:j_end_f-2, i_start:i_end],
                      u[kts:ktf, j_start_f-1:j_end_f-1, i_start:i_end],u[kts:ktf, j_start_f:j_end_f, i_start:i_end],
                      u[kts:ktf, j_start_f+1:j_end_f+1, i_start:i_end],u[kts:ktf, j_start_f+2:j_end_f+2, i_start:i_end],
                      vel_modified8[kts:ktf, j_start_f:j_end_f, i_start:i_end])
    fqy[jp1,kts:ktf, jds+1, i_start:i_end] = 0.25 * (rv[kts:ktf, jds+1, i_start:i_end] + rv[kts:ktf, jds+1, i_start-1:i_end-1]) * \
                                                    (u[kts:ktf, jds+1, i_start:i_end] + u[kts:ktf, jds, i_start:i_end])
    vel[kts:ktf, jds+2, i_start:i_end] = 0.5*(rv[kts:ktf, jds+2, i_start:i_end] + \
                                              rv[kts:ktf, jds+2, i_start-1:i_end-1])
    vel_modified7 = vel.clone()
    fqy[jp1,kts:ktf, jds+2, i_start:i_end] = vel_modified7[kts:ktf, jds+2, i_start:i_end] * \
              flux3_u(u[kts:ktf, jds, i_start:i_end],u[kts:ktf, jds+1, i_start:i_end],
                      u[kts:ktf, jds+2, i_start:i_end],u[kts:ktf, jds+3, i_start:i_end],
                      vel_modified7[kts:ktf, jds+2, i_start:i_end])
    fqy[jp1,kts:ktf, jde-2, i_start:i_end] = 0.25 * (rv[kts:ktf, jde-2, i_start:i_end] + rv[kts:ktf, jde-2, i_start-1:i_end-1]) * \
                                                    (u[kts:ktf, jde-2, i_start:i_end] + u[kts:ktf, jde-3, i_start:i_end])
    vel[kts:ktf, jde-3, i_start:i_end] = 0.5*(rv[kts:ktf, jde-3, i_start:i_end] + \
                                              rv[kts:ktf, jde-3, i_start-1:i_end-1])     #注意单独引用ite，ide，jte，jde等末尾索引时，python中需再减1
    vel_modified6 = vel.clone()
    fqy[jp1,kts:ktf, jde-3, i_start:i_end] = vel_modified6[kts:ktf, jde-3, i_start:i_end] * \
              flux3_u(u[kts:ktf, jde-5, i_start:i_end],u[kts:ktf, jde-4, i_start:i_end],
                      u[kts:ktf, jde-3, i_start:i_end],u[kts:ktf, jde-2, i_start:i_end],
                      vel_modified6[kts:ktf, jde-2, i_start:i_end])
    mrdy[j_start+1:j_end+1, i_start:i_end] = msfux[j_start:j_end, i_start:i_end] * rdy
    mrdy_e = mrdy.repeat(nzall,1,1)
    tendency[kts:ktf, j_start:j_end, i_start:i_end] = tendency[kts:ktf, j_start:j_end, i_start:i_end] - \
             mrdy_e[kts:ktf, j_start+1:j_end+1, i_start:i_end] * (fqy[jp1,kts:ktf, j_start+1:j_end+1, i_start:i_end] -
                             fqy[jp1,kts:ktf, j_start:j_end, i_start:i_end])  #### is it right?
    #print("in advect ru_tend: ", tendency[38, 480, 386])
    #print("in advect ru_tend: ", fqy[jp1,20, 603, 480],fqy[jp1,20,602,480],mrdy[603,480],msfux[602,480],j_end)
    #print("in advect ru_tend 1: ", tendency[20, 309, 603])
    #fqy[jp0, kts:ktf, jds+1:jde, i_start:i_end] = fqy[jp1, kts:ktf, jds+1:jde, i_start:i_end]
    #jtmp = jp1
    #jp1 = jp0
    #jp0 = jtmp
    #print("in advect ru_tend 1: ", tendency[20,443,603])
    # x advection
    i_start = its
    i_end   = ite

    j_start = jts
    j_end   = min(jte,jde-1)
    
    i_start_f = i_start
    i_end_f   = i_end+1
    
    i_start = max(ids+1,its)
    i_start_f = ids+3
    
    i_end = min(ide-1,ite)
    i_end_f = ide-2
    
    vel[kts:ktf, j_start:j_end, i_start_f:i_end_f] = 0.5*(ru[kts:ktf, j_start:j_end, i_start_f:i_end_f] + \
                                                      ru[kts:ktf, j_start:j_end, i_start_f-1:i_end_f-1])
    vel_modified5 = vel.clone()
    fqx[kts:ktf, j_start:j_end, i_start_f:i_end_f] = vel_modified5[kts:ktf, j_start:j_end, i_start_f:i_end_f] * \
              flux5_u(u[kts:ktf, j_start:j_end, i_start_f-3:i_end_f-3],u[kts:ktf, j_start:j_end, i_start_f-2:i_end_f-2],
                      u[kts:ktf, j_start:j_end, i_start_f-1:i_end_f-1],u[kts:ktf, j_start:j_end, i_start_f:i_end_f],
                      u[kts:ktf, j_start:j_end, i_start_f+1:i_end_f+1],u[kts:ktf, j_start:j_end, i_start_f+2:i_end_f+2],
                      vel_modified5[kts:ktf, j_start:j_end, i_start_f:i_end_f])
    #print("in advect ru_tend: ", fqx[20, 309, 603])
    #print("in advect ru_tend 000: ", fqx[20,443,603:605])
    ub = torch.zeros((nzall,nyall))
    condition = u[kts:ktf,j_start:j_end,ids+1] < 0.
    ub[kts:ktf,j_start:j_end] = torch.where(condition, u[kts:ktf,j_start:j_end,ids+1], u[kts:ktf,j_start:j_end,ids])
    fqx[kts:ktf,j_start:j_end,ids+1] = 0.25 * (ru[kts:ktf,j_start:j_end,ids+1] + ru[kts:ktf,j_start:j_end,ids])* \
                                            (u[kts:ktf,j_start:j_end,ids+1] + ub[kts:ktf,j_start:j_end])
    #print("in advect ru_tend 000: ", fqx[20,443,603:605])
    vel[kts:ktf, j_start:j_end, ids+2] = 0.5 * (ru[kts:ktf, j_start:j_end, ids+2] + ru[kts:ktf, j_start:j_end, ids+1])
    vel_modified4 = vel.clone()
    fqx[kts:ktf, j_start:j_end, ids+2] = vel_modified4[kts:ktf, j_start:j_end, ids+2] * \
            flux3_u(u[kts:ktf, j_start:j_end, ids],u[kts:ktf, j_start:j_end, ids+1],
                    u[kts:ktf, j_start:j_end, ids+2],u[kts:ktf, j_start:j_end, ids+3],
                    vel_modified4[kts:ktf, j_start:j_end, ids+2])
    #print("in advect ru_tend 00: ", fqx[20,443,603:605])
    condition = u[kts:ktf,j_start:j_end,ide-2] > 0.
    ub[kts:ktf,j_start:j_end] = torch.where(condition, u[kts:ktf,j_start:j_end,ide-2], u[kts:ktf,j_start:j_end,ide-1])
    fqx[kts:ktf,j_start:j_end,ide-1] = 0.25 * (ru[kts:ktf,j_start:j_end,ide-1] + ru[kts:ktf,j_start:j_end,ide-2])* \
                                            (u[kts:ktf,j_start:j_end,ide-2] + ub[kts:ktf,j_start:j_end])
    #print("in advect ru_tend 1: ", ru[20,309,604],ru[20,309,603],u[20,309,603],ub[20,309],fqx[20,309,604])
    #print("in advect ru_tend 0: ", fqx[20,443,604],ub[20,443],u[20,443,603],ru[20,443,604],ru[20,443,603])
    vel[kts:ktf, j_start:j_end, ide-2] = 0.5 * (ru[kts:ktf, j_start:j_end, ide-2] + ru[kts:ktf, j_start:j_end, ide-3])
    vel_modified3 = vel.clone()
    fqx[kts:ktf, j_start:j_end, ide-2] = vel_modified3[kts:ktf, j_start:j_end, ide-2] * \
            flux3_u(u[kts:ktf, j_start:j_end, ide-4],u[kts:ktf, j_start:j_end, ide-3],
                    u[kts:ktf, j_start:j_end, ide-2],u[kts:ktf, j_start:j_end, ide-1],
                    vel_modified3[kts:ktf, j_start:j_end, ide-2])
    #print("in advect ru_tend 1: ", ru[20,309,603],ru[20,309,602],fqx[20,309,603],vel[20,309,603])
    mrdx[j_start:j_end, i_start:i_end] = msfux[j_start:j_end, i_start:i_end] * rdx
    mrdx_e = mrdx.repeat(nzall,1,1)
    tendency[kts:ktf, j_start:j_end, i_start:i_end] = tendency[kts:ktf, j_start:j_end, i_start:i_end] - \
             mrdx_e[kts:ktf, j_start:j_end, i_start:i_end] * (fqx[kts:ktf, j_start:j_end, i_start+1:i_end+1] -
                                                              fqx[kts:ktf, j_start:j_end, i_start:i_end])
    #print("in advect ru_tend: ", tendency[38, 480, 386])
    #print("in advect ru_tend 2: ", tendency[20, 309, 603])
    #print("in advect ru_tend 2: ", tendency[20,443,603],fqx[20,443,603:605],mrdx_e[20,443,603:605])
    # z advection
    i_start = its
    i_end   = ite
    j_start = jts
    j_end   = min(jte,jde-1)
    
    i_start = max(ids+1,its)
    i_end   = min(ide-1,ite)
    
    vel[kts+3:ktf-2, j_start:j_end, i_start:i_end] = 0.5 * (rom[kts+3:ktf-2, j_start:j_end, i_start-1:i_end-1] +
                rom[kts+3:ktf-2, j_start:j_end, i_start:i_end])
    vel_modified2 = vel.clone()
    vflux[kts+3:ktf-2, j_start:j_end, i_start:i_end] = vel_modified2[kts+3:ktf-2, j_start:j_end, i_start:i_end] * \
                flux5_u(u[kts:ktf-5, j_start:j_end, i_start:i_end], u[kts+1:ktf-4, j_start:j_end, i_start:i_end],
                        u[kts+2:ktf-3, j_start:j_end, i_start:i_end], u[kts+3:ktf-2, j_start:j_end, i_start:i_end],
                        u[kts+4:ktf-1, j_start:j_end, i_start:i_end], u[kts+5:ktf, j_start:j_end, i_start:i_end],
                        -vel_modified2[kts+3:ktf-2, j_start:j_end, i_start:i_end])
    fzm_e = fzm.unsqueeze(1).unsqueeze(2).repeat(1,nyfull,nxfull)
    fzp_e = fzp.unsqueeze(1).unsqueeze(2).repeat(1,nyfull,nxfull)
    vflux[kts+1,j_start:j_end, i_start:i_end] = 0.5 * (rom[kts+1,j_start:j_end, i_start:i_end] +
                                                       rom[kts+1,j_start:j_end, i_start-1:i_end-1]) * \
                                                      (fzm_e[kts+1,j_start:j_end, i_start:i_end] * 
                                                       u[kts+1,j_start:j_end, i_start:i_end] + 
                                                       fzp_e[kts+1,j_start:j_end, i_start:i_end] *
                                                       u[kts,j_start:j_end, i_start:i_end])
    vel[kts+2,j_start:j_end, i_start:i_end] = 0.5 * (rom[kts+2,j_start:j_end, i_start:i_end] +
                                                     rom[kts+2,j_start:j_end, i_start-1:i_end-1])
    vel_modified1 = vel.clone()
    vflux[kts+2,j_start:j_end, i_start:i_end] = vel_modified1[kts+2,j_start:j_end, i_start:i_end] * \
                flux3_u(u[kts,j_start:j_end, i_start:i_end],u[kts+1,j_start:j_end, i_start:i_end],
                        u[kts+2,j_start:j_end, i_start:i_end],u[kts+3,j_start:j_end, i_start:i_end],
                        -vel_modified1[kts+2,j_start:j_end, i_start:i_end])
    vel[ktf-2,j_start:j_end, i_start:i_end] = 0.5 * (rom[ktf-2,j_start:j_end, i_start:i_end] +
                                                     rom[ktf-2,j_start:j_end, i_start-1:i_end-1])
    vflux[ktf-2,j_start:j_end, i_start:i_end] = vel[ktf-2,j_start:j_end, i_start:i_end] * \
                flux3_u(u[ktf-4,j_start:j_end, i_start:i_end],u[ktf-3,j_start:j_end, i_start:i_end],
                        u[ktf-2,j_start:j_end, i_start:i_end],u[ktf-1,j_start:j_end, i_start:i_end],
                        -vel[ktf-2,j_start:j_end, i_start:i_end])
    #print("in advect ru_tend 11: ", ktf, vel[38,480,386],u[36,480,386],u[37,480,386],u[38,480,386],u[39,480,386],rom[39,480,386],rom[39,480,385])
    vflux[ktf-1,j_start:j_end, i_start:i_end] = 0.5 * (rom[ktf-1, j_start:j_end, i_start:i_end] +
                                                       rom[ktf-1, j_start:j_end, i_start-1:i_end-1]) * \
                                                      (fzm_e[ktf-1,j_start:j_end, i_start:i_end] * 
                                                       u[ktf-1,j_start:j_end, i_start:i_end] + 
                                                       fzp_e[ktf-1,j_start:j_end, i_start:i_end] *
                                                       u[ktf-2,j_start:j_end, i_start:i_end])
    tendency[kts:ktf, j_start:j_end, i_start:i_end] = tendency[kts:ktf, j_start:j_end, i_start:i_end] - \
             rdzw_e[kts:ktf, j_start:j_end, i_start:i_end] * (vflux[kts+1:ktf+1, j_start:j_end, i_start:i_end] -
                                                              vflux[kts:ktf, j_start:j_end, i_start:i_end])
    #print("in advect ru_tend: ", tendency[38, 480, 386],vflux[39,480,386],vflux[38,480,386])
    #print("in advect ru_tend 3: ", tendency[20, 309, 603])
    #print("in advect ru_tend 3: ", tendency[20,443,603])
    return tendency

def fortran_sign(a,b):
    a = a.expand_as(b)
    return torch.where(b>=0, a.abs(), -a.abs())

def flux4_u(q_im2, q_im1, q_i, q_ip1, ua):
    ans = ( 7 * (q_i + q_im1) - (q_ip1 + q_im2))/12.0
    return ans

def flux3_u(q_im2, q_im1, q_i, q_ip1, ua):
    ans = flux4_u(q_im2, q_im1, q_i, q_ip1, ua) + fortran_sign(torch.tensor(1),torch.tensor(time_step)) * \
          fortran_sign(torch.tensor(1.),ua) * ((q_ip1 - q_im2) - 3.*(q_i - q_im1))/12.0
    return ans

def flux6_u(q_im3, q_im2, q_im1, q_i, q_ip1, q_ip2, ua):
    ans = (37.*(q_i + q_im1) - 8.*(q_ip1+q_im2) + (q_ip2 + q_im3))/60.0
    return ans

def flux5_u(q_im3, q_im2, q_im1, q_i, q_ip1, q_ip2, ua):
    ans = flux6_u(q_im3, q_im2, q_im1, q_i, q_ip1, q_ip2, ua) - fortran_sign(torch.tensor(1),torch.tensor(time_step)) * \
          fortran_sign(torch.tensor(1.),ua) * ((q_ip2-q_im3)-5.*(q_ip1-q_im2)+10.*(q_i-q_im1))/60.0
    return ans
    
# Advection tendency for v.
def advect_v(v, v_old, tendency,            \
             ru, rv, rom,                   \
             c1, c2,                        \
             mut, time_step,                \
             msfux, msfuy, msfvx, msfvy,    \
             msftx, msfty,                  \
             fzm, fzp,                      \
             rdx, rdy, rdzw,                \
             ids, ide, jds, jde, kds, kde,  \
             ims, ime, jms, jme, kms, kme,  \
             its, ite, jts, jte, kts, kte):
    ktf=min(kte,kde-1)
    
    # order = 5
    # y advection
    i_start = its
    i_end   = min(ite,ide-1)
    j_start = jts
    j_end   = jte
    
    j_start_f = j_start
    j_end_f   = j_end+1
    
    j_start = max(jts,jds+1)
    j_start_f = jds+3
    
    j_end = min(jte,jde-1)
    j_end_f = jde-2
    
    jp1 = 1
    jp0 = 0
    
    fqy = torch.zeros((2,nzall,nyall,nxall))
    vel = torch.zeros((nzall,nyall,nxall))
    mrdy = torch.zeros((nyall,nxall))
    fqx = torch.zeros((nzall,nyall,nxall))
    mrdx = torch.zeros((nyall,nxall))
    vflux = torch.zeros((nzall,nyall,nxall))
    
    rdzw_e = rdzw.unsqueeze(1).unsqueeze(2).repeat(1,nyall,nxall)
    
    vel[kts:ktf, j_start_f:j_end_f, i_start:i_end] = 0.5*(rv[kts:ktf, j_start_f:j_end_f, i_start:i_end] + \
                                                      rv[kts:ktf, j_start_f-1:j_end_f-1, i_start:i_end])
    vel_modified8 = vel.clone()
    fqy[jp1,kts:ktf, j_start_f:j_end_f, i_start:i_end] = vel_modified8[kts:ktf, j_start_f:j_end_f, i_start:i_end] * \
              flux5_u(v[kts:ktf, j_start_f-3:j_end_f-3, i_start:i_end],v[kts:ktf, j_start_f-2:j_end_f-2, i_start:i_end],
                      v[kts:ktf, j_start_f-1:j_end_f-1, i_start:i_end],v[kts:ktf, j_start_f:j_end_f, i_start:i_end],
                      v[kts:ktf, j_start_f+1:j_end_f+1, i_start:i_end],v[kts:ktf, j_start_f+2:j_end_f+2, i_start:i_end],
                      vel_modified8[kts:ktf, j_start_f:j_end_f, i_start:i_end])
    #print("advect v")
    #print(j_end_f)
    #print(v[kts:ktf, j_end_f-3, i_start:i_end])
    vb = torch.zeros((nzall,nxall))
    condition = v[kts:ktf,jds+1,i_start:i_end] < 0.
    vb[kts:ktf,i_start:i_end] = torch.where(condition, v[kts:ktf,jds+1,i_start:i_end], v[kts:ktf,jds,i_start:i_end])
    fqy[jp1, kts:ktf, jds+1, i_start:i_end] =  0.25 * (rv[kts:ktf,jds+1,i_start:i_end] + rv[kts:ktf,jds, i_start:i_end])* \
                                            (v[kts:ktf,jds+1,i_start:i_end] + vb[kts:ktf,i_start:i_end])

    vel[kts:ktf, jds+2, i_start:i_end] = 0.5 * (rv[kts:ktf,jds+2,i_start:i_end] + rv[kts:ktf,jds+1, i_start:i_end])
    vel_modified7 = vel.clone()
    fqy[jp1,kts:ktf, jds+2, i_start:i_end] = vel_modified7[kts:ktf, jds+2, i_start:i_end] * \
              flux3_u(v[kts:ktf, jds, i_start:i_end],v[kts:ktf, jds+1, i_start:i_end],
                      v[kts:ktf, jds+2, i_start:i_end],v[kts:ktf, jds+3, i_start:i_end],
                      vel_modified7[kts:ktf, jds+2, i_start:i_end])

    condition = v[kts:ktf,jde-2,i_start:i_end] > 0.
    vb[kts:ktf,i_start:i_end] = torch.where(condition, v[kts:ktf,jde-2,i_start:i_end], v[kts:ktf,jde-1,i_start:i_end])
    fqy[jp1, kts:ktf, jde-1, i_start:i_end] =  0.25 * (rv[kts:ktf,jde-1,i_start:i_end] + rv[kts:ktf,jde-2, i_start:i_end])* \
                                            (v[kts:ktf,jde-2,i_start:i_end] + vb[kts:ktf,i_start:i_end])

    vel[kts:ktf, jde-2, i_start:i_end] = 0.5 * (rv[kts:ktf,jde-2,i_start:i_end] + rv[kts:ktf,jde-3, i_start:i_end])
    vel_modified6 = vel.clone()
    fqy[jp1,kts:ktf, jde-2, i_start:i_end] = vel_modified6[kts:ktf, jde-2, i_start:i_end] * \
              flux3_u(v[kts:ktf, jde-4, i_start:i_end],v[kts:ktf, jde-3, i_start:i_end],
                      v[kts:ktf, jde-2, i_start:i_end],v[kts:ktf, jde-1, i_start:i_end],
                      vel_modified6[kts:ktf, jde-2, i_start:i_end])

    mrdy[j_start+1:j_end+1, i_start:i_end] = msfvy[j_start:j_end, i_start:i_end] * rdy
    mrdy_e = mrdy.repeat(nzall,1,1)
    tendency[kts:ktf, j_start:j_end, i_start:i_end] = tendency[kts:ktf, j_start:j_end, i_start:i_end] - \
             mrdy_e[kts:ktf, j_start+1:j_end+1, i_start:i_end] * (fqy[jp1, kts:ktf, j_start+1:j_end+1, i_start:i_end] -
                                                                fqy[jp1, kts:ktf, j_start:j_end, i_start:i_end])
    #print("in advect rv_tend: ", tendency[20, 480, 600:603],fqy[jp1,20,481, 601:604],fqy[jp1,20,480, 600:603])
    #print("in advect",tendency[0:3,528:531,10:15])
    # x advection
    i_start = its
    i_end   = min(ite,ide-1)
    j_start = jts
    j_end   = jte
    
    j_start = max(jds+1,jts)
    j_end   = min(jde-1,jte)
    
    i_start_f = i_start
    i_end_f   = i_end+1
    
    i_start = max(ids+1,its)
    i_start_f = min(i_start+2,ids+3)
    
    i_end = min(ide-2,ite)
    i_end_f = ide-3
    
    vel[kts:ktf, j_start:j_end, i_start_f:i_end_f] = 0.5 * (ru[kts:ktf, j_start:j_end, i_start_f:i_end_f] +
                                                            ru[kts:ktf, j_start-1:j_end-1, i_start_f:i_end_f])
    vel_modified5 = vel.clone()
    fqx[kts:ktf, j_start:j_end, i_start_f:i_end_f] = vel_modified5[kts:ktf, j_start:j_end, i_start_f:i_end_f] * \
            flux5_u(v[kts:ktf, j_start:j_end, i_start_f-3:i_end_f-3],v[kts:ktf, j_start:j_end, i_start_f-2:i_end_f-2],
                    v[kts:ktf, j_start:j_end, i_start_f-1:i_end_f-1],v[kts:ktf, j_start:j_end, i_start_f:i_end_f],
                    v[kts:ktf, j_start:j_end, i_start_f+1:i_end_f+1],v[kts:ktf, j_start:j_end, i_start_f+2:i_end_f+2],
                    vel_modified5[kts:ktf, j_start:j_end, i_start_f:i_end_f])
    fqx[kts:ktf, j_start:j_end, ids+1] = 0.25 * (ru[kts:ktf, j_start:j_end, ids+1] + 
                                                 ru[kts:ktf, j_start-1:j_end-1, ids+1]) * \
                                                (v[kts:ktf, j_start:j_end, ids+1] +
                                                 v[kts:ktf, j_start:j_end, ids])
    vel[kts:ktf, j_start:j_end, ids+2] = 0.5 * (ru[kts:ktf, j_start:j_end, ids+2] +
                                                ru[kts:ktf, j_start-1:j_end-1, ids+2])
    vel_modified4 = vel.clone()
    fqx[kts:ktf, j_start:j_end, ids+2] = vel_modified4[kts:ktf, j_start:j_end, ids+2] * \
            flux3_u(v[kts:ktf, j_start:j_end, ids],v[kts:ktf, j_start:j_end, ids+1],
                    v[kts:ktf, j_start:j_end, ids+2],v[kts:ktf, j_start:j_end, ids+3],
                    vel_modified4[kts:ktf, j_start:j_end, ids+2])
    fqx[kts:ktf, j_start:j_end, ide-2] = 0.25 * (ru[kts:ktf, j_start:j_end, i_end] + 
                                                 ru[kts:ktf, j_start-1:j_end-1, i_end]) * \
                                                (v[kts:ktf, j_start:j_end, i_end] +
                                                 v[kts:ktf, j_start:j_end, i_end-1])
    vel[kts:ktf, j_start:j_end, ide-3] = 0.5 * (ru[kts:ktf, j_start:j_end, ide-3] + 
                                                ru[kts:ktf, j_start-1:j_end-1, ide-3])
    vel_modified3 = vel.clone()
    fqx[kts:ktf, j_start:j_end, ide-3] = vel_modified3[kts:ktf, j_start:j_end, ide-3] * \
            flux3_u(v[kts:ktf, j_start:j_end, ide-5],v[kts:ktf, j_start:j_end, ide-4],
                    v[kts:ktf, j_start:j_end, ide-3],v[kts:ktf, j_start:j_end, ide-2],
                    vel_modified3[kts:ktf, j_start:j_end, ide-3])
    mrdx[j_start:j_end, i_start:i_end] = msfvy[j_start:j_end, i_start:i_end] * rdx
    mrdx_e = mrdx.repeat(nzall,1,1)
    tendency[kts:ktf, j_start:j_end, i_start:i_end] = tendency[kts:ktf, j_start:j_end, i_start:i_end] - \
             mrdx_e[kts:ktf, j_start:j_end, i_start:i_end] * (fqx[kts:ktf, j_start:j_end, i_start+1:i_end+1] -
                                                              fqx[kts:ktf, j_start:j_end, i_start:i_end])
    #print("in advect rv_tend: ", tendency[20, 480, 601],fqx[20,480, 602],fqx[20,480, 601],ru[20,480,602],ru[20,479,602],v[20,480,599:603])
    #print("in advect",tendency[0:3,528:531,10:15])
    # z advection
    i_start = its
    i_end   = min(ite,ide-1)
    j_start = jts
    j_end   = jte
    
    j_start = max(jds+1,jts)
    j_end   = min(jde-1,jte)
    
    #vflux[:,:,:] = 0.
    
    vel[kts+3:ktf-2, j_start:j_end, i_start:i_end] = 0.5 * (rom[kts+3:ktf-2, j_start:j_end, i_start:i_end] +
                                                            rom[kts+3:ktf-2, j_start-1:j_end-1, i_start:i_end])
    vel_modified2 = vel.clone()
    vflux[kts+3:ktf-2, j_start:j_end, i_start:i_end] = vel_modified2[kts+3:ktf-2, j_start:j_end, i_start:i_end] * \
             flux5_u(v[kts:ktf-5, j_start:j_end, i_start:i_end],v[kts+1:ktf-4, j_start:j_end, i_start:i_end],
                     v[kts+2:ktf-3, j_start:j_end, i_start:i_end],v[kts+3:ktf-2, j_start:j_end, i_start:i_end],
                     v[kts+4:ktf-1, j_start:j_end, i_start:i_end],v[kts+5:ktf, j_start:j_end, i_start:i_end],
                     -vel_modified2[kts+3:ktf-2, j_start:j_end, i_start:i_end])
    fzm_e = fzm.unsqueeze(1).unsqueeze(2).repeat(1,nyfull,nxfull)
    fzp_e = fzp.unsqueeze(1).unsqueeze(2).repeat(1,nyfull,nxfull)
    #rom_modified = rom.clone()
    vflux[kts+1, j_start:j_end, i_start:i_end] = 0.5 * (rom[kts+1, j_start:j_end, i_start:i_end] +
                                                        rom[kts+1, j_start-1:j_end-1, i_start:i_end]) * \
                                                       (fzm_e[kts+1, j_start:j_end, i_start:i_end] *
                                                        v[kts+1, j_start:j_end, i_start:i_end] +
                                                        fzp_e[kts+1, j_start:j_end, i_start:i_end] *
                                                        v[kts, j_start:j_end, i_start:i_end])
    vel[kts+2, j_start:j_end, i_start:i_end] = 0.5 * (rom[kts+2, j_start:j_end, i_start:i_end] +
                                                      rom[kts+2, j_start-1:j_end-1, i_start:i_end])
    #vflux_new_slice = vel[kts+2, j_start:j_end, i_start:i_end] * \
    #         flux3_u(v[kts, j_start:j_end, i_start:i_end],v[kts+1, j_start:j_end, i_start:i_end],
    #                 v[kts+2, j_start:j_end, i_start:i_end],v[kts+3, j_start:j_end, i_start:i_end],
    #                 -vel[kts+2, j_start:j_end, i_start:i_end])
    #vflux = vflux.clone()
    #vflux[kts+2, j_start:j_end, i_start:i_end] = vflux_new_slice
    vel_modified1 = vel.clone()
    vflux[kts+2, j_start:j_end, i_start:i_end] = vel_modified1[kts+2, j_start:j_end, i_start:i_end] * \
             flux3_u(v[kts, j_start:j_end, i_start:i_end],v[kts+1, j_start:j_end, i_start:i_end],
                     v[kts+2, j_start:j_end, i_start:i_end],v[kts+3, j_start:j_end, i_start:i_end],
                     -vel_modified1[kts+2, j_start:j_end, i_start:i_end])
    
    vel[ktf-2, j_start:j_end, i_start:i_end] = 0.5 * (rom[ktf-2, j_start:j_end, i_start:i_end] +
                                                      rom[ktf-2, j_start-1:j_end-1, i_start:i_end])
    vflux[ktf-2, j_start:j_end, i_start:i_end] = vel[ktf-2, j_start:j_end, i_start:i_end] * \
             flux3_u(v[ktf-4, j_start:j_end, i_start:i_end],v[ktf-3, j_start:j_end, i_start:i_end],
                     v[ktf-2, j_start:j_end, i_start:i_end],v[ktf-1, j_start:j_end, i_start:i_end],
                     vel[ktf-2, j_start:j_end, i_start:i_end])
    vflux[ktf-1, j_start:j_end, i_start:i_end] = 0.5 * (rom[ktf-1, j_start:j_end, i_start:i_end] +
                                                      rom[ktf-1, j_start-1:j_end-1, i_start:i_end]) * \
                                                     (fzm_e[ktf-1, j_start:j_end, i_start:i_end] * 
                                                      v[ktf-1, j_start:j_end, i_start:i_end] + 
                                                      fzp_e[ktf-1, j_start:j_end, i_start:i_end] * 
                                                      v[ktf-2, j_start:j_end, i_start:i_end])
    msfvy_e = msfvy.repeat(nzall,1,1)
    msfvx_e = msfvx.repeat(nzall,1,1)
    tendency[kts:ktf, j_start:j_end, i_start:i_end] = tendency[kts:ktf, j_start:j_end, i_start:i_end] - \
             (msfvy_e[kts:ktf, j_start:j_end, i_start:i_end]/msfvx_e[kts:ktf, j_start:j_end, i_start:i_end]) * \
             rdzw_e[kts:ktf, j_start:j_end, i_start:i_end] * (vflux[kts+1:ktf+1, j_start:j_end, i_start:i_end] -
                                                              vflux[kts:ktf, j_start:j_end, i_start:i_end])
    #print("in advect rv_tend: ", tendency[20, 480, 600:603],vflux[21,480, 600:603],vflux[20,480, 600:603])
    return tendency

# Advection tendency for w.
def advect_w(w, w_old, tendency,            \
             ru, rv, rom,                   \
             c1, c2,                        \
             mut, time_step,                \
             msfux, msfuy, msfvx, msfvy,    \
             msftx, msfty,                  \
             fzm, fzp,                      \
             rdx, rdy, rdzu,                \
             ids, ide, jds, jde, kds, kde,  \
             ims, ime, jms, jme, kms, kme,  \
             its, ite, jts, jte, kts, kte):
    ktf=min(kte,kde-1)
    
    # y advection
    i_start = its
    i_end   = min(ite,ide-1)
    j_start = jts
    j_end   = min(jte,jde-1)
    
    j_start_f = j_start
    j_end_f   = j_end+1
    
    j_start = max(jts,jds+1)
    j_start_f = jds+3
    
    j_end = min(jte,jde-2)
    j_end_f = jde-3
    
    jp1 = 1
    jp0 = 0
    
    fqy = torch.zeros((2,nzall,nyall,nxall))
    vel = torch.zeros((nzall,nyall,nxall))
    mrdy = torch.zeros((nyall,nxall))
    fqx = torch.zeros((nzall,nyall,nxall))
    mrdx = torch.zeros((nyall,nxall))
    vflux = torch.zeros((nzall,nyall,nxall))
    
    rdzu_e = rdzu.unsqueeze(1).unsqueeze(2).repeat(1,nyall,nxall)
    
    fzm_e = fzm.unsqueeze(1).unsqueeze(2).repeat(1,nyfull,nxfull)
    fzp_e = fzp.unsqueeze(1).unsqueeze(2).repeat(1,nyfull,nxfull)
    
    vel[kts+1:ktf, j_start_f:j_end_f, i_start:i_end] = fzm_e[kts+1:ktf, j_start_f:j_end_f, i_start:i_end] * \
                                                       rv[kts+1:ktf, j_start_f:j_end_f, i_start:i_end] + \
                                                       fzp_e[kts+1:ktf, j_start_f:j_end_f, i_start:i_end] * \
                                                       rv[kts:ktf-1, j_start_f:j_end_f, i_start:i_end]
    fqy[jp1,kts+1:ktf, j_start_f:j_end_f, i_start:i_end] = vel[kts+1:ktf, j_start_f:j_end_f, i_start:i_end] * \
            flux5_u(w[kts+1:ktf, j_start_f-3:j_end_f-3, i_start:i_end],w[kts+1:ktf, j_start_f-2:j_end_f-2, i_start:i_end],
                    w[kts+1:ktf, j_start_f-1:j_end_f-1, i_start:i_end],w[kts+1:ktf, j_start_f:j_end_f, i_start:i_end],
                    w[kts+1:ktf, j_start_f+1:j_end_f+1, i_start:i_end],w[kts+1:ktf, j_start_f+2:j_end_f+2, i_start:i_end],
                    vel[kts+1:ktf, j_start_f:j_end_f, i_start:i_end])
    # ktf + 1 
    vel[ktf, j_start_f:j_end_f, i_start:i_end] = (2. - fzm_e[ktf-1, j_start_f:j_end_f, i_start:i_end]) * \
                                                   rv[ktf-1, j_start_f:j_end_f, i_start:i_end] - \
                                                   fzp_e[ktf-1, j_start_f:j_end_f, i_start:i_end] * \
                                                   rv[ktf-2, j_start_f:j_end_f, i_start:i_end]
    fqy[jp1,ktf, j_start_f:j_end_f, i_start:i_end] = vel[ktf, j_start_f:j_end_f, i_start:i_end] * \
            flux5_u(w[ktf, j_start_f-3:j_end_f-3, i_start:i_end],w[ktf, j_start_f-2:j_end_f-2, i_start:i_end],
                    w[ktf, j_start_f-1:j_end_f-1, i_start:i_end],w[ktf, j_start_f:j_end_f, i_start:i_end],
                    w[ktf, j_start_f+1:j_end_f+1, i_start:i_end],w[ktf, j_start_f+2:j_end_f+2, i_start:i_end],
                    vel[ktf, j_start_f:j_end_f, i_start:i_end])
    # jds+1
    fqy[jp1, kts+1:ktf, jds+1, i_start:i_end] = 0.5 * (fzm_e[kts+1:ktf, jds+1, i_start:i_end] *
                                                       rv[kts+1:ktf, jds+1, i_start:i_end] +
                                                       fzp_e[kts+1:ktf, jds+1, i_start:i_end] *
                                                       rv[kts:ktf-1, jds+1, i_start:i_end]) * \
                                                      (w[kts+1:ktf, jds+1, i_start:i_end] +
                                                       w[kts+1:ktf, jds, i_start:i_end])
    fqy[jp1, ktf, jds+1, i_start:i_end] = 0.5 * ((2. - fzm_e[ktf-1, jds+1, i_start:i_end]) * 
                                                   rv[ktf-1, jds+1, i_start:i_end] -
                                                   fzp_e[ktf-1, jds+1, i_start:i_end] * 
                                                   rv[ktf-2, jds+1, i_start:i_end]) * \
                                                  (w[ktf, jds+1, i_start:i_end] + 
                                                   w[ktf, jds, i_start:i_end])
    # jds+2
    vel[kts+1:ktf, jds+2, i_start:i_end] = fzm_e[kts+1:ktf, jds+2, i_start:i_end] * \
                                           rv[kts+1:ktf, jds+2, i_start:i_end] + \
                                           fzp_e[kts+1:ktf, jds+2, i_start:i_end] * \
                                           rv[kts:ktf-1, jds+2, i_start:i_end]
    fqy[jp1, kts+1:ktf, jds+2, i_start:i_end] = vel[kts+1:ktf, jds+2, i_start:i_end] * \
            flux3_u(w[kts+1:ktf, jds, i_start:i_end],w[kts+1:ktf, jds+1, i_start:i_end],
                    w[kts+1:ktf, jds+2, i_start:i_end],w[kts+1:ktf, jds+3, i_start:i_end],
                    vel[kts+1:ktf, jds+2, i_start:i_end])
    vel[ktf, jds+2, i_start:i_end] = (2. - fzm_e[ktf-1, jds+2, i_start:i_end]) * \
                                        rv[ktf-1, jds+2, i_start:i_end] - \
                                        fzp_e[ktf-1, jds+2, i_start:i_end] * \
                                        rv[ktf-2, jds+2, i_start:i_end]
    fqy[jp1, ktf, jds+2, i_start:i_end] = vel[ktf, jds+2, i_start:i_end] * \
            flux3_u(w[ktf, jds, i_start:i_end],w[ktf, jds+1, i_start:i_end],
                    w[ktf, jds+2, i_start:i_end],w[ktf, jds+3, i_start:i_end],
                    vel[ktf, jds+2, i_start:i_end])
    # jde-1
    fqy[jp1, kts+1:ktf, jde-2, i_start:i_end] = 0.5 * (fzm_e[kts+1:ktf, jde-2, i_start:i_end] *
                                                       rv[kts+1:ktf, jde-2, i_start:i_end] + 
                                                       fzp_e[kts+1:ktf, jde-2, i_start:i_end] *
                                                       rv[kts:ktf-1, jde-2, i_start:i_end]) * \
                                                      (w[kts+1:ktf, jde-2, i_start:i_end] + \
                                                       w[kts+1:ktf, jde-3, i_start:i_end])
    fqy[jp1, ktf, jde-2, i_start:i_end] = 0.5 * ((2. - fzm_e[ktf-1, jde-2, i_start:i_end]) * 
                                                   rv[ktf-1, jde-2, i_start:i_end] - 
                                                   fzp_e[ktf-1, jde-2, i_start:i_end] * 
                                                   rv[ktf-2, jde-2, i_start:i_end]) * \
                                                  (w[ktf, jde-2, i_start:i_end] + 
                                                   w[ktf, jde-3, i_start:i_end])
    # jde-2
    vel[kts+1:ktf, jde-3, i_start:i_end] = fzm_e[kts+1:ktf, jde-3, i_start:i_end] * \
                                           rv[kts+1:ktf, jde-3, i_start:i_end] + \
                                           fzp_e[kts+1:ktf, jde-3, i_start:i_end] * \
                                           rv[kts:ktf-1, jde-3, i_start:i_end]
    fqy[jp1, kts+1:ktf, jde-3, i_start:i_end] = vel[kts+1:ktf, jde-3, i_start:i_end] * \
            flux3_u(w[kts+1:ktf, jde-5, i_start:i_end],w[kts+1:ktf, jde-4, i_start:i_end],
                    w[kts+1:ktf, jde-3, i_start:i_end],w[kts+1:ktf, jde-2, i_start:i_end],
                    vel[kts+1:ktf, jde-3, i_start:i_end])
    vel[ktf, jde-3, i_start:i_end] = (2. - fzm_e[ktf-1, jde-3, i_start:i_end]) * \
                                      rv[ktf-1, jde-3, i_start:i_end] - \
                                      fzp_e[ktf-1, jde-3, i_start:i_end] * \
                                      rv[ktf-2, jde-3, i_start:i_end]
    fqy[jp1, ktf, jde-3, i_start:i_end] = vel[ktf, jde-3, i_start:i_end] * \
            flux3_u(w[ktf, jde-5, i_start:i_end],w[ktf, jde-4, i_start:i_end],
                    w[ktf, jde-3, i_start:i_end],w[ktf, jde-2, i_start:i_end],
                    vel[ktf, jde-3, i_start:i_end])
    
    mrdy[j_start+1:j_end+1, i_start:i_end] = msftx[j_start:j_end, i_start:i_end] * rdy
    mrdy_e = mrdy.repeat(nzall,1,1)
    tendency[kts+1:ktf+1,j_start:j_end,i_start:i_end] = tendency[kts+1:ktf+1,j_start:j_end,i_start:i_end] - \
             mrdy_e[kts+1:ktf+1, j_start+1:j_end+1, i_start:i_end] * (fqy[jp1, kts+1:ktf+1, j_start+1:j_end+1, i_start:i_end] -
                                                       fqy[jp1, kts+1:ktf+1, j_start:j_end, i_start:i_end])
    #print("in advect rw_tend: ", tendency[20, 480, 6])
    # x advection
    i_start = its
    i_end   = min(ite,ide-1)
    
    j_start = jts
    j_end   = min(jte,jde-1)
    
    i_start_f = i_start
    i_end_f   = i_end+1
    
    i_start = max(ids+1,its)
    i_start_f = min(i_start+2,ids+3)
      
    i_end = min(ide-2,ite) 
    i_end_f = ide-3
    
    ###L5153
    vel[kts+1:ktf, j_start:j_end, i_start_f:i_end_f] = fzm_e[kts+1:ktf, j_start:j_end, i_start_f:i_end_f] * \
                                                       ru[kts+1:ktf, j_start:j_end, i_start_f:i_end_f] + \
                                                       fzp_e[kts+1:ktf, j_start:j_end, i_start_f:i_end_f] * \
                                                       ru[kts:ktf-1, j_start:j_end, i_start_f:i_end_f]
    fqx[kts+1:ktf, j_start:j_end, i_start_f:i_end_f] = vel[kts+1:ktf, j_start:j_end, i_start_f:i_end_f] * \
              flux5_u(w[kts+1:ktf, j_start:j_end, i_start_f-3:i_end_f-3],w[kts+1:ktf, j_start:j_end, i_start_f-2:i_end_f-2],
                      w[kts+1:ktf, j_start:j_end, i_start_f-1:i_end_f-1],w[kts+1:ktf, j_start:j_end, i_start_f:i_end_f],
                      w[kts+1:ktf, j_start:j_end, i_start_f+1:i_end_f+1],w[kts+1:ktf, j_start:j_end, i_start_f+2:i_end_f+2],
                      vel[kts+1:ktf, j_start:j_end, i_start_f:i_end_f])
    # ktf+1
    vel[ktf, j_start:j_end, i_start_f:i_end_f] = (2. - fzm_e[ktf-1, j_start:j_end, i_start_f:i_end_f]) * \
                                                   ru[ktf-1, j_start:j_end, i_start_f:i_end_f] - \
                                                   fzp_e[ktf-1, j_start:j_end, i_start_f:i_end_f] * \
                                                   ru[ktf-2, j_start:j_end, i_start_f:i_end_f]
    fqx[ktf, j_start:j_end, i_start_f:i_end_f] = vel[ktf, j_start:j_end, i_start_f:i_end_f] * \
              flux5_u(w[ktf, j_start:j_end, i_start_f-3:i_end_f-3],w[ktf, j_start:j_end, i_start_f-2:i_end_f-2],
                      w[ktf, j_start:j_end, i_start_f-1:i_end_f-1],w[ktf, j_start:j_end, i_start_f:i_end_f],
                      w[ktf, j_start:j_end, i_start_f+1:i_end_f+1],w[ktf, j_start:j_end, i_start_f+2:i_end_f+2],
                      vel[ktf, j_start:j_end, i_start_f:i_end_f])
    # ids+1
    fqx[kts+1:ktf, j_start:j_end, ids+1] = 0.5 * (fzm_e[kts+1:ktf, j_start:j_end, ids+1] *
                                                  ru[kts+1:ktf, j_start:j_end, ids+1] +
                                                  fzp_e[kts+1:ktf, j_start:j_end, ids+1] *
                                                  ru[kts:ktf-1, j_start:j_end, ids+1]) * \
                                                 (w[kts+1:ktf, j_start:j_end, ids+1] +
                                                  w[kts+1:ktf, j_start:j_end, ids])
    fqx[ktf, j_start:j_end, ids+1] = 0.5 * ((2. - fzm_e[ktf-1, j_start:j_end, ids+1]) *
                                              ru[ktf-1, j_start:j_end, ids+1] -
                                              fzp_e[ktf-1, j_start:j_end, ids+1] *
                                              ru[ktf-2, j_start:j_end, ids+1]) * \
                                             (w[ktf, j_start:j_end, ids+1] +
                                              w[ktf, j_start:j_end, ids])
    #print("in advect rw_tend 111: ", fqx[20,480,6],ru[20,480,6],ru[19,480,6],w[20,480,6],w[20,480,5])
    # ids+2
    vel[kts+1:ktf, j_start:j_end, ids+2] = fzm_e[kts+1:ktf, j_start:j_end, ids+2] * \
                                           ru[kts+1:ktf, j_start:j_end, ids+2] + \
                                           fzp_e[kts+1:ktf, j_start:j_end, ids+2] * \
                                           ru[kts:ktf-1, j_start:j_end, ids+2]
    fqx[kts+1:ktf, j_start:j_end, ids+2] = vel[kts+1:ktf, j_start:j_end, ids+2] * \
              flux3_u(w[kts+1:ktf, j_start:j_end, ids],w[kts+1:ktf, j_start:j_end, ids+1],
                      w[kts+1:ktf, j_start:j_end, ids+2],w[kts+1:ktf, j_start:j_end, ids+3],
                      vel[kts+1:ktf, j_start:j_end, ids+2])
    vel[ktf, j_start:j_end, ids+2] = (2. - fzm_e[ktf-1, j_start:j_end, ids+2]) * ru[ktf-1, j_start:j_end, ids+2] \
                                     - fzp_e[ktf-1, j_start:j_end, ids+2] * ru[ktf-2, j_start:j_end, ids+2]
    fqx[ktf, j_start:j_end, ids+2] = vel[ktf, j_start:j_end, ids+2] * \
              flux3_u(w[ktf, j_start:j_end, ids],w[ktf, j_start:j_end, ids+1],
                      w[ktf, j_start:j_end, ids+2],w[ktf, j_start:j_end, ids+3],
                      vel[ktf, j_start:j_end, ids+2])
    # ide -1
    fqx[kts+1:ktf, j_start:j_end, ide-2] = 0.5 * (fzm_e[kts+1:ktf, j_start:j_end, ide-2] *
                                                  ru[kts+1:ktf, j_start:j_end, ide-2] + 
                                                  fzp_e[kts+1:ktf, j_start:j_end, ide-2] * 
                                                  ru[kts:ktf-1, j_start:j_end, ide-2]) * \
                                                 (w[kts+1:ktf, j_start:j_end, ide-2] + 
                                                  w[kts+1:ktf, j_start:j_end, ide-3])
    fqx[ktf, j_start:j_end, ide-2] = 0.5 * ((2. - fzm_e[ktf-1, j_start:j_end, ide-2]) *
                                              ru[ktf-1, j_start:j_end, ide-2] - 
                                              fzp_e[ktf-1, j_start:j_end, ide-2] * 
                                              ru[ktf-2, j_start:j_end, ide-2]) * \
                                             (w[ktf, j_start:j_end, ide-2] + 
                                              w[ktf, j_start:j_end, ide-3])
    # ide -2
    vel[kts+1:ktf, j_start:j_end, ide-3] = fzm_e[kts+1:ktf, j_start:j_end, ide-3] * \
                                           ru[kts+1:ktf, j_start:j_end, ide-3] + \
                                           fzp_e[kts+1:ktf, j_start:j_end, ide-3] * \
                                           ru[kts:ktf-1, j_start:j_end, ide-3]
    fqx[kts+1:ktf, j_start:j_end, ide-3] = vel[kts+1:ktf, j_start:j_end, ide-3] * \
              flux3_u(w[kts+1:ktf, j_start:j_end, ide-5],w[kts+1:ktf, j_start:j_end, ide-4],
                      w[kts+1:ktf, j_start:j_end, ide-3],w[kts+1:ktf, j_start:j_end, ide-2],
                      vel[kts+1:ktf, j_start:j_end, ide-3])
    vel[ktf,j_start:j_end, ide-3] = (2. - fzm_e[ktf-1,j_start:j_end, ide-3]) * \
                                      ru[ktf-1,j_start:j_end, ide-3] - \
                                      fzp_e[ktf-1,j_start:j_end, ide-3] * \
                                      ru[ktf-2,j_start:j_end, ide-3]
    fqx[ktf,j_start:j_end, ide-3] = vel[ktf,j_start:j_end, ide-3] * \
              flux3_u(w[ktf,j_start:j_end, ide-5],w[ktf,j_start:j_end, ide-4],
                      w[ktf,j_start:j_end, ide-3],w[ktf,j_start:j_end, ide-2],
                      vel[ktf,j_start:j_end, ide-3])
    
    mrdx[j_start:j_end, i_start:i_end] = msftx[j_start:j_end, i_start:i_end] * rdx
    mrdx_e = mrdx.repeat(nzall,1,1)
    tendency[kts+1:ktf+1, j_start:j_end, i_start:i_end] = tendency[kts+1:ktf+1, j_start:j_end, i_start:i_end] - \
                                                          mrdx_e[kts+1:ktf+1, j_start:j_end, i_start:i_end] * \
                                                          (fqx[kts+1:ktf+1, j_start:j_end, i_start+1:i_end+1] -
                                                           fqx[kts+1:ktf+1, j_start:j_end, i_start:i_end])
    #print("in advect rw_tend: ", tendency[20, 480, 6],fqx[20,480,7],fqx[20,480,6])
    # z advection
    i_start = its
    i_end   = min(ite,ide-1)
    j_start = jts
    j_end   = min(jte,jde-1)
    
    vel[kts+3:ktf-1, j_start:j_end, i_start:i_end] = 0.5 * (rom[kts+3:ktf-1, j_start:j_end, i_start:i_end] + 
                                                            rom[kts+2:ktf-2, j_start:j_end, i_start:i_end])
    vflux[kts+3:ktf-1, j_start:j_end, i_start:i_end] = vel[kts+3:ktf-1, j_start:j_end, i_start:i_end] * \
              flux5_u(w[kts:ktf-4, j_start:j_end, i_start:i_end],w[kts+1:ktf-3, j_start:j_end, i_start:i_end],
                    w[kts+2:ktf-2, j_start:j_end, i_start:i_end],w[kts+3:ktf-1, j_start:j_end, i_start:i_end],
                    w[kts+4:ktf, j_start:j_end, i_start:i_end],w[kts+5:ktf+1, j_start:j_end, i_start:i_end],
                    -vel[kts+3:ktf-1, j_start:j_end, i_start:i_end])
    # kts+1
    vflux[kts+1, j_start:j_end, i_start:i_end] = 0.25 * (rom[kts+1, j_start:j_end, i_start:i_end] + 
                                                         rom[kts, j_start:j_end, i_start:i_end]) * \
                                                        (w[kts+1, j_start:j_end, i_start:i_end] + 
                                                         w[kts, j_start:j_end, i_start:i_end])
    # kts+2
    vel[kts+2, j_start:j_end, i_start:i_end] = 0.5 * (rom[kts+2, j_start:j_end, i_start:i_end] + 
                                                      rom[kts+1, j_start:j_end, i_start:i_end])
    vflux_new_slice = vel[kts+2, j_start:j_end, i_start:i_end] * \
              flux3_u(w[kts, j_start:j_end, i_start:i_end],w[kts+1, j_start:j_end, i_start:i_end],
                      w[kts+2, j_start:j_end, i_start:i_end],w[kts+3, j_start:j_end, i_start:i_end],
                      -vel[kts+2, j_start:j_end, i_start:i_end])
    vflux = vflux.clone()
    vflux[kts+2, j_start:j_end, i_start:i_end] = vflux_new_slice
    """
    vflux[kts+2, j_start:j_end, i_start:i_end] = vel[kts+2, j_start:j_end, i_start:i_end] * \
              flux3_u(w[kts, j_start:j_end, i_start:i_end],w[kts+1, j_start:j_end, i_start:i_end],
                      w[kts+2, j_start:j_end, i_start:i_end],w[kts+3, j_start:j_end, i_start:i_end],
                      -vel[kts+2, j_start:j_end, i_start:i_end])
    """
    # ktf
    vel[ktf-1, j_start:j_end, i_start:i_end] = 0.5 * (rom[ktf-1, j_start:j_end, i_start:i_end] + 
                                                      rom[ktf-2, j_start:j_end, i_start:i_end])
    vflux[ktf-1, j_start:j_end, i_start:i_end] = vel[ktf-1, j_start:j_end, i_start:i_end] * \
              flux3_u(w[ktf-3, j_start:j_end, i_start:i_end],w[ktf-2, j_start:j_end, i_start:i_end],
                      w[ktf-1, j_start:j_end, i_start:i_end],w[ktf, j_start:j_end, i_start:i_end],
                      vel[ktf-1, j_start:j_end, i_start:i_end])
    # ktf+1
    vflux[ktf, j_start:j_end, i_start:i_end] = 0.25 * (rom[ktf, j_start:j_end, i_start:i_end] +
                                                         rom[ktf-1, j_start:j_end, i_start:i_end]) * \
                                                        (w[ktf, j_start:j_end, i_start:i_end] +
                                                         w[ktf-1, j_start:j_end, i_start:i_end])
                                                        
    
    tendency[kts+1:ktf, j_start:j_end, i_start:i_end] = tendency[kts+1:ktf, j_start:j_end, i_start:i_end] - \
                                                        rdzu_e[kts+1:ktf, j_start:j_end, i_start:i_end] * \
                                                        (vflux[kts+2:ktf+1, j_start:j_end, i_start:i_end] -
                                                         vflux[kts+1:ktf, j_start:j_end, i_start:i_end])
    tendency[ktf, j_start:j_end, i_start:i_end] = tendency[ktf, j_start:j_end, i_start:i_end] + \
                                                    2. * rdzu_e[ktf-1, j_start:j_end, i_start:i_end] * \
                                                    vflux[ktf, j_start:j_end, i_start:i_end]
    #print("in advect rw_tend: ", tendency[20, 480, 6])
    return tendency

def advect_scalar_pd( field, field_old, tendency,    
                      h_tendency, z_tendency,        
                      ru, rv, rom,
                      c1, c2,
                      mut, mub, mu_old,
                      time_step, config_flags,
                      tenddec,
                      msfux, msfuy, msfvx, msfvy,
                      msftx, msfty,
                      fzm, fzp,
                      rdx, rdy, rdzw, dt,
                      ids, ide, jds, jde, kds, kde,
                      ims, ime, jms, jme, kms, kme,
                      its, ite, jts, jte, kts, kte  ):
    return tendency

# Advection tendency for a scalar field (e.g. theta).
def advect_scalar(field, field_old, tendency,    \
                  ru, rv, rom,                   \
                  c1, c2,                        \
                  mut, time_step,                \
                  msfux, msfuy, msfvx, msfvy,    \
                  msftx, msfty,                  \
                  fzm, fzp,                      \
                  rdx, rdy, rdzw,                \
                  ids, ide, jds, jde, kds, kde,  \
                  ims, ime, jms, jme, kms, kme,  \
                  its, ite, jts, jte, kts, kte  ):
        
    # y advection
    ktf=min(kte,kde-1)
    i_start = its
    i_end   = min(ite,ide-1)
    j_start = jts
    j_end   = min(jte,jde-1)
    
    j_start_f = j_start
    j_end_f   = j_end+1
    
    j_start = max(jts,jds+1)
    j_start_f = jds+3
    
    j_end = min(jte,jde-2)
    j_end_f = jde-3
    
    jp1 = 1
    jp0 = 0
    
    fqy = torch.zeros((2,nzall,nyall,nxall))
    
    fzm_e = fzm.unsqueeze(1).unsqueeze(2).repeat(1,nyfull,nxfull)
    fzp_e = fzp.unsqueeze(1).unsqueeze(2).repeat(1,nyfull,nxfull)
    
    vel = torch.zeros((nzall,nyall,nxall))
    mrdy = torch.zeros((nyall,nxall))
    fqx = torch.zeros((nzall,nyall,nxall))
    mrdx = torch.zeros((nyall,nxall))
    vflux = torch.zeros((nzall,nyall,nxall))
    
    ru_modified = ru.clone()
    rv_modified = rv.clone()
    rom_modified = rom.clone()
    
    rdzw_e = rdzw.unsqueeze(1).unsqueeze(2).repeat(1,nyall,nxall)
    #print("in advect scalar 0:", tendency[3,300:316,305])
    vel[kts:ktf, j_start_f:j_end_f, i_start:i_end] = rv_modified[kts:ktf, j_start_f:j_end_f, i_start:i_end] + 0.0
    fqy[jp1, kts:ktf, j_start_f:j_end_f, i_start:i_end] = vel[kts:ktf, j_start_f:j_end_f, i_start:i_end] * \
              flux5_u(field[kts:ktf, j_start_f-3:j_end_f-3, i_start:i_end],field[kts:ktf, j_start_f-2:j_end_f-2, i_start:i_end],
                      field[kts:ktf, j_start_f-1:j_end_f-1, i_start:i_end],field[kts:ktf, j_start_f:j_end_f, i_start:i_end],
                      field[kts:ktf, j_start_f+1:j_end_f+1, i_start:i_end],field[kts:ktf, j_start_f+2:j_end_f+2, i_start:i_end],
                      vel[kts:ktf, j_start_f:j_end_f, i_start:i_end])
    # jds+1
    fqy[jp1, kts:ktf, jds+1, i_start:i_end] = 0.5 * rv_modified[kts:ktf, jds+1, i_start:i_end] * \
                           (field[kts:ktf, jds+1, i_start:i_end] + field[kts:ktf, jds, i_start:i_end])
    # jds+2
    fqy[jp1, kts:ktf, jds+2, i_start:i_end] = rv_modified[kts:ktf, jds+2, i_start:i_end] * \
              flux3_u(field[kts:ktf, jds, i_start:i_end],field[kts:ktf, jds+1, i_start:i_end],
                    field[kts:ktf, jds+2, i_start:i_end],field[kts:ktf, jds+3, i_start:i_end],
                    rv_modified[kts:ktf, jds+2, i_start:i_end])
    # jde-1
    fqy[jp1, kts:ktf, jde-2, i_start:i_end] = 0.5 * rv_modified[kts:ktf, jde-2, i_start:i_end] * \
                           (field[kts:ktf, jde-2, i_start:i_end] + field[kts:ktf, jde-3, i_start:i_end]) 
    # jde-2
    fqy[jp1, kts:ktf, jde-3, i_start:i_end] = rv_modified[kts:ktf, jde-3, i_start:i_end] * \
              flux3_u(field[kts:ktf, jde-5, i_start:i_end],field[kts:ktf, jde-4, i_start:i_end],
                      field[kts:ktf, jde-3, i_start:i_end],field[kts:ktf, jde-2, i_start:i_end],
                      rv_modified[kts:ktf, jde-3, i_start:i_end])
    
    mrdy[j_start+1 : j_end+1, i_start:i_end] = msftx[j_start : j_end, i_start:i_end] * rdy
    mrdy_e = mrdy.repeat(nzall,1,1)
    tendency[kts:ktf, j_start:j_end, i_start:i_end] = tendency[kts:ktf, j_start : j_end, i_start:i_end] - \
                      mrdy_e[kts:ktf, j_start+1 : j_end+1, i_start:i_end] * (fqy[jp1,kts:ktf, j_start+1:j_end+1, i_start:i_end] -
                                                                             fqy[jp1,kts:ktf, j_start:j_end, i_start:i_end])
    #print("in advect scalar 1:", tendency[16,323,602])
    #print("in rk scalar tend:",im, advect_tend[im,16,323,602])
    #print("in advect t_tend 1: ", tendency[20, 480, 602],fqy[jp1,20,481,602],fqy[jp1,20,480,602])
    #print("advect scalar def a",tendency[1,602,12:32]/mut[602,12:32])
    #print("advect scalar def a",rv[1,602,12:32]/mut[602,12:32])
    #print("advect scalar def a",fqy[1,1,602,12:32]/mut[602,12:32])
    #print("advect scalar def a",fqy[1,1,603,12:32]/mut[603,12:32])
    #print(fqy[1,:,602,601])
    # x advection
    i_start = its
    i_end   = min(ite,ide-1)

    j_start = jts
    j_end   = min(jte,jde-1)
      
    i_start_f = i_start
    i_end_f   = i_end+1
    
    i_start = max(ids+1,its)
    i_start_f = min(i_start+2,ids+3)
    
    i_end = min(ide-2,ite)
    i_end_f = ide-3
        
    fqx[kts:ktf, j_start:j_end, i_start_f:i_end_f] = ru_modified[kts:ktf, j_start:j_end, i_start_f:i_end_f] * \
              flux5_u(field[kts:ktf, j_start:j_end, i_start_f-3:i_end_f-3],field[kts:ktf, j_start:j_end, i_start_f-2:i_end_f-2],
                      field[kts:ktf, j_start:j_end, i_start_f-1:i_end_f-1],field[kts:ktf, j_start:j_end, i_start_f:i_end_f],
                      field[kts:ktf, j_start:j_end, i_start_f+1:i_end_f+1],field[kts:ktf, j_start:j_end, i_start_f+2:i_end_f+2],
                      ru_modified[kts:ktf, j_start:j_end, i_start_f:i_end_f])
    #print("advect scalar def fqx",fqx[0:4,600,5:10]/mut[600,5:10])
    #print("advect scalar def fqx",field[0:4,600,5:10]/mut[600,5:10])
    
    # ids+1
    fqx[kts:ktf, j_start:j_end, ids+1] = 0.5 * ru_modified[kts:ktf, j_start:j_end, ids+1] * \
                      (field[kts:ktf, j_start:j_end, ids+1] + field[kts:ktf, j_start:j_end, ids])
    # ids+2
    fqx[kts:ktf, j_start:j_end, ids+2] = ru_modified[kts:ktf, j_start:j_end, ids+2] * \
              flux3_u(field[kts:ktf, j_start:j_end, ids],field[kts:ktf, j_start:j_end, ids+1],
                      field[kts:ktf, j_start:j_end, ids+2],field[kts:ktf, j_start:j_end, ids+3],
                      ru_modified[kts:ktf, j_start:j_end, ids+2])
    #print("advect scalar def fqx",fqx[:,600,5:605]/mut[600,5:605])
    #print("advect scalar def fqx",field[0:3,600,7:11])
    #print(ru[0:3,600,9])
    #        ru[:,600,9])
    #print("advect scalar def fqx",flux3_u(field[:,600,7],field[:,600,8],
    #        field[:,600,9],field[:,600,10],
    #        ru[:,600,9]))
    # ide-1
    fqx[kts:ktf, j_start:j_end, ide-2] = 0.5 * ru_modified[kts:ktf, j_start:j_end, ide-2] * \
                      (field[kts:ktf, j_start:j_end, ide-2] + field[kts:ktf, j_start:j_end, ide-3])
    # ide-2
    fqx[kts:ktf, j_start:j_end, ide-3] = ru_modified[kts:ktf, j_start:j_end, ide-3] * \
              flux3_u(field[kts:ktf, j_start:j_end, ide-5],field[kts:ktf, j_start:j_end, ide-4],
                      field[kts:ktf, j_start:j_end, ide-3],field[kts:ktf, j_start:j_end, ide-2],
                      ru_modified[kts:ktf, j_start:j_end, ide-2])
    
    mrdx[j_start:j_end, i_start:i_end] = msftx[j_start:j_end, i_start:i_end] * rdx
    mrdx_e = mrdx.repeat(nzall,1,1)
    tendency[kts:ktf, j_start:j_end, i_start:i_end] = tendency[kts:ktf, j_start:j_end, i_start:i_end] - \
                      mrdx_e[kts:ktf, j_start:j_end, i_start:i_end] * (fqx[kts:ktf, j_start:j_end, i_start+1:i_end+1] -
                                                                       fqx[kts:ktf, j_start:j_end, i_start:i_end])
    #print("in advect scalar 2:", tendency[16,323,602],fqx[16,323,603],ru[16,323,603],field[16,323,603],field[16,323,602])
    #print("in advect scalar 2:", tendency[3,302,305],field[3,302,302:308])
    #print("in advect t_tend 2: ", tendency[20, 480, 602],fqx[20,480,603],fqx[20,480,602])
    #print("in advect t_tend 2 1: ", ru[20,480,602], field[20,480,600:604])
    #print("advect scalar def",tendency[:,602,601])
    #print("advect scalar def a",tendency[1,602,12:32]/mut[602,12:32])
    #print("advect scalar def a",tendency[:,600,5:605]/mut[600,5:605])
    # z advection
    i_start = its
    i_end   = min(ite,ide-1)
    j_start = jts
    j_end   = min(jte,jde-1)
    
    vflux[kts+3:ktf-2, j_start:j_end, i_start:i_end] = rom_modified[kts+3:ktf-2, j_start:j_end, i_start:i_end] * \
              flux5_u(field[kts:ktf-5, j_start:j_end, i_start:i_end],field[kts+1:ktf-4, j_start:j_end, i_start:i_end],
                      field[kts+2:ktf-3, j_start:j_end, i_start:i_end],field[kts+3:ktf-2, j_start:j_end, i_start:i_end],
                      field[kts+4:ktf-1, j_start:j_end, i_start:i_end],field[kts+5:ktf, j_start:j_end, i_start:i_end],
                      -rom_modified[kts+3:ktf-2, j_start:j_end, i_start:i_end])
    
    # kts+1
    vflux[kts+1, j_start:j_end, i_start:i_end] = rom_modified[kts+1, j_start:j_end, i_start:i_end] * \
                      (fzm_e[kts+1, j_start:j_end, i_start:i_end] * field[kts+1, j_start:j_end, i_start:i_end] +
                       fzp_e[kts+1, j_start:j_end, i_start:i_end] * field[kts, j_start:j_end, i_start:i_end])
    # kts+2
    vflux[kts+2, j_start:j_end, i_start:i_end] = rom_modified[kts+2, j_start:j_end, i_start:i_end] * \
              flux3_u(field[kts, j_start:j_end, i_start:i_end],field[kts+1, j_start:j_end, i_start:i_end],
                      field[kts+2, j_start:j_end, i_start:i_end],field[kts+3, j_start:j_end, i_start:i_end],
                      -rom_modified[kts+2, j_start:j_end, i_start:i_end])
    # ktf-1
    vflux[ktf-2, j_start:j_end, i_start:i_end] = rom_modified[ktf-2, j_start:j_end, i_start:i_end] * \
              flux3_u(field[ktf-4, j_start:j_end, i_start:i_end],field[ktf-3, j_start:j_end, i_start:i_end],
                      field[ktf-2, j_start:j_end, i_start:i_end],field[ktf-1, j_start:j_end, i_start:i_end],
                      -rom_modified[ktf-2, j_start:j_end, i_start:i_end])
    # ktf
    #vflux_modified = vflux.clone()
    vflux[ktf-1, j_start:j_end, i_start:i_end] = rom_modified[ktf-1, j_start:j_end, i_start:i_end] * \
                    (fzm_e[ktf-1, j_start:j_end, i_start:i_end] * field[ktf-1, j_start:j_end, i_start:i_end] +
                     fzp_e[ktf-1, j_start:j_end, i_start:i_end] * field[ktf-2, j_start:j_end, i_start:i_end])
    #vflux = vflux_modified
    tendency[kts:ktf, j_start:j_end, i_start:i_end] = tendency[kts:ktf, j_start:j_end, i_start:i_end] - \
                    rdzw_e[kts:ktf, j_start:j_end, i_start:i_end] * (vflux[kts+1:ktf+1, j_start:j_end, i_start:i_end] -
                                                                     vflux[kts:ktf, j_start:j_end, i_start:i_end])
    #print("in advect scalar 3:", tendency[16,323,602])
    #print("in advect scalar 3:", tendency[3,302,305],field[0:3,302,305])
    #print("in advect t_tend 3: ", tendency[20, 480, 602],vflux[21,480,602],vflux[20,480,602])
    #print("advect scalar def a",tendency[1,602,12:32]/mut[602,12:32])
    #print("advect scalar def a",tendency[:,600,5:605]/mut[600,5:605])
    #print("advect scalar def a",rom[:,600,5:605]/mut[600,5:605])
    #print("advect scalar def",rom[:,602,601])
    #print("rom3", rom[:,603,601])    
    return tendency

# Geopotential equation right-hand side.
def rhs_ph(ph_tend, u, v, ww,               \
           ph, ph_old, phb, w,              \
           mut, muuf, muvf,                 \
           c1f, c2f,                        \
           fnm, fnp,                        \
           rdnw, cfn, cfn1, rdx, rdy,       \
           msfux, msfuy, msfvx,             \
           msfvx_inv, msfvy,                \
           msftx, msfty,                    \
           non_hydrostatic,                 \
           ids, ide, jds, jde, kds, kde,    \
           ims, ime, jms, jme, kms, kme,    \
           its, ite, jts, jte, kts, kte):
    
    itf=min(ite,ide-1)
    jtf=min(jte,jde-1)
    ktf=min(kte,kde-1)
    
    rdnw_e = rdnw.unsqueeze(1).unsqueeze(2).repeat(1,nyfull,nxfull)
    fnm_e = fnm.unsqueeze(1).unsqueeze(2).repeat(1,nyfull,nxfull)
    fnp_e = fnp.unsqueeze(1).unsqueeze(2).repeat(1,nyfull,nxfull)
    mut_e = mut.repeat(nzall,1,1)
    muuf_e = muuf.repeat(nzall,1,1)
    muvf_e = muvf.repeat(nzall,1,1)
    msfty_e = msfty.repeat(nzall,1,1)
    msfvy_e = msfvy.repeat(nzall,1,1)
    msfux_e = msfux.repeat(nzall,1,1)
    
    wdwn = torch.zeros((nzall,nyall,nxall))
        
    wdwn[1:kte, jts:jtf, its:itf] = 0.5 * (ww[1:kte, jts:jtf, its:itf] + ww[0:kte-1, jts:jtf, its:itf]) * \
                                    rdnw_e[0:kte-1, jts:jtf, its:itf] * \
                                    (ph[1:kte, jts:jtf, its:itf] - ph[0:kte-1, jts:jtf, its:itf] + 
                                     phb[1:kte, jts:jtf, its:itf] - phb[0:kte-1, jts:jtf, its:itf])
    ph_tend[1:kte-1, jts:jtf, its:itf] = ph_tend[1:kte-1, jts:jtf, its:itf] - \
                    (fnm_e[1:kte-1, jts:jtf, its:itf] * wdwn[2:kte, jts:jtf, its:itf] + 
                     fnp_e[1:kte-1, jts:jtf, its:itf] * wdwn[1:kte-1, jts:jtf, its:itf])
    
    ph_tend[kte-1, jts:jtf, its:itf] = 0.
    ph_tend[1:kte, jts:jtf, its:itf] = ph_tend[1:kte, jts:jtf, its:itf] + mut_e[1:kte, jts:jtf, its:itf] * \
                    g * w[1:kte, jts:jtf, its:itf] / msfty_e[1:kte, jts:jtf, its:itf]
    
    # y advection
    i_start = its 
    j_start = jts 
    itf=min(ite,ide-1)
    jtf=min(jte,jde-1)
    
    j_start = max(jts,jds+3)
    jtf     = min(jtf,jde-4)
    
    ph_tend[1:kte-1, j_start:jtf, i_start:itf] = ph_tend[1:kte-1, j_start:jtf, i_start:itf] - \
                     (0.25*rdy/msfty_e[1:kte-1, j_start:jtf, i_start:itf]) * ((muvf_e[1:kte-1, j_start+1:jtf+1, i_start:itf] * 
                     (v[1:kte-1, j_start+1:jtf+1, i_start:itf] + v[0:kte-2, j_start+1:jtf+1, i_start:itf]) * \
                     msfvy_e[1:kte-1, j_start+1:jtf+1, i_start:itf] + muvf_e[1:kte-1, j_start:jtf, i_start:itf] * \
                     (v[1:kte-1, j_start:jtf, i_start:itf] + v[0:kte-2, j_start:jtf, i_start:itf]) * \
                     msfvy_e[1:kte-1, j_start:jtf, i_start:itf]) * (1./60.) \
                     * (45. * (ph[1:kte-1, j_start+1:jtf+1, i_start:itf] - ph[1:kte-1, j_start-1:jtf-1, i_start:itf]) - \
                        9. * (ph[1:kte-1, j_start+2:jtf+2, i_start:itf] - ph[1:kte-1, j_start-2:jtf-2, i_start:itf]) \
                        + (ph[1:kte-1, j_start+3:jtf+3, i_start:itf] - ph[1:kte-1, j_start-3:jtf-3, i_start:itf]) + \
                        45. * (phb[1:kte-1, j_start+1:jtf+1, i_start:itf] - phb[1:kte-1, j_start-1:jtf-1, i_start:itf]) - \
                        9. * (phb[1:kte-1, j_start+2:jtf+2, i_start:itf] - phb[1:kte-1, j_start-2:jtf-2, i_start:itf]) + \
                        (phb[1:kte-1, j_start+3:jtf+3, i_start:itf] - phb[1:kte-1, j_start-3:jtf-3, i_start:itf])))
    
    # kte
    ph_tend[kte-1, j_start:jtf, i_start:itf] = ph_tend[kte-1, j_start:jtf, i_start:itf] - \
                     (0.5 * rdy / msfty_e[kte-1, j_start:jtf, i_start:itf]) * ((muvf_e[kte-1, j_start+1:jtf+1, i_start:itf] * \
                     (cfn * v[kte-2, j_start+1:jtf+1, i_start:itf] + cfn1 * v[kte-3, j_start+1:jtf+1, i_start:itf]) * \
                     msfvy_e[kte-1, j_start+1:jtf+1, i_start:itf] + muvf_e[kte-1, j_start:jtf, i_start:itf] * \
                     (cfn * v[kte-2, j_start:jtf, i_start:itf] + cfn1 *v[kte-3, j_start:jtf, i_start:itf]) * \
                     msfvy_e[kte-1, j_start:jtf, i_start:itf]) * (1./60.) * \
                     (45. * (ph[kte-1, j_start+1:jtf+1, i_start:itf] - ph[kte-1, j_start-1:jtf-1, i_start:itf]) - \
                      9. * (ph[kte-1, j_start+2:jtf+2, i_start:itf] - ph[kte-1, j_start-2:jtf-2, i_start:itf]) + \
                      (ph[kte-1, j_start+3:jtf+3, i_start:itf] - ph[kte-1, j_start-3:jtf-3, i_start:itf]) + \
                      45. * (phb[kte-1, j_start+1:jtf+1, i_start:itf] - phb[kte-1, j_start-1:jtf-1, i_start:itf]) - \
                      9. * (phb[kte-1, j_start+2:jtf+2, i_start:itf] - phb[kte-1, j_start-2:jtf-2, i_start:itf]) + \
                      (phb[kte-1, j_start+3:jtf+3, i_start:itf] - phb[kte-1, j_start-3:jtf-3, i_start:itf])))
    # jds +2
    ph_tend[1:kte-1, jds+2, i_start:itf] = ph_tend[1:kte-1, jds+2, i_start:itf] - \
                     (0.25 * rdy / msfty_e[1:kte-1, jds+2, i_start:itf]) * ((muvf_e[1:kte-1, jds+3, i_start:itf] * 
                     (v[1:kte-1, jds+3, i_start:itf] + v[0:kte-2, jds+3, i_start:itf]) * \
                     msfvy_e[1:kte-1, jds+3, i_start:itf] + muvf_e[1:kte-1, jds+2, i_start:itf] * \
                     (v[1:kte-1, jds+2, i_start:itf] + v[0:kte-2, jds+2, i_start:itf]) * \
                     msfvy_e[1:kte-1, jds+2, i_start:itf]) * (1./12.) * \
                     (8. * (ph[1:kte-1, jds+3, i_start:itf] - ph[1:kte-1, jds+1, i_start:itf]) - \
                      (ph[1:kte-1, jds+4, i_start:itf] - ph[1:kte-1, jds, i_start:itf]) + \
                      8. * (phb[1:kte-1, jds+3, i_start:itf] - phb[1:kte-1, jds+1, i_start:itf]) - \
                      (phb[1:kte-1, jds+4, i_start:itf] - phb[1:kte-1, jds, i_start:itf])))
    
    # jds +2, kte
    ph_tend[kte-1, jds+2, i_start:itf] = ph_tend[kte-1, jds+2, i_start:itf] - (0.5 * rdy / msfty_e[kte-1, jds+2, i_start:itf]) * \
                     ((muvf_e[kte-1, jds+3, i_start:itf] * (cfn * v[kte-2, jds+3, i_start:itf] + \
                       cfn1 * v[kte-3, jds+3, i_start:itf]) * msfvy_e[kte-1, jds+3, i_start:itf] +
                       muvf_e[kte-1, jds+2, i_start:itf] * (cfn * v[kte-2, jds+2, i_start:itf] + \
                       cfn1 * v[kte-3, jds+2, i_start:itf]) * msfvy_e[kte-1, jds+2, i_start:itf]) * (1./12.) * \
                     (8. * (ph[kte-1, jds+3, i_start:itf] - ph[kte-1, jds+1, i_start:itf]) - \
                      (ph[kte-1, jds+4, i_start:itf] - ph[kte-1, jds, i_start:itf]) + \
                      8. * (phb[kte-1, jds+3, i_start:itf] - phb[kte-1, jds+1, i_start:itf]) - \
                      (phb[kte-1, jds+4, i_start:itf] - phb[kte-1, jds, i_start:itf]) ))
    # jde -3                
    ph_tend[1:kte-1, jde-4, i_start:itf] = ph_tend[1:kte-1, jde-4, i_start:itf] - \
                     (0.25 * rdy / msfty_e[1:kte-1, jde-4, i_start:itf]) * ((muvf_e[1:kte-1, jde-3, i_start:itf] * \
                     (v[1:kte-1, jde-3, i_start:itf] + v[0:kte-2, jde-3, i_start:itf]) * \
                     msfvy_e[1:kte-1, jde-3, i_start:itf] + muvf_e[1:kte-1, jde-4, i_start:itf] * \
                     (v[1:kte-1, jde-4, i_start:itf] + v[0:kte-2, jde-4, i_start:itf]) * \
                     msfvy_e[1:kte-1, jde-4, i_start:itf]) * (1./12.) * \
                     (8. * (ph[1:kte-1, jde-3, i_start:itf] - ph[1:kte-1, jde-5, i_start:itf]) - \
                      (ph[1:kte-1, jde-2, i_start:itf] - ph[1:kte-1, jde-6, i_start:itf]) + \
                      8. * (phb[1:kte-1, jde-3, i_start:itf] - phb[1:kte-1, jde-5, i_start:itf]) - \
                      (phb[1:kte-1, jde-2, i_start:itf] - phb[1:kte-1, jde-6, i_start:itf])))
    # jde -3, kte
    ph_tend[kte-1, jde-4, i_start:itf] = ph_tend[kte-1, jde-4, i_start:itf] - \
                     (0.5 * rdy / msfty_e[kte-1, jde-4, i_start:itf]) * ((muvf_e[kte-1, jde-3, i_start:itf] * \
                     (cfn * v[kte-2, jde-3, i_start:itf] + cfn1 * v[kte-3, jde-3, i_start:itf]) * \
                     msfvy_e[kte-1, jde-3, i_start:itf] + muvf_e[kte-1, jde-4, i_start:itf] * \
                     (cfn * v[kte-2, jde-4, i_start:itf] + cfn1 * v[kte-3, jde-4, i_start:itf]) * \
                     msfvy_e[kte-1, jde-4, i_start:itf]) * (1./12.) * \
                     (8. * (ph[kte-1, jde-3, i_start:itf] - ph[kte-1, jde-5, i_start:itf]) - \
                      (ph[kte-1, jde-2, i_start:itf] - ph[kte-1, jde-6, i_start:itf]) + \
                      8. * (phb[kte-1, jde-3, i_start:itf] - phb[kte-1, jde-5, i_start:itf]) - \
                      (phb[kte-1, jde-2, i_start:itf] - phb[kte-1, jde-6, i_start:itf])))
    # jds +1
    ph_tend[1:kte-1, jds+1, i_start:itf] = ph_tend[1:kte-1, jds+1, i_start:itf] - \
                     (0.25 * rdy / msfty_e[1:kte-1, jds+1, i_start:itf]) * \
                     (muvf_e[1:kte-1, jds+2, i_start:itf] * \
                      (v[1:kte-1, jds+2, i_start:itf] + v[0:kte-2, jds+2, i_start:itf]) * \
                      msfvy_e[1:kte-1, jds+2, i_start:itf] * \
                      (phb[1:kte-1, jds+2, i_start:itf] - phb[1:kte-1, jds+1, i_start:itf] + \
                       ph[1:kte-1, jds+2, i_start:itf] - ph[1:kte-1, jds+1, i_start:itf]) + 
                      muvf_e[1:kte-1, jds+1, i_start:itf] * \
                      (v[1:kte-1, jds+1, i_start:itf] + v[0:kte-2, jds+1, i_start:itf]) * \
                      msfvy_e[1:kte-1, jds+1, i_start:itf] * \
                      (phb[1:kte-1, jds+1, i_start:itf] - phb[1:kte-1, jds, i_start:itf] + \
                       ph[1:kte-1, jds+1, i_start:itf] - ph[1:kte-1, jds, i_start:itf]))
    
    # jds +1, kte
    ph_tend[kte-1, jds+1, i_start:itf] = ph_tend[kte-1, jds+1, i_start:itf] - \
                     (0.5 * rdy / msfty_e[kte-1, jds+1, i_start:itf]) * \
                     (muvf_e[kte-1, jds+2, i_start:itf] * \
                      (cfn * v[kte-2, jds+2, i_start:itf] + cfn1 * v[kte-3, jds+2, i_start:itf]) * \
                      msfvy_e[kte-1, jds+2, i_start:itf] * \
                      (phb[kte-1, jds+2, i_start:itf] - phb[kte-1, jds+1, i_start:itf] + \
                       ph[kte-1, jds+2, i_start:itf] - ph[kte-1, jds+1, i_start:itf]) +
                      muvf_e[kte-1, jds+1, i_start:itf] * \
                      (cfn * v[kte-2, jds+1, i_start:itf] + cfn1 * v[kte-3, jds+1, i_start:itf]) * \
                      msfvy_e[kte-1, jds+1, i_start:itf] * \
                      (phb[kte-1, jds+1, i_start:itf] - phb[kte-1, jds, i_start:itf] + \
                       ph[kte-1, jds+1, i_start:itf] - ph[kte-1, jds, i_start:itf]))
    # jde -2
    ph_tend[1:kte-1, jde-3, i_start:itf] = ph_tend[1:kte-1, jde-3, i_start:itf] - \
                     (0.25 * rdy / msfty_e[1:kte-1, jde-3, i_start:itf]) * \
                     (muvf_e[1:kte-1, jde-2, i_start:itf] * \
                      (v[1:kte-1, jde-2, i_start:itf] + v[0:kte-2, jde-2, i_start:itf]) * \
                      msfvy_e[1:kte-1, jde-2, i_start:itf] * \
                     (phb[1:kte-1, jde-2, i_start:itf] -phb[1:kte-1, jde-3, i_start:itf] + \
                      ph[1:kte-1, jde-2, i_start:itf] - ph[1:kte-1, jde-3, i_start:itf]) + \
                      muvf_e[1:kte-1, jde-3, i_start:itf] * \
                      (v[1:kte-1, jde-3, i_start:itf] + v[0:kte-2, jde-3, i_start:itf]) * \
                      msfvy_e[1:kte-1, jde-3, i_start:itf] * \
                     (phb[1:kte-1, jde-3, i_start:itf] -phb[1:kte-1, jde-4, i_start:itf] + \
                      ph[1:kte-1, jde-3, i_start:itf] - ph[1:kte-1, jde-4, i_start:itf]))
    # jde -2, kte
    ph_tend[kte-1, jde-3, i_start:itf] = ph_tend[kte-1, jde-3, i_start:itf] - \
                     (0.5 * rdy / msfty_e[kte-1, jde-3, i_start:itf]) * \
                     (muvf_e[kte-1, jde-2, i_start:itf] * \
                      (cfn * v[kte-2, jde-2, i_start:itf] + cfn1 * v[kte-3, jde-2, i_start:itf]) * \
                      msfvy_e[kte-1, jde-2, i_start:itf] * \
                      (phb[kte-1, jde-2, i_start:itf] -phb[kte-1, jde-3, i_start:itf] + \
                       ph[kte-1, jde-2, i_start:itf] - ph[kte-1, jde-3, i_start:itf]) +
                      muvf_e[kte-1, jde-3, i_start:itf] * \
                      (cfn * v[kte-2, jde-3, i_start:itf] + cfn1 * v[kte-3, jde-3, i_start:itf]) * \
                      msfvy_e[kte-1, jde-3, i_start:itf] * \
                      (phb[kte-1, jde-3, i_start:itf] - phb[kte-1, jde-4, i_start:itf] + \
                       ph[kte-1, jde-3, i_start:itf] - ph[kte-1, jde-4, i_start:itf]))
    
    # x advection
    i_start = its
    j_start = jts
    itf=min(ite,ide-1)
    jtf=min(jte,jde-1)
    
    i_start = max(its,ids+3)
    itf     = min(itf,ide-4)
    
    ph_tend[1:kte-1, j_start:jtf, i_start:itf] = ph_tend[1:kte-1, j_start:jtf, i_start:itf] - \
                     (0.25 * rdx / msfty_e[1:kte-1, j_start:jtf, i_start:itf]) * \
                     ((muuf_e[1:kte-1, j_start:jtf, i_start+1:itf+1] * \
                       (u[1:kte-1, j_start:jtf, i_start+1:itf+1] + u[0:kte-2, j_start:jtf, i_start+1:itf+1]) * \
                       msfux_e[1:kte-1, j_start:jtf, i_start+1:itf+1] +
                       muuf_e[1:kte-1, j_start:jtf, i_start:itf] * \
                       (u[1:kte-1, j_start:jtf, i_start:itf] + u[0:kte-2, j_start:jtf, i_start:itf]) * \
                       msfux_e[1:kte-1, j_start:jtf, i_start:itf]) * (1./60.) *
                      (45. * (ph[1:kte-1, j_start:jtf, i_start+1:itf+1] - ph[1:kte-1, j_start:jtf, i_start-1:itf-1]) - \
                       9. * (ph[1:kte-1, j_start:jtf, i_start+2:itf+2] - ph[1:kte-1, j_start:jtf, i_start-2:itf-2]) + \
                       (ph[1:kte-1, j_start:jtf, i_start+3:itf+3] - ph[1:kte-1, j_start:jtf, i_start-3:itf-3]) + \
                       45. * (phb[1:kte-1, j_start:jtf, i_start+1:itf+1] - phb[1:kte-1, j_start:jtf, i_start-1:itf-1]) - \
                       9. * (phb[1:kte-1, j_start:jtf, i_start+2:itf+2] - phb[1:kte-1, j_start:jtf, i_start-2:itf-2]) + \
                       (phb[1:kte-1, j_start:jtf, i_start+3:itf+3] - phb[1:kte-1, j_start:jtf, i_start-3:itf-3])))
    
    # kte
    ph_tend[kte-1, j_start:jtf, i_start:itf] = ph_tend[kte-1, j_start:jtf, i_start:itf] - \
                     (0.5 * rdx / msfty_e[kte-1, j_start:jtf, i_start:itf]) * ((muuf_e[kte-1, j_start:jtf, i_start+1:itf+1] * \
                     (cfn * u[kte-2, j_start:jtf, i_start+1:itf+1] + cfn1 * u[kte-3, j_start:jtf, i_start+1:itf+1]) * \
                     msfux_e[kte-1, j_start:jtf, i_start+1:itf+1] + muuf_e[kte-1, j_start:jtf, i_start:itf] * \
                     (cfn * u[kte-2, j_start:jtf, i_start:itf] + cfn1 * u[kte-3, j_start:jtf, i_start:itf]) * \
                     msfux_e[kte-1, j_start:jtf, i_start:itf]) * (1./60.) * \
                     (45. * (ph[kte-1, j_start:jtf, i_start+1:itf+1] - ph[kte-1, j_start:jtf, i_start-1:itf-1]) - \
                      9. * (ph[kte-1, j_start:jtf, i_start+2:itf+2] - ph[kte-1, j_start:jtf, i_start-2:itf-2]) + \
                      (ph[kte-1, j_start:jtf, i_start+3:itf+3] - ph[kte-1, j_start:jtf, i_start-3:itf-3]) + \
                      45. * (phb[kte-1, j_start:jtf, i_start+1:itf+1] - phb[kte-1, j_start:jtf, i_start-1:itf-1]) - \
                      9. * (phb[kte-1, j_start:jtf, i_start+2:itf+2] - phb[kte-1, j_start:jtf, i_start-2:itf-2]) + \
                      (phb[kte-1, j_start:jtf, i_start+3:itf+3] - phb[kte-1, j_start:jtf, i_start-3:itf-3])))
    
    # ids +1
    ph_tend[1:kte-1, j_start:jtf, ids+1] = ph_tend[1:kte-1, j_start:jtf, ids+1] - \
                     (0.25 * rdx / msfty_e[1:kte-1, j_start:jtf, ids+1]) * \
                     (muuf_e[1:kte-1, j_start:jtf, ids+2] * \
                      (u[1:kte-1, j_start:jtf, ids+2] + u[0:kte-2, j_start:jtf, ids+2]) * \
                      msfux_e[1:kte-1, j_start:jtf, ids+2] * 
                      (phb[1:kte-1, j_start:jtf, ids+2] - phb[1:kte-1, j_start:jtf, ids+1] + 
                       ph[1:kte-1, j_start:jtf, ids+2] - ph[1:kte-1, j_start:jtf, ids+1]) +
                      muuf_e[1:kte-1, j_start:jtf, ids+1] * \
                      (u[1:kte-1, j_start:jtf, ids+1] + u[0:kte-2, j_start:jtf, ids+1]) * \
                      msfux_e[1:kte-1, j_start:jtf, ids+1] * 
                      (phb[1:kte-1, j_start:jtf, ids+1] - phb[1:kte-1, j_start:jtf, ids] + 
                       ph[1:kte-1, j_start:jtf, ids+1] - ph[1:kte-1, j_start:jtf, ids])) 
    # ids +1, kte
    ph_tend[kte-1, j_start:jtf, ids+1] = ph_tend[kte-1, j_start:jtf, ids+1] - \
                     (0.5 * rdx / msfty_e[kte-1, j_start:jtf, ids+1]) * \
                     (muuf_e[kte-1, j_start:jtf, ids+2] * \
                      (cfn * u[kte-2, j_start:jtf, ids+2] + cfn1 * u[kte-3, j_start:jtf, ids+2]) * 
                      msfux_e[kte-1, j_start:jtf, ids+2] * 
                      (phb[kte-1, j_start:jtf, ids+2] - phb[kte-1, j_start:jtf, ids+1] + 
                       ph[kte-1, j_start:jtf, ids+2] - ph[kte-1, j_start:jtf, ids+1]) +
                      muuf_e[kte-1, j_start:jtf, ids+1] * \
                      (cfn * u[kte-2, j_start:jtf, ids+1] + cfn1 * u[kte-3, j_start:jtf, ids+1]) * 
                      msfux_e[kte-1, j_start:jtf, ids+1] * 
                       (phb[kte-1, j_start:jtf, ids+1] - phb[kte-1, j_start:jtf, ids] + 
                        ph[kte-1, j_start:jtf, ids+1] - ph[kte-1, j_start:jtf, ids]))
    # ide -2
    ph_tend[1:kte-1, j_start:jtf, ide-3] = ph_tend[1:kte-1, j_start:jtf, ide-3] - \
                     (0.25 * rdx / msfty_e[1:kte-1, j_start:jtf, ide-3]) * \
                     (muuf_e[1:kte-1, j_start:jtf, ide-2] * \
                      (u[1:kte-1, j_start:jtf, ide-2] + u[0:kte-2, j_start:jtf, ide-2]) * \
                      msfux_e[1:kte-1, j_start:jtf, ide-2] * 
                      (phb[1:kte-1, j_start:jtf, ide-2] - phb[1:kte-1, j_start:jtf, ide-3] + 
                       ph[1:kte-1, j_start:jtf, ide-2] - ph[1:kte-1, j_start:jtf, ide-3]) +
                      muuf_e[1:kte-1, j_start:jtf, ide-3] * \
                      (u[1:kte-1, j_start:jtf, ide-3] + u[0:kte-2, j_start:jtf, ide-3]) * \
                      msfux_e[1:kte-1, j_start:jtf, ide-3] * 
                      (phb[1:kte-1, j_start:jtf, ide-3] - phb[1:kte-1, j_start:jtf, ide-4] + 
                       ph[1:kte-1, j_start:jtf, ide-3] - ph[1:kte-1, j_start:jtf, ide-4]))
    # ide -2, kte
    ph_tend[kte-1, j_start:jtf, ide-3] = ph_tend[kte-1, j_start:jtf, ide-3] - \
                     (0.5 * rdx / msfty_e[kte-1, j_start:jtf, ide-3]) * \
                     (muuf_e[kte-1, j_start:jtf, ide-2] * \
                      (cfn * u[kte-2, j_start:jtf, ide-2] + cfn1 * u[kte-3, j_start:jtf, ide-2]) * 
                      msfux_e[kte-1, j_start:jtf, ide-2] * 
                      (phb[kte-1, j_start:jtf, ide-2] - phb[kte-1, j_start:jtf, ide-3] + 
                       ph[kte-1, j_start:jtf, ide-2] - ph[kte-1, j_start:jtf, ide-3]) +
                      muuf_e[kte-1, j_start:jtf, ide-3] * \
                      (cfn * u[kte-2, j_start:jtf, ide-3] + cfn1 * u[kte-3, j_start:jtf, ide-3]) * 
                      msfux_e[kte-1, j_start:jtf, ide-3] * 
                       (phb[kte-1, j_start:jtf, ide-3] - phb[kte-1, j_start:jtf, ide-4] + 
                        ph[kte-1, j_start:jtf, ide-3] - ph[kte-1, j_start:jtf, ide-4]))
    
    return ph_tend

# Horizontal pressure-gradient force tendency.
def horizontal_pressure_gradient(ru_tend,rv_tend,                 \
                                 ph,alt,p,pb,al,php,cqu,cqv,      \
                                 muu,muv,mu,c1h,c2h,fnm,fnp,rdnw, \
                                 cf1,cf2,cf3,cfn,cfn1,            \
                                 rdx,rdy,msfux,msfuy,             \
                                 msfvx,msfvy,msftx,msfty,         \
				 non_hydrostatic,top_lid,         \
                                 ids, ide, jds, jde, kds, kde,    \
                                 ims, ime, jms, jme, kms, kme,    \
                                 its, ite, jts, jte, kts, kte):
    
    dpn = torch.zeros((nzall,nyall,nxall))
    dpy = torch.zeros((nzall,nyall,nxall))
    dpx = torch.zeros((nzall,nyall,nxall))
    
    p_modified = p.clone()
    ph_modified = ph.clone()
    php_modified = php.clone()
    # y pressure gradient
    itf=min(ite,ide-1)
    jtf=jte
    ktf=min(kte,kde-1)
    i_start = its
    j_start = jts
    
    j_start = jts+1
    jtf = jtf-1
    
    rdnw_e = rdnw.unsqueeze(1).unsqueeze(2).repeat(1,nyfull,nxfull)
    fnm_e = fnm.unsqueeze(1).unsqueeze(2).repeat(1,nyfull,nxfull)
    fnp_e = fnp.unsqueeze(1).unsqueeze(2).repeat(1,nyfull,nxfull)
    
    msfvy_e = msfvy.repeat(nzall,1,1)
    msfvx_e = msfvx.repeat(nzall,1,1)
    msfuy_e = msfuy.repeat(nzall,1,1)
    msfux_e = msfux.repeat(nzall,1,1)
    
    muv_e = muv.repeat(nzall,1,1)
    muu_e = muu.repeat(nzall,1,1)
    mu_e = mu.repeat(nzall,1,1)
    
    k = 0
    dpn[k, j_start:jtf, i_start:itf] = 0.5 * (cf1 * (p_modified[k, j_start-1:jtf-1, i_start:itf] + p_modified[k, j_start:jtf, i_start:itf]) + \
                                              cf2 * (p_modified[k+1, j_start-1:jtf-1, i_start:itf] + p_modified[k+1, j_start:jtf, i_start:itf]) + \
                                              cf3 * (p_modified[k+2, j_start-1:jtf-1, i_start:itf] + p_modified[k+2, j_start:jtf, i_start:itf]))
    dpn[kde-1, j_start:jtf, i_start:itf] = 0.
    
    k_start = 1
    dpn[k_start:ktf, j_start:jtf, i_start:itf] = 0.5 * (fnm_e[k_start:ktf, j_start:jtf, i_start:itf] * 
                                                  (p_modified[k_start:ktf, j_start-1:jtf-1, i_start:itf] + p_modified[k_start:ktf, j_start:jtf, i_start:itf]) + 
                                                  fnp_e[k_start:ktf, j_start:jtf, i_start:itf] * 
                                                  (p_modified[k_start-1:ktf-1, j_start-1:jtf-1, i_start:itf] + p_modified[k_start-1:ktf-1, j_start:jtf, i_start:itf]))
    
    k_start = 0
    dpy[k_start:ktf, j_start:jtf, i_start:itf] = (msfvy_e[k_start:ktf, j_start:jtf, i_start:itf] / \
                      msfvx_e[k_start:ktf, j_start:jtf, i_start:itf]) * 0.5 * rdy * muv_e[k_start:ktf, j_start:jtf, i_start:itf] * \
                      ((ph_modified[k_start+1:ktf+1, j_start:jtf, i_start:itf] - ph_modified[k_start+1:ktf+1, j_start-1:jtf-1, i_start:itf] + 
                        ph_modified[k_start:ktf, j_start:jtf, i_start:itf] - ph_modified[k_start:ktf, j_start-1:jtf-1, i_start:itf]) +
                       (alt[k_start:ktf, j_start:jtf, i_start:itf] + alt[k_start:ktf, j_start-1:jtf-1, i_start:itf]) * 
                       (p_modified[k_start:ktf, j_start:jtf, i_start:itf] - p_modified[k_start:ktf, j_start-1:jtf-1, i_start:itf]) +
                       (al[k_start:ktf, j_start:jtf, i_start:itf] + al[k_start:ktf, j_start-1:jtf-1, i_start:itf]) * 
                       (pb[k_start:ktf, j_start:jtf, i_start:itf] - pb[k_start:ktf, j_start-1:jtf-1, i_start:itf]))
    
    dpy_modified = dpy.clone()
    dpy_modified[k_start:ktf, j_start:jtf, i_start:itf] = dpy[k_start:ktf, j_start:jtf, i_start:itf] + \
                      (msfvy_e[k_start:ktf, j_start:jtf, i_start:itf] / msfvx_e[k_start:ktf, j_start:jtf, i_start:itf]) * rdy * \
                      (php_modified[k_start:ktf, j_start:jtf, i_start:itf] - php_modified[k_start:ktf, j_start-1:jtf-1, i_start:itf]) * \
                      (rdnw_e[k_start:ktf, j_start:jtf, i_start:itf] * 
                       (dpn[k_start+1:ktf+1, j_start:jtf, i_start:itf] - dpn[k_start:ktf, j_start:jtf, i_start:itf]) - 0.5 * 
                       (mu_e[k_start:ktf, j_start-1:jtf-1, i_start:itf] + mu_e[k_start:ktf, j_start:jtf, i_start:itf]))
    dpy = dpy_modified
    
    cqv_modified = cqv.clone()
    rv_tend[k_start:ktf, j_start:jtf, i_start:itf] = rv_tend[k_start:ktf, j_start:jtf, i_start:itf] - \
                      cqv_modified[k_start:ktf, j_start:jtf, i_start:itf] * dpy[k_start:ktf, j_start:jtf, i_start:itf]
    
    # x pressure gradient
    itf=ite
    jtf=min(jte,jde-1)
    ktf=min(kte,kde-1)
    i_start = its
    j_start = jts
    
    i_start = its+1
    itf = itf-1
    
    k=0
    dpn[k, j_start:jtf, i_start:itf] = 0.5 * (cf1 * (p_modified[k, j_start:jtf, i_start-1:itf-1] + p_modified[k, j_start:jtf, i_start:itf]) + \
                                              cf2 * (p_modified[k+1, j_start:jtf, i_start-1:itf-1] + p_modified[k+1, j_start:jtf, i_start:itf]) + \
                                              cf3 * (p_modified[k+2, j_start:jtf, i_start-1:itf-1] + p_modified[k+2, j_start:jtf, i_start:itf]))
    dpn[kde-1, j_start:jtf, i_start:itf] = 0.
    k_start = 1
    dpn[k_start:ktf, j_start:jtf, i_start:itf] = 0.5 * (fnm_e[k_start:ktf, j_start:jtf, i_start:itf] * 
                                                  (p_modified[k_start:ktf, j_start:jtf, i_start-1:itf-1] + p_modified[k_start:ktf, j_start:jtf, i_start:itf]) + 
                                                  fnp_e[k_start:ktf, j_start:jtf, i_start:itf] * 
                                                  (p_modified[k_start-1:ktf-1, j_start:jtf, i_start-1:itf-1] + p_modified[k_start-1:ktf-1, j_start:jtf, i_start:itf]))
    
    k_start = 0
    dpx[k_start:ktf, j_start:jtf, i_start:itf] = (msfux_e[k_start:ktf, j_start:jtf, i_start:itf] / \
                       msfuy_e[k_start:ktf, j_start:jtf, i_start:itf]) * 0.5 * rdx * muu_e[k_start:ktf, j_start:jtf, i_start:itf] * \
                       ((ph_modified[k_start+1:ktf+1, j_start:jtf, i_start:itf] - ph_modified[k_start+1:ktf+1, j_start:jtf, i_start-1:itf-1] + 
                        ph_modified[k_start:ktf, j_start:jtf, i_start:itf] - ph_modified[k_start:ktf, j_start:jtf, i_start-1:itf-1]) +
                       (alt[k_start:ktf, j_start:jtf, i_start:itf] + alt[k_start:ktf, j_start:jtf, i_start-1:itf-1]) * 
                       (p_modified[k_start:ktf, j_start:jtf, i_start:itf] - p_modified[k_start:ktf, j_start:jtf, i_start-1:itf-1]) +
                       (al[k_start:ktf, j_start:jtf, i_start:itf] + al[k_start:ktf, j_start:jtf, i_start-1:itf-1]) * 
                       (pb[k_start:ktf, j_start:jtf, i_start:itf] - pb[k_start:ktf, j_start:jtf, i_start-1:itf-1]))
    
    dpx_modified = dpx.clone()
    dpx_modified[k_start:ktf, j_start:jtf, i_start:itf] = dpx_modified[k_start:ktf, j_start:jtf, i_start:itf] + \
                       (msfux_e[k_start:ktf, j_start:jtf, i_start:itf] / msfuy_e[k_start:ktf, j_start:jtf, i_start:itf]) * rdx * \
                       (php_modified[k_start:ktf, j_start:jtf, i_start:itf] - php_modified[k_start:ktf, j_start:jtf, i_start-1:itf-1]) * \
                       (rdnw_e[k_start:ktf, j_start:jtf, i_start:itf] * 
                       (dpn[k_start+1:ktf+1, j_start:jtf, i_start:itf] - dpn[k_start:ktf, j_start:jtf, i_start:itf]) - 0.5 * 
                       (mu_e[k_start:ktf, j_start:jtf, i_start-1:itf-1] + mu_e[k_start:ktf, j_start:jtf, i_start:itf]))
    dpx = dpx_modified
    
    cqu_modified = cqu.clone()
    
    ru_tend[k_start:ktf, j_start:jtf, i_start:itf] = ru_tend[k_start:ktf, j_start:jtf, i_start:itf] - \
                       cqu_modified[k_start:ktf, j_start:jtf, i_start:itf] * dpx[k_start:ktf, j_start:jtf, i_start:itf]
    
    return ru_tend, rv_tend

# Pressure-gradient + buoyancy force on w.
def pg_buoy_w(rw_tend, p, cqw, muf, mubf,     \
              c1f, c2f,                       \
              rdnw, rdn, g, msftx, msfty,     \
              ids, ide, jds, jde, kds, kde,   \
              ims, ime, jms, jme, kms, kme,   \
              its, ite, jts, jte, kts, kte):
    itf=min(ite,ide-1)
    jtf=min(jte,jde-1)
    
    rdnw_e = rdnw.unsqueeze(1).unsqueeze(2).repeat(1,nyfull,nxfull)
    rdn_e = rdn.unsqueeze(1).unsqueeze(2).repeat(1,nyfull,nxfull)
    msfty_e = msfty.repeat(nzall,1,1)
    muf_e = muf.repeat(nzall,1,1)
    mubf_e = mubf.repeat(nzall,1,1)
    
    cq1 = torch.zeros((nzall,nyall,nxall))
    cq2 = torch.zeros((nzall,nyall,nxall))
    
    cq1[kde-1, jts:jtf, its:itf] = 1./(1. + cqw[kde-2, jts:jtf, its:itf])
    cq2[kde-1, jts:jtf, its:itf] = cqw[kde-2, jts:jtf, its:itf] * cq1[kde-1, jts:jtf, its:itf]
    rw_tend[kde-1, jts:jtf, its:itf] = rw_tend[kde-1, jts:jtf, its:itf] + (1. / msfty_e[kde-1, jts:jtf, its:itf]) * \
                   g * (cq1[kde-1, jts:jtf, its:itf] * 2. * rdnw_e[kde-2, jts:jtf, its:itf] * \
                        (-p[kde-2, jts:jtf, its:itf]) - muf_e[kde-1, jts:jtf, its:itf] - \
                        cq2[kde-1, jts:jtf, its:itf] * mubf_e[kde-1, jts:jtf, its:itf])
    
    k_start = 1
    
    cq1[k_start:kde-1, jts:jtf, its:itf] = 1./(1. + cqw[k_start:kde-1, jts:jtf, its:itf])
    cq2[k_start:kde-1, jts:jtf, its:itf] = cqw[k_start:kde-1, jts:jtf, its:itf] * cq1[k_start:kde-1, jts:jtf, its:itf]
    cqw[k_start:kde-1, jts:jtf, its:itf] = cq1[k_start:kde-1, jts:jtf, its:itf]
    rw_tend[k_start:kde-1, jts:jtf, its:itf] = rw_tend[k_start:kde-1, jts:jtf, its:itf] + \
                   (1. / msfty_e[k_start:kde-1, jts:jtf, its:itf]) * g * \
                   (cq1[k_start:kde-1, jts:jtf, its:itf] * rdn_e[k_start:kde-1, jts:jtf, its:itf] * 
                    (p[k_start:kde-1, jts:jtf, its:itf] - p[k_start-1:kde-2, jts:jtf, its:itf]) - 
                    muf_e[k_start:kde-1, jts:jtf, its:itf] - cq2[k_start:kde-1, jts:jtf, its:itf] * mubf_e[k_start:kde-1, jts:jtf, its:itf])
    
    return rw_tend, cqw

# Coriolis acceleration terms.
def coriolis(ru, rv, rw,                   \
             ru_tend,  rv_tend,  rw_tend,  \
             msftx, msfty, msfux, msfuy,   \
             msfvx, msfvy,                 \
             f, e, sina, cosa, fzm, fzp,   \
             ids, ide, jds, jde, kds, kde, \
             ims, ime, jms, jme, kms, kme, \
             its, ite, jts, jte, kts, kte):
    ktf=min(kte,kde-1)
    
    f_e = f.repeat(nzall,1,1)
    e_e = e.repeat(nzall,1,1)
    cosa_e = cosa.repeat(nzall,1,1)
    sina_e = sina.repeat(nzall,1,1)
    
    msfux_e = msfux.repeat(nzall,1,1)
    msfuy_e = msfuy.repeat(nzall,1,1)
    msfvy_e = msfvy.repeat(nzall,1,1)
    msfvx_e = msfvx.repeat(nzall,1,1)
    msftx_e = msftx.repeat(nzall,1,1)
    msfty_e = msfty.repeat(nzall,1,1)
    
    fzm_e = fzm.unsqueeze(1).unsqueeze(2).repeat(1,nyfull,nxfull)
    fzp_e = fzp.unsqueeze(1).unsqueeze(2).repeat(1,nyfull,nxfull)
    # coriolis for u
    i_start = its
    i_end   = ite
    
    i_start = max(ids+1,its)
    i_end   = min(ide-1,ite)
    
    ru_tend[kts:ktf, jts:jte-1, i_start:i_end] = ru_tend[kts:ktf, jts:jte-1, i_start:i_end] + \
              (msfux_e[kts:ktf, jts:jte-1, i_start:i_end] / msfuy_e[kts:ktf, jts:jte-1, i_start:i_end]) * 0.5 * \
              (f_e[kts:ktf, jts:jte-1, i_start:i_end] + f_e[kts:ktf, jts:jte-1, i_start-1:i_end-1]) * 0.25 * \
              (rv[kts:ktf, jts+1:jte, i_start-1:i_end-1] + rv[kts:ktf, jts+1:jte, i_start:i_end] + 
               rv[kts:ktf, jts:jte-1, i_start-1:i_end-1] + rv[kts:ktf, jts:jte-1, i_start:i_end]) - 0.5* \
              (e_e[kts:ktf, jts:jte-1, i_start:i_end] + e_e[kts:ktf, jts:jte-1, i_start-1:i_end-1]) * 0.5 * \
              (cosa_e[kts:ktf, jts:jte-1, i_start:i_end] + cosa_e[kts:ktf, jts:jte-1, i_start-1:i_end-1]) * 0.25 * \
              (rw[kts+1:ktf+1, jts:jte-1, i_start-1:i_end-1] + rw[kts:ktf, jts:jte-1, i_start-1:i_end-1] + 
               rw[kts+1:ktf+1, jts:jte-1, i_start:i_end] + rw[kts:ktf, jts:jte-1, i_start:i_end])
    # coriolis for v
    j_start = jts
    j_end   = jte
    
    j_start = max(jds+1,jts)
    j_end   = min(jde-1,jte)
    
    rv_tend[kts:ktf, j_start:j_end, its:ite-1] = rv_tend[kts:ktf, j_start:j_end, its:ite-1] - \
              (msfvy_e[kts:ktf, j_start:j_end, its:ite-1] / msfvx_e[kts:ktf, j_start:j_end, its:ite-1]) * 0.5 * \
              (f_e[kts:ktf, j_start:j_end, its:ite-1] + f_e[kts:ktf, j_start-1:j_end-1, its:ite-1]) * 0.25 * \
              (ru[kts:ktf, j_start:j_end, its:ite-1] + ru[kts:ktf, j_start:j_end, its+1:ite] + 
               ru[kts:ktf, j_start-1:j_end-1, its:ite-1] + ru[kts:ktf, j_start-1:j_end-1, its+1:ite]) + \
              (msfvy_e[kts:ktf, j_start:j_end, its:ite-1] / msfvx_e[kts:ktf, j_start:j_end, its:ite-1]) * 0.5 * \
              (e_e[kts:ktf, j_start:j_end, its:ite-1] + e_e[kts:ktf, j_start-1:j_end-1, its:ite-1]) * 0.5 * \
              (sina_e[kts:ktf, j_start:j_end, its:ite-1] + sina_e[kts:ktf, j_start-1:j_end-1, its:ite-1]) * 0.25 * \
              (rw[kts+1:ktf+1, j_start-1:j_end-1, its:ite-1] + rw[kts:ktf, j_start-1:j_end-1, its:ite-1] + 
               rw[kts+1:ktf+1, j_start:j_end, its:ite-1] + rw[kts:ktf, j_start:j_end, its:ite-1])
    # coriolis for w
    rw_tend[kts+1:ktf, jts:jte-1, its:ite-1] = rw_tend[kts+1:ktf, jts:jte-1, its:ite-1] + \
              e_e[kts+1:ktf, jts:jte-1, its:ite-1] * (cosa_e[kts+1:ktf, jts:jte-1, its:ite-1] * 0.5 * 
              (fzm_e[kts+1:ktf, jts:jte-1, its:ite-1] * (ru[kts+1:ktf, jts:jte-1, its:ite-1] + 
               ru[kts+1:ktf, jts:jte-1, its+1:ite]) + fzp_e[kts+1:ktf, jts:jte-1, its:ite-1] * 
               (ru[kts:ktf-1, jts:jte-1, its:ite-1] + ru[kts:ktf-1, jts:jte-1, its+1:ite])) - \
              (msftx_e[kts+1:ktf, jts:jte-1, its:ite-1] / msfty_e[kts+1:ktf, jts:jte-1, its:ite-1]) * \
              sina_e[kts+1:ktf, jts:jte-1, its:ite-1] * 0.5 * (fzm_e[kts+1:ktf, jts:jte-1, its:ite-1] * 
               (rv[kts+1:ktf, jts:jte-1, its:ite-1] + rv[kts+1:ktf, jts+1:jte, its:ite-1]) + 
               fzp_e[kts+1:ktf, jts:jte-1, its:ite-1] * 
               (rv[kts:ktf-1, jts:jte-1, its:ite-1] +rv[kts:ktf-1, jts+1:jte, its:ite-1])))
       
    return ru_tend, rv_tend, rw_tend

# Curvature terms in the momentum equations.
def curvature(ru, rv, rw, u, v, w, ru_tend, rv_tend, rw_tend, \
              msfux, msfuy, msfvx, msfvy, msftx, msfty,       \
              xlat, fzm, fzp, rdx, rdy,                       \
              ids, ide, jds, jde, kds, kde,                   \
              ims, ime, jms, jme, kms, kme,                   \
              its, ite, jts, jte, kts, kte ):
    itf=min(ite,ide-1)
    jtf=min(jte,jde-1)
    ktf=min(kte,kde-1)
    
    i_start = its-1
    i_end   = ite
    j_start = jts-1
    j_end   = jte
    
    i_start = its
    i_end   = ite-1
    j_start = jts
    j_end   = jte-1
    
    msfvx_e = msfvx.repeat(nzall,1,1)
    msfuy_e = msfuy.repeat(nzall,1,1)
    msfvy_e = msfvy.repeat(nzall,1,1)
    msftx_e = msftx.repeat(nzall,1,1)
    msfty_e = msfty.repeat(nzall,1,1)
    
    fzm_e = fzm.unsqueeze(1).unsqueeze(2).repeat(1,nyfull,nxfull)
    fzp_e = fzp.unsqueeze(1).unsqueeze(2).repeat(1,nyfull,nxfull)
    
    vxgm = torch.zeros((nzall,nyall,nxall))
    
    vxgm[kts:ktf, j_start:j_end, i_start:i_end] = 0.5 * (u[kts:ktf, j_start:j_end, i_start:i_end] + u[kts:ktf, j_start:j_end, i_start+1:i_end+1]) * \
                                                        (msfvx_e[kts:ktf, j_start+1:j_end+1, i_start:i_end] - msfvx_e[kts:ktf, j_start:j_end, i_start:i_end]) * rdy - \
                                                  0.5 * (v[kts:ktf, j_start:j_end, i_start:i_end] + v[kts:ktf, j_start+1:j_end+1, i_start:i_end]) * \
                                                        (msfuy_e[kts:ktf, j_start:j_end, i_start+1:i_end+1] - msfuy_e[kts:ktf, j_start:j_end, i_start:i_end]) * rdx
    vxgm[kts:ktf, jts:jte-1, its-1] = vxgm[kts:ktf, jts:jte-1, its]
    vxgm[kts:ktf, jts:jte-1, ite-1] = vxgm[kts:ktf, jts:jte-1, ite-2]
    vxgm[kts:ktf, jts-1, its-1:ite] = vxgm[kts:ktf, jts, its-1:ite]
    vxgm[kts:ktf, jte-1, its-1:ite] = vxgm[kts:ktf, jte-2, its-1:ite]
    
    # curvature for u
    i_start  = its
    i_start = max( ids+1 , its )
    i_end   = min( ide-1 , ite )
    
    ru_tend[kts:ktf, jts:jte-1, i_start:i_end] = ru_tend[kts:ktf, jts:jte-1, i_start:i_end] + 0.5 * \
               (vxgm[kts:ktf, jts:jte-1, i_start:i_end] + vxgm[kts:ktf, jts:jte-1, i_start-1:i_end-1]) * 0.25 * \
                (rv[kts:ktf, jts+1:jte, i_start-1:i_end-1] + rv[kts:ktf, jts+1:jte, i_start:i_end] + 
                 rv[kts:ktf, jts:jte-1, i_start-1:i_end-1] + rv[kts:ktf, jts:jte-1, i_start:i_end]) - \
               u[kts:ktf, jts:jte-1, i_start:i_end] * reradius * 0.25 * \
                (rw[kts+1:ktf+1, jts:jte-1, i_start-1:i_end-1] + rw[kts:ktf, jts:jte-1, i_start-1:i_end-1] + 
                 rw[kts+1:ktf+1, jts:jte-1, i_start:i_end] + rw[kts:ktf, jts:jte-1, i_start:i_end])
    # curvature for v
    j_start = jts
    j_start = max ( jds+1 , jts )
    j_end   = min ( jde-1 , jte )
    
    rv_tend[kts:ktf, j_start:j_end, its:ite-1] = rv_tend[kts:ktf, j_start:j_end, its:ite-1] - 0.5 * \
               (vxgm[kts:ktf, j_start:j_end, its:ite-1] + vxgm[kts:ktf, j_start-1:j_end-1, its:ite-1]) * 0.25 * \
                (ru[kts:ktf, j_start:j_end, its:ite-1] + ru[kts:ktf, j_start:j_end, its+1:ite] + 
                 ru[kts:ktf, j_start-1:j_end-1, its:ite-1] + ru[kts:ktf, j_start-1:j_end-1, its+1:ite]) - \
               (msfvy_e[kts:ktf, j_start:j_end, its:ite-1] / msfvx_e[kts:ktf, j_start:j_end, its:ite-1]) * \
               v[kts:ktf, j_start:j_end, its:ite-1] * reradius * 0.25 * \
                (rw[kts+1:ktf+1, j_start-1:j_end-1, its:ite-1] + rw[kts:ktf, j_start-1:j_end-1, its:ite-1] + 
                 rw[kts+1:ktf+1, j_start:j_end, its:ite-1] + rw[kts:ktf, j_start:j_end, its:ite-1])
    # curvature for w
    rw_tend[kts+1:ktf, jts:jte-1, its:ite-1] = rw_tend[kts+1:ktf, jts:jte-1, its:ite-1] + \
               reradius * (0.5 * (fzm_e[kts+1:ktf, jts:jte-1, its:ite-1] * 
                                  (ru[kts+1:ktf, jts:jte-1, its:ite-1] + ru[kts+1:ktf, jts:jte-1, its+1:ite]) + 
                                  fzp_e[kts+1:ktf, jts:jte-1, its:ite-1] * 
                                  (ru[kts:ktf-1, jts:jte-1, its:ite-1] + ru[kts:ktf-1, jts:jte-1, its+1:ite])) * 
                           0.5 * (fzm_e[kts+1:ktf, jts:jte-1, its:ite-1] * 
                                  (u[kts+1:ktf, jts:jte-1, its:ite-1] + u[kts+1:ktf, jts:jte-1, its+1:ite]) + 
                                  fzp_e[kts+1:ktf, jts:jte-1, its:ite-1] * 
                                  (u[kts:ktf-1, jts:jte-1, its:ite-1] + u[kts:ktf-1, jts:jte-1, its+1:ite])) +
                          (msftx_e[kts+1:ktf, jts:jte-1, its:ite-1] / msfty_e[kts+1:ktf, jts:jte-1, its:ite-1]) * 
                           0.5 * (fzm_e[kts+1:ktf, jts:jte-1, its:ite-1] * 
                                  (rv[kts+1:ktf, jts:jte-1, its:ite-1] + rv[kts+1:ktf, jts+1:jte, its:ite-1]) + 
                                  fzp_e[kts+1:ktf, jts:jte-1, its:ite-1] * 
                                  (rv[kts:ktf-1, jts:jte-1, its:ite-1] + rv[kts:ktf-1, jts+1:jte, its:ite-1])) *
                           0.5 * (fzm_e[kts+1:ktf, jts:jte-1, its:ite-1] * 
                                  (v[kts+1:ktf, jts:jte-1, its:ite-1] + v[kts+1:ktf, jts+1:jte, its:ite-1]) + 
                                  fzp_e[kts+1:ktf, jts:jte-1, its:ite-1] * 
                                  (v[kts:ktf-1, jts:jte-1, its:ite-1] + v[kts:ktf-1, jts+1:jte, its:ite-1])))
                
    return ru_tend, rv_tend, rw_tend

# Horizontal diffusion tendency (2nd / 6th order).
def horizontal_diffusion(name, field, tendency, mut, c1, c2,  \
                         msfux, msfuy, msfvx, msfvx_inv,      \
                         msfvy, msftx, msfty,                 \
                         khdif, xkmhd, rdx, rdy,              \
                         ids, ide, jds, jde, kds, kde,        \
                         ims, ime, jms, jme, kms, kme,        \
                         its, ite, jts, jte, kts, kte,):
    ktf=min(kte,kde-1)
    
    msfux_e = msfux.repeat(nzall,1,1)
    msfuy_e = msfuy.repeat(nzall,1,1)
    msfvx_e = msfvx.repeat(nzall,1,1)
    msfvy_e = msfvy.repeat(nzall,1,1)
    msftx_e = msftx.repeat(nzall,1,1)
    msfty_e = msfty.repeat(nzall,1,1)
    msfvx_inv_e = msfvx_inv.repeat(nzall,1,1)
    
    mut_e = mut.repeat(nzall,1,1)
    
    mkrdxm = torch.zeros((nzall,nyall,nxall))
    mkrdxp = torch.zeros((nzall,nyall,nxall))
    mrdx = torch.zeros((nzall,nyall,nxall))
    mkrdym = torch.zeros((nzall,nyall,nxall))
    mkrdyp = torch.zeros((nzall,nyall,nxall))
    mrdy = torch.zeros((nzall,nyall,nxall))
    
    if name=="u":
        i_start = its
        i_end   = ite
        j_start = jts
        j_end   = min(jte,jde-1)
        
        i_start = max(ids+1,its)
        i_end   = min(ide-1,ite)
        j_start = max(jds+1,jts)
        j_end   = min(jde-2,jte)
        
        mkrdxm[kts:ktf, j_start:j_end, i_start:i_end] = (msftx_e[kts:ktf, j_start:j_end, i_start-1:i_end-1] / \
                   msfty_e[kts:ktf, j_start:j_end, i_start-1:i_end-1]) * mut_e[kts:ktf, j_start:j_end, i_start-1:i_end-1] * \
                   xkmhd[kts:ktf, j_start:j_end, i_start-1:i_end-1] * rdx
        mkrdxp[kts:ktf, j_start:j_end, i_start:i_end] = (msftx_e[kts:ktf, j_start:j_end, i_start:i_end] / \
                   msfty_e[kts:ktf, j_start:j_end, i_start:i_end]) * mut_e[kts:ktf, j_start:j_end, i_start:i_end] * \
                   xkmhd[kts:ktf, j_start:j_end, i_start:i_end] * rdx
        
        mrdx[kts:ktf, j_start:j_end, i_start:i_end] = msfux_e[kts:ktf, j_start:j_end, i_start:i_end] * \
                   msfuy_e[kts:ktf, j_start:j_end, i_start:i_end] * rdx
        mkrdym[kts:ktf, j_start:j_end, i_start:i_end] = ((msfuy_e[kts:ktf, j_start:j_end, i_start:i_end] + 
                   msfuy_e[kts:ktf, j_start-1:j_end-1, i_start:i_end]) / \
                   (msfux_e[kts:ktf, j_start:j_end, i_start:i_end] + msfux_e[kts:ktf, j_start-1:j_end-1, i_start:i_end])) * \
                   0.25 * (mut_e[kts:ktf, j_start:j_end, i_start:i_end] + mut_e[kts:ktf, j_start-1:j_end-1, i_start:i_end] + 
                           mut_e[kts:ktf, j_start-1:j_end-1, i_start-1:i_end-1] + mut_e[kts:ktf, j_start:j_end, i_start-1:i_end-1]) * \
                   0.25 * (xkmhd[kts:ktf, j_start:j_end, i_start:i_end] + xkmhd[kts:ktf, j_start-1:j_end-1, i_start:i_end] + 
                           xkmhd[kts:ktf, j_start-1:j_end-1, i_start-1:i_end-1] + xkmhd[kts:ktf, j_start:j_end, i_start-1:i_end-1]) * rdy
        mkrdyp[kts:ktf, j_start:j_end, i_start:i_end] = ((msfuy_e[kts:ktf, j_start:j_end, i_start:i_end] + 
                   msfuy_e[kts:ktf, j_start+1:j_end+1, i_start:i_end]) / \
                   (msfux_e[kts:ktf, j_start:j_end, i_start:i_end] + msfux_e[kts:ktf, j_start+1:j_end+1, i_start:i_end])) * \
                   0.25 * (mut_e[kts:ktf, j_start:j_end, i_start:i_end] + mut_e[kts:ktf, j_start+1:j_end+1, i_start:i_end] + 
                           mut_e[kts:ktf, j_start+1:j_end+1, i_start-1:i_end-1] + mut_e[kts:ktf, j_start:j_end, i_start-1:i_end-1]) * \
                   0.25 * (xkmhd[kts:ktf, j_start:j_end, i_start:i_end] + xkmhd[kts:ktf, j_start+1:j_end+1, i_start:i_end] + 
                           xkmhd[kts:ktf, j_start+1:j_end+1, i_start-1:i_end-1] + xkmhd[kts:ktf, j_start:j_end, i_start-1:i_end-1]) * rdy
        mrdy[kts:ktf, j_start:j_end, i_start:i_end] = msfux_e[kts:ktf, j_start:j_end, i_start:i_end] * \
                   msfuy_e[kts:ktf, j_start:j_end, i_start:i_end] * rdy
                   
        tendency[kts:ktf, j_start:j_end, i_start:i_end] = tendency[kts:ktf, j_start:j_end, i_start:i_end] + \
                   (mrdx[kts:ktf, j_start:j_end, i_start:i_end] * (mkrdxp[kts:ktf, j_start:j_end, i_start:i_end] * 
                    (field[kts:ktf, j_start:j_end, i_start+1:i_end+1] - field[kts:ktf, j_start:j_end, i_start:i_end]) - 
                    mkrdxm[kts:ktf, j_start:j_end, i_start:i_end] * 
                    (field[kts:ktf, j_start:j_end, i_start:i_end] - field[kts:ktf, j_start:j_end, i_start-1:i_end-1])) +
                    mrdy[kts:ktf, j_start:j_end, i_start:i_end] * (mkrdyp[kts:ktf, j_start:j_end, i_start:i_end] * 
                    (field[kts:ktf, j_start+1:j_end+1, i_start:i_end] - field[kts:ktf, j_start:j_end, i_start:i_end]) -
                    mkrdym[kts:ktf, j_start:j_end, i_start:i_end] * 
                    (field[kts:ktf, j_start:j_end, i_start:i_end] - field[kts:ktf, j_start-1:j_end-1, i_start:i_end])))
        
    elif name=="v":
        i_start = its
        i_end   = min(ite,ide-1)
        j_start = jts
        j_end   = jte
        
        i_start = max(ids+1,its)
        i_end   = min(ide-2,ite)
        j_start = max(jds+1,jts)
        j_end   = min(jde-1,jte)
        
        mkrdxm[kts:ktf, j_start:j_end, i_start:i_end] = ((msfvx_e[kts:ktf, j_start:j_end, i_start:i_end] + 
                    msfvx_e[kts:ktf, j_start:j_end, i_start-1:i_end-1]) / \
                    (msfvy_e[kts:ktf, j_start:j_end, i_start:i_end] + msfvy_e[kts:ktf, j_start:j_end, i_start-1:i_end-1])) * \
                    0.25 * (mut_e[kts:ktf, j_start:j_end, i_start:i_end] + mut_e[kts:ktf, j_start-1:j_end-1, i_start:i_end] + 
                            mut_e[kts:ktf, j_start-1:j_end-1, i_start-1:i_end-1] + mut_e[kts:ktf, j_start:j_end, i_start-1:i_end-1]) * \
                    0.25 * (xkmhd[kts:ktf, j_start:j_end, i_start:i_end] + xkmhd[kts:ktf, j_start-1:j_end-1, i_start:i_end] + 
                            xkmhd[kts:ktf, j_start-1:j_end-1, i_start-1:i_end-1] + xkmhd[kts:ktf, j_start:j_end, i_start-1:i_end-1]) * rdx
        mkrdxp[kts:ktf, j_start:j_end, i_start:i_end] = ((msfvx_e[kts:ktf, j_start:j_end, i_start:i_end] + 
                    msfvx_e[kts:ktf, j_start:j_end, i_start+1:i_end+1]) / \
                    (msfvy_e[kts:ktf, j_start:j_end, i_start:i_end] + msfvy_e[kts:ktf, j_start:j_end, i_start+1:i_end+1])) * \
                    0.25 * (mut_e[kts:ktf, j_start:j_end, i_start:i_end] + mut_e[kts:ktf, j_start-1:j_end-1, i_start:i_end] + 
                            mut_e[kts:ktf, j_start-1:j_end-1, i_start+1:i_end+1] + mut_e[kts:ktf, j_start:j_end, i_start+1:i_end+1]) * \
                    0.25 * (xkmhd[kts:ktf, j_start:j_end, i_start:i_end] + xkmhd[kts:ktf, j_start-1:j_end-1, i_start:i_end] + 
                            xkmhd[kts:ktf, j_start-1:j_end-1, i_start+1:i_end+1] + xkmhd[kts:ktf, j_start:j_end, i_start+1:i_end+1]) * rdx
        mrdx[kts:ktf, j_start:j_end, i_start:i_end] = msfvx_e[kts:ktf, j_start:j_end, i_start:i_end] * \
                    msfvy_e[kts:ktf, j_start:j_end, i_start:i_end] * rdx
        # 注意，mkrdym,mkrdyp是否要乘mut？

        mkrdym[kts:ktf, j_start:j_end, i_start:i_end] = (msfty_e[kts:ktf, j_start-1:j_end-1, i_start:i_end] / \
                    msftx_e[kts:ktf, j_start-1:j_end-1, i_start:i_end]) *  \
                    xkmhd[kts:ktf, j_start-1:j_end-1, i_start:i_end] * rdy
        
        mkrdyp[kts:ktf, j_start:j_end, i_start:i_end] = (msfty_e[kts:ktf, j_start:j_end, i_start:i_end] / \
                    msftx_e[kts:ktf, j_start:j_end, i_start:i_end]) *  \
                    xkmhd[kts:ktf, j_start:j_end, i_start:i_end] * rdy
        mrdy[kts:ktf, j_start:j_end, i_start:i_end] = msfvx_e[kts:ktf, j_start:j_end, i_start:i_end] * \
                    msfvy_e[kts:ktf, j_start:j_end, i_start:i_end] * rdy
        
        tendency[kts:ktf, j_start:j_end, i_start:i_end] = tendency[kts:ktf, j_start:j_end, i_start:i_end] + \
                   (mrdx[kts:ktf, j_start:j_end, i_start:i_end] * (mkrdxp[kts:ktf, j_start:j_end, i_start:i_end] * 
                    (field[kts:ktf, j_start:j_end, i_start+1:i_end+1] - field[kts:ktf, j_start:j_end, i_start:i_end]) - 
                    mkrdxm[kts:ktf, j_start:j_end, i_start:i_end] * 
                    (field[kts:ktf, j_start:j_end, i_start:i_end] - field[kts:ktf, j_start:j_end, i_start-1:i_end-1])) +
                    mrdy[kts:ktf, j_start:j_end, i_start:i_end] * (mkrdyp[kts:ktf, j_start:j_end, i_start:i_end] * 
                    (field[kts:ktf, j_start+1:j_end+1, i_start:i_end] - field[kts:ktf, j_start:j_end, i_start:i_end]) - 
                    mkrdym[kts:ktf, j_start:j_end, i_start:i_end] * 
                    (field[kts:ktf, j_start:j_end, i_start:i_end] - field[kts:ktf, j_start-1:j_end-1, i_start:i_end])))
        
    elif name=="w":
        i_start = its
        i_end   = min(ite,ide-1)
        j_start = jts
        j_end   = min(jte,jde-1)
        
        i_start = max(ids+1,its)
        i_end   = min(ide-2,ite)
        j_start = max(jds+1,jts)
        j_end   = min(jde-2,jte)
        
        mkrdxm[kts+1:ktf, j_start:j_end, i_start:i_end] = (msfux_e[kts+1:ktf, j_start:j_end, i_start:i_end] / 
                  msfuy_e[kts+1:ktf, j_start:j_end, i_start:i_end]) * \
                  0.25 * (mut_e[kts+1:ktf, j_start:j_end, i_start:i_end] + mut_e[kts+1:ktf, j_start:j_end, i_start-1:i_end-1] + 
                          mut_e[kts+1:ktf, j_start:j_end, i_start:i_end] + mut_e[kts+1:ktf, j_start:j_end, i_start-1:i_end-1]) * \
                  0.25 * (xkmhd[kts+1:ktf, j_start:j_end, i_start:i_end] + xkmhd[kts+1:ktf, j_start:j_end, i_start-1:i_end-1] + 
                          xkmhd[kts:ktf-1, j_start:j_end, i_start:i_end] + xkmhd[kts:ktf-1, j_start:j_end, i_start-1:i_end-1]) * rdx
        mkrdxp[kts+1:ktf, j_start:j_end, i_start:i_end] = (msfux_e[kts+1:ktf, j_start:j_end, i_start+1:i_end+1] / 
                  msfuy_e[kts+1:ktf, j_start:j_end, i_start+1:i_end+1]) * \
                  0.25 * (mut_e[kts+1:ktf, j_start:j_end, i_start+1:i_end+1] + mut_e[kts+1:ktf, j_start:j_end, i_start:i_end] + 
                          mut_e[kts+1:ktf, j_start:j_end, i_start+1:i_end+1] + mut_e[kts+1:ktf, j_start:j_end, i_start:i_end]) * \
                  0.25 * (xkmhd[kts+1:ktf, j_start:j_end, i_start+1:i_end+1] + xkmhd[kts+1:ktf, j_start:j_end, i_start:i_end] + 
                          xkmhd[kts:ktf-1, j_start:j_end, i_start+1:i_end+1] + xkmhd[kts:ktf-1, j_start:j_end, i_start:i_end]) * rdx
        mrdx[kts+1:ktf, j_start:j_end, i_start:i_end] = msftx_e[kts+1:ktf, j_start:j_end, i_start:i_end] * \
                  msfty_e[kts+1:ktf, j_start:j_end, i_start:i_end] * rdx
        mkrdym[kts+1:ktf, j_start:j_end, i_start:i_end] = (msfvy_e[kts+1:ktf, j_start:j_end, i_start:i_end] / \
                  msfvx_e[kts+1:ktf, j_start:j_end, i_start:i_end]) * \
                  0.25 * (mut_e[kts+1:ktf, j_start:j_end, i_start:i_end] + mut_e[kts+1:ktf, j_start-1:j_end-1, i_start:i_end] + 
                          mut_e[kts+1:ktf, j_start:j_end, i_start:i_end] + mut_e[kts+1:ktf, j_start-1:j_end-1, i_start:i_end]) * \
                  0.25 * (xkmhd[kts+1:ktf, j_start:j_end, i_start:i_end] + xkmhd[kts+1:ktf, j_start-1:j_end-1, i_start:i_end] + 
                          xkmhd[kts:ktf-1, j_start:j_end, i_start:i_end] + xkmhd[kts:ktf-1, j_start-1:j_end-1, i_start:i_end]) * rdy
        mkrdyp[kts+1:ktf, j_start:j_end, i_start:i_end] = (msfvy_e[kts+1:ktf, j_start+1:j_end+1, i_start:i_end] / \
                  msfvx_e[kts+1:ktf, j_start+1:j_end+1, i_start:i_end]) * \
                  0.25 * (mut_e[kts+1:ktf, j_start+1:j_end+1, i_start:i_end] + mut_e[kts+1:ktf, j_start:j_end, i_start:i_end] + 
                          mut_e[kts+1:ktf, j_start+1:j_end+1, i_start:i_end] + mut_e[kts+1:ktf, j_start:j_end, i_start:i_end]) * \
                  0.25 * (xkmhd[kts+1:ktf, j_start+1:j_end+1, i_start:i_end] + xkmhd[kts+1:ktf, j_start:j_end, i_start:i_end] + 
                          xkmhd[kts:ktf-1, j_start+1:j_end+1, i_start:i_end] + xkmhd[kts:ktf-1, j_start:j_end, i_start:i_end]) * rdy
        mrdy[kts+1:ktf, j_start:j_end, i_start:i_end] = msftx_e[kts+1:ktf, j_start:j_end, i_start:i_end] * \
                  msfty_e[kts+1:ktf, j_start:j_end, i_start:i_end] * rdy
        
        tendency[kts+1:ktf, j_start:j_end, i_start:i_end] = tendency[kts+1:ktf, j_start:j_end, i_start:i_end] + \
                 (mrdx[kts+1:ktf, j_start:j_end, i_start:i_end] * (mkrdxp[kts+1:ktf, j_start:j_end, i_start:i_end] * 
                  (field[kts+1:ktf, j_start:j_end, i_start+1:i_end+1] - field[kts+1:ktf, j_start:j_end, i_start:i_end]) - 
                  mkrdxm[kts+1:ktf, j_start:j_end, i_start:i_end] * 
                  (field[kts+1:ktf, j_start:j_end, i_start:i_end] - field[kts+1:ktf, j_start:j_end, i_start-1:i_end-1])) +
                  mrdy[kts+1:ktf, j_start:j_end, i_start:i_end] * (mkrdyp[kts+1:ktf, j_start:j_end, i_start:i_end] * 
                  (field[kts+1:ktf, j_start+1:j_end+1, i_start:i_end] - field[kts+1:ktf, j_start:j_end, i_start:i_end]) - 
                  mkrdym[kts+1:ktf, j_start:j_end, i_start:i_end] * 
                  (field[kts+1:ktf, j_start:j_end, i_start:i_end] - field[kts+1:ktf, j_start-1:j_end-1, i_start:i_end])))
    
    else:
        i_start = its
        i_end   = min(ite,ide-1)
        j_start = jts
        j_end   = min(jte,jde-1)
        
        i_start = max(ids+1,its)
        i_end   = min(ide-2,ite)
        j_start = max(jds+1,jts)
        j_end   = min(jde-2,jte)
        
        mkrdxm[kts:ktf, j_start:j_end, i_start:i_end] = (msfux_e[kts:ktf, j_start:j_end, i_start:i_end] / \
                  msfuy_e[kts:ktf, j_start:j_end, i_start:i_end]) * 0.5 * \
                  (xkmhd[kts:ktf, j_start:j_end, i_start:i_end] + xkmhd[kts:ktf, j_start:j_end, i_start-1:i_end-1]) * 0.5 * \
                  (mut_e[kts:ktf, j_start:j_end, i_start:i_end] + mut_e[kts:ktf, j_start:j_end, i_start-1:i_end-1]) * rdx
        mkrdxp[kts:ktf, j_start:j_end, i_start:i_end] = (msfux_e[kts:ktf, j_start:j_end, i_start+1:i_end+1] / \
                  msfuy_e[kts:ktf, j_start:j_end, i_start+1:i_end+1]) * 0.5 * \
                  (xkmhd[kts:ktf, j_start:j_end, i_start+1:i_end+1] + xkmhd[kts:ktf, j_start:j_end, i_start:i_end]) * 0.5 * \
                  (mut_e[kts:ktf, j_start:j_end, i_start+1:i_end+1] + mut_e[kts:ktf, j_start:j_end, i_start:i_end]) * rdx
        
        mrdx[kts:ktf, j_start:j_end, i_start:i_end] = msftx_e[kts:ktf, j_start:j_end, i_start:i_end] * \
                  msfty_e[kts:ktf, j_start:j_end, i_start:i_end] * rdx
        mkrdym[kts:ktf, j_start:j_end, i_start:i_end] = (msfvy_e[kts:ktf, j_start:j_end, i_start:i_end] * \
                  msfvx_inv_e[kts:ktf, j_start:j_end, i_start:i_end]) * 0.5 * \
                  (xkmhd[kts:ktf, j_start:j_end, i_start:i_end] + xkmhd[kts:ktf, j_start-1:j_end-1, i_start:i_end]) * 0.5 * \
                  (mut_e[kts:ktf, j_start:j_end, i_start:i_end] + mut_e[kts:ktf, j_start-1:j_end-1, i_start:i_end]) * rdy
        mkrdyp[kts:ktf, j_start:j_end, i_start:i_end] = (msfvy_e[kts:ktf, j_start+1:j_end+1, i_start:i_end] * \
                  msfvx_inv_e[kts:ktf, j_start+1:j_end+1, i_start:i_end]) * 0.5 * \
                  (xkmhd[kts:ktf, j_start+1:j_end+1, i_start:i_end] + xkmhd[kts:ktf, j_start:j_end, i_start:i_end]) * 0.5 * \
                  (mut_e[kts:ktf, j_start+1:j_end+1, i_start:i_end] + mut_e[kts:ktf, j_start:j_end, i_start:i_end]) * rdy
        mrdy[kts:ktf, j_start:j_end, i_start:i_end] = msftx_e[kts:ktf, j_start:j_end, i_start:i_end] * \
                  msfty_e[kts:ktf, j_start:j_end, i_start:i_end] * rdy
        
        tendency[kts:ktf, j_start:j_end, i_start:i_end] = tendency[kts:ktf, j_start:j_end, i_start:i_end] + \
                 (mrdx[kts:ktf, j_start:j_end, i_start:i_end] * (mkrdxp[kts:ktf, j_start:j_end, i_start:i_end] * 
                  (field[kts:ktf, j_start:j_end, i_start+1:i_end+1] - field[kts:ktf, j_start:j_end, i_start:i_end]) - 
                  mkrdxm[kts:ktf, j_start:j_end, i_start:i_end] * 
                  (field[kts:ktf, j_start:j_end, i_start:i_end] - field[kts:ktf, j_start:j_end, i_start-1:i_end-1])) + 
                  mrdy[kts:ktf, j_start:j_end, i_start:i_end] * (mkrdyp[kts:ktf, j_start:j_end, i_start:i_end] * 
                  (field[kts:ktf, j_start+1:j_end+1, i_start:i_end] - field[kts:ktf, j_start:j_end, i_start:i_end]) - 
                  mkrdym[kts:ktf, j_start:j_end, i_start:i_end] * 
                  (field[kts:ktf, j_start:j_end, i_start:i_end] - field[kts:ktf, j_start-1:j_end-1, i_start:i_end])))
    
    return tendency

# Horizontal diffusion for a 3-D field.
def horizontal_diffusion_3dmp(name, field, tendency, mut, c1, c2,  \
                              base_3d,                             \
                              msfux, msfuy, msfvx, msfvx_inv,      \
                              msfvy, msftx, msfty,                 \
                              khdif, xkmhd, rdx, rdy,              \
                              ids, ide, jds, jde, kds, kde,        \
                              ims, ime, jms, jme, kms, kme,        \
                              its, ite, jts, jte, kts, kte):
    ktf=min(kte,kde-1)
    
    i_start = its
    i_end   = min(ite,ide-1)
    j_start = jts
    j_end   = min(jte,jde-1)
    
    i_start = max(ids+1,its)
    i_end   = min(ide-2,ite)
    j_start = max(jds+1,jts)
    j_end   = min(jde-2,jte)
    
    msfux_e = msfux.repeat(nzall,1,1)
    msfuy_e = msfuy.repeat(nzall,1,1)
    msfvy_e = msfvy.repeat(nzall,1,1)
    msftx_e = msftx.repeat(nzall,1,1)
    msfty_e = msfty.repeat(nzall,1,1)
    msfvx_inv_e = msfvx_inv.repeat(nzall,1,1)
    
    mkrdxm = torch.zeros((nzall,nyall,nxall))
    mkrdxp = torch.zeros((nzall,nyall,nxall))
    mrdx = torch.zeros((nzall,nyall,nxall))
    mkrdym = torch.zeros((nzall,nyall,nxall))
    mkrdyp = torch.zeros((nzall,nyall,nxall))
    mrdy = torch.zeros((nzall,nyall,nxall))
    
    mut_e = mut.repeat(nzall,1,1)
    
    mkrdxm[kts:ktf, j_start:j_end, i_start:i_end] = (msfux_e[kts:ktf, j_start:j_end, i_start:i_end] / \
                    msfuy_e[kts:ktf, j_start:j_end, i_start:i_end]) * 0.5 * \
                    (xkmhd[kts:ktf, j_start:j_end, i_start:i_end] + xkmhd[kts:ktf, j_start:j_end, i_start-1:i_end-1]) * 0.5 * \
                    (mut_e[kts:ktf, j_start:j_end, i_start:i_end] + mut_e[kts:ktf, j_start:j_end, i_start-1:i_end-1]) *rdx
    
    mkrdxp[kts:ktf, j_start:j_end, i_start:i_end] = (msfux_e[kts:ktf, j_start:j_end, i_start+1:i_end+1] / \
                    msfuy_e[kts:ktf, j_start:j_end, i_start+1:i_end+1]) * 0.5 * \
                    (xkmhd[kts:ktf, j_start:j_end, i_start+1:i_end+1] + xkmhd[kts:ktf, j_start:j_end, i_start:i_end]) * 0.5 * \
                    (mut_e[kts:ktf, j_start:j_end, i_start+1:i_end+1] + mut_e[kts:ktf, j_start:j_end, i_start:i_end]) *rdx
    
    mrdx[kts:ktf, j_start:j_end, i_start:i_end] = msftx_e[kts:ktf, j_start:j_end, i_start:i_end] * \
                    msfty_e[kts:ktf, j_start:j_end, i_start:i_end] * rdx
    mkrdym[kts:ktf, j_start:j_end, i_start:i_end] = (msfvy_e[kts:ktf, j_start:j_end, i_start:i_end] * 
                    msfvx_inv_e[kts:ktf, j_start:j_end, i_start:i_end]) * 0.5 * \
                    (xkmhd[kts:ktf, j_start:j_end, i_start:i_end] + xkmhd[kts:ktf, j_start-1:j_end-1, i_start:i_end]) * 0.5 * \
                    (mut_e[kts:ktf, j_start:j_end, i_start:i_end] + mut_e[kts:ktf, j_start-1:j_end-1, i_start:i_end]) * rdy
    
    mkrdyp[kts:ktf, j_start:j_end, i_start:i_end] = (msfvy_e[kts:ktf, j_start+1:j_end+1, i_start:i_end] * 
                    msfvx_inv_e[kts:ktf, j_start+1:j_end+1, i_start:i_end]) * 0.5 * \
                    (xkmhd[kts:ktf, j_start+1:j_end+1, i_start:i_end] + xkmhd[kts:ktf, j_start:j_end, i_start:i_end]) * 0.5 * \
                    (mut_e[kts:ktf, j_start+1:j_end+1, i_start:i_end] + mut_e[kts:ktf, j_start:j_end, i_start:i_end]) * rdy
    
    mrdy[kts:ktf, j_start:j_end, i_start:i_end] = msftx_e[kts:ktf, j_start:j_end, i_start:i_end] * \
                    msfty_e[kts:ktf, j_start:j_end, i_start:i_end] * rdy
    tendency[kts:ktf, j_start:j_end, i_start:i_end] = tendency[kts:ktf, j_start:j_end, i_start:i_end] + \
                   (mrdx[kts:ktf, j_start:j_end, i_start:i_end] * (mkrdxp[kts:ktf, j_start:j_end, i_start:i_end] * 
                    (field[kts:ktf, j_start:j_end, i_start+1:i_end+1] - field[kts:ktf, j_start:j_end, i_start:i_end] - 
                     base_3d[kts:ktf, j_start:j_end, i_start+1:i_end+1] + base_3d[kts:ktf, j_start:j_end, i_start:i_end]) - 
                    mkrdxm[kts:ktf, j_start:j_end, i_start:i_end] * 
                    (field[kts:ktf, j_start:j_end, i_start:i_end] - field[kts:ktf, j_start:j_end, i_start-1:i_end-1] - 
                     base_3d[kts:ktf, j_start:j_end, i_start:i_end] + base_3d[kts:ktf, j_start:j_end, i_start-1:i_end-1])) +
                    mrdy[kts:ktf, j_start:j_end, i_start:i_end] * (mkrdyp[kts:ktf, j_start:j_end, i_start:i_end] * 
                    (field[kts:ktf, j_start+1:j_end+1, i_start:i_end] - field[kts:ktf, j_start:j_end, i_start:i_end] - 
                     base_3d[kts:ktf, j_start+1:j_end+1, i_start:i_end] + base_3d[kts:ktf, j_start:j_end, i_start:i_end]) -
                    mkrdym[kts:ktf, j_start:j_end, i_start:i_end] * 
                    (field[kts:ktf, j_start:j_end, i_start:i_end] - field[kts:ktf, j_start-1:j_end-1, i_start:i_end] - 
                     base_3d[kts:ktf, j_start:j_end, i_start:i_end] + base_3d[kts:ktf, j_start-1:j_end-1, i_start:i_end])))
    
    return tendency

def vertical_diffusion(name, field, tendency,        \
                       c1, c2,                       \
                       alt, MUT, rdn, rdnw, kvdif,   \
                       ids, ide, jds, jde, kds, kde, \
                       ims, ime, jms, jme, kms, kme, \
                       its, ite, jts, jte, kts, kte):
    ktf=min(kte,kde-1)
    
    vflux = torch.zeros((nzall,nyall,nxall)).to(device)
    rdnw_e = rdnw.unsqueeze(1).unsqueeze(2).repeat(1,nyfull,nxfull)
    rdn_e = rdn.unsqueeze(1).unsqueeze(2).repeat(1,nyfull,nxfull)
    mut_e = MUT.repeat(nzall,1,1)
    
    if name == "w":
        i_start = its
        i_end   = min(ite,ide-1)
        j_start = jts
        j_end   = min(jte,jde-1)
        
        vflux[kts:ktf-1, j_start:j_end, i_start:i_end] = (kvdif / alt[kts:ktf-1, j_start:j_end, i_start:i_end]) * \
                rdnw_e[kts:ktf-1, j_start:j_end, i_start:i_end] * \
                (field[kts+1:ktf, j_start:j_end, i_start:i_end] - field[kts:ktf-1, j_start:j_end, i_start:i_end])
        vflux[ktf-1, j_start:j_end, i_start:i_end] = 0.
        tendency[kts+1:ktf, j_start:j_end, i_start:i_end] = tendency[kts+1:ktf, j_start:j_end, i_start:i_end] + \
                rdn_e[kts+1:ktf, j_start:j_end, i_start:i_end] * g * g / mut_e[kts+1:ktf, j_start:j_end, i_start:i_end] / \
                (0.5 * (alt[kts+1:ktf, j_start:j_end, i_start:i_end] + alt[kts:ktf-1, j_start:j_end, i_start:i_end])) * \
                (vflux[kts+1:ktf, j_start:j_end, i_start:i_end] - vflux[kts:ktf-1, j_start:j_end, i_start:i_end])
    elif name == "m":
        i_start = its
        i_end   = min(ite,ide-1)
        j_start = jts
        j_end   = min(jte,jde-1)
        vflux[kts:ktf-1, j_start:j_end, i_start:i_end] = kvdif * rdn_e[kts+1:ktf, j_start:j_end, i_start:i_end] / \
                (0.5 * (alt[kts:ktf-1, j_start:j_end, i_start:i_end] + alt[kts+1:ktf, j_start:j_end, i_start:i_end])) * \
                (field[kts+1:ktf, j_start:j_end, i_start:i_end] - field[kts:ktf-1, j_start:j_end, i_start:i_end])
        vflux[kts, j_start:j_end, i_start:i_end] = vflux[kts+1, j_start:j_end, i_start:i_end]
        vflux[ktf-1, j_start:j_end, i_start:i_end] = 0.
        tendency[kts+1:ktf, j_start:j_end, i_start:i_end] = tendency[kts+1:ktf, j_start:j_end, i_start:i_end] + \
                g * g / mut_e[kts+1:ktf, j_start:j_end, i_start:i_end] / alt[kts+1:ktf, j_start:j_end, i_start:i_end] * \
                rdnw_e[kts+1:ktf, j_start:j_end, i_start:i_end] * \
                (vflux[kts+1:ktf, j_start:j_end, i_start:i_end] - vflux[kts:ktf-1, j_start:j_end, i_start:i_end])
            
    return tendency

# Vertical diffusion tendency for u.
def vertical_diffusion_u(field, tendency,              \
                         u_base, c1h,c2h,              \
                         alt, muu, rdn, rdnw, kvdif,   \
                         ids, ide, jds, jde, kds, kde, \
                         ims, ime, jms, jme, kms, kme, \
                         its, ite, jts, jte, kts, kte):
    ktf=min(kte,kde-1)
    i_start = max(ids+1,its)
    i_end   = min(ide-1,ite)
    j_start = jts
    j_end   = min(jte,jde-1)
    
    vflux = torch.zeros((nzall,nyall,nxall)).to(device)
    rdnw_e = rdnw.unsqueeze(1).unsqueeze(2).repeat(1,nyfull,nxfull)
    rdn_e = rdn.unsqueeze(1).unsqueeze(2).repeat(1,nyfull,nxfull)
    muu_e = muu.repeat(nzall,1,1)
    u_base_e = u_base.unsqueeze(1).unsqueeze(2).repeat(1,nyfull,nxfull)
    
    vflux[kts:ktf-1, j_start:j_end, i_start:i_end] = kvdif * rdn_e[kts+1:ktf, j_start:j_end, i_start:i_end] / \
            (0.25 * (alt[kts:ktf-1, j_start:j_end, i_start:i_end] + alt[kts:ktf-1, j_start:j_end, i_start-1:i_end-1] + 
                     alt[kts+1:ktf, j_start:j_end, i_start:i_end] + alt[kts+1:ktf, j_start:j_end, i_start-1:i_end-1])) * \
            (field[kts+1:ktf, j_start:j_end, i_start:i_end] - field[kts:ktf-1, j_start:j_end, i_start:i_end] - 
             u_base_e[kts+1:ktf, j_start:j_end, i_start:i_end] + u_base_e[kts:ktf-1, j_start:j_end, i_start:i_end])
    vflux[ktf-1, j_start:j_end, i_start:i_end] = 0.
    tendency[kts+1:ktf-1, j_start:j_end, i_start:i_end] = tendency[kts+1:ktf-1, j_start:j_end, i_start:i_end] + g * g * \
            rdnw_e[kts+1:ktf-1, j_start:j_end, i_start:i_end] / muu_e[kts+1:ktf-1, j_start:j_end, i_start:i_end] / (
            0.5 * (alt[kts+1:ktf-1, j_start:j_end, i_start-1:i_end-1] + alt[kts+1:ktf-1, j_start:j_end, i_start:i_end])) * (
            vflux[kts+1:ktf-1, j_start:j_end, i_start:i_end] - vflux[kts:ktf-2, j_start:j_end, i_start:i_end])
        
    return tendency

# Vertical diffusion tendency for v.
def vertical_diffusion_v(field, tendency,              \
                         v_base, c1h,c2h,              \
                         alt, muv, rdn, rdnw, kvdif,   \
                         ids, ide, jds, jde, kds, kde, \
                         ims, ime, jms, jme, kms, kme, \
                         its, ite, jts, jte, kts, kte):
    
    ktf=min(kte,kde-1)
    i_start = its
    i_end   = min(ite,ide-1)
    j_start = jts
    j_end   = min(jte,jde-1)
    
    vflux = torch.zeros((nzall,nyall,nxall)).to(device)
    rdnw_e = rdnw.unsqueeze(1).unsqueeze(2).repeat(1,nyfull,nxfull)
    rdn_e = rdn.unsqueeze(1).unsqueeze(2).repeat(1,nyfull,nxfull)
    muv_e = muv.repeat(nzall,1,1)
    v_base_e = v_base.unsqueeze(1).unsqueeze(2).repeat(1,nyfull,nxfull)
    
    vflux[kts:ktf-1, j_start:j_end, i_start:i_end] = kvdif * rdn_e[kts+1:ktf, j_start:j_end, i_start:i_end] / \
            (0.25 * (alt[kts:ktf-1, j_start:j_end, i_start:i_end] + alt[kts:ktf-1, j_start-1:j_end-1, i_start:i_end] + 
                     alt[kts+1:ktf, j_start:j_end, i_start:i_end] + alt[kts+1:ktf, j_start-1:j_end-1, i_start:i_end])) * \
            (field[kts+1:ktf, j_start:j_end, i_start:i_end] - field[kts:ktf-1, j_start:j_end, i_start:i_end] - 
             v_base_e[kts+1:ktf, j_start:j_end, i_start:i_end] + v_base_e[kts:ktf-1, j_start:j_end, i_start:i_end])
    vflux[ktf-1, j_start:j_end, i_start:i_end] = 0.
    tendency[kts+1:ktf-1, j_start:j_end, i_start:i_end] = tendency[kts+1:ktf-1, j_start:j_end, i_start:i_end] + g * g * \
            rdnw_e[kts+1:ktf-1, j_start:j_end, i_start:i_end] / muv_e[kts+1:ktf-1, j_start:j_end, i_start:i_end] / (
            0.5 * (alt[kts+1:ktf-1, j_start-1:j_end-1, i_start:i_end] + alt[kts+1:ktf-1, j_start:j_end, i_start:i_end])) * (
            vflux[kts+1:ktf-1, j_start:j_end, i_start:i_end] - vflux[kts:ktf-2, j_start:j_end, i_start:i_end])
    
    return tendency

def vertical_diffusion_mp(field, tendency,               \
                          base, c1, c2,                  \
                          alt, MUT, rdn, rdnw, kvdif,    \
                          ids, ide, jds, jde, kds, kde,  \
                          ims, ime, jms, jme, kms, kme,  \
                          its, ite, jts, jte, kts, kte):
    ktf=min(kte,kde-1)
   
    i_start = its
    i_end   = min(ite,ide-1)
    j_start = jts
    j_end   = min(jte,jde-1)
    
    vflux = torch.zeros((nzall,nyall,nxall)).to(device)
    rdnw_e = rdnw.unsqueeze(1).unsqueeze(2).repeat(1,nyfull,nxfull)
    rdn_e = rdn.unsqueeze(1).unsqueeze(2).repeat(1,nyfull,nxfull)
    mut_e = MUT.repeat(nzall,1,1)
    base_e = base.unsqueeze(1).unsqueeze(2).repeat(1,nyfull,nxfull)
    
    vflux[kts:ktf-1, j_start:j_end, i_start:i_end] = kvdif * rdn_e[kts+1:ktf, j_start:j_end, i_start:i_end] / \
            (0.5 * (alt[kts:ktf-1, j_start:j_end, i_start:i_end] + alt[kts+1:ktf, j_start:j_end, i_start:i_end] )) * \
            (field[kts+1:ktf, j_start:j_end, i_start:i_end] - field[kts:ktf-1, j_start:j_end, i_start:i_end] - 
             base_e[kts+1:ktf, j_start:j_end, i_start:i_end] + base_e[kts:ktf-1, j_start:j_end, i_start:i_end])
    vflux[ktf-1, j_start:j_end, i_start:i_end] = 0.
    tendency[kts+1:ktf, j_start:j_end, i_start:i_end] = tendency[kts+1:ktf, j_start:j_end, i_start:i_end] + g * g * \
            mut_e[kts+1:ktf, j_start:j_end, i_start:i_end] / alt[kts+1:ktf, j_start:j_end, i_start:i_end] * \
            rdnw_e[kts+1:ktf, j_start:j_end, i_start:i_end] * (
            vflux[kts+1:ktf, j_start:j_end, i_start:i_end] - vflux[kts:ktf-1, j_start:j_end, i_start:i_end])
        
    return tendency

# Vertical diffusion for a 3-D field.
def vertical_diffusion_3dmp(field, tendency,               \
                            base_3d, c1, c2,               \
                            alt, MUT, rdn, rdnw, kvdif,    \
                            ids, ide, jds, jde, kds, kde,  \
                            ims, ime, jms, jme, kms, kme,  \
                            its, ite, jts, jte, kts, kte):
    ktf=min(kte,kde-1)
   
    i_start = its
    i_end   = min(ite,ide-1)
    j_start = jts
    j_end   = min(jte,jde-1)
    
    vflux = torch.zeros((nzall,nyall,nxall)).to(device)
    rdnw_e = rdnw.unsqueeze(1).unsqueeze(2).repeat(1,nyfull,nxfull)
    rdn_e = rdn.unsqueeze(1).unsqueeze(2).repeat(1,nyfull,nxfull)
    mut_e = MUT.repeat(nzall,1,1)
        
    vflux[kts:ktf-1, j_start:j_end, i_start:i_end] = kvdif * rdn_e[kts+1:ktf, j_start:j_end, i_start:i_end] / \
            (0.5 * (alt[kts:ktf-1, j_start:j_end, i_start:i_end] + alt[kts+1:ktf, j_start:j_end, i_start:i_end] )) * \
            (field[kts+1:ktf, j_start:j_end, i_start:i_end] - field[kts:ktf-1, j_start:j_end, i_start:i_end] - 
             base_3d[kts+1:ktf, j_start:j_end, i_start:i_end] + base_3d[kts:ktf-1, j_start:j_end, i_start:i_end])
    vflux[ktf-1, j_start:j_end, i_start:i_end] = 0.
    tendency[kts+1:ktf, j_start:j_end, i_start:i_end] = tendency[kts+1:ktf, j_start:j_end, i_start:i_end] + g * g * \
            mut_e[kts+1:ktf, j_start:j_end, i_start:i_end] / alt[kts+1:ktf, j_start:j_end, i_start:i_end] * \
            rdnw_e[kts+1:ktf, j_start:j_end, i_start:i_end] * (
            vflux[kts+1:ktf, j_start:j_end, i_start:i_end] - vflux[kts:ktf-1, j_start:j_end, i_start:i_end])
    
    return tendency
    
def sixth_order_diffusion(name, field, tendency, MUT, dt, \
                        config_flags, c1, c2,           \
                        diff_6th_opt, diff_6th_factor,  \
                        ids, ide, jds, jde, kds, kde,   \
                        ims, ime, jms, jme, kms, kme,   \
                        its, ite, jts, jte, kts, kte):
    diff_6th_coef = diff_6th_factor * 0.015625 / ( 2.0 * dt ) 
    ktf = min( kte, kde-1 )
    
    return

# Merge RK tendencies into the accumulated dry tendencies.
def rk_addtend_dry(ru_tend, rv_tend, rw_tend, ph_tend, t_tend,      \
                   ru_tendf, rv_tendf, rw_tendf, ph_tendf, t_tendf, \
                   u_save, v_save, w_save, ph_save, t_save,         \
                   mu_tend, mu_tendf, rk_step, c1, c2,              \
                   h_diabatic, mut, msftx, msfty, msfux, msfuy,     \
                   msfvx, msfvx_inv, msfvy,                         \
                   ids,ide, jds,jde, kds,kde,                       \
                   ims,ime, jms,jme, kms,kme,                       \
                   ips,ipe, jps,jpe, kps,kpe,                       \
                   its,ite, jts,jte, kts,kte):
    
    msfuy_e = msfuy.repeat(nzall,1,1)
    msfvx_e = msfvx.repeat(nzall,1,1)
    msfty_e = msfty.repeat(nzall,1,1)
    msfvx_inv_e = msfvx_inv.repeat(nzall,1,1)
    
    mut_e = mut.repeat(nzall,1,1)
    
    if rk_step == 1:
        ru_tendf[kts:kte-1, jts:jde-1, its:ite] = ru_tendf[kts:kte-1, jts:jde-1, its:ite] + \
              u_save[kts:kte-1, jts:jde-1, its:ite] * msfuy_e[kts:kte-1, jts:jde-1, its:ite]
    ru_tend[kts:kte-1, jts:jde-1, its:ite] = ru_tend[kts:kte-1, jts:jde-1, its:ite] + \
             ru_tendf[kts:kte-1, jts:jde-1, its:ite] / msfuy_e[kts:kte-1, jts:jde-1, its:ite]
    if rk_step ==1:
        rv_tendf[kts:kte-1, jts:jte, its:ide-1] = rv_tendf[kts:kte-1, jts:jte, its:ide-1] + \
              v_save[kts:kte-1, jts:jte, its:ide-1] * msfvx_e[kts:kte-1, jts:jte, its:ide-1]
    rv_tend[kts:kte-1, jts:jte, its:ide-1] = rv_tend[kts:kte-1, jts:jte, its:ide-1] + \
             rv_tendf[kts:kte-1, jts:jte, its:ide-1] * msfvx_inv_e[kts:kte-1, jts:jte, its:ide-1]
    if rk_step ==1:
        rw_tendf[kts:kte, jts:jde-1, its:ide-1] = rw_tendf[kts:kte, jts:jde-1, its:ide-1] + \
             w_save[kts:kte, jts:jde-1, its:ide-1] * msfty_e[kts:kte, jts:jde-1, its:ide-1]
    rw_tend[kts:kte, jts:jde-1, its:ide-1] = rw_tend[kts:kte, jts:jde-1, its:ide-1] + \
             rw_tendf[kts:kte, jts:jde-1, its:ide-1] / msfty_e[kts:kte, jts:jde-1, its:ide-1]
    if rk_step ==1:
        ph_tendf[kts:kte, jts:jde-1, its:ide-1] = ph_tendf[kts:kte, jts:jde-1, its:ide-1] + \
             ph_save[kts:kte, jts:jde-1, its:ide-1]
    ph_tend[kts:kte, jts:jde-1, its:ide-1] = ph_tend[kts:kte, jts:jde-1, its:ide-1] + \
             ph_tendf[kts:kte, jts:jde-1, its:ide-1] / msfty_e[kts:kte, jts:jde-1, its:ide-1]
    if rk_step ==1:
        t_tendf[kts:kte-1, jts:jde-1, its:ide-1] = t_tendf[kts:kte-1, jts:jde-1, its:ide-1] + \
             t_save[kts:kte-1, jts:jde-1, its:ide-1]
    t_tend[kts:kte-1, jts:jde-1, its:ide-1] = t_tend[kts:kte-1, jts:jde-1, its:ide-1] + \
             t_tendf[kts:kte-1, jts:jde-1, its:ide-1] / msfty_e[kts:kte-1, jts:jde-1, its:ide-1] + \
             mut_e[kts:kte-1, jts:jde-1, its:ide-1] * h_diabatic[kts:kte-1, jts:jde-1, its:ide-1] / \
             msfty_e[kts:kte-1, jts:jde-1, its:ide-1]
    
    mu_tend[jts:jde-1,its:ide-1] = mu_tend[jts:jde-1,its:ide-1] + mu_tendf[jts:jde-1,its:ide-1]
    
    return ru_tend, rv_tend, rw_tend, ph_tend, t_tend, mu_tend


# Rebind state for the small (acoustic) steps.
def small_step_prep(u_1, u_2, v_1, v_2, w_1, w_2, \
                    t_1, t_2, ph_1, ph_2,         \
                    mub, mu_1, mu_2,              \
                    muu, muus, muv, muvs,         \
                    mut, muts, mudf,              \
                    c1h, c2h, c1f, c2f,           \
                    c3h, c4h, c3f, c4f,           \
                    u_save, v_save, w_save,       \
                    t_save, ph_save, mu_save,     \
                    ww, ww_save,                  \
                    c2a, pb, p, alt,              \
                    msfux, msfuy, msfvx,          \
                    msfvx_inv,                    \
                    msfvy, msftx, msfty,          \
                    rdx, rdy,                     \
                    rk_step,                      \
                    ids,ide, jds,jde, kds,kde,    \
                    ims,ime, jms,jme, kms,kme,    \
                    its,ite, jts,jte, kts,kte):
    i_start = its
    i_end   = min(ite,ide-1)
    j_start = jts
    j_end   = min(jte,jde-1)
    k_start = kts
    k_end = min(kte,kde-1)

    i_endu = ite
    j_endv = jte
    
    msfuy_e = msfuy.repeat(nzall,1,1)
    msfty_e = msfty.repeat(nzall,1,1)
    msfvx_inv_e = msfvx_inv.repeat(nzall,1,1)
    u_2_s = u_2.clone()
    v_2_s = v_2.clone()
    w_2_s = w_2.clone()
    t_2_s = t_2.clone()
    ph_2_s = ph_2.clone()

    if rk_step==1:
        mu_1[j_start:j_end, i_start:i_end] = mu_2[j_start:j_end, i_start:i_end] + 0.0
        ww_save[kde-1,j_start:j_end, i_start:i_end] = 0.0
        ww_save[0,j_start:j_end, i_start:i_end] = 0.0
        mudf[j_start:j_end, i_start:i_end] = 0.0
        u_1[k_start:k_end, j_start:j_end, i_start:i_endu] = u_2[k_start:k_end, j_start:j_end, i_start:i_endu] + 0.0
        v_1[k_start:k_end, j_start:j_endv, i_start:i_end] = v_2[k_start:k_end, j_start:j_endv, i_start:i_end] + 0.0
        t_1[k_start:k_end, j_start:j_end, i_start:i_end] = t_2[k_start:k_end, j_start:j_end, i_start:i_end] + 0.0
        w_1[k_start:kde, j_start:j_end, i_start:i_end] = w_2[k_start:kde, j_start:j_end, i_start:i_end] + 0.0
        ph_1[k_start:kde, j_start:j_end, i_start:i_end] = ph_2[k_start:kde, j_start:j_end, i_start:i_end] + 0.0
        muts[j_start:j_end, i_start:i_end] = mub[j_start:j_end, i_start:i_end] + \
                                             mu_2[j_start:j_end, i_start:i_end]
        muus[j_start:j_end, i_start:i_endu] = muu[j_start:j_end, i_start:i_endu] + 0.0
        muvs[j_start:j_endv, i_start:i_end] = muv[j_start:j_endv, i_start:i_end] + 0.0
        mu_save[j_start:j_end, i_start:i_end] = mu_2[j_start:j_end, i_start:i_end] + 0.0
        mu_2[j_start:j_end, i_start:i_end] = 0.0
    else:
        muts[j_start:j_end, i_start:i_end] = mub[j_start:j_end, i_start:i_end] + \
                                             mu_1[j_start:j_end, i_start:i_end]
        muus[j_start:j_end, i_start:i_endu] = 0.5 * (mub[j_start:j_end, i_start:i_endu] + mu_1[j_start:j_end, i_start:i_endu] + 
                                                     mub[j_start:j_end, i_start-1:i_endu-1] + mu_1[j_start:j_end, i_start-1:i_endu-1]) 
        #print("in small step prep",mub[443,604], mub[443,580])
        #print("in small step prep",mu_1[443,604], mu_1[443,580])
        muvs[j_start:j_endv, i_start:i_endu] = 0.5 * (mub[j_start:j_endv, i_start:i_endu] + mu_1[j_start:j_endv, i_start:i_endu] + 
                                                      mub[j_start-1:j_endv-1, i_start:i_endu] + mu_1[j_start-1:j_endv-1, i_start:i_endu])
        mu_save[j_start:j_end, i_start:i_end] = mu_2[j_start:j_end, i_start:i_end] + 0.0
        mu_2[j_start:j_end, i_start:i_end] = mu_1[j_start:j_end, i_start:i_end] - mu_2[j_start:j_end, i_start:i_end]

    muus_e = muus.repeat(nzall,1,1)
    muu_e = muu.repeat(nzall,1,1)
    muvs_e = muvs.repeat(nzall,1,1)
    muv_e = muv.repeat(nzall,1,1)
    muts_e = muts.repeat(nzall,1,1)
    mut_e = mut.repeat(nzall,1,1)
    
    ww_save[kde-1, j_start:j_end, i_start:i_end] = 0.0
    ww_save[0, j_start:j_end, i_start:i_end] = 0.0
    c2a[k_start:k_end, j_start:j_end, i_start:i_end] = cpovcv * (pb[k_start:k_end, j_start:j_end, i_start:i_end] + 
                                                                 p[k_start:k_end, j_start:j_end, i_start:i_end]) / \
                                                                alt[k_start:k_end, j_start:j_end, i_start:i_end]
    u_save[k_start:k_end, j_start:j_end, i_start:i_endu] = u_2[k_start:k_end, j_start:j_end, i_start:i_endu] + 0.0
    #print("in small step prep",u_2[20,443,604], u_2[20,443,580])
    #print("in small step prep",u_1[20,443,604], u_1[20,443,580])
    #print("in small step prep",muus_e[20,443,604], muus_e[20,443,580])
    #print("in small step prep",muu_e[20,443,604], muu_e[20,443,580])
    u_2_s[k_start:k_end, j_start:j_end, i_start:i_endu] = (muus_e[k_start:k_end, j_start:j_end, i_start:i_endu] * 
                                                         u_1[k_start:k_end, j_start:j_end, i_start:i_endu] -
                                                         muu_e[k_start:k_end, j_start:j_end, i_start:i_endu] *
                                                         u_2[k_start:k_end, j_start:j_end, i_start:i_endu]) / \
                                                         msfuy_e[k_start:k_end, j_start:j_end, i_start:i_endu]
    v_save[k_start:k_end, j_start:j_endv, i_start:i_end] = v_2[k_start:k_end, j_start:j_endv, i_start:i_end] + 0.0
    v_2_s[k_start:k_end, j_start:j_endv, i_start:i_end] = (muvs_e[k_start:k_end, j_start:j_endv, i_start:i_end] * 
                                                         v_1[k_start:k_end, j_start:j_endv, i_start:i_end] -
                                                         muv_e[k_start:k_end, j_start:j_endv, i_start:i_end] *
                                                         v_2[k_start:k_end, j_start:j_endv, i_start:i_end])  * \
                                                         msfvx_inv_e[k_start:k_end, j_start:j_endv, i_start:i_end]
    #print("11 in small prep v: ", v_2[20, 3:9, 480])
    t_save[k_start:k_end, j_start:j_end, i_start:i_end] = t_2[k_start:k_end, j_start:j_end, i_start:i_end] + 0.0
    #print("in small step prep", t_2[24:26,443,580],t_save[24:26,443,580], t_1[24:26,443,580],muts_e[24:26,443,580],mut_e[24:26,443,580])
    t_2_s[k_start:k_end, j_start:j_end, i_start:i_end] = (muts_e[k_start:k_end, j_start:j_end, i_start:i_end] * 
                                                        t_1[k_start:k_end, j_start:j_end, i_start:i_end] -
                                                        mut_e[k_start:k_end, j_start:j_end, i_start:i_end] *
                                                        t_2[k_start:k_end, j_start:j_end, i_start:i_end])
    #print("in small step prep", t_2[24:26,443,580],t_save[24:26,443,580], t_1[24:26,443,580],muts_e[24:26,443,580],mut_e[24:26,443,580])
    w_save[k_start:kde, j_start:j_end, i_start:i_end] = w_2[k_start:kde, j_start:j_end, i_start:i_end] + 0.0
    w_2_s[k_start:kde, j_start:j_end, i_start:i_end] = (muts_e[k_start:kde, j_start:j_end, i_start:i_end] * 
                                                      w_1[k_start:kde, j_start:j_end, i_start:i_end] - 
                                                      mut_e[k_start:kde, j_start:j_end, i_start:i_end] * 
                                                      w_2[k_start:kde, j_start:j_end, i_start:i_end]) / \
                                                      msfty_e[k_start:kde, j_start:j_end, i_start:i_end]
    ph_save[k_start:kde, j_start:j_end, i_start:i_end] = ph_2[k_start:kde, j_start:j_end, i_start:i_end] + 0.0
    ph_2_s[k_start:kde, j_start:j_end, i_start:i_end] = ph_1[k_start:kde, j_start:j_end, i_start:i_end] - \
                                                      ph_2[k_start:kde, j_start:j_end, i_start:i_end]
    ww_save[k_start:kde, j_start:j_end, i_start:i_end] = ww[k_start:kde, j_start:j_end, i_start:i_end]

    return u_1, v_1, w_1, t_1, ph_1, u_save, v_save, w_save, t_save, ph_save, u_2_s, v_2_s, w_2_s, t_2_s, ph_2_s, \
           c2a, ww_save, mu_1, mu_2, muus, muvs, muts, mudf, mu_save
          
# Diagnose pressure and density.
def calc_p_rho(al_ori, p_ori, ph,                    \
               alt, t_2, t_1, c2a, pm1,      \
               mu, mut,                      \
               c1h, c2h, c1f, c2f,           \
               c3h, c4h, c3f, c4f,           \
               znu, t0,                      \
               rdnw, dnw, smdiv,             \
               non_hydrostatic, step,        \
               ids, ide, jds, jde, kds, kde, \
               ims, ime, jms, jme, kms, kme, \
               its,ite, jts,jte, kts,kte):
    i_start = its
    i_end   = min(ite,ide-1)
    j_start = jts
    j_end   = min(jte,jde-1)
    k_start = kts
    k_end = min(kte,kde-1)
    
    mu_e = mu.repeat(nzall,1,1)
    mut_e = mut.repeat(nzall,1,1)
    rdnw_e = rdnw.unsqueeze(1).unsqueeze(2).repeat(1,nyfull,nxfull)
    
    al = al_ori.clone()
    p = p_ori.clone()
    
    ptmp = torch.zeros((nzall,nyall,nxall)).to(device)
    
    al[k_start:k_end, j_start:j_end, i_start:i_end] = -1. / mut_e[k_start:k_end, j_start:j_end, i_start:i_end] * \
              (alt[k_start:k_end, j_start:j_end, i_start:i_end] * mu_e[k_start:k_end, j_start:j_end, i_start:i_end] + 
               rdnw_e[k_start:k_end, j_start:j_end, i_start:i_end] * (ph[k_start+1:k_end+1, j_start:j_end, i_start:i_end] - 
                                                                      ph[k_start:k_end, j_start:j_end, i_start:i_end]))
    p[k_start:k_end, j_start:j_end, i_start:i_end] = c2a[k_start:k_end, j_start:j_end, i_start:i_end] * \
              (alt[k_start:k_end, j_start:j_end, i_start:i_end] * (t_2[k_start:k_end, j_start:j_end, i_start:i_end] - 
               mu_e[k_start:k_end, j_start:j_end, i_start:i_end] * t_1[k_start:k_end, j_start:j_end, i_start:i_end]) / \
               (mut_e[k_start:k_end, j_start:j_end, i_start:i_end] * (t0 + t_1[k_start:k_end, j_start:j_end, i_start:i_end])) - 
               al[k_start:k_end, j_start:j_end, i_start:i_end])
    if step == 0:
        pm1 = p + 0.0
    else:
        ptmp[k_start:k_end, j_start:j_end, i_start:i_end] = p[k_start:k_end, j_start:j_end, i_start:i_end] + 0.0
        p[k_start:k_end, j_start:j_end, i_start:i_end] = p[k_start:k_end, j_start:j_end, i_start:i_end] + smdiv * (
                                                         p[k_start:k_end, j_start:j_end, i_start:i_end] - 
                                                         pm1[k_start:k_end, j_start:j_end, i_start:i_end])
        pm1[k_start:k_end, j_start:j_end, i_start:i_end] = ptmp[k_start:k_end, j_start:j_end, i_start:i_end] + 0.0
    return al, p, pm1

# Coefficients a / alpha / gamma for the w equation.
def calc_coef_w(a,alpha,gamma,              \
                mut,                        \
                c1h, c2h, c1f, c2f,         \
                c3h, c4h, c3f, c4f,         \
                cqw,                        \
                rdn, rdnw, c2a,             \
                dts, g, epssm, top_lid,     \
                ids,ide, jds,jde, kds,kde,  \
                ims,ime, jms,jme, kms,kme,  \
                its,ite, jts,jte, kts,kte):
    
    i_start = its
    i_end   = min(ite,ide-1)
    j_start = jts
    j_end   = min(jte,jde-1)
    k_start = kts
    k_end   = kte-1
    
    mut_e = mut.repeat(nzall,1,1)
    rdnw_e = rdnw.unsqueeze(1).unsqueeze(2).repeat(1,nyfull,nxfull)
    rdn_e = rdn.unsqueeze(1).unsqueeze(2).repeat(1,nyfull,nxfull)

    
    b = torch.zeros((nzall,nyall,nxall)).to(device)
    c = torch.zeros((nzall,nyall,nxall)).to(device)
        
    lid_flag = 1
    if top_lid:
        lid_flag = 0
    
    a[1, j_start:j_end, i_start:i_end] = 0.0
    a[kde-1, j_start:j_end, i_start:i_end] = -2. * ((0.5 * dts * g * (1. + epssm)) ** 2)  * \
             rdnw_e[kde-2, j_start:j_end, i_start:i_end] ** 2 * c2a[kde-2, j_start:j_end, i_start:i_end] * \
             lid_flag / ( mut_e[kde-1, j_start:j_end, i_start:i_end] * mut_e[kde-1, j_start:j_end, i_start:i_end])
             #(mut_e[kde-1, j_start:j_end, i_start:i_end] * mut_e[kde-1, j_start:j_end, i_start:i_end])
    gamma[0, j_start:j_end, i_start:i_end] = 0.0
    
    a[2:kde-1, j_start:j_end, i_start:i_end] = -cqw[2:kde-1, j_start:j_end, i_start:i_end] * \
             ((0.5 * dts * g * (1. + epssm)) ** 2) * rdn_e[2:kde-1, j_start:j_end, i_start:i_end] * \
             rdnw_e[1:kde-2, j_start:j_end, i_start:i_end] * c2a[1:kde-2, j_start:j_end, i_start:i_end] / \
             ( mut_e[2:kde-1, j_start:j_end, i_start:i_end] * mut_e[2:kde-1, j_start:j_end, i_start:i_end] )
             #(mut_e[2:kde-1, j_start:j_end, i_start:i_end] * mut_e[2:kde-1, j_start:j_end, i_start:i_end])
    b[1:kde-1, j_start:j_end, i_start:i_end] = 1. + cqw[1:kde-1, j_start:j_end, i_start:i_end] * \
             ((0.5 * dts * g * (1. + epssm)) ** 2) * rdn_e[1:kde-1, j_start:j_end, i_start:i_end] * (
             rdnw_e[1:kde-1, j_start:j_end, i_start:i_end] * c2a[1:kde-1, j_start:j_end, i_start:i_end] / \
             ( mut_e[1:kde-1, j_start:j_end, i_start:i_end] * mut_e[1:kde-1, j_start:j_end, i_start:i_end] ) +
             #(mut_e[1:kde-1, j_start:j_end, i_start:i_end] * mut_e[1:kde-1, j_start:j_end, i_start:i_end]) + \
             rdnw_e[0:kde-2, j_start:j_end, i_start:i_end] * c2a[0:kde-2, j_start:j_end, i_start:i_end] / \
             ( mut_e[1:kde-1, j_start:j_end, i_start:i_end] * mut_e[1:kde-1, j_start:j_end, i_start:i_end] ))
             #(mut_e[1:kde-1, j_start:j_end, i_start:i_end] * mut_e[1:kde-1, j_start:j_end, i_start:i_end]))
    c[1:kde-1, j_start:j_end, i_start:i_end] = -cqw[1:kde-1, j_start:j_end, i_start:i_end] * \
             ((0.5 * dts * g * (1. + epssm)) ** 2) * rdn_e[1:kde-1, j_start:j_end, i_start:i_end] * \
             rdnw_e[1:kde-1, j_start:j_end, i_start:i_end] * c2a[1:kde-1, j_start:j_end, i_start:i_end] / \
             ( mut_e[1:kde-1, j_start:j_end, i_start:i_end] * mut_e[1:kde-1, j_start:j_end, i_start:i_end] )
             #(mut_e[1:kde-1, j_start:j_end, i_start:i_end] * mut_e[1:kde-1, j_start:j_end, i_start:i_end])
    for k in range(1,kde-1):
        alpha[k, j_start:j_end, i_start:i_end] = 1. / (b[k, j_start:j_end, i_start:i_end] - 
                                                          a[k, j_start:j_end, i_start:i_end] * 
                                                          gamma[k-1, j_start:j_end, i_start:i_end])
        gamma[k, j_start:j_end, i_start:i_end] = c[k, j_start:j_end, i_start:i_end] * \
                                                    alpha[k, j_start:j_end, i_start:i_end]
    
    b[kde-1, j_start:j_end, i_start:i_end] = 1. + 2. * \
             ((0.5 * dts * g * (1. + epssm)) ** 2) * (rdnw_e[kde-2, j_start:j_end, i_start:i_end] ** 2) * \
             c2a[kde-2, j_start:j_end, i_start:i_end] / \
             ( mut_e[kde-1, j_start:j_end, i_start:i_end] * mut_e[kde-1, j_start:j_end, i_start:i_end] )
             #(mut_e[kde-1, j_start:j_end, i_start:i_end] * mut_e[kde-1, j_start:j_end, i_start:i_end])
    alpha[kde-1, j_start:j_end, i_start:i_end] = 1. / (b[kde-1, j_start:j_end, i_start:i_end] - 
                                                       a[kde-1, j_start:j_end, i_start:i_end] * 
                                                       gamma[kde-2, j_start:j_end, i_start:i_end])
    gamma[kde-1, j_start:j_end, i_start:i_end] = 0.0
       
    return a, alpha, gamma

# Advance u / v one small step.
def advance_uv(u_ori, ru_tend, v_ori, rv_tend,        \
               p, pb,                         \
               ph, php, alt, al, mu,          \
               muu, cqu, muv, cqv, mudf,      \
               c1h, c2h, c1f, c2f,            \
               c3h, c4h, c3f, c4f,            \
               msfux, msfuy, msfvx,           \
               msfvx_inv, msfvy,              \
               rdx, rdy, dts,                 \
               cf1, cf2, cf3, fnm, fnp,       \
               emdiv,                         \
               rdnw, spec_zone,               \
               non_hydrostatic, top_lid,      \
               ids, ide, jds, jde, kds, kde,  \
               ims, ime, jms, jme, kms, kme,  \
               its, ite, jts, jte, kts, kte):
    i_start = max( its,ids+spec_zone )
    i_end   = min( ite,ide-spec_zone-1 )
    j_start = max( jts,jds+spec_zone )
    j_end   = min( jte,jde-spec_zone-1 )
    k_start = kts
    k_end   = min( kte, kde-1 )

    i_endu = min( ite,ide-spec_zone )
    j_endv = min( jte,jde-spec_zone )
    k_endw = k_end
    
    i_start_up = i_start
    i_end_up   = i_endu
    j_start_up = j_start
    j_end_up   = j_end
    
    i_start_vp = i_start
    i_end_vp   = i_end
    j_start_vp = j_start
    j_end_vp   = j_endv
    
    i_start_u_tend = i_start
    i_end_u_tend   = i_endu
    j_start_v_tend = j_start
    j_end_v_tend   = j_endv
    
    dx = 1./rdx
    dy = 1./rdy
    
    msfux_e = msfux.repeat(nzall,1,1)
    msfuy_e = msfuy.repeat(nzall,1,1)
    msfvx_e = msfvx.repeat(nzall,1,1)
    msfvy_e = msfvy.repeat(nzall,1,1)
    
    muu_e = muu.repeat(nzall,1,1)
    muv_e = muv.repeat(nzall,1,1)
    mu_e = mu.repeat(nzall,1,1)
    
    fnm_e = fnm.unsqueeze(1).unsqueeze(2).repeat(1,nyfull,nxfull)
    fnp_e = fnp.unsqueeze(1).unsqueeze(2).repeat(1,nyfull,nxfull)
    rdnw_e = rdnw.unsqueeze(1).unsqueeze(2).repeat(1,nyfull,nxfull)
    
    mudf_xy = torch.zeros((nyall,nxall)).to(device)
    dpxy = torch.zeros((nzall,nyall,nxall)).to(device)
    dpn = torch.zeros((nzall,nyall,nxall)).to(device)
    
    u = u_ori.clone()
    v = v_ori.clone()
    
    # u
    u[k_start:k_end, j_start:j_end, i_start_u_tend:i_end_u_tend] = u[k_start:k_end, j_start:j_end, i_start_u_tend:i_end_u_tend] + \
             dts * ru_tend[k_start:k_end, j_start:j_end, i_start_u_tend:i_end_u_tend]
    mudf_xy[j_start:j_end, i_start_up:i_end_up] = -emdiv * dx * (mudf[j_start:j_end, i_start_up:i_end_up] - 
             mudf[j_start:j_end, i_start_up-1:i_end_up-1] ) / msfuy[j_start:j_end, i_start_up:i_end_up]
    dpxy[k_start:k_end, j_start:j_end, i_start_up:i_end_up] = (msfux_e[k_start:k_end, j_start:j_end, i_start_up:i_end_up] / 
             msfuy_e[k_start:k_end, j_start:j_end, i_start_up:i_end_up]) * 0.5 * rdx * muu_e[k_start:k_end, j_start:j_end, i_start_up:i_end_up] * \
             (((ph[k_start+1:k_end+1, j_start:j_end, i_start_up:i_end_up] - ph[k_start+1:k_end+1, j_start:j_end, i_start_up-1:i_end_up-1]) +
               (ph[k_start:k_end, j_start:j_end, i_start_up:i_end_up] - ph[k_start:k_end, j_start:j_end, i_start_up-1:i_end_up-1])) + 
               (alt[k_start:k_end, j_start:j_end, i_start_up:i_end_up] + alt[k_start:k_end, j_start:j_end, i_start_up-1:i_end_up-1]) * 
               (p[k_start:k_end, j_start:j_end, i_start_up:i_end_up] - p[k_start:k_end, j_start:j_end, i_start_up-1:i_end_up-1]) + 
               (al[k_start:k_end, j_start:j_end, i_start_up:i_end_up] + al[k_start:k_end, j_start:j_end, i_start_up-1:i_end_up-1]) * 
               (pb[k_start:k_end, j_start:j_end, i_start_up:i_end_up] - pb[k_start:k_end, j_start:j_end, i_start_up-1:i_end_up-1]))

    dpn[0, j_start:j_end, i_start_up:i_end_up] = 0.5 * (cf1 * (p[0, j_start:j_end, i_start_up:i_end_up] + p[0, j_start:j_end, i_start_up-1:i_end_up-1]) +
                                                        cf2 * (p[1, j_start:j_end, i_start_up:i_end_up] + p[1, j_start:j_end, i_start_up-1:i_end_up-1]) +
                                                        cf3 * (p[2, j_start:j_end, i_start_up:i_end_up] + p[2, j_start:j_end, i_start_up-1:i_end_up-1]))
    dpn[kde-1, j_start:j_end, i_start_up:i_end_up] = 0.
    dpn[k_start+1:k_end, j_start:j_end, i_start_up:i_end_up] = 0.5 * (fnm_e[k_start+1:k_end, j_start:j_end, i_start_up:i_end_up] * 
               (p[k_start+1:k_end, j_start:j_end, i_start_up:i_end_up] + p[k_start+1:k_end, j_start:j_end, i_start_up-1:i_end_up-1]) + 
               fnp_e[k_start+1:k_end, j_start:j_end, i_start_up:i_end_up] * 
               (p[k_start:k_end-1, j_start:j_end, i_start_up:i_end_up] + p[k_start:k_end-1, j_start:j_end, i_start_up-1:i_end_up-1]))

    dpxy[k_start:k_end, j_start:j_end, i_start_up:i_end_up] = dpxy[k_start:k_end, j_start:j_end, i_start_up:i_end_up] + \
               (msfux_e[k_start:k_end, j_start:j_end, i_start_up:i_end_up] / msfuy_e[k_start:k_end, j_start:j_end, i_start_up:i_end_up]) * \
               rdx * (php[k_start:k_end, j_start:j_end, i_start_up:i_end_up] - php[k_start:k_end, j_start:j_end, i_start_up-1:i_end_up-1]) * \
               (rdnw_e[k_start:k_end, j_start:j_end, i_start_up:i_end_up] * 
                (dpn[k_start+1:k_end+1, j_start:j_end, i_start_up:i_end_up] - dpn[k_start:k_end, j_start:j_end, i_start_up:i_end_up]) - 0.5 * 
                (mu_e[k_start:k_end, j_start:j_end, i_start_up-1:i_end_up-1] + mu_e[k_start:k_end, j_start:j_end, i_start_up:i_end_up]))
    
    mudf_xy_e = mudf_xy.repeat(nzall,1,1)
    u[k_start:k_end, j_start:j_end, i_start_up:i_end_up] = u[k_start:k_end, j_start:j_end, i_start_up:i_end_up] - \
               dts * cqu[k_start:k_end, j_start:j_end, i_start_up:i_end_up] * dpxy[k_start:k_end, j_start:j_end, i_start_up:i_end_up] + \
               mudf_xy_e[k_start:k_end, j_start:j_end, i_start_up:i_end_up]

    # v           
    v[k_start:k_end, j_start_v_tend:j_end_v_tend, i_start:i_end] = v[k_start:k_end, j_start_v_tend:j_end_v_tend, i_start:i_end] + \
               dts * rv_tend[k_start:k_end, j_start_v_tend:j_end_v_tend, i_start:i_end]
    
    mudf_xy[j_start_v_tend:j_end_v_tend, i_start:i_end] = -emdiv * dy * (mudf[j_start_v_tend:j_end_v_tend, i_start:i_end] - 
               mudf[j_start_v_tend-1:j_end_v_tend-1, i_start:i_end]) * msfvx_inv[j_start_v_tend:j_end_v_tend, i_start:i_end]
    dpxy[k_start:k_end, j_start_vp:j_end_vp, i_start:i_end] = (msfvy_e[k_start:k_end, j_start_vp:j_end_vp, i_start:i_end] / 
             msfvx_e[k_start:k_end, j_start_vp:j_end_vp, i_start:i_end]) * 0.5 * rdy * muv_e[k_start:k_end, j_start_vp:j_end_vp, i_start:i_end] * \
             (((ph[k_start+1:k_end+1, j_start_vp:j_end_vp, i_start:i_end] - ph[k_start+1:k_end+1, j_start_vp-1:j_end_vp-1, i_start:i_end]) + 
               (ph[k_start:k_end, j_start_vp:j_end_vp, i_start:i_end] - ph[k_start:k_end, j_start_vp-1:j_end_vp-1, i_start:i_end])) + 
               (alt[k_start:k_end, j_start_vp:j_end_vp, i_start:i_end] +alt[k_start:k_end, j_start_vp-1:j_end_vp-1, i_start:i_end]) * 
               (p[k_start:k_end, j_start_vp:j_end_vp, i_start:i_end] - p[k_start:k_end, j_start_vp-1:j_end_vp-1, i_start:i_end]) + 
               (al[k_start:k_end, j_start_vp:j_end_vp, i_start:i_end] + al[k_start:k_end, j_start_vp-1:j_end_vp-1, i_start:i_end]) * 
               (pb[k_start:k_end, j_start_vp:j_end_vp, i_start:i_end] -pb[k_start:k_end, j_start_vp-1:j_end_vp-1, i_start:i_end]))

    dpn[0, j_start_vp:j_end_vp, i_start:i_end] = 0.5 * (cf1 * (p[0, j_start_vp:j_end_vp, i_start:i_end] + p[0, j_start_vp-1:j_end_vp-1, i_start:i_end]) +
                                                        cf2 * (p[1, j_start_vp:j_end_vp, i_start:i_end] + p[1, j_start_vp-1:j_end_vp-1, i_start:i_end]) +
                                                        cf3 * (p[2, j_start_vp:j_end_vp, i_start:i_end] + p[2, j_start_vp-1:j_end_vp-1, i_start:i_end]))
    dpn[kde-1, j_start_vp:j_end_vp, i_start:i_end] = 0.
    dpn[k_start+1:k_end, j_start_vp:j_end_vp, i_start:i_end] = 0.5 * (fnm_e[k_start+1:k_end, j_start_vp:j_end_vp, i_start:i_end] * 
               (p[k_start+1:k_end, j_start_vp:j_end_vp, i_start:i_end] + p[k_start+1:k_end, j_start_vp-1:j_end_vp-1, i_start:i_end]) + 
               fnp_e[k_start+1:k_end, j_start_vp:j_end_vp, i_start:i_end] * 
               (p[k_start:k_end-1, j_start_vp:j_end_vp, i_start:i_end] + p[k_start:k_end-1, j_start_vp-1:j_end_vp-1, i_start:i_end]))
    dpxy[k_start:k_end, j_start_vp:j_end_vp, i_start:i_end] = dpxy[k_start:k_end, j_start_vp:j_end_vp, i_start:i_end] + \
               (msfvy_e[k_start:k_end, j_start_vp:j_end_vp, i_start:i_end] / msfvx_e[k_start:k_end, j_start_vp:j_end_vp, i_start:i_end]) * \
               rdy * (php[k_start:k_end, j_start_vp:j_end_vp, i_start:i_end] - php[k_start:k_end, j_start_vp-1:j_end_vp-1, i_start:i_end]) * \
               (rdnw_e[k_start:k_end, j_start_vp:j_end_vp, i_start:i_end] * 
                (dpn[k_start+1:k_end+1, j_start_vp:j_end_vp, i_start:i_end] - dpn[k_start:k_end, j_start_vp:j_end_vp, i_start:i_end]) - 0.5 * 
                (mu_e[k_start:k_end, j_start_vp-1:j_end_vp-1, i_start:i_end] + mu_e[k_start:k_end, j_start_vp:j_end_vp, i_start:i_end]))
    mudf_xy_e = mudf_xy.repeat(nzall,1,1)
    v[k_start:k_end, j_start_vp:j_end_vp, i_start:i_end] = v[k_start:k_end, j_start_vp:j_end_vp, i_start:i_end] - \
               dts * cqv[k_start:k_end, j_start_vp:j_end_vp, i_start:i_end] * dpxy[k_start:k_end, j_start_vp:j_end_vp, i_start:i_end] + \
               mudf_xy_e[k_start:k_end, j_start_vp:j_end_vp, i_start:i_end]
        
    return u,v

# Advance mu and theta one small step.
def advance_mu_t(ww_ori, ww_1, u, u_1, v, v_1,            \
                 mu_ori, mut, muave_ori, muts_ori, muu, muv, mudf_ori,\
                 c1h, c2h, c1f, c2f,                  \
                 c3h, c4h, c3f, c4f,                  \
                 uam, vam, wwam, t_ori, t_1,              \
                 t_ave_ori, ft, mu_tend,                  \
                 rdx, rdy, dts, epssm,                \
                 dnw, fnm, fnp, rdnw,                 \
                 msfux, msfuy, msfvx, msfvx_inv,      \
                 msfvy, msftx, msfty,                 \
                 step,                                \
                 ids, ide, jds, jde, kds, kde,        \
                 ims, ime, jms, jme, kms, kme,        \
                 its, ite, jts, jte, kts, kte):
    i_start = its 
    i_end   = min(ite,ide-1)
    j_start = jts
    j_end   = min(jte,jde-1)
    k_start = kts
    k_end   = kte-1
    
    i_start = max(its,ids+1)
    i_end   = min(ite,ide-2)
    j_start = max(jts,jds+1)
    j_end   = min(jte,jde-2)
    
    i_endu = ite  
    j_endv = jte
    
    msfuy_e = msfuy.repeat(nzall,1,1)
    msftx_e = msftx.repeat(nzall,1,1)
    msfty_e = msfty.repeat(nzall,1,1)
    msfvx_inv_e = msfvx_inv.repeat(nzall,1,1)
    
    muu_e = muu.repeat(nzall,1,1)
    muv_e = muv.repeat(nzall,1,1)
    
    dnw_e = dnw.unsqueeze(1).unsqueeze(2).repeat(1,nyfull,nxfull)
    
    fnm_e = fnm.unsqueeze(1).unsqueeze(2).repeat(1,nyfull,nxfull)
    fnp_e = fnp.unsqueeze(1).unsqueeze(2).repeat(1,nyfull,nxfull)
    rdnw_e = rdnw.unsqueeze(1).unsqueeze(2).repeat(1,nyfull,nxfull)
    
    dvdxi = torch.zeros((nzall,nyall,nxall)).to(device)
    dmdt_k = torch.zeros((nzall,nyall,nxall)).to(device)
    wdtn = torch.zeros((nzall,nyall,nxall)).to(device)
    
    ww = ww_ori.clone()
    t = t_ori.clone()
    t_ave = t_ave_ori.clone()
    muave = muave_ori.clone()
    muts = muts_ori.clone()
    mudf = mudf_ori.clone()
    mu = mu_ori.clone()
        
    dvdxi[k_start:k_end, j_start:j_end, i_start:i_end] = msftx_e[k_start:k_end, j_start:j_end, i_start:i_end] * \
            msfty_e[k_start:k_end, j_start:j_end, i_start:i_end] * (rdy * ((v[k_start:k_end, j_start+1:j_end+1, i_start:i_end] + 
            muv_e[k_start:k_end, j_start+1:j_end+1, i_start:i_end] * v_1[k_start:k_end, j_start+1:j_end+1, i_start:i_end] * 
            msfvx_inv_e[k_start:k_end, j_start+1:j_end+1, i_start:i_end]) - (v[k_start:k_end, j_start:j_end, i_start:i_end] + 
            muv_e[k_start:k_end, j_start:j_end, i_start:i_end] * v_1[k_start:k_end, j_start:j_end, i_start:i_end] * 
            msfvx_inv_e[k_start:k_end, j_start:j_end, i_start:i_end])) + rdx * ((u[k_start:k_end, j_start:j_end, i_start+1:i_end+1] + 
            muu_e[k_start:k_end, j_start:j_end, i_start+1:i_end+1] * u_1[k_start:k_end, j_start:j_end, i_start+1:i_end+1] / 
            msfuy_e[k_start:k_end, j_start:j_end, i_start+1:i_end+1]    ) - (u[k_start:k_end, j_start:j_end, i_start:i_end] + 
            muu_e[k_start:k_end, j_start:j_end, i_start:i_end] * u_1[k_start:k_end, j_start:j_end, i_start:i_end] / 
            msfuy_e[k_start:k_end, j_start:j_end, i_start:i_end])))
    
    dmdt_k[k_start:k_end, j_start:j_end, i_start:i_end] = dnw_e[k_start:k_end, j_start:j_end, i_start:i_end] * \
            dvdxi[k_start:k_end, j_start:j_end, i_start:i_end]
    dmdt = dmdt_k.sum(dim=0)
    with torch.no_grad():
        muave[j_start:j_end, i_start:i_end] = mu[j_start:j_end, i_start:i_end] + 0.0
        mu[j_start:j_end, i_start:i_end] = mu[j_start:j_end, i_start:i_end] + dts * (dmdt[j_start:j_end, i_start:i_end] + \
            mu_tend[j_start:j_end, i_start:i_end])
        mudf[j_start:j_end, i_start:i_end] = (dmdt[j_start:j_end, i_start:i_end] + mu_tend[j_start:j_end, i_start:i_end])
        muts[j_start:j_end, i_start:i_end] = mut[j_start:j_end, i_start:i_end] + mu[j_start:j_end, i_start:i_end]
        muave[j_start:j_end, i_start:i_end] = 0.5 * ((1. + epssm) * mu[j_start:j_end, i_start:i_end] + 
                                                 (1. - epssm) * muave[j_start:j_end, i_start:i_end])
    
    for k in range(1,k_end):
        ww[k, j_start:j_end, i_start:i_end] = ww[k-1, j_start:j_end, i_start:i_end] - \
            dnw_e[k-1, j_start:j_end, i_start:i_end] * (dmdt[j_start:j_end, i_start:i_end] + 
            dvdxi[k-1, j_start:j_end, i_start:i_end] + mu_tend[j_start:j_end, i_start:i_end]) / \
            msfty_e[k-1, j_start:j_end, i_start:i_end]
    ww[0:k_end, j_start:j_end, i_start:i_end] = ww[0:k_end, j_start:j_end, i_start:i_end] - \
                                                ww_1[0:k_end, j_start:j_end, i_start:i_end]   # ww_1: large timestep ww
    
    # t
    t_ave[0:k_end, j_start:j_end, i_start:i_end] = t[0:k_end, j_start:j_end, i_start:i_end] + 0.0

    t[0:k_end, j_start:j_end, i_start:i_end] = t[0:k_end, j_start:j_end, i_start:i_end] + \
            msfty_e[0:k_end, j_start:j_end, i_start:i_end] * dts * ft[0:k_end, j_start:j_end, i_start:i_end]
    wdtn[0, j_start:j_end, i_start:i_end] = 0.
    wdtn[kde-1, j_start:j_end, i_start:i_end] = 0.
    wdtn[1:k_end, j_start:j_end, i_start:i_end] = ww[1:k_end, j_start:j_end, i_start:i_end] * \
            (fnm_e[1:k_end, j_start:j_end, i_start:i_end] * t_1[1:k_end, j_start:j_end, i_start:i_end] + 
             fnp_e[1:k_end, j_start:j_end, i_start:i_end] * t_1[0:k_end-1, j_start:j_end, i_start:i_end])
    
    t[0:k_end, j_start:j_end, i_start:i_end] = t[0:k_end, j_start:j_end, i_start:i_end] - dts * \
            msfty_e[0:k_end, j_start:j_end, i_start:i_end] * (msftx_e[0:k_end, j_start:j_end, i_start:i_end] * 
            (0.5 * rdy * (v[0:k_end, j_start+1:j_end+1, i_start:i_end] * 
            (t_1[0:k_end, j_start+1:j_end+1, i_start:i_end] + t_1[0:k_end, j_start:j_end, i_start:i_end]) - 
            v[0:k_end, j_start:j_end, i_start:i_end] * 
            (t_1[0:k_end, j_start:j_end, i_start:i_end] + t_1[0:k_end, j_start-1:j_end-1, i_start:i_end])) + 
            0.5 * rdx * (u[0:k_end, j_start:j_end, i_start+1:i_end+1] * 
            (t_1[0:k_end, j_start:j_end, i_start+1:i_end+1] + t_1[0:k_end, j_start:j_end, i_start:i_end]) - 
            u[0:k_end, j_start:j_end, i_start:i_end] * 
            (t_1[0:k_end, j_start:j_end, i_start:i_end] + t_1[0:k_end, j_start:j_end, i_start-1:i_end-1]))) +
            rdnw_e[0:k_end, j_start:j_end, i_start:i_end] * 
            (wdtn[1:k_end+1, j_start:j_end, i_start:i_end] - wdtn[0:k_end, j_start:j_end, i_start:i_end]))
    
    return ww,ww_1,t,t_ave,uam,vam,wwam,muave,muts,mudf,mu

# Advance w one small step.
def advance_w(w_ori, rw_tend, ww, w_save, u, v,  \
              mu1, mut, muave, muts,      \
              c1h, c2h, c1f, c2f,         \
              c3h, c4h, c3f, c4f,         \
              t_2ave_ori, t_2, t_1,           \
              ph_ori, ph_1, phb, ph_tend,     \
              ht, c2a, cqw, alt, alb,     \
              a, alpha, gamma,            \
              rdx, rdy, dts, t0, epssm,   \
              dnw, fnm, fnp, rdnw, rdn,   \
              cf1, cf2, cf3, msftx, msfty,\
              top_lid,                    \
              ids,ide, jds,jde, kds,kde,  \
              ims,ime, jms,jme, kms,kme,  \
              its,ite, jts,jte, kts,kte):
    i_start = its
    i_end   = min(ite,ide-1)
    j_start = jts
    j_end   = min(jte,jde-1)
    k_start = kts
    k_end   = kte-1
    
    i_start = max(its,ids+1)
    i_end   = min(ite,ide-2)
    
    j_start = max(jts,jds+1)
    j_end   = min(jte,jde-2)
    
    pi = 3.14159265359
    dampmag = dts* 0.2 # config_flags%dampcoef
    hdepth=5000.  # config_flags%zdamp
    
    t_2ave = t_2ave_ori.clone()
    w = w_ori.clone()
    ph = ph_ori.clone()
    
    msfty_e = msfty.repeat(nzall,1,1)
    rdnw_e = rdnw.unsqueeze(1).unsqueeze(2).repeat(1,nyfull,nxfull)
    
    muave_e = muave.repeat(nzall,1,1)
    muts_e = muts.repeat(nzall,1,1)
    mut_e = mut.repeat(nzall,1,1)
    
    fnm_e = fnm.unsqueeze(1).unsqueeze(2).repeat(1,nyfull,nxfull)
    fnp_e = fnp.unsqueeze(1).unsqueeze(2).repeat(1,nyfull,nxfull)
    rdn_e = rdn.unsqueeze(1).unsqueeze(2).repeat(1,nyfull,nxfull)
    
    rhs = torch.zeros((nzall,nyall,nxall)).to(device)
    msft_inv_e = torch.zeros((nzall,nyall,nxall)).to(device)
    wdwn = torch.zeros((nzall,nyall,nxall)).to(device)

    rhs[0, j_start:j_end, i_start:i_end] = 0.
    msft_inv_e[:, j_start:j_end, i_start:i_end] = 1./msfty_e[:, j_start:j_end, i_start:i_end]
    
    t_2ave[0:k_end, j_start:j_end, i_start:i_end] = 0.5 * ((1. + epssm) * t_2[0:k_end, j_start:j_end, i_start:i_end] + 
                                                           (1. - epssm) * t_2ave[0:k_end, j_start:j_end, i_start:i_end])
    t_2ave[0:k_end, j_start:j_end, i_start:i_end] = (t_2ave[0:k_end, j_start:j_end, i_start:i_end] + 
                                                     muave_e[0:k_end, j_start:j_end, i_start:i_end] * t0) / \
                                                     (muts_e[0:k_end, j_start:j_end, i_start:i_end] * 
                                                     (t0 + t_1[0:k_end, j_start:j_end, i_start:i_end]))
    wdwn[1:k_end+1, j_start:j_end, i_start:i_end] = 0.5 * (ww[1:k_end+1, j_start:j_end, i_start:i_end] + 
                ww[0:k_end, j_start:j_end, i_start:i_end]) * rdnw_e[0:k_end, j_start:j_end, i_start:i_end] * \
                (ph_1[1:k_end+1, j_start:j_end, i_start:i_end] - ph_1[0:k_end, j_start:j_end, i_start:i_end] + 
                 phb[1:k_end+1, j_start:j_end, i_start:i_end] - phb[0:k_end, j_start:j_end, i_start:i_end])
    rhs[1:k_end+1, j_start:j_end, i_start:i_end] = dts * (ph_tend[1:k_end+1, j_start:j_end, i_start:i_end] + \
                                                   0.5 * g * (1. - epssm) * w[1:k_end+1, j_start:j_end, i_start:i_end])
    
    rhs[1:k_end, j_start:j_end, i_start:i_end] = rhs[1:k_end, j_start:j_end, i_start:i_end] - dts * (
                fnm_e[1:k_end, j_start:j_end, i_start:i_end] * wdwn[2:k_end+1, j_start:j_end, i_start:i_end] + 
                fnp_e[1:k_end, j_start:j_end, i_start:i_end] * wdwn[1:k_end, j_start:j_end, i_start:i_end])
    
    rhs[1:k_end+1, j_start:j_end, i_start:i_end] = ph[1:k_end+1, j_start:j_end, i_start:i_end] + \
                msfty_e[1:k_end+1, j_start:j_end, i_start:i_end] * rhs[1:k_end+1, j_start:j_end, i_start:i_end] / \
                mut_e[1:k_end+1, j_start:j_end, i_start:i_end]
    #rhs[k_end,:,:] = 0.
    
    w[0,j_start:j_end, i_start:i_end] = msfty[j_start:j_end, i_start:i_end] * 0.5 * rdy * (
                (ht[j_start+1:j_end+1, i_start:i_end] - ht[j_start:j_end, i_start:i_end]) * 
                (cf1 * v[0, j_start+1:j_end+1, i_start:i_end] + cf2 * v[1, j_start+1:j_end+1, i_start:i_end] + 
                 cf3 * v[2, j_start+1:j_end+1, i_start:i_end]) + 
                (ht[j_start:j_end, i_start:i_end] - ht[j_start-1:j_end-1, i_start:i_end]) * 
                (cf1 * v[0, j_start:j_end, i_start:i_end] + cf2 * v[1, j_start:j_end, i_start:i_end] + 
                 cf3 * v[2, j_start:j_end, i_start:i_end])) + \
                msftx[j_start:j_end, i_start:i_end] * 0.5 * rdx * (
                (ht[j_start:j_end, i_start+1:i_end+1] - ht[j_start:j_end, i_start:i_end]) * 
                (cf1 * u[0, j_start:j_end, i_start+1:i_end+1] + cf2 * u[1, j_start:j_end, i_start+1:i_end+1] + 
                 cf3 * u[2, j_start:j_end, i_start+1:i_end+1]) + 
                (ht[j_start:j_end, i_start:i_end] - ht[j_start:j_end, i_start-1:i_end-1]) * 
                (cf1 * u[0, j_start:j_end, i_start:i_end] + cf2 * u[1, j_start:j_end, i_start:i_end] + 
                 cf3 * u[2, j_start:j_end, i_start:i_end]))
           
    MUTHK = mut_e + 0.0
    MUTHKM1 = mut_e + 0.0
    
    w[1:k_end, j_start:j_end, i_start:i_end] = w[1:k_end, j_start:j_end, i_start:i_end] + dts * \
                rw_tend[1:k_end, j_start:j_end, i_start:i_end] + msft_inv_e[1:k_end, j_start:j_end, i_start:i_end] * \
                cqw[1:k_end, j_start:j_end, i_start:i_end] * ( 0.5 * dts * g * rdn_e[1:k_end, j_start:j_end, i_start:i_end] * (
                c2a[1:k_end, j_start:j_end, i_start:i_end] * rdnw_e[1:k_end, j_start:j_end, i_start:i_end] / MUTHK[1:k_end, j_start:j_end, i_start:i_end] * 
                ((1. + epssm) * (rhs[2:k_end+1, j_start:j_end, i_start:i_end] - rhs[1:k_end, j_start:j_end, i_start:i_end]) + 
                 (1. - epssm) * (ph[2:k_end+1, j_start:j_end, i_start:i_end] - ph[1:k_end, j_start:j_end, i_start:i_end])) -
                c2a[0:k_end-1, j_start:j_end, i_start:i_end] * rdnw_e[0:k_end-1, j_start:j_end, i_start:i_end] / MUTHKM1[1:k_end, j_start:j_end, i_start:i_end] *
                ((1. + epssm) * (rhs[1:k_end, j_start:j_end, i_start:i_end] - rhs[0:k_end-1, j_start:j_end, i_start:i_end]) + 
                 (1. - epssm) * (ph[1:k_end, j_start:j_end, i_start:i_end] - ph[0:k_end-1, j_start:j_end, i_start:i_end])))) \
               + dts * g * msft_inv_e[1:k_end, j_start:j_end, i_start:i_end] * (rdn_e[1:k_end, j_start:j_end, i_start:i_end] * 
                (c2a[1:k_end, j_start:j_end, i_start:i_end] * alt[1:k_end, j_start:j_end, i_start:i_end] * 
                 t_2ave[1:k_end, j_start:j_end, i_start:i_end] - c2a[0:k_end-1, j_start:j_end, i_start:i_end] * 
                 alt[0:k_end-1, j_start:j_end, i_start:i_end] * t_2ave[0:k_end-1, j_start:j_end, i_start:i_end]) - 
                 muave_e[1:k_end, j_start:j_end, i_start:i_end])
    
    MUTHKM1 = mut_e + 0.0
    w[k_end, j_start:j_end, i_start:i_end] = w[k_end, j_start:j_end, i_start:i_end] + dts * \
                rw_tend[k_end, j_start:j_end, i_start:i_end] + msft_inv_e[k_end, j_start:j_end, i_start:i_end] * (
                -0.5 * dts * g / MUTHKM1[k_end, j_start:j_end, i_start:i_end] * rdnw_e[k_end-1, j_start:j_end, i_start:i_end] ** 2 * 2. * 
                c2a[k_end-1, j_start:j_end, i_start:i_end] * ((1. + epssm) * 
                (rhs[k_end, j_start:j_end, i_start:i_end] - rhs[k_end-1, j_start:j_end, i_start:i_end]) + (1. - epssm) * 
                (ph[k_end, j_start:j_end, i_start:i_end] - ph[k_end-1, j_start:j_end, i_start:i_end])) \
              - dts * g * (2. * rdnw_e[k_end-1, j_start:j_end, i_start:i_end] * c2a[k_end-1, j_start:j_end, i_start:i_end] * 
                alt[k_end-1, j_start:j_end, i_start:i_end] * t_2ave[k_end-1, j_start:j_end, i_start:i_end] + 
                muave_e[k_end, j_start:j_end, i_start:i_end]))
    
    for k in range(1,k_end+1):
        w[k, j_start:j_end, i_start:i_end] = (w[k, j_start:j_end, i_start:i_end] - a[k, j_start:j_end, i_start:i_end] * \
                w[k-1, j_start:j_end, i_start:i_end]) * alpha[k, j_start:j_end, i_start:i_end]
    
    for k in range(k_end-1, 0, -1):   #k_end -1 ~ 1
        w[k, j_start:j_end, i_start:i_end] = w[k, j_start:j_end, i_start:i_end] - gamma[k, j_start:j_end, i_start:i_end] * \
                w[k+1, j_start:j_end, i_start:i_end]
    #print("in advance_w w111",w[0:3,528,10:12])
    #damp_opt = 3 
    htop = torch.zeros((nyall,nxall)).to(device)
    hbot = torch.zeros((nyall,nxall)).to(device)
    dampwt = torch.zeros((nzall,nyall,nxall)).to(device)
    dampwt0 = torch.zeros((nzall,nyall,nxall)).to(device)
    hk = torch.zeros((nzall,nyall,nxall)).to(device)
    
    htop[j_start:j_end, i_start:i_end] = (ph_1[k_end, j_start:j_end, i_start:i_end] + 
                                                 phb[k_end, j_start:j_end, i_start:i_end]) / g
    hbot[j_start:j_end, i_start:i_end] = htop[j_start:j_end, i_start:i_end] - hdepth
    hbot_e = hbot.repeat(nzall,1,1)
    
    hk[1:k_end+1, j_start:j_end, i_start:i_end] = (ph_1[1:k_end+1, j_start:j_end, i_start:i_end] + 
                                                   phb[1:k_end+1, j_start:j_end, i_start:i_end]) / g
    condition = hk >= hbot_e
    dampwt[1:k_end+1, j_start:j_end, i_start:i_end] = dampmag * (torch.sin(0.5 * pi * (hk[1:k_end+1, j_start:j_end, i_start:i_end] - 
                                                      hbot_e[1:k_end+1, j_start:j_end, i_start:i_end]) / hdepth))**2
    dampwt = torch.where(condition, dampwt, dampwt0)
    w[1:k_end+1, j_start:j_end, i_start:i_end] = (w[1:k_end+1, j_start:j_end, i_start:i_end] - 
                dampwt[1:k_end+1, j_start:j_end, i_start:i_end] * mut_e[1:k_end+1, j_start:j_end, i_start:i_end] * 
                w_save[1:k_end+1, j_start:j_end, i_start:i_end]) / (1. + dampwt[1:k_end+1, j_start:j_end, i_start:i_end])
    
    # ph
    ph[1:k_end+1, j_start:j_end, i_start:i_end] = rhs[1:k_end+1, j_start:j_end, i_start:i_end] + \
            msfty_e[1:k_end+1, j_start:j_end, i_start:i_end] * 0.5 * dts * g * (1. + epssm) * \
            w[1:k_end+1, j_start:j_end, i_start:i_end] / muts_e[1:k_end+1, j_start:j_end, i_start:i_end]
    
    return t_2ave,w,ph

# Flux sums for w.
def sumflux(ru, rv, ww,                             \
            u_lin, v_lin, ww_lin,                   \
            muu, muv,                               \
            c1h, c2h, c1f, c2f,                     \
            c3h, c4h, c3f, c4f,                     \
            ru_m, rv_m, ww_m, epssm,                \
            msfux, msfuy, msfvx, msfvx_inv, msfvy,  \
            iteration , number_of_small_timesteps,  \
            ids,ide, jds,jde, kds,kde,              \
            ims,ime, jms,jme, kms,kme,              \
            its,ite, jts,jte, kts,kte):
    if iteration == 1:
        ru_m[kts:kte, jts:jte, its:ite] = 0.
        rv_m[kts:kte, jts:jte, its:ite] = 0.
        ww_m[kts:kte, jts:jte, its:ite] = 0.
    
    mini = min(ide-1,ite)
    minj = min(jde-1,jte)
    mink = min(kde-1,kte)
    
    ru_m[kts:mink, jts:minj, its:ite] = ru_m[kts:mink, jts:minj, its:ite] + ru[kts:mink, jts:minj, its:ite]
    rv_m[kts:mink, jts:jte, its:mini] = rv_m[kts:mink, jts:jte, its:mini] + rv[kts:mink, jts:jte, its:mini]
    ww_m[kts:kte, jts:minj, its:mini] = ww_m[kts:kte, jts:minj, its:mini] + ww[kts:kte, jts:minj, its:mini]
    
    if iteration == number_of_small_timesteps:
        muu_e = muu.repeat(nzall,1,1)
        muv_e = muv.repeat(nzall,1,1)
        
        msfuy_e = msfuy.repeat(nzall,1,1)
        msfvx_inv_e = msfvx_inv.repeat(nzall,1,1)
        
        ru_m[kts:mink, jts:minj, its:ite] = ru_m[kts:mink, jts:minj, its:ite] / number_of_small_timesteps + \
                 muu_e[kts:mink, jts:minj, its:ite] * u_lin[kts:mink, jts:minj, its:ite] / \
                 msfuy_e[kts:mink, jts:minj, its:ite]
        rv_m[kts:mink, jts:jte, its:mini] = rv_m[kts:mink, jts:jte, its:mini] / number_of_small_timesteps + \
                 muv_e[kts:mink, jts:jte, its:mini] * v_lin[kts:mink, jts:jte, its:mini] * \
                 msfvx_inv_e[kts:mink, jts:jte, its:mini]
        ww_m[kts:kte, jts:minj, its:mini] = ww_m[kts:kte, jts:minj, its:mini] / number_of_small_timesteps + \
                 ww_lin[kts:kte, jts:minj, its:mini]
    
    return ru_m, rv_m, ww_m



# Recompute corner masses after the small steps.
def calc_mu_uv_1(mu, muu, muv,                 \
                 ids, ide, jds, jde, kds, kde, \
                 ims, ime, jms, jme, kms, kme, \
                 its, ite, jts, jte, kts, kte):
    itf=ite
    jtf=min(jte,jde-1)
    muu[jts:jtf, its+1:itf-1] = 0.5 * (mu[jts:jtf, its+1:itf-1] + mu[jts:jtf, its:itf-2])
    muu[jts:jtf, its] = mu[jts:jtf, its]
    muu[jts:jtf, ite-1] = mu[jts:jtf, ite-2] #+ mu[jts:jtf, ite-3])
    
    itf=min(ite,ide-1)
    jtf=jte
    muv[jts+1:jtf-1, its:itf] = 0.5 * (mu[jts+1:jtf-1, its:itf] + mu[jts:jtf-2, its:itf])
    muv[jts, its:itf] = mu[jts, its:itf]
    muv[jte-1, its:itf] = mu[jte-2, its:itf] #+ mu[jte-3, its:itf])
    
    return muu, muv

# Finish the small-step loop (final tendencies).
def small_step_finish(u_2, u_1, v_2, v_1, w_2, w_1,    \
                      t_2, t_1, ph_2, ph_1, ww, ww1,   \
                      mu_2, mu_1,                      \
                      mut, muts, muu, muus, muv, muvs, \
                      c1h, c2h, c1f, c2f,              \
                      c3h, c4h, c3f, c4f,              \
                      u_save, v_save, w_save,          \
                      t_save, ph_save, mu_save,        \
                      msfux, msfuy, msfvx, msfvy,      \
                      msftx, msfty,                    \
                      h_diabatic,                      \
                      number_of_small_timesteps,dts,   \
                      rk_step, rk_order,               \
                      ids,ide, jds,jde, kds,kde,       \
                      ims,ime, jms,jme, kms,kme,       \
                      its,ite, jts,jte, kts,kte):
    i_start = its  
    i_end   = min(ite,ide-1)
    j_start = jts
    j_end   = min(jte,jde-1)
    
    i_endu = ite
    j_endv = jte
    
    msfuy_e = msfuy.repeat(nzall,1,1)
    msfvx_e = msfvx.repeat(nzall,1,1)
    msfty_e = msfty.repeat(nzall,1,1)
    
    muv_e = muv.repeat(nzall,1,1)
    muvs_e = muvs.repeat(nzall,1,1)
    muu_e = muu.repeat(nzall,1,1)
    muus_e = muus.repeat(nzall,1,1)
    mut_e = mut.repeat(nzall,1,1)
    muts_e = muts.repeat(nzall,1,1)
    
    u_2_b = u_2.clone()
    v_2_b = v_2.clone()
    w_2_b = w_2.clone()
    t_2_b = t_2.clone()
    ph_2_b = ph_2.clone()
    
    v_2_b[kds:kde-1, j_start:j_endv, i_start:i_end] = (msfvx_e[kds:kde-1, j_start:j_endv, i_start:i_end] * 
            v_2[kds:kde-1, j_start:j_endv, i_start:i_end] + v_save[kds:kde-1, j_start:j_endv, i_start:i_end] * 
            muv_e[kds:kde-1, j_start:j_endv, i_start:i_end] )/ muvs_e[kds:kde-1, j_start:j_endv, i_start:i_end]
    u_2_b[kds:kde-1, j_start:j_end, i_start:i_endu] = (msfuy_e[kds:kde-1, j_start:j_end, i_start:i_endu] * 
            u_2[kds:kde-1, j_start:j_end, i_start:i_endu] + u_save[kds:kde-1, j_start:j_end, i_start:i_endu] * 
            muu_e[kds:kde-1, j_start:j_end, i_start:i_endu]) / muus_e[kds:kde-1, j_start:j_end, i_start:i_endu]
    w_2_b[kds:kde, j_start:j_end, i_start:i_end] = (msfty_e[kds:kde, j_start:j_end, i_start:i_end] * 
            w_2[kds:kde, j_start:j_end, i_start:i_end] + w_save[kds:kde, j_start:j_end, i_start:i_end] * 
            mut_e[kds:kde, j_start:j_end, i_start:i_end]) / muts_e[kds:kde, j_start:j_end, i_start:i_end]
    ph_2_b[kds:kde, j_start:j_end, i_start:i_end] = ph_2[kds:kde, j_start:j_end, i_start:i_end] + \
            ph_save[kds:kde, j_start:j_end, i_start:i_end]
    ww[kds:kde, j_start:j_end, i_start:i_end] = ww[kds:kde, j_start:j_end, i_start:i_end] + \
            ww1[kds:kde, j_start:j_end, i_start:i_end]
    
    if rk_step < rk_order:
        t_2_b[kds:kde-1, j_start:j_end, i_start:i_end] = (t_2[kds:kde-1, j_start:j_end, i_start:i_end] + 
            t_save[kds:kde-1, j_start:j_end, i_start:i_end] * mut_e[kds:kde-1, j_start:j_end, i_start:i_end]) / \
            muts_e[kds:kde-1, j_start:j_end, i_start:i_end]
    else:
        t_2_b[kds:kde-1, j_start:j_end, i_start:i_end] = (t_2[kds:kde-1, j_start:j_end, i_start:i_end] - 
            dts * number_of_small_timesteps * mut_e[kds:kde-1, j_start:j_end, i_start:i_end] * 
            h_diabatic[kds:kde-1, j_start:j_end, i_start:i_end] + t_save[kds:kde-1, j_start:j_end, i_start:i_end] * 
            mut_e[kds:kde-1, j_start:j_end, i_start:i_end]) / muts_e[kds:kde-1, j_start:j_end, i_start:i_end]
    
    mu_2[j_start:j_end, i_start:i_end] = mu_2[j_start:j_end, i_start:i_end] + mu_save[j_start:j_end, i_start:i_end]
    
    return u_2_b, v_2_b, w_2_b, ph_2_b, ww, t_2_b, mu_2

def rk_update_scalar_pd(scs, sce,                      \
                        scalar, sc_tend,               \
                        c1, c2,                        \
                        mu_old, mu_new, mu_base,       \
                        rk_step, dt, spec_zone,        \
                        ids, ide, jds, jde, kds, kde,  \
                        ims, ime, jms, jme, kms, kme,  \
                        its, ite, jts, jte, kts, kte):
    i_start = its
    i_end   = min(ite,ide-1)
    j_start = jts
    j_end   = min(jte,jde-1)
    k_start = kts
    k_end   = kte-1
    
    i_start_spc = i_start
    i_end_spc   = i_end
    j_start_spc = j_start
    j_end_spc   = j_end
    k_start_spc = k_start
    k_end_spc   = k_end
    
    i_start = max( its,ids+spec_zone )
    i_end   = min( ite,ide-spec_zone-1 )
    j_start = max( jts,jds+spec_zone )
    j_end   = min( jte,jde-spec_zone-1 )
    k_start = kts
    k_end   = min( kte, kde-1 )
    
    muold = torch.zeros((nyall,nxall)).to(device)
    munew = torch.zeros((nyall,nxall)).to(device)
    tendency = torch.zeros((nzall,nyall,nxall)).to(device)
    
    muold[jts:jde-1, its:ide-1] = muold[jts:jde-1, its:ide-1] + mu_base[jts:jde-1, its:ide-1]
    munew[jts:jde-1, its:ide-1] = munew[jts:jde-1, its:ide-1] + mu_base[jts:jde-1, its:ide-1]
    
    muold_e = muold.repeat(nzall,1,1)
    munew_e = munew.repeat(nzall,1,1)
    
    for im in range(scs,sce+1):  #  可能有问题
        tendency[kts:kde-1, jts:jde-1, its:ide-1] = 0.
        tendency[k_start_spc:k_end_spc, j_start_spc:j_end_spc, i_start_spc:i_end_spc] = tendency[k_start_spc:k_end_spc, j_start_spc:j_end_spc, i_start_spc:i_end_spc] + \
            sc_tend[im, k_start_spc:k_end_spc, j_start_spc:j_end_spc, i_start_spc:i_end_spc]
        sc_tend[im, k_start_spc:k_end_spc, j_start_spc:j_end_spc, i_start_spc:i_end_spc] = 0.
        scalar[im, kts:kde-1, jts:jde-1, its:ide-1] = (muold_e[kts:kde-1, jts:jde-1, its:ide-1] * scalar[im, kts:kde-1, jts:jde-1, its:ide-1] + \
            dt * tendency[kts:kde-1, jts:jde-1, its:ide-1]) / munew_e[kts:kde-1, jts:jde-1, its:ide-1]
        
    return scalar, sc_tend

# Advection tendency for moisture / scalar species.
def rk_scalar_tend(scs, sce,                        \
                   tenddec,                         \
                   rk_step, dt,                     \
                   ru, rv, ww, mut, mub, mu_old,    \
                   c1h, c2h, alt,                   \
                   scalar_old, scalar,              \
                   scalar_tends, advect_tend,       \
                   h_tendency, z_tendency,          \
                   RQVFTEN,                         \
                   base, moist_step, fnm, fnp,      \
                   msfux, msfuy, msfvx, msfvx_inv,  \
                   msfvy, msftx, msfty,             \
                   rdx, rdy, rdn, rdnw,             \
                   khdif, kvdif, xkmhd,             \
                   diff_6th_opt, diff_6th_factor,   \
                   adv_opt,                         \
                   ids, ide, jds, jde, kds, kde,    \
                   ims, ime, jms, jme, kms, kme,    \
                   its, ite, jts, jte, kts, kte):
    
    khdq = khdif/prandtl
    kvdq = kvdif/prandtl
    
    for im in range(scs,sce+1):
        advect_tend[im, kts:kte, jts:jte, its:ite] = 0.
        h_tendency[:,:,:] = 0.
        z_tendency[:,:,:] = 0.
        
        advect_tend[im,:,:,:] = advect_scalar(scalar[im,:,:,:],  \
                               scalar[im,:,:,:],              \
                               advect_tend[im,:,:,:],         \
                               ru, rv, ww, c1h, c2h,          \
                               mut, time_step,                \
                               msfux, msfuy, msfvx, msfvy,    \
                               msftx, msfty, fnm, fnp,        \
                               rdx, rdy, rdnw,                \
                               ids, ide, jds, jde, kds, kde,  \
                               ims, ime, jms, jme, kms, kme,  \
                               its, ite, jts, jte, kts, kte)
        
        if rk_step == 1:
            scalar_tends[im,:,:,:] = horizontal_diffusion( 'm', scalar[im,:,:,:],  \
                                   scalar_tends[im,:,:,:], mut,       \
                                   c1h, c2h,                          \
                                   msfux, msfuy, msfvx, msfvx_inv,    \
                                   msfvy, msftx, msfty,               \
                                   khdq , xkmhd, rdx, rdy,            \
                                   ids, ide, jds, jde, kds, kde,      \
                                   ims, ime, jms, jme, kms, kme,      \
                                   its, ite, jts, jte, kts, kte      )
            
            if moist_step and im == P_QV:
                scalar_tends[im,:,:,:] = vertical_diffusion_mp(scalar[im,:,:,:],             \
                                                               scalar_tends[im,:,:,:],       \
                                                               base, c1h, c2h,               \
                                                               alt, mut, rdn, rdnw, kvdq ,   \
                                                               ids, ide, jds, jde, kds, kde, \
                                                               ims, ime, jms, jme, kms, kme, \
                                                               its, ite, jts, jte, kts, kte )
                
            else:
                scalar_tends[im,:,:,:] = vertical_diffusion('m', scalar[im,:,:,:],        \
                                                            scalar_tends[im,:,:,:],       \
                                                            c1h, c2h,                     \
                                                            alt, mut, rdn, rdnw, kvdq,    \
                                                            ids, ide, jds, jde, kds, kde, \
                                                            ims, ime, jms, jme, kms, kme, \
                                                            its, ite, jts, jte, kts, kte )
    
    return scalar_tends, advect_tend, h_tendency, z_tendency


# RK update of moisture / scalar species.
def rk_update_scalar(scs, sce,                      \
                     scalar_1, scalar_2, sc_tend,   \
                     #advh_t, advz_t,                \  assume no advh_t advz_t
                     advect_tend,                   \
                     h_tendency, z_tendency,        \
                     msftx, msfty, c1, c2,          \
                     mu_old, mu_new, mu_base,       \
                     rk_step, dt, spec_zone,        \
                     tenddec,                       \
                     ids, ide, jds, jde, kds, kde,  \
                     ims, ime, jms, jme, kms, kme,  \
                     its, ite, jts, jte, kts, kte):
    i_start = its
    i_end   = min(ite,ide-1)
    j_start = jts
    j_end   = min(jte,jde-1)
    k_start = kts
    k_end   = kte-1
    
    i_start_spc = i_start
    i_end_spc   = i_end
    j_start_spc = j_start
    j_end_spc   = j_end
    k_start_spc = k_start
    k_end_spc   = k_end
    
    i_start = max( its,ids+spec_zone )
    i_end   = min( ite,ide-spec_zone-1 )
    j_start = max( jts,jds+spec_zone )
    j_end   = min( jte,jde-spec_zone-1 )
    k_start = kts
    k_end   = min( kte, kde-1 )
    
    msfty_e = msfty.repeat(nzall,1,1)
    
    muold = torch.zeros((nyall,nxall)).to(device)
    munew = torch.zeros((nyall,nxall)).to(device)
    tendency = torch.zeros((nzall,nyall,nxall)).to(device)
    
    muold[jts:jde-1, its:ide-1] = mu_old[jts:jde-1, its:ide-1] + mu_base[jts:jde-1, its:ide-1]
    munew[jts:jde-1, its:ide-1] = mu_new[jts:jde-1, its:ide-1] + mu_base[jts:jde-1, its:ide-1]
    muold_e = muold.repeat(nzall,1,1)
    munew_e = munew.repeat(nzall,1,1)
    
    if rk_step == 1:
        for im in range(scs,sce+1):
            tendency[kts:kde-1, jts:jde-1, its:ide-1] = 0.
            tendency[k_start:k_end, j_start:j_end, i_start:i_end] = advect_tend[im, k_start:k_end, j_start:j_end, i_start:i_end] * \
                    msfty_e[k_start:k_end, j_start:j_end, i_start:i_end]
            tendency[k_start_spc:k_end_spc, j_start_spc:j_end_spc, i_start_spc:i_end_spc] = tendency[k_start_spc:k_end_spc, j_start_spc:j_end_spc, i_start_spc:i_end_spc] + \
                    sc_tend[im, k_start_spc:k_end_spc, j_start_spc:j_end_spc, i_start_spc:i_end_spc]
            scalar_1[im, :, :, :] = scalar_2[im, :, :, :] + 0.0
            scalar_2[im, kts:kde-1, jts:jde-1, its:ide-1] = (muold_e[kts:kde-1, jts:jde-1, its:ide-1] * scalar_1[im, kts:kde-1, jts:jde-1, its:ide-1] + \
                    dt * tendency[kts:kde-1, jts:jde-1, its:ide-1]) / munew_e[kts:kde-1, jts:jde-1, its:ide-1]
    else:
        for im in range(scs,sce+1):
            tendency[kts:kde-1, jts:jde-1, its:ide-1] = 0.
            tendency[k_start:k_end, j_start:j_end, i_start:i_end] = advect_tend[im, k_start:k_end, j_start:j_end, i_start:i_end] * \
                    msfty_e[k_start:k_end, j_start:j_end, i_start:i_end]
            tendency[k_start_spc:k_end_spc, j_start_spc:j_end_spc, i_start_spc:i_end_spc] = tendency[k_start_spc:k_end_spc, j_start_spc:j_end_spc, i_start_spc:i_end_spc] + \
                    sc_tend[im, k_start_spc:k_end_spc, j_start_spc:j_end_spc, i_start_spc:i_end_spc]
            scalar_2[im, kts:kde-1, jts:jde-1, its:ide-1] = (muold_e[kts:kde-1, jts:jde-1, its:ide-1] * scalar_1[im, kts:kde-1, jts:jde-1, its:ide-1] + \
                    dt * tendency[kts:kde-1, jts:jde-1, its:ide-1]) / munew_e[kts:kde-1, jts:jde-1, its:ide-1]
    
    return scalar_1, scalar_2



# Diagnose pressure, density and geopotential phi.
def calc_p_rho_phi(moist, n_moist, hypsometric_opt,      \
                   al, alb, mu, muts,                    \
                   c1, c2, c3h, c4h, c3f, c4f,           \
                   ph, phb, p, pb,                       \
                   t, p0, t0, ptop, znu, znw, dnw, rdnw, \
                   rdn, non_hydrostatic,          \
                   ids, ide, jds, jde, kds, kde,  \
                   ims, ime, jms, jme, kms, kme,  \
                   its, ite, jts, jte, kts, kte):
    itf=min(ite,ide-1) 
    jtf=min(jte,jde-1)
    ktf=min(kte,kde-1)
    
    muts_e = muts.repeat(nzall,1,1)
    mu_e = mu.repeat(nzall,1,1)
    rdnw_e = rdnw.unsqueeze(1).unsqueeze(2).repeat(1,nyall,nxall)
    
    qvf = torch.zeros((nzall,nyall,nxall)).to(device)
    temp = torch.zeros((nzall,nyall,nxall)).to(device)
    
    al[kts:ktf, jts:jtf, its:itf] = -1. / muts_e[kts:ktf, jts:jtf, its:itf] * \
            (alb[kts:ktf, jts:jtf, its:itf] * mu_e[kts:ktf, jts:jtf, its:itf] + rdnw_e[kts:ktf, jts:jtf, its:itf] * 
             (ph[kts+1:ktf+1, jts:jtf, its:itf] - ph[kts:ktf, jts:jtf, its:itf]))
    # moist
    qvf[kts:ktf, jts:jtf, its:itf] = 1. + rvovrd * moist[P_QV,kts:ktf, jts:jtf, its:itf]
    temp[kts:ktf, jts:jtf, its:itf] = (r_d * (t0 + t[kts:ktf, jts:jtf, its:itf]) * qvf[kts:ktf, jts:jtf, its:itf]) / \
            (p0 * (al[kts:ktf, jts:jtf, its:itf] + alb[kts:ktf, jts:jtf, its:itf]))
    
    
    p[kts:ktf, jts:jtf, its:itf] = temp[kts:ktf, jts:jtf, its:itf] ** cpovcv
    
    p[kts:ktf, jts:jtf, its:itf] = p[kts:ktf, jts:jtf, its:itf] * p0 - pb[kts:ktf, jts:jtf, its:itf]
    
    # hypsometric opt == 1
    #dnw_e = dnw.unsqueeze(1).unsqueeze(2).repeat(1,nyfull,nxfull)
    #for k in range(1,ktf+1):
    #    ph[k, jts:jtf, its:itf] = ph[k-1, jts:jtf, its:itf] - (dnw_e[k-1, jts:jtf, its:itf]) * \
    #            ((muts_e[k, jts:jtf, its:itf]) * al[k-1, jts:jtf, its:itf] + 
    #             mu_e[k, jts:jtf, its:itf] * alb[k-1, jts:jtf, its:itf])
    
    return al, p, ph

def advance_ppt(RTHCUTEN,RQVCUTEN,RQCCUTEN,RQRCUTEN,           \
                CLDFRA_CUP,                                    \
                RQICUTEN,RQSCUTEN,                             \
                RAINC,RAINCV,RAINSH,PRATEC,PRATESH,            \
                NCA, HTOP,HBOT,CUTOP,CUBOT,                    \
                CUPPT, DT, config_flags,                       \
                ids,ide, jds,jde, kds,kde,                     \
                ims,ime, jms,jme, kms,kme,                     \
                its,ite, jts,jte, kts,kte):
    
    return

# Post-RK physics preparation (PBL tendencies).
def phy_prep_part2(mut,muu,muv,                                 \
                   c1h, c2h, c1f, c2f,                          \
                   RTHBLTEN, RUBLTEN, RVBLTEN,                  \
                   RQVBLTEN, RQCBLTEN, RQIBLTEN,                \
                   ids, ide, jds, jde, kds, kde,                \
                   ims, ime, jms, jme, kms, kme,                \
                   its, ite, jts, jte, kts, kte):
    i_start = its
    i_end   = min( ite,ide-1 )
    j_start = jts
    j_end   = min( jte,jde-1 )

    k_start = kts
    k_end = min( kte, kde-1 )

    c1 = c1h
    c2 = c2h
    
    # only do for pbl
    mut_e = mut.repeat(nzall,1,1)

    RUBLTEN[k_start:k_end, j_start:j_end, i_start:i_end] = RUBLTEN[k_start:k_end, j_start:j_end, i_start:i_end] / \
            mut_e[k_start:k_end, j_start:j_end, i_start:i_end]
    RVBLTEN[k_start:k_end, j_start:j_end, i_start:i_end] = RVBLTEN[k_start:k_end, j_start:j_end, i_start:i_end] / \
            mut_e[k_start:k_end, j_start:j_end, i_start:i_end]
    RTHBLTEN[k_start:k_end, j_start:j_end, i_start:i_end] = RTHBLTEN[k_start:k_end, j_start:j_end, i_start:i_end] / \
            mut_e[k_start:k_end, j_start:j_end, i_start:i_end]
    RQVBLTEN[k_start:k_end, j_start:j_end, i_start:i_end] = RQVBLTEN[k_start:k_end, j_start:j_end, i_start:i_end] / \
            mut_e[k_start:k_end, j_start:j_end, i_start:i_end]
    RQCBLTEN[k_start:k_end, j_start:j_end, i_start:i_end] = RQCBLTEN[k_start:k_end, j_start:j_end, i_start:i_end] / \
            mut_e[k_start:k_end, j_start:j_end, i_start:i_end]
    RQIBLTEN[k_start:k_end, j_start:j_end, i_start:i_end] = RQIBLTEN[k_start:k_end, j_start:j_end, i_start:i_end] / \
            mut_e[k_start:k_end, j_start:j_end, i_start:i_end]
       
    return RUBLTEN, RVBLTEN, RTHBLTEN, RQVBLTEN, RQCBLTEN, RQIBLTEN



# Reset w to zero at the surface (no-flow lower boundary).
def set_w_surface(znw, fill_w_flag,                            \
                  w, ht, u, v, cf1, cf2, cf3, rdx, rdy,        \
                  msftx, msfty,                                \
                  ids, ide, jds, jde, kds, kde,                \
                  ims, ime, jms, jme, kms, kme,                \
                  its, ite, jts, jte, kts, kte):
    jm1_limit = jds
    jp1_limit = jde-1
    im1_limit = ids
    ip1_limit = ide-1
    
    #msfty_e = msfty.repeat(kte-kts,1,1)
    #msftx_e = msftx.repeat(kte-kts,1,1)
    #ht_e = ht.repeat(kte-kts,1,1)
            
    w[0, jts+1:jte-2, its+1:ite-2] = msfty[jts+1:jte-2, its+1:ite-2] * 0.5 * rdy * (
            (ht[jts+2:jte-1, its+1:ite-2] - ht[jts+1:jte-2, its+1:ite-2]) * (cf1 * v[0, jts+2:jte-1, its+1:ite-2] + 
                                                                             cf2 * v[1, jts+2:jte-1, its+1:ite-2] + 
                                                                             cf3 * v[2, jts+2:jte-1, its+1:ite-2]) + 
            (ht[jts+1:jte-2, its+1:ite-2] - ht[jts:jte-3, its+1:ite-2]) * (cf1 * v[0, jts+1:jte-2, its+1:ite-2] + 
                                                                           cf2 * v[1, jts+1:jte-2, its+1:ite-2] + 
                                                                           cf3 * v[2, jts+1:jte-2, its+1:ite-2])) + \
                                     msftx[jts+1:jte-2, its+1:ite-2] * 0.5 * rdx * (
            (ht[jts+1:jte-2, its+2:ite-1] - ht[jts+1:jte-2, its+1:ite-2]) * (cf1 * u[0, jts+1:jte-2, its+2:ite-1] + 
                                                                             cf2 * u[1, jts+1:jte-2, its+2:ite-1] + 
                                                                             cf3 * u[2, jts+1:jte-2, its+2:ite-1]) + 
            (ht[jts+1:jte-2, its+1:ite-2] - ht[jts+1:jte-2, its:ite-3]) * (cf1 * u[0, jts+1:jte-2, its+1:ite-2] + 
                                                                           cf2 * u[1, jts+1:jte-2, its+1:ite-2] + 
                                                                           cf3 * u[2, jts+1:jte-2, its+1:ite-2]))
    w[0, jts, its+1:ite-2] = msfty[jts, its+1:ite-2] * 0.5 * rdy * (
            (ht[jts+1, its+1:ite-2] - ht[jts, its+1:ite-2]) * (cf1 * v[0, jts+1, its+1:ite-2] + 
                                                               cf2 * v[1, jts+1, its+1:ite-2] + 
                                                               cf3 * v[2, jts+1, its+1:ite-2])) + \
                             msftx[jts, its+1:ite-2] * 0.5 * rdx * (
            (ht[jts, its+2:ite-1] - ht[jts, its+1:ite-2]) * (cf1 * u[0, jts, its+2:ite-1] + 
                                                             cf2 * u[1, jts, its+2:ite-1] + 
                                                             cf3 * u[2, jts, its+2:ite-1]) + 
            (ht[jts, its+1:ite-2] - ht[jts, its:ite-3]) * (cf1 * u[0, jts, its+1:ite-2] + 
                                                           cf2 * u[1, jts, its+1:ite-2] + 
                                                           cf3 * u[2, jts, its+1:ite-2]))
    w[0, jte-2, its+1:ite-2] = msfty[jte-2, its+1:ite-2] * 0.5 * rdy * (
            (ht[jte-2, its+1:ite-2] - ht[jte-3, its+1:ite-2]) * (cf1 * v[0, jte-2, its+1:ite-2] + 
                                                                 cf2 * v[1, jte-2, its+1:ite-2] + 
                                                                 cf3 * v[2, jte-2, its+1:ite-2])) + \
                               msftx[jte-2, its+1:ite-2] * 0.5 * rdx * (
            (ht[jte-2, its+2:ite-1] - ht[jte-2, its+1:ite-2]) * (cf1 * u[0, jte-2, its+2:ite-1] + 
                                                                 cf2 * u[1, jte-2, its+2:ite-1] + 
                                                                 cf3 * u[2, jte-2, its+2:ite-1]) + 
            (ht[jte-2, its+1:ite-2] - ht[jte-2, its:ite-3]) * (cf1 * u[0, jte-2, its+1:ite-2] + 
                                                               cf2 * u[1, jte-2, its+1:ite-2] + 
                                                               cf3 * u[2, jte-2, its+1:ite-2]))
    w[0, jts+1:jte-2, its] = msfty[jts+1:jte-2, its] * 0.5 * rdy * (
            (ht[jts+2:jte-1, its] - ht[jts+1:jte-2, its]) * (cf1 * v[0, jts+2:jte-1, its] + 
                                                             cf2 * v[1, jts+2:jte-1, its] + 
                                                             cf3 * v[2, jts+2:jte-1, its]) + 
            (ht[jts+1:jte-2, its] - ht[jts:jte-3, its]) * (cf1 * v[0, jts+1:jte-2, its] + 
                                                           cf2 * v[1, jts+1:jte-2, its] + 
                                                           cf3 * v[2, jts+1:jte-2, its])) + \
                             msftx[jts+1:jte-2, its] * 0.5 * rdx * (
            (ht[jts+1:jte-2, its+1] - ht[jts+1:jte-2, its]) * (cf1 * u[0, jts+1:jte-2, its+1] + 
                                                               cf2 * u[1, jts+1:jte-2, its+1] + 
                                                               cf3 * u[2, jts+1:jte-2, its+1]) )
    w[0, jts+1:jte-2, ite-2] = msfty[jts+1:jte-2, ite-2] * 0.5 * rdy * (
            (ht[jts+2:jte-1, ite-2] - ht[jts+1:jte-2, ite-2]) * (cf1 * v[0, jts+2:jte-1, ite-2] + 
                                                             cf2 * v[1, jts+2:jte-1, ite-2] + 
                                                             cf3 * v[2, jts+2:jte-1, ite-2]) + 
            (ht[jts+1:jte-2, ite-2] - ht[jts:jte-3, ite-2]) * (cf1 * v[0, jts+1:jte-2, ite-2] + 
                                                           cf2 * v[1, jts+1:jte-2, ite-2] + 
                                                           cf3 * v[2, jts+1:jte-2, ite-2])) + \
                             msftx[jts+1:jte-2, ite-2] * 0.5 * rdx * (
            (ht[jts+1:jte-2, ite-2] - ht[jts+1:jte-2, ite-3]) * (cf1 * u[0, jts+1:jte-2, ite-2] + 
                                                                 cf2 * u[1, jts+1:jte-2, ite-2] + 
                                                                 cf3 * u[2, jts+1:jte-2, ite-2]))
    w[0, jts, its] = msfty[jts, its] * 0.5 * rdy * (
            (ht[jts+1, its] - ht[jts, its]) * (cf1 * v[0, jts+1, its] + 
                                               cf2 * v[1, jts+1, its] + 
                                               cf3 * v[2, jts+1, its])) + \
                     msftx[jts, its] * 0.5 * rdx * (
            (ht[jts, its+1] - ht[jts, its]) * (cf1 * u[0, jts, its+1] + 
                                               cf2 * u[1, jts, its+1] + 
                                               cf3 * u[2, jts, its+1]) )
    w[0, jte-2, its] = msfty[jte-2, its] * 0.5 * rdy * (
            (ht[jte-2, its] - ht[jte-3, its]) * (cf1 * v[0, jte-2, its] + 
                                                 cf2 * v[1, jte-2, its] + 
                                                 cf3 * v[2, jte-2, its])) + \
                       msftx[jte-2, its] * 0.5 * rdx * (
            (ht[jte-2, its+1] - ht[jte-2, its]) * (cf1 * u[0, jte-2, its+1] + 
                                                   cf2 * u[1, jte-2, its+1] + 
                                                   cf3 * u[2, jte-2, its+1]) )
    w[0, jts, ite-2] = msfty[jts, ite-2] * 0.5 * rdy * (
            (ht[jts+1, ite-2] - ht[jts+1, ite-2]) * (cf1 * v[0, jts+1, ite-2] + 
                                               cf2 * v[1, jts+1, ite-2] + 
                                               cf3 * v[2, jts+1, ite-2]) ) + \
                       msftx[jts, ite-2] * 0.5 * rdx * (
            (ht[jts, ite-2] - ht[jts, ite-3]) * (cf1 * u[0, jts, ite-2] + 
                                                 cf2 * u[1, jts, ite-2] + 
                                                 cf3 * u[2, jts, ite-2]) )
    w[0, jte-2, ite-2] = msfty[jte-2, ite-2] * 0.5 * rdy * (
            (ht[jte-2, ite-2] - ht[jte-3, ite-2]) * (cf1 * v[0, jte-2, ite-2] + 
                                                     cf2 * v[1, jte-2, ite-2] + 
                                                     cf3 * v[2, jte-2, ite-2])) + \
                       msftx[jts, its] * 0.5 * rdx * (
            (ht[jte-2, ite-2] - ht[jte-2, ite-3]) * (cf1 * u[0, jte-2, ite-2] + 
                                                     cf2 * u[1, jte-2, ite-2] + 
                                                     cf3 * u[2, jte-2, ite-2]) )
    
    return w
