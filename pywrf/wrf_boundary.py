"""PyWRF lateral boundary conditions.

Specified (relaxation + specified zones) and flow-dependent boundary
updates, plus mass-weighting helpers. Constants come from
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

# Relaxation boundary tendencies for dry variables.
def relax_bdy_dry(ru_tendf, rv_tendf, ph_tendf, t_tendf,           \
                  rw_tendf, mu_tend, c1h, c2h, c1f, c2f,           \
                  ru, rv, ph, t,                                   \
                  w, mu, mut,                                      \
                  u_bxs,u_bxe,u_bys,u_bye,                         \
                  v_bxs,v_bxe,v_bys,v_bye,                         \
                  ph_bxs,ph_bxe,ph_bys,ph_bye,                     \
                  t_bxs,t_bxe,t_bys,t_bye,                         \
                  w_bxs,w_bxe,w_bys,w_bye,                         \
                  mu_bxs,mu_bxe,mu_bys,mu_bye,                     \
                  u_btxs,u_btxe,u_btys,u_btye,                     \
                  v_btxs,v_btxe,v_btys,v_btye,                     \
                  ph_btxs,ph_btxe,ph_btys,ph_btye,                 \
                  t_btxs,t_btxe,t_btys,t_btye,                     \
                  w_btxs,w_btxe,w_btys,w_btye,                     \
                  mu_btxs,mu_btxe,mu_btys,mu_btye,                 \
                  spec_bdy_width, spec_zone, relax_zone,           \
                  dtbc, fcx, gcx,             \
                  ids,ide, jds,jde, kds,kde,  \
                  ims,ime, jms,jme, kms,kme,  \
                  ips,ipe, jps,jpe, kps,kpe,  \
                  its, ite, jts, jte, kts, kte):
    
    return

# Add relaxation boundary tendencies to a field.
def relax_bdytend(field, field_tend,                     \
                  field_bdy_xs, field_bdy_xe,            \
                  field_bdy_ys, field_bdy_ye,            \
                  field_bdy_tend_xs, field_bdy_tend_xe,  \
                  field_bdy_tend_ys, field_bdy_tend_ye,  \
                  variable_in,                           \
                  spec_bdy_width, spec_zone, relax_zone, \
                  dtbc, fcx, gcx,             \
                  ids,ide, jds,jde, kds,kde,  \
                  ims,ime, jms,jme, kms,kme,  \
                  ips,ipe, jps,jpe, kps,kpe,  \
                  its,ite, jts,jte, kts,kte):
    ibs = ids
    ibe = ide-1
    itf = min(ite,ide-1)
    jbs = jds
    jbe = jde-1
    jtf = min(jte,jde-1)
    ktf = kde-1
    
    fls0 = torch.zeros((nzall,nyall,nxall))
    fls1 = torch.zeros((nzall,nyall,nxall))
    fls2 = torch.zeros((nzall,nyall,nxall))
    fls3 = torch.zeros((nzall,nyall,nxall))
    fls4 = torch.zeros((nzall,nyall,nxall))
    
    variable = variable_in
    if variable == 'u':
        ibe = ide
        itf = min(ite,ide)
    if variable == 'v':
        jbe = jde
        jtf = min(jte,jde)
    if variable == 'm':
        ktf = kte
    if variable == 'h':
        ktf = kte
    # relax_zone = 4, in fact jbs+1:jbs+4-1+1
    
    # y boundary    
    for b_dist in range(1,4,1):
        fls0[kts:ktf, jbs+b_dist, ibs+b_dist:ibe-b_dist] = field_bdy_ys[kts:ktf, b_dist, ibs+b_dist:ibe-b_dist] + \
              dtbc * field_bdy_tend_ys[kts:ktf, b_dist, ibs+b_dist:ibe-b_dist] - field[kts:ktf, jbs+b_dist, ibs+b_dist:ibe-b_dist]
        fls1[kts:ktf, jbs+b_dist, ibs+b_dist:ibe-b_dist] = field_bdy_ys[kts:ktf, b_dist, ibs+b_dist-1:ibe-b_dist-1] + \
              dtbc * field_bdy_tend_ys[kts:ktf, b_dist, ibs+b_dist-1:ibe-b_dist-1] - field[kts:ktf, jbs+b_dist, ibs+b_dist-1:ibe-b_dist-1]
        fls2[kts:ktf, jbs+b_dist, ibs+b_dist:ibe-b_dist] = field_bdy_ys[kts:ktf, b_dist, ibs+b_dist+1:ibe-b_dist+1] + \
              dtbc * field_bdy_tend_ys[kts:ktf, b_dist, ibs+b_dist+1:ibe-b_dist+1] - field[kts:ktf, jbs+b_dist, ibs+b_dist+1:ibe-b_dist+1]
        fls3[kts:ktf, jbs+b_dist, ibs+b_dist:ibe-b_dist] = field_bdy_ys[kts:ktf, b_dist-1, ibs+b_dist:ibe-b_dist] + \
              dtbc * field_bdy_tend_ys[kts:ktf, b_dist-1, ibs+b_dist:ibe-b_dist] - field[kts:ktf, jbs+b_dist-1, ibs+b_dist:ibe-b_dist]
        fls4[kts:ktf, jbs+b_dist, ibs+b_dist:ibe-b_dist] = field_bdy_ys[kts:ktf, b_dist+1, ibs+b_dist:ibe-b_dist] + \
              dtbc * field_bdy_tend_ys[kts:ktf, b_dist+1, ibs+b_dist:ibe-b_dist] - field[kts:ktf, jbs+b_dist+1, ibs+b_dist:ibe-b_dist]
        field_tend[kts:ktf, jbs+b_dist,ibs+b_dist:ibe-b_dist] = field_tend[kts:ktf, jbs+b_dist,ibs+b_dist:ibe-b_dist] + \
              fcx[b_dist] * fls0[kts:ktf, jbs+b_dist,ibs+b_dist:ibe-b_dist] - gcx[b_dist] * \
              (fls1[kts:ktf, jbs+b_dist,ibs+b_dist:ibe-b_dist] + fls2[kts:ktf, jbs+b_dist,ibs+b_dist:ibe-b_dist] + 
               fls3[kts:ktf, jbs+b_dist,ibs+b_dist:ibe-b_dist] + fls4[kts:ktf, jbs+b_dist,ibs+b_dist:ibe-b_dist] - 
               4. * fls0[kts:ktf, jbs+b_dist,ibs+b_dist:ibe-b_dist])
              
    for b_dist in range(1,4,1):
        fls0[kts:ktf, jbe-1-b_dist, ibs+b_dist:ibe-b_dist] = field_bdy_ye[kts:ktf, b_dist, ibs+b_dist:ibe-b_dist] + \
              dtbc * field_bdy_tend_ye[kts:ktf, b_dist, ibs+b_dist:ibe-b_dist] - field[kts:ktf, jbe-1-b_dist, ibs+b_dist:ibe-b_dist]
        fls1[kts:ktf, jbe-1-b_dist, ibs+b_dist:ibe-b_dist] = field_bdy_ye[kts:ktf, b_dist, ibs+b_dist-1:ibe-b_dist-1] + \
              dtbc * field_bdy_tend_ye[kts:ktf, b_dist, ibs+b_dist-1:ibe-b_dist-1] - field[kts:ktf, jbe-1-b_dist, ibs+b_dist-1:ibe-b_dist-1]
        fls2[kts:ktf, jbe-1-b_dist, ibs+b_dist:ibe-b_dist] = field_bdy_ye[kts:ktf, b_dist, ibs+b_dist+1:ibe-b_dist+1] + \
              dtbc * field_bdy_tend_ye[kts:ktf, b_dist, ibs+b_dist+1:ibe-b_dist+1] - field[kts:ktf, jbe-1-b_dist, ibs+b_dist+1:ibe-b_dist+1]
        fls3[kts:ktf, jbe-1-b_dist, ibs+b_dist:ibe-b_dist] = field_bdy_ye[kts:ktf, b_dist-1, ibs+b_dist:ibe-b_dist] + \
              dtbc * field_bdy_tend_ye[kts:ktf, b_dist-1, ibs+b_dist:ibe-b_dist] - field[kts:ktf, jbe-1-b_dist+1, ibs+b_dist:ibe-b_dist]
        fls4[kts:ktf, jbe-1-b_dist, ibs+b_dist:ibe-b_dist] = field_bdy_ye[kts:ktf, b_dist+1, ibs+b_dist:ibe-b_dist] + \
              dtbc * field_bdy_tend_ye[kts:ktf, b_dist+1, ibs+b_dist:ibe-b_dist] - field[kts:ktf, jbe-1-b_dist-1, ibs+b_dist:ibe-b_dist]
        field_tend[kts:ktf, jbe-1-b_dist, ibs+b_dist:ibe-b_dist] = field_tend[kts:ktf, jbe-1-b_dist, ibs+b_dist:ibe-b_dist] + \
              fcx[b_dist] * fls0[kts:ktf, jbe-1-b_dist, ibs+b_dist:ibe-b_dist] - gcx[b_dist] * \
              (fls1[kts:ktf, jbe-1-b_dist, ibs+b_dist:ibe-b_dist] + fls2[kts:ktf, jbe-1-b_dist, ibs+b_dist:ibe-b_dist] + 
               fls3[kts:ktf, jbe-1-b_dist, ibs+b_dist:ibe-b_dist] + fls4[kts:ktf, jbe-1-b_dist, ibs+b_dist:ibe-b_dist] - 
               4. * fls0[kts:ktf, jbe-1-b_dist, ibs+b_dist:ibe-b_dist])
              
    # x boundary
    for b_dist in range(1,4,1):
        fls0[kts:ktf, jbs+b_dist+1:jbe-b_dist-1, ibs+b_dist] = field_bdy_xs[kts:ktf, jbs+b_dist+1:jbe-b_dist-1, b_dist] + \
              dtbc * field_bdy_tend_xs[kts:ktf, jbs+b_dist+1:jbe-b_dist-1, b_dist] - field[kts:ktf, jbs+b_dist+1:jbe-b_dist-1, ibs+b_dist]
        fls1[kts:ktf, jbs+b_dist+1:jbe-b_dist-1, ibs+b_dist] = field_bdy_xs[kts:ktf, jbs+b_dist+1-1:jbe-b_dist-1-1, b_dist] + \
              dtbc * field_bdy_tend_xs[kts:ktf, jbs+b_dist+1-1:jbe-b_dist-1-1, b_dist] - field[kts:ktf, jbs+b_dist+1-1:jbe-b_dist-1-1, ibs+b_dist]
        fls2[kts:ktf, jbs+b_dist+1:jbe-b_dist-1, ibs+b_dist] = field_bdy_xs[kts:ktf, jbs+b_dist+1+1:jbe-b_dist-1+1, b_dist] + \
              dtbc * field_bdy_tend_xs[kts:ktf, jbs+b_dist+1+1:jbe-b_dist-1+1, b_dist] - field[kts:ktf, jbs+b_dist+1+1:jbe-b_dist-1+1, ibs+b_dist]
        fls3[kts:ktf, jbs+b_dist+1:jbe-b_dist-1, ibs+b_dist] = field_bdy_xs[kts:ktf, jbs+b_dist+1:jbe-b_dist-1, b_dist-1] + \
              dtbc * field_bdy_tend_xs[kts:ktf, jbs+b_dist+1:jbe-b_dist-1, b_dist-1] - field[kts:ktf, jbs+b_dist+1:jbe-b_dist-1, ibs+b_dist-1]
        fls4[kts:ktf, jbs+b_dist+1:jbe-b_dist-1, ibs+b_dist] = field_bdy_xs[kts:ktf, jbs+b_dist+1:jbe-b_dist-1, b_dist+1] + \
              dtbc * field_bdy_tend_xs[kts:ktf, jbs+b_dist+1:jbe-b_dist-1, b_dist+1] - field[kts:ktf, jbs+b_dist+1:jbe-b_dist-1, ibs+b_dist+1]
        field_tend[kts:ktf, jbs+b_dist+1:jbe-b_dist-1, ibs+b_dist] = field_tend[kts:ktf, jbs+b_dist+1:jbe-b_dist-1, ibs+b_dist] + \
              fcx[b_dist] * fls0[kts:ktf, jbs+b_dist+1:jbe-b_dist-1, ibs+b_dist] - gcx[b_dist] * \
              (fls1[kts:ktf, jbs+b_dist+1:jbe-b_dist-1, ibs+b_dist] + fls2[kts:ktf, jbs+b_dist+1:jbe-b_dist-1, ibs+b_dist] + 
               fls3[kts:ktf, jbs+b_dist+1:jbe-b_dist-1, ibs+b_dist] + fls4[kts:ktf, jbs+b_dist+1:jbe-b_dist-1, ibs+b_dist] - 
               4. * fls0[kts:ktf, jbs+b_dist+1:jbe-b_dist-1, ibs+b_dist])
              
    for b_dist in range(1,4,1):
        fls0[kts:ktf, jbs+b_dist+1:jbe-b_dist-1, ibe-1-b_dist] = field_bdy_xe[kts:ktf, jbs+b_dist+1:jbe-b_dist-1, b_dist] + \
              dtbc * field_bdy_tend_xe[kts:ktf, jbs+b_dist+1:jbe-b_dist-1, b_dist] - field[kts:ktf, jbs+b_dist+1:jbe-b_dist-1, ibe-1-b_dist]
        fls1[kts:ktf, jbs+b_dist+1:jbe-b_dist-1, ibe-1-b_dist] = field_bdy_xe[kts:ktf, jbs+b_dist+1-1:jbe-b_dist-1-1, b_dist] + \
              dtbc * field_bdy_tend_xe[kts:ktf, jbs+b_dist+1-1:jbe-b_dist-1-1, b_dist] - field[kts:ktf, jbs+b_dist+1-1:jbe-b_dist-1-1, ibe-1-b_dist]
        fls2[kts:ktf, jbs+b_dist+1:jbe-b_dist-1, ibe-1-b_dist] = field_bdy_xe[kts:ktf, jbs+b_dist+1+1:jbe-b_dist-1+1, b_dist] + \
              dtbc * field_bdy_tend_xe[kts:ktf, jbs+b_dist+1+1:jbe-b_dist-1+1, b_dist] - field[kts:ktf, jbs+b_dist+1+1:jbe-b_dist-1+1, ibe-1-b_dist]
        fls3[kts:ktf, jbs+b_dist+1:jbe-b_dist-1, ibe-1-b_dist] = field_bdy_xe[kts:ktf, jbs+b_dist+1:jbe-b_dist-1, b_dist-1] + \
              dtbc * field_bdy_tend_xe[kts:ktf, jbs+b_dist+1:jbe-b_dist-1, b_dist-1] - field[kts:ktf, jbs+b_dist+1:jbe-b_dist-1, ibe-1-b_dist+1]
        fls4[kts:ktf, jbs+b_dist+1:jbe-b_dist-1, ibe-1-b_dist] = field_bdy_xe[kts:ktf, jbs+b_dist+1:jbe-b_dist-1, b_dist+1] + \
              dtbc * field_bdy_tend_xe[kts:ktf, jbs+b_dist+1:jbe-b_dist-1, b_dist+1] - field[kts:ktf, jbs+b_dist+1:jbe-b_dist-1, ibe-1-b_dist-1]
        field_tend[kts:ktf, jbs+b_dist+1:jbe-b_dist-1, ibe-1-b_dist] = field_tend[kts:ktf, jbs+b_dist+1:jbe-b_dist-1, ibe-1-b_dist] + \
              fcx[b_dist] * fls0[kts:ktf, jbs+b_dist+1:jbe-b_dist-1, ibe-1-b_dist] - gcx[b_dist] * \
              (fls1[kts:ktf, jbs+b_dist+1:jbe-b_dist-1, ibe-1-b_dist] + fls2[kts:ktf, jbs+b_dist+1:jbe-b_dist-1, ibe-1-b_dist] + 
               fls3[kts:ktf, jbs+b_dist+1:jbe-b_dist-1, ibe-1-b_dist] + fls4[kts:ktf, jbs+b_dist+1:jbe-b_dist-1, ibe-1-b_dist] - 
               4. * fls0[kts:ktf, jbs+b_dist+1:jbe-b_dist-1, ibe-1-b_dist])
      
    return field_tend

# Specified boundary tendencies for dry variables.
def spec_bdy_dry(ru_tend, rv_tend, ph_tend, t_tend,   \
                 rw_tend, mu_tend,                    \
                 u_bxs,u_bxe,u_bys,u_bye,             \
                 v_bxs,v_bxe,v_bys,v_bye,             \
                 ph_bxs,ph_bxe,ph_bys,ph_bye,         \
                 t_bxs,t_bxe,t_bys,t_bye,             \
                 w_bxs,w_bxe,w_bys,w_bye,             \
                 mu_bxs,mu_bxe,mu_bys,mu_bye,         \
                 u_btxs,u_btxe,u_btys,u_btye,         \
                 v_btxs,v_btxe,v_btys,v_btye,         \
                 ph_btxs,ph_btxe,ph_btys,ph_btye,     \
                 t_btxs,t_btxe,t_btys,t_btye,         \
                 w_btxs,w_btxe,w_btys,w_btye,         \
                 mu_btxs,mu_btxe,mu_btys,mu_btye,     \
                 spec_bdy_width, spec_zone,           \
                 ids,ide, jds,jde, kds,kde,  \
                 ims,ime, jms,jme, kms,kme,  \
                 ips,ipe, jps,jpe, kps,kpe,  \
                 its, ite, jts, jte, kts, kte):
    
    
    return

# Add specified boundary tendencies to a field.
def spec_bdytend(field_tend,                           \
                 field_bdy_xs, field_bdy_xe,           \
                 field_bdy_ys, field_bdy_ye,           \
                 field_bdy_tend_xs, field_bdy_tend_xe, \
                 field_bdy_tend_ys, field_bdy_tend_ye, \
                 variable_in,               \
                 spec_bdy_width, spec_zone, \
                 ids,ide, jds,jde, kds,kde,  \
                 ims,ime, jms,jme, kms,kme,  \
                 ips,ipe, jps,jpe, kps,kpe,  \
                 its,ite, jts,jte, kts,kte):
    variable = variable_in
    
    ibs = ids
    ibe = ide-1
    itf = min(ite,ide-1)
    jbs = jds
    jbe = jde-1
    jtf = min(jte,jde-1)
    ktf = kde-1
    if variable == 'u':
        ibe = ide
    if variable == 'u':
        itf = min(ite,ide)
    if variable == 'v':
        jbe = jde
    if variable == 'v':
        jtf = min(jte,jde)
    if variable == 'm':
        ktf = kte
    if variable == 'h':
        ktf = kte
    field_tend[kts:ktf, jbs, ibs:ibe] = field_bdy_tend_ys[kts:ktf, 0, ibs:ibe]
    field_tend[kts:ktf, jbe-1, ibs:ibe] = field_bdy_tend_ye[kts:ktf, 0, ibs:ibe]
    field_tend[kts:ktf, jbs:jbe, ibs] = field_bdy_tend_xs[kts:ktf, jbs:jbe, 0]
    field_tend[kts:ktf, jbs:jbe, ibe-1] = field_bdy_tend_xe[kts:ktf, jbs:jbe, 0]
    
    return field_tend

# Update a field at the specified boundary zones.
def spec_bdyupdate(field,
                   field_tend, dt,             \
                   variable,                \
                   spec_zone,                  \
                   ids,ide, jds,jde, kds,kde,  \
                   ims,ime, jms,jme, kms,kme,  \
                   ips,ipe, jps,jpe, kps,kpe,  \
                   its,ite, jts,jte, kts,kte):
    if (variable == 'U'):
        variable = 'u'
    if (variable == 'V'):
        variable = 'v'
    if (variable == 'M'):
        variable = 'm'
    if (variable == 'H'):
        variable = 'h'
        
    ibs = ids
    ibe = ide-1
    itf = min(ite,ide-1)
    jbs = jds
    jbe = jde-1
    jtf = min(jte,jde-1)
    ktf = kde-1

    if (variable == 'u'):
        ibe = ide
    if (variable == 'u'):
        itf = min(ite,ide)
    if (variable == 'v'):
        jbe = jde
    if (variable == 'v'):
        jtf = min(jte,jde)
    if (variable == 'm'):
        ktf = kte
    if (variable == 'h'):
        ktf = kte
        
    if kds == 0 and kde == 0:
        field = field.unsqueeze(0)
        field_tend = field_tend.unsqueeze(0)
        ktf = 1
        
    # Y-start boundary
    field[kts:ktf, jts, its:itf] = field[kts:ktf, jts, its:itf] + dt * field_tend[kts:ktf, jts, its:itf]
    # Y-end boundary
    field[kts:ktf, jtf-1, its:itf] = field[kts:ktf, jtf-1, its:itf] + dt * field_tend[kts:ktf, jtf-1, its:itf]
    # X-start boundary
    field[kts:ktf, jbs+1:jbe-1, its] = field[kts:ktf, jbs+1:jbe-1, its] + dt * field_tend[kts:ktf, jbs+1:jbe-1, its]
    # X-end boundary
    field[kts:ktf, jbs+1:jbe-1, itf-1] = field[kts:ktf, jbs+1:jbe-1, itf-1] + dt * field_tend[kts:ktf, jbs+1:jbe-1, itf-1]
    
    if kds == 0 and kde ==0:
        field = field.squeeze(0)
        field_tend = field_tend.squeeze(0)
    
    return field

# Update geopotential at the specified boundary zones.
def spec_bdyupdate_ph(ph_save, field,           \
                      field_tend, mu_tend, muts,  \
                      c1, c2, dt,                 \
                      variable_in,                \
                      spec_zone,                  \
                      ids,ide, jds,jde, kds,kde,  \
                      ims,ime, jms,jme, kms,kme,  \
                      ips,ipe, jps,jpe, kps,kpe,  \
                      its,ite, jts,jte, kts,kte):
    variable = variable_in
    if variable == 'U':
        variable = 'u'
    if variable == 'V':
        variable = 'v'
    if variable == 'M':
        variable = 'm'
    if variable == 'H':
        variable = 'h'
    
    ibs = ids
    ibe = ide-1
    itf = min(ite,ide-1)
    jbs = jds
    jbe = jde-1
    jtf = min(jte,jde-1)
    ktf = kde-1
    
    if variable == 'u':
        ibe = ide
    if variable == 'u':
        itf = min(ite,ide)
    if variable == 'v':
        jbe = jde
    if variable == 'v':
        jtf = min(jte,jde)
    if variable == 'm':
        ktf = kte
    if variable == 'h':
        ktf = kte
    
    mu_old = torch.zeros((nyall,nxall)).to(device)
    # y start
    mu_old[jts, ibs:ibe] = muts[jts, ibs:ibe] - dt * mu_tend[jts, ibs:ibe]
    mu_old_e = mu_old.repeat(nzall,1,1)
    muts_e = muts.repeat(nzall,1,1)
    field[kts:ktf, jts, ibs:ibe] = field[kts:ktf, jts, ibs:ibe] * mu_old_e[kts:ktf, jts, ibs:ibe] / \
            muts_e[kts:ktf, jts, ibs:ibe] + dt * field_tend[kts:ktf, jts, ibs:ibe] / \
            muts_e[kts:ktf, jts, ibs:ibe] + ph_save[kts:ktf, jts, ibs:ibe] * \
            (mu_old_e[kts:ktf, jts, ibs:ibe] / muts_e[kts:ktf, jts, ibs:ibe] - 1.)
    # y end
    mu_old[jbe-1, ibs:ibe] = muts[jbe-1, ibs:ibe] - dt * mu_tend[jbe-1, ibs:ibe]
    mu_old_e = mu_old.repeat(nzall,1,1)
    field[kts:ktf, jbe-1, ibs:ibe] = field[kts:ktf, jbe-1, ibs:ibe] * mu_old_e[kts:ktf, jbe-1, ibs:ibe] / \
            muts_e[kts:ktf, jbe-1, ibs:ibe] + dt * field_tend[kts:ktf, jbe-1, ibs:ibe] / \
            muts_e[kts:ktf, jbe-1, ibs:ibe] + ph_save[kts:ktf, jbe-1, ibs:ibe] * \
            (mu_old_e[kts:ktf, jbe-1, ibs:ibe] / muts_e[kts:ktf, jbe-1, ibs:ibe] - 1.)
    # x start
    mu_old[jbs+1:jbe-1, ibs] = muts[jbs+1:jbe-1, ibs] - dt * mu_tend[jbs+1:jbe-1, ibs]
    mu_old_e = mu_old.repeat(nzall,1,1)
    field[kts:ktf, jbs+1:jbe-1, ibs] = field[kts:ktf, jbs+1:jbe-1, ibs] * mu_old_e[kts:ktf, jbs+1:jbe-1, ibs] / \
            muts_e[kts:ktf, jbs+1:jbe-1, ibs] + dt * field_tend[kts:ktf, jbs+1:jbe-1, ibs] / \
            muts_e[kts:ktf, jbs+1:jbe-1, ibs] + ph_save[kts:ktf, jbs+1:jbe-1, ibs] * \
            (mu_old_e[kts:ktf, jbs+1:jbe-1, ibs] / muts_e[kts:ktf, jbs+1:jbe-1, ibs] - 1.)
    # x end
    mu_old[jbs+1:jbe-1, ibe-1] = muts[jbs+1:jbe-1, ibe-1] - dt * mu_tend[jbs+1:jbe-1, ibe-1]
    mu_old_e = mu_old.repeat(nzall,1,1)
    field[kts:ktf, jbs+1:jbe-1, ibe-1] = field[kts:ktf, jbs+1:jbe-1, ibe-1] * mu_old_e[kts:ktf, jbs+1:jbe-1, ibe-1] / \
            muts_e[kts:ktf, jbs+1:jbe-1, ibe-1] + dt * field_tend[kts:ktf, jbs+1:jbe-1, ibe-1] / \
            muts_e[kts:ktf, jbs+1:jbe-1, ibe-1] + ph_save[kts:ktf, jbs+1:jbe-1, ibe-1] * \
            (mu_old_e[kts:ktf, jbs+1:jbe-1, ibe-1] / muts_e[kts:ktf, jbs+1:jbe-1, ibe-1] - 1.)
        
    return field

# Zero-gradient boundary condition for a field.
def zero_grad_bdy(field,                      \
                  variable_in,                \
                  spec_zone,                  \
                  ids,ide, jds,jde, kds,kde,  \
                  ims,ime, jms,jme, kms,kme,  \
                  ips,ipe, jps,jpe, kps,kpe,  \
                  its,ite, jts,jte, kts,kte):
    variable = variable_in
    if variable == 'U':
        variable = 'u'
    if variable == 'V':
        variable = 'v'
    
    ibs = ids
    ibe = ide-1
    itf = min(ite,ide-1)
    jbs = jds
    jbe = jde-1
    jtf = min(jte,jde-1)
    ktf = kde-1
    
    if variable == 'u':
        ibe = ide
    if variable == 'u':
        itf = min(ite,ide)
    if variable == 'v':
        jbe = jde
    if variable == 'v':
        jtf = min(jte,jde)
    if variable == 'w':
        ktf = kde
    # y start
    field[kts:ktf, jbs, ibs] = field[kts:ktf, jbs+1, ibs+1]
    field[kts:ktf, jbs, ibe-1] = field[kts:ktf, jbs+1, ibe-2]
    field[kts:ktf, jbs, ibs+1:ibe-1] = field[kts:ktf, jbs+1, ibs+1:ibe-1]
    # y end
    field[kts:ktf, jbe-1, ibs] = field[kts:ktf, jbe-2, ibs+1]
    field[kts:ktf, jbe-1, ibe-1] = field[kts:ktf, jbe-2, ibe-2]
    field[kts:ktf, jbe-1, ibs+1:ibe-1] = field[kts:ktf, jbe-2, ibs+1:ibe-1]
    # x start
    field[kts:ktf, jbs+1:jbe-1, ibs] = field[kts:ktf, jbs+1:jbe-1, ibs+1]
    # x end
    field[kts:ktf, jbs+1:jbe-1, ibe-1] = field[kts:ktf, jbs+1:jbe-1, ibe-2]
        
    return field

# Mass-weight a boundary tendency.
def mass_weight(field , mut, rfield , c1 , c2 , \
                ids,ide, jds,jde, kds,kde,      \
                ims,ime, jms,jme, kms,kme,      \
                irs,ire, jrs,jre, krs,kre,      \
                its,ite, jts,jte, kts,kte ):
    mut_e = mut.repeat(nzall,1,1)
    rfield[kts:kte, jts:jte, its:ite] = field[kts:kte, jts:jte, its:ite] * mut_e[kts:kte, jts:jte, its:ite]
    return rfield

# Relaxation boundary for moisture / scalar fields.
def relax_bdy_scalar(scalar_tend,                \
                     scalar, mu, c1h, c2h,       \
                     scalar_bxs,scalar_bxe,scalar_bys,scalar_bye, \
                     scalar_btxs,scalar_btxe,scalar_btys,scalar_btye, \
                     spec_bdy_width, spec_zone, relax_zone,       \
                     dtbc, fcx, gcx,             \
                     ids,ide, jds,jde, kds,kde,  \
                     ims,ime, jms,jme, kms,kme,  \
                     ips,ipe, jps,jpe, kps,kpe,  \
                     its, ite, jts, jte, kts, kte):
    i_start = max(its-1, ids)
    i_end = min(ite+1, ide-1)
    j_start = max(jts-1, jds)
    j_end = min(jte+1, jde-1)
    
    rscalar = torch.zeros((nzall,nyall,nxall)).to(device)
    
    rscalar = mass_weight(scalar , mu , rscalar, c1h, c2h, \
                          ids,ide, jds,jde, kds,kde,   \
                          ims,ime, jms,jme, kms,kme,   \
                          ims,ime, jms,jme, kms,kme,   \
                          i_start,i_end, j_start,j_end, kts,kte-1)
    scalar_tend = relax_bdytend(rscalar, scalar_tend,             \
                               scalar_bxs,scalar_bxe,scalar_bys,scalar_bye, \
                               scalar_btxs,scalar_btxe,scalar_btys,scalar_btye,       \
                               'q',                                   \
                               spec_bdy_width, spec_zone, relax_zone, \
                               dtbc, fcx, gcx,             \
                               ids,ide, jds,jde, kds,kde,  \
                               ims,ime, jms,jme, kms,kme,  \
                               ips,ipe, jps,jpe, kps,kpe,  \
                               its,ite, jts,jte, kts,kte)
    
    return scalar_tend

# Specified boundary for moisture / scalar fields.
def spec_bdy_scalar(scalar_tend,    \
                    scalar_bxs,scalar_bxe,scalar_bys,scalar_bye,  \
                    scalar_btxs,scalar_btxe,scalar_btys,scalar_btye, \
                    spec_bdy_width, spec_zone,                   \
                    ids,ide, jds,jde, kds,kde,  \
                    ims,ime, jms,jme, kms,kme,  \
                    ips,ipe, jps,jpe, kps,kpe,  \
                    its, ite, jts, jte, kts, kte):
    scalar_tend = spec_bdytend(scalar_tend,                \
                               scalar_bxs,scalar_bxe,scalar_bys,scalar_bye, \
                               scalar_btxs,scalar_btxe,scalar_btys,scalar_btye,    \
                               'q',                       \
                               spec_bdy_width, spec_zone, \
                               ids,ide, jds,jde, kds,kde,  \
                               ims,ime, jms,jme, kms,kme,  \
                               ips,ipe, jps,jpe, kps,kpe,  \
                               its,ite, jts,jte, kts,kte)
    
    return scalar_tend

# Flow-dependent boundary update.
def flow_dep_bdy(field,                        \
                 u, v,                         \
                 spec_zone,                    \
                 ids,ide, jds,jde, kds,kde,    \
                 ims,ime, jms,jme, kms,kme,    \
                 ips,ipe, jps,jpe, kps,kpe,    \
                 its,ite, jts,jte, kts,kte ):
    ibs = ids
    ibe = ide-1
    itf = min(ite,ide-1)
    jbs = jds
    jbe = jde-1
    jtf = min(jte,jde-1)
    ktf = kde-1
    
    condition = v[kts:ktf, jts, its:itf] < 0.
    field[kts:ktf, jts, its:itf] = torch.where(condition, field[kts:ktf, jts+1, its:itf], torch.tensor(0.))
    field[kts:ktf, jts, its] = field[kts:ktf, jts+1, its+1] + 0.0
    field[kts:ktf, jts, itf-1] = field[kts:ktf, jts+1, itf-2] + 0.0
    
    condition = v[kts:ktf, jtf, its:itf] > 0.
    field[kts:ktf, jtf-1, its:itf] = torch.where(condition, field[kts:ktf, jtf-2, its:itf],  torch.tensor(0.))
    field[kts:ktf, jtf-1, its] = field[kts:ktf, jtf-2, its+1] + 0.0
    field[kts:ktf, jtf-1, itf-1] = field[kts:ktf, jtf-2, itf-2] + 0.0
    
    condition = u[kts:ktf, jts:jtf, its] < 0.
    field[kts:ktf, jts:jtf, its] = torch.where(condition, field[kts:ktf, jts:jtf, its+1], torch.tensor(0.))
    field[kts:ktf, jts, its] = field[kts:ktf, jts+1, its+1] + 0.0
    field[kts:ktf, jtf-1, its] = field[kts:ktf, jtf-2, its+1] + 0.0
    
    condition = u[kts:ktf, jts:jtf, itf] > 0.
    field[kts:ktf, jts:jtf, itf-1] = torch.where(condition, field[kts:ktf, jts:jtf, itf-2], torch.tensor(0.))
    field[kts:ktf, jts, itf-1] = field[kts:ktf, jts+1, itf-2] + 0.0
    field[kts:ktf, jtf-1, itf-1] = field[kts:ktf, jtf-2, itf-2] + 0.0
    
    return field

# Final specified-boundary update at the end of a step.
def spec_bdy_final(field, mu, c1, c2, msf,                \
                   field_bdy_xs, field_bdy_xe,            \
                   field_bdy_ys, field_bdy_ye,            \
                   field_bdy_tend_xs, field_bdy_tend_xe,  \
                   field_bdy_tend_ys, field_bdy_tend_ye,  \
                   variable_in,                           \
                   spec_bdy_width, spec_zone,             \
                   dtbc,                       \
                   ids,ide, jds,jde, kds,kde,  \
                   ims,ime, jms,jme, kms,kme,  \
                   ips,ipe, jps,jpe, kps,kpe,  \
                   its,ite, jts,jte, kts,kte):
    
    variable = variable_in
    
    if (variable == 'U'):
        variable = 'u'
    if (variable == 'V'):
        variable = 'v'
    if (variable == 'W'):
        variable = 'w'
    if (variable == 'M'):
        variable = 'm'
    if (variable == 'T'):
        variable = 't'
    if (variable == 'H'):
        variable = 'h'
    ibs = ids
    ibe = ide-1
    itf = min(ite,ide-1)
    jbs = jds
    jbe = jde-1
    jtf = min(jte,jde-1)
    ktf = kde-1
    if (variable == 'u'):
        ibe = ide
    if (variable == 'u'):
        itf = min(ite,ide)
    if (variable == 'v'):
        jbe = jde
    if (variable == 'v'):
        jtf = min(jte,jde)
    if (variable == 'm'):
        ktf = kde
    if (variable == 'h'):
        ktf = kde
    if (variable == 'w'):
        ktf = kde
        
    msfcouple = False
    mucouple = True
    if (variable == 'u' or variable == 'v' or variable == 'w'):
        msfcouple = True
    if (variable == 'm' ):
        mucouple = False
    xmsf = torch.ones((nzfull,nyfull,nxfull)).to(device)
    xmu = torch.ones((nzfull,nyfull,nxfull)).to(device)
    # y-start boundary
    if (msfcouple):
        xmsf = msf.repeat(nzall,1,1)
    if (mucouple):
        xmu = mu.repeat(nzall,1,1)
    field[kts:ktf, jts, its:ibe] = xmsf[kts:ktf, jts, its:ibe] * (field_bdy_ys[kts:ktf, 0, its:ibe] + \
            dtbc * field_bdy_tend_ys[kts:ktf, 0, its:ibe]) / xmu[kts:ktf, jts, its:ibe]
    # y-end boundary
    field[kts:ktf, jtf-1, its:ibe] = xmsf[kts:ktf, jtf-1, its:ibe] * (field_bdy_ye[kts:ktf, 0, its:ibe] + \
            dtbc * field_bdy_tend_ye[kts:ktf, 0, its:ibe]) / xmu[kts:ktf, jtf-1, its:ibe]
    # x-start boundary
    field[kts:ktf, jts:jbe, its] = xmsf[kts:ktf, jts:jbe, its] * (field_bdy_xs[kts:ktf, jts:jbe, 0] + \
            dtbc * field_bdy_tend_xs[kts:ktf, jts:jbe, 0]) / xmu[kts:ktf, jts:jbe, its]
    # x-end boudary
    field[kts:ktf, jts:jbe, itf-1] = xmsf[kts:ktf, jts:jbe, itf-1] * (field_bdy_xe[kts:ktf, jts:jbe, 0] + \
            dtbc * field_bdy_tend_xe[kts:ktf, jts:jbe, 0]) / xmu[kts:ktf, jts:jbe, itf-1]
        
    return field
