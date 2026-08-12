"""PyWRF solver: time integration of the WRF core.

This module runs one full model integration: load the WRF input / boundary /
reference NetCDF files, allocate the model state, and advance it through the
RK3 time loop with WSM6 microphysics and lateral boundary updates.

The heavy numerical kernels live in :mod:`pywrf.wrf_dynamics`,
:mod:`pywrf.wrf_physics` and :mod:`pywrf.wrf_boundary`; all run parameters and
constants come from :mod:`pywrf.config_params`.
"""

import os
import sys
import numpy as np
import torch
import xarray as xr

from pywrf.wrf_physics import *
from pywrf.wrf_dynamics import *
from pywrf.wrf_boundary import *

# --- Input data paths ----------------------------------------------------
# The real 9 km case data ships with the repo in data/ (see data/README.md).
# Override with PYWRF_DATA_DIR / PYWRF_RUN_NAME, or edit below.
DATA_DIR   = os.environ.get("PYWRF_DATA_DIR", "data")
RUN_NAME   = os.environ.get("PYWRF_RUN_NAME", "2024020106")
WRF_INPUT  = os.path.join(DATA_DIR, f"wrf_inout_step3_{RUN_NAME}")
WRF_BDY    = os.path.join(DATA_DIR, f"wrfbdy_d01_{RUN_NAME}")
WRF_OUTPUT = os.path.join(DATA_DIR, f"wrfout_d01_{RUN_NAME}")


class WrfSolver:
    """Orchestrate one full WRF run.

    Thin wrapper over the solver: create a :class:`WrfSolver` and call
    :meth:`WrfSolver.solve`. All run parameters and physical constants
    live in :mod:`pywrf.config_params`.
    """

    def _load_datasets(self):
        """Open the three WRF NetCDF datasets (input / boundary / reference)."""
        wrfinput = xr.open_dataset(WRF_INPUT)
        wrfbdy = xr.open_dataset(WRF_BDY)
        wrfoutput = xr.open_dataset(WRF_OUTPUT)
        return wrfinput, wrfbdy, wrfoutput

    def solve(self):
        """Integrate the 9 km WRF case.

        Reads the WRF input / boundary / reference NetCDF files, allocates the
        model state and advances it through the RK3 time loop with WSM6
        microphysics and lateral boundary updates.

        Returns the final key fields (useful for verification / downstream use).
        """
        #### load input/bdy ####
        wrfinput, wrfbdy, wrfoutput = self._load_datasets()

        # --- input fields: extract every variable used by the core ---
        u_input = wrfinput["U"]
        v_input = wrfinput["V"]
        w_input = wrfoutput["W"]
        mu_input = wrfinput["MU"]
        mub_input = wrfinput["MUB"]

        msfu_input = wrfinput["MAPFAC_U"]
        msfv_input = wrfinput["MAPFAC_V"]
        msfvx_inv_input = wrfinput["MF_VX_INV"]
        msft_input = wrfinput["MAPFAC_M"]
        msftx_input = wrfinput["MAPFAC_MX"]
        msfty_input = wrfinput["MAPFAC_MY"]
        msfux_input = wrfinput["MAPFAC_UX"]
        msfuy_input = wrfinput["MAPFAC_UY"]
        msfvx_input = wrfinput["MAPFAC_VX"]
        msfvy_input = wrfinput["MAPFAC_VY"]

        qv_input = wrfinput["QVAPOR"]
        qc_input = wrfinput["QCLOUD"]
        qr_input = wrfinput["QRAIN"]
        qi_input = wrfinput["QICE"]
        qs_input = wrfinput["QSNOW"]
        qg_input = wrfinput["QGRAUP"]

        pb_input = wrfinput["PB"]
        p_input = wrfinput["P"]

        t_init_input = wrfinput["T_INIT"]

        phb_input = wrfinput["PHB"]
        ph_input = wrfinput["PH"]

        rdnw_input = wrfinput["RDNW"]
        rdn_input = wrfinput["RDN"]
        c1h_input = wrfinput["C1H"]
        c2h_input = wrfinput["C2H"]
        c1f_input = wrfinput["C1F"]
        c2f_input = wrfinput["C2F"]
        c3h_input = wrfinput["C3H"]
        c4h_input = wrfinput["C4H"]
        c3f_input = wrfinput["C3F"]
        c4f_input = wrfinput["C4F"]

        t_input = wrfinput["T"]

        fnm_input = wrfinput["FNM"]
        fnp_input = wrfinput["FNP"]

        dn_input = wrfinput["DN"]
        dnw_input = wrfinput["DNW"]

        cf1_input = wrfinput["CF1"]
        cf2_input = wrfinput["CF2"]
        cf3_input = wrfinput["CF3"]

        cfn_input = wrfinput["CFN"]
        cfn1_input = wrfinput["CFN1"]

        f_input = wrfinput["F"]
        e_input = wrfinput["E"]
        sina_input = wrfinput["SINALPHA"]
        cosa_input = wrfinput["COSALPHA"]
        clat_input = wrfinput["CLAT"]

        znu_input = wrfinput["ZNU"]
        znw_input = wrfinput["ZNW"]

        ht_input = wrfinput["HGT"]

        #qni_input = wrfinput["QNICE"]
        #qnr_input = wrfinput["QNRAIN"]

        u_base_input = wrfinput["U_BASE"]
        v_base_input = wrfinput["V_BASE"]
        qv_base_input = wrfinput["QV_BASE"]



        ### (2) Allocate model state and workspace (once, reused for all steps) ---
        # define full domain
        u = torch.zeros((nzall,nyall,nxall)).to(device)   # 41,610,610
        v = torch.zeros((nzall,nyall,nxall)).to(device)
        w = torch.zeros((nzall,nyall,nxall)).to(device)
        mu = torch.zeros((nyall,nxall)).to(device)              # 610,610
        mub = torch.zeros((nyall,nxall)).to(device)

        msfu = torch.zeros((nyall,nxall)).to(device)
        msfv = torch.zeros((nyall,nxall)).to(device)
        msfvx_inv = torch.zeros((nyall,nxall)).to(device)
        msft = torch.zeros((nyall,nxall)).to(device)
        msftx = torch.zeros((nyall,nxall)).to(device)
        msfty = torch.zeros((nyall,nxall)).to(device)
        msfux = torch.zeros((nyall,nxall)).to(device)
        msfuy = torch.zeros((nyall,nxall)).to(device)
        msfvx = torch.zeros((nyall,nxall)).to(device)
        msfvy = torch.zeros((nyall,nxall)).to(device)

        moist = torch.zeros((7, nzall,nyall,nxall)).to(device)
        scalar = torch.zeros((3, nzall,nyall,nxall)).to(device)
        pb = torch.zeros((nzall,nyall,nxall)).to(device)
        p = torch.zeros((nzall,nyall,nxall)).to(device)
        t_init = torch.zeros((nzall,nyall,nxall)).to(device)
        phb = torch.zeros((nzall,nyall,nxall)).to(device)
        ph = torch.zeros((nzall,nyall,nxall)).to(device)
        rdnw = torch.zeros(nzall).to(device)
        rdn = torch.zeros(nzall).to(device)
        c1h = torch.zeros(nzall).to(device)
        c2h = torch.zeros(nzall).to(device)
        c1f = torch.zeros(nzall).to(device)
        c2f = torch.zeros(nzall).to(device)
        c3h = torch.zeros(nzall).to(device)
        c4h = torch.zeros(nzall).to(device)
        c3f = torch.zeros(nzall).to(device)
        c4f = torch.zeros(nzall).to(device)
        t = torch.zeros((nzall,nyall,nxall)).to(device)
        fnm = torch.zeros(nzall).to(device)
        fnp = torch.zeros(nzall).to(device)
        znu = torch.zeros(nzall).to(device)
        znw = torch.zeros(nzall).to(device)
        dn = torch.zeros(nzall).to(device)
        dnw = torch.zeros(nzall).to(device)
        u_base = torch.zeros((nzall)).to(device)
        v_base = torch.zeros((nzall)).to(device)
        qv_base = torch.zeros((nzall)).to(device)

        cf1 = torch.tensor(0).to(device)
        cf2 = torch.tensor(0).to(device)
        cf3 = torch.tensor(0).to(device)
        cfn = torch.tensor(0).to(device)
        cfn1 = torch.tensor(0).to(device)

        f = torch.zeros((nyall,nxall)).to(device)
        e = torch.zeros((nyall,nxall)).to(device)
        sina = torch.zeros((nyall,nxall)).to(device)
        cosa = torch.zeros((nyall,nxall)).to(device)
        clat = torch.zeros((nyall,nxall)).to(device)

        ht = torch.zeros((nyall,nxall)).to(device)

        rainnc = torch.zeros((nyall,nxall)).to(device)
        rainncv = torch.zeros((nyall,nxall)).to(device)
        snownc = torch.zeros((nyall,nxall)).to(device)
        snowncv = torch.zeros((nyall,nxall)).to(device)
        graupelnc = torch.zeros((nyall,nxall)).to(device)
        graupelncv = torch.zeros((nyall,nxall)).to(device)
        sr = torch.zeros((nyall,nxall)).to(device)
        refl_10cm = torch.zeros((nzall,nyall,nxall)).to(device)

        #qni = torch.zeros((nzall,nyall,nxall)).to(device)
        #qnr = torch.zeros((nzall,nyall,nxall)).to(device)

        # patch data  #bdy width 5

        u[0:nzall-1,5:nyall-6,5:nxall-5] = torch.tensor(u_input[0,:,:,:].data)
        v[0:nzall-1,5:nyall-5,5:nxall-6] = torch.tensor(v_input[0,:,:,:].data)
        w[0:nzall,5:nyall-6,5:nxall-6] = torch.tensor(w_input[0,:,:,:].data)
        mu[5:nyall-6,5:nxall-6] = torch.tensor(mu_input[0,:,:].data)
        mub[5:nyall-6,5:nxall-6] = torch.tensor(mub_input[0,:,:].data)
        msfu[5:nyall-6,5:nxall-5] = torch.tensor(msfu_input[0,:,:].data)
        msfv[5:nyall-5,5:nxall-6] = torch.tensor(msfv_input[0,:,:].data)
        msfvx_inv[5:nyall-5,5:nxall-6] = torch.tensor(msfvx_inv_input[0,:,:].data)
        msft[5:nyall-6,5:nxall-6] = torch.tensor(msft_input[0,:,:].data)
        msftx[5:nyall-6,5:nxall-6] = torch.tensor(msftx_input[0,:,:].data)
        msfty[5:nyall-6,5:nxall-6] = torch.tensor(msfty_input[0,:,:].data)
        msfux[5:nyall-6,5:nxall-5] = torch.tensor(msfux_input[0,:,:].data)
        msfuy[5:nyall-6,5:nxall-5] = torch.tensor(msfuy_input[0,:,:].data)
        msfvx[5:nyall-5,5:nxall-6] = torch.tensor(msfvx_input[0,:,:].data)
        msfvy[5:nyall-5,5:nxall-6] = torch.tensor(msfvy_input[0,:,:].data)

        moist[P_QV,0:nzall-1,5:nyall-6,5:nxall-6] = torch.tensor(qv_input[0,:,:,:].data)
        moist[P_QC,0:nzall-1,5:nyall-6,5:nxall-6] = torch.tensor(qc_input[0,:,:,:].data)
        moist[P_QR,0:nzall-1,5:nyall-6,5:nxall-6] = torch.tensor(qr_input[0,:,:,:].data)
        moist[P_QI,0:nzall-1,5:nyall-6,5:nxall-6] = torch.tensor(qi_input[0,:,:,:].data)
        moist[P_QS,0:nzall-1,5:nyall-6,5:nxall-6] = torch.tensor(qs_input[0,:,:,:].data)
        moist[P_QG,0:nzall-1,5:nyall-6,5:nxall-6] = torch.tensor(qg_input[0,:,:,:].data)

        #scalar[P_QNI,0:nzall-1,5:nyall-6,5:nxall-6] = torch.tensor(qni_input[0,:,:,:].data)
        #scalar[P_QNR,0:nzall-1,5:nyall-6,5:nxall-6] = torch.tensor(qnr_input[0,:,:,:].data)

        pb[0:nzall-1,5:nyall-6,5:nxall-6] = torch.tensor(pb_input[0,:,:,:].data)
        p[0:nzall-1,5:nyall-6,5:nxall-6] = torch.tensor(p_input[0,:,:,:].data)
        t_init[0:nzall-1,5:nyall-6,5:nxall-6] = torch.tensor(t_init_input[0,:,:,:].data)
        phb[0:nzall,5:nyall-6,5:nxall-6] = torch.tensor(phb_input[0,:,:,:].data) ####
        ph[0:nzall,5:nyall-6,5:nxall-6] = torch.tensor(ph_input[0,:,:,:].data)

        rdnw[0:nzall-1] = torch.tensor(rdnw_input[0,:].data)
        rdn[0:nzall-1] = torch.tensor(rdn_input[0,:].data)
        c1h[0:nzall-1] = torch.tensor(c1h_input[0,:].data)
        c2h[0:nzall-1] = torch.tensor(c2h_input[0,:].data)
        c3h[0:nzall-1] = torch.tensor(c3h_input[0,:].data)
        c4h[0:nzall-1] = torch.tensor(c4h_input[0,:].data)
        c1f[:] = torch.tensor(c1f_input[0,:].data)
        c2f[:] = torch.tensor(c2f_input[0,:].data)
        c3f[:] = torch.tensor(c3f_input[0,:].data)
        c4f[:] = torch.tensor(c4f_input[0,:].data)
        t[0:nzall-1,5:nyall-6,5:nxall-6] = torch.tensor(t_input[0,:,:,:].data)

        fnm[0:nzall-1] = torch.tensor(fnm_input[0,:].data)
        fnp[0:nzall-1] = torch.tensor(fnp_input[0,:].data)

        dn[0:nzall-1] = torch.tensor(dn_input[0,:].data)
        dnw[0:nzall-1] = torch.tensor(dnw_input[0,:].data)

        u_base[0:nzall-1] = torch.tensor(u_base_input[0,:].data)
        v_base[0:nzall-1] = torch.tensor(v_base_input[0,:].data)
        qv_base[0:nzall-1] = torch.tensor(qv_base_input[0,:].data)

        cf1 = torch.tensor(cf1_input[0].data)
        cf2 = torch.tensor(cf2_input[0].data)
        cf3 = torch.tensor(cf3_input[0].data)
        cfn = torch.tensor(cfn_input[0].data)
        cfn1 = torch.tensor(cfn1_input[0].data)

        f[5:nyall-6,5:nxall-6] = torch.tensor(f_input[0,:,:].data)
        e[5:nyall-6,5:nxall-6] = torch.tensor(e_input[0,:,:].data)
        sina[5:nyall-6,5:nxall-6] = torch.tensor(sina_input[0,:,:].data)
        cosa[5:nyall-6,5:nxall-6] = torch.tensor(cosa_input[0,:,:].data)
        clat[5:nyall-6,5:nxall-6] = torch.tensor(clat_input[0,:,:].data)

        znu[0:nzall-1] = torch.tensor(znu_input[0,:].data)
        znw[0:nzall] = torch.tensor(znw_input[0,:].data)
        ht[5:nyall-6,5:nxall-6] = torch.tensor(ht_input[0,:,:].data)

        ### calculate al alb
        al = torch.zeros((nzall,nyall,nxall)).to(device)
        alb = torch.zeros((nzall,nyall,nxall)).to(device)
        rdnw_e = rdnw.unsqueeze(1).unsqueeze(2).repeat(1,nyfull,nxfull)
        mub_e = mub.repeat(kte-kts,1,1)
        mu_e = mu.repeat(kte-kts,1,1)

        alb = r_d/p1000mb*(t_init + t0)*(pb/p1000mb)**cvpm

        c1h_e = c1h.unsqueeze(1).unsqueeze(2).repeat(1,nyfull,nxfull)
        c2h_e = c2h.unsqueeze(1).unsqueeze(2).repeat(1,nyfull,nxfull)
        c1f_e = c1f.unsqueeze(1).unsqueeze(2).repeat(1,nyfull,nxfull)
        c2f_e = c2f.unsqueeze(1).unsqueeze(2).repeat(1,nyfull,nxfull)

        al[kts:kte-1,jts:jde-1,its:ide-1] = -1./(mub_e[kts:kte-1,jts:jde-1,its:ide-1] \
        +mu_e[kts:kte-1,jts:jde-1,its:ide-1])*(alb[kts:kte-1,jts:jde-1,its:ide-1]* \
            mu_e[kts:kte-1,jts:jde-1,its:ide-1] + rdnw_e[kts:kte-1,jts:jde-1,its:ide-1]* \
            (ph[kts+1:kte,jts:jde-1,its:ide-1]-ph[kts:kte-1,jts:jde-1,its:ide-1]))

        #### end load input/bdy ####

        mut = torch.zeros((nyall,nxall)).to(device)
        muu = torch.zeros((nyall,nxall)).to(device)
        muv = torch.zeros((nyall,nxall)).to(device)
        ru = torch.zeros((nzall,nyall,nxall)).to(device)
        rv = torch.zeros((nzall,nyall,nxall)).to(device)
        rw = torch.zeros((nzall,nyall,nxall)).to(device)
        ww = torch.zeros((nzfull,nyfull,nxfull)).to(device)
        ww1 = torch.zeros((nzfull,nyfull,nxfull)).to(device)
        cqu = torch.zeros((nzall,nyall,nxall)).to(device)
        cqv = torch.zeros((nzall,nyall,nxall)).to(device)
        cqw = torch.zeros((nzall,nyall,nxall)).to(device)
        alt = torch.zeros((nzall,nyall,nxall)).to(device)
        php = torch.zeros((nzall,nyall,nxall)).to(device)

        ru_tendf = torch.zeros((nzall,nyall,nxall)).to(device)
        rv_tendf = torch.zeros((nzall,nyall,nxall)).to(device)
        rw_tendf = torch.zeros((nzall,nyall,nxall)).to(device)
        ph_tendf = torch.zeros((nzall,nyall,nxall)).to(device)
        t_tendf = torch.zeros((nzall,nyall,nxall)).to(device)
        tke_tend = torch.zeros((nzall,nyall,nxall)).to(device)
        mu_tendf = torch.zeros((nyall,nxall)).to(device)
        moist_tend = torch.zeros((7,nzall,nyall,nxall)).to(device)
        advect_tend = torch.zeros((7,nzall,nyall,nxall)).to(device)
        h_tendency = torch.zeros((nzall,nyall,nxall)).to(device)
        z_tendency = torch.zeros((nzall,nyall,nxall)).to(device)
        scalar_tend = torch.zeros((3,nzall,nyall,nxall)).to(device)
        scalar_old = torch.zeros((3,nzall,nyall,nxall)).to(device)
        advect_tend_sca = torch.zeros((3,nzall,nyall,nxall)).to(device)

        ru_tend = torch.zeros((nzall,nyall,nxall)).to(device)
        rv_tend = torch.zeros((nzall,nyall,nxall)).to(device)
        rw_tend = torch.zeros((nzall,nyall,nxall)).to(device)
        ph_tend = torch.zeros((nzall,nyall,nxall)).to(device)
        t_tend = torch.zeros((nzall,nyall,nxall)).to(device)
        mu_tend = torch.zeros((nyall,nxall)).to(device)

        u_save = torch.zeros((nzall,nyall,nxall)).to(device)
        v_save = torch.zeros((nzall,nyall,nxall)).to(device)
        w_save = torch.zeros((nzall,nyall,nxall)).to(device)
        ph_save = torch.zeros((nzall,nyall,nxall)).to(device)
        t_save = torch.zeros((nzall,nyall,nxall)).to(device)
        mu_save = torch.zeros((nyall,nxall)).to(device)

        rho = torch.zeros((nzall,nyall,nxall)).to(device)
        u_phy = torch.zeros((nzall,nyall,nxall)).to(device)
        v_phy = torch.zeros((nzall,nyall,nxall)).to(device)
        z_at_w = torch.zeros((nzall,nyall,nxall)).to(device)
        dz8w = torch.zeros((nzall,nyall,nxall)).to(device)
        z = torch.zeros((nzall,nyall,nxall)).to(device)
        p8w = torch.zeros((nzall,nyall,nxall)).to(device)
        t8w = torch.zeros((nzall,nyall,nxall)).to(device)
        p_hyd_w = torch.zeros((nzall,nyall,nxall)).to(device)
        p_hyd = torch.zeros((nzall,nyall,nxall)).to(device)

        rdz = torch.zeros((nzall,nyall,nxall)).to(device)
        rdzw = torch.zeros((nzall,nyall,nxall)).to(device)
        zx = torch.zeros((nzall,nyall,nxall)).to(device)
        zy = torch.zeros((nzall,nyall,nxall)).to(device)

        defor11 = torch.zeros((nzall,nyall,nxall)).to(device)
        defor22 = torch.zeros((nzall,nyall,nxall)).to(device)
        defor33 = torch.zeros((nzall,nyall,nxall)).to(device)
        defor12 = torch.zeros((nzall,nyall,nxall)).to(device)
        defor13 = torch.zeros((nzall,nyall,nxall)).to(device)
        defor23 = torch.zeros((nzall,nyall,nxall)).to(device)
        div = torch.zeros((nzall,nyall,nxall)).to(device)

        BN2 = torch.zeros((nzall,nyall,nxall)).to(device)
        xkmh = torch.zeros((nzall,nyall,nxall)).to(device)
        xkmv = torch.zeros((nzall,nyall,nxall)).to(device)
        xkhh = torch.zeros((nzall,nyall,nxall)).to(device)
        xkhv = torch.zeros((nzall,nyall,nxall)).to(device)
        tke = torch.zeros((nzall,nyall,nxall)).to(device)

        rthblten = torch.zeros((nzall,nyall,nxall)).to(device)
        rublten = torch.zeros((nzall,nyall,nxall)).to(device)
        rvblten = torch.zeros((nzall,nyall,nxall)).to(device)
        rqvblten = torch.zeros((nzall,nyall,nxall)).to(device)
        rqcblten = torch.zeros((nzall,nyall,nxall)).to(device)
        rqiblten = torch.zeros((nzall,nyall,nxall)).to(device)

        rfield = torch.zeros((nzall,nyall,nxall)).to(device)

        u_1 = torch.zeros((nzall,nyall,nxall)).to(device)
        v_1 = torch.zeros((nzall,nyall,nxall)).to(device)
        w_1 = torch.zeros((nzall,nyall,nxall)).to(device)
        t_1 = torch.zeros((nzall,nyall,nxall)).to(device)
        ph_1 = torch.zeros((nzall,nyall,nxall)).to(device)
        tke_1 = torch.zeros((nzall,nyall,nxall)).to(device)
        u_save = torch.zeros((nzall,nyall,nxall)).to(device)
        v_save = torch.zeros((nzall,nyall,nxall)).to(device)
        w_save = torch.zeros((nzall,nyall,nxall)).to(device)
        t_save = torch.zeros((nzall,nyall,nxall)).to(device)
        ph_save = torch.zeros((nzall,nyall,nxall)).to(device)
        c2a = torch.zeros((nzall,nyall,nxall)).to(device)
        ww_1 = torch.zeros((nzall,nyall,nxall)).to(device)
        mu_1 = torch.zeros((nyall,nxall)).to(device)
        muus = torch.zeros((nyall,nxall)).to(device)
        muvs = torch.zeros((nyall,nxall)).to(device)
        muts = torch.zeros((nyall,nxall)).to(device)
        mudf = torch.zeros((nyall,nxall)).to(device)
        mu_save = torch.zeros((nyall,nxall)).to(device)

        pm1 = torch.zeros((nzall,nyall,nxall)).to(device)

        a = torch.zeros((nzall,nyall,nxall)).to(device)
        alpha = torch.zeros((nzall,nyall,nxall)).to(device)
        gamma = torch.zeros((nzall,nyall,nxall)).to(device)

        muave = torch.zeros((nyall,nxall)).to(device)
        ru_m = torch.zeros((nzall,nyall,nxall)).to(device)
        rv_m = torch.zeros((nzall,nyall,nxall)).to(device)
        ww_m = torch.zeros((nzall,nyall,nxall)).to(device)
        t_2save = torch.zeros((nzall,nyall,nxall)).to(device)

        moist_old = torch.zeros((7, nzall,nyall,nxall)).to(device)

        th_phy = torch.zeros((nzall,nyall,nxall)).to(device)
        p_phy = torch.zeros((nzall,nyall,nxall)).to(device)
        pi_phy = torch.zeros((nzall,nyall,nxall)).to(device)
        t_phy = torch.zeros((nzall,nyall,nxall)).to(device)


        RQVFTEN = torch.zeros((nzall,nyall,nxall)).to(device)

        rainnc = torch.zeros((nyall,nxall)).to(device)
        rainncv = torch.zeros((nyall,nxall)).to(device)
        snownc = torch.zeros((nyall,nxall)).to(device)
        snowncv = torch.zeros((nyall,nxall)).to(device)
        graupelnc = torch.zeros((nyall,nxall)).to(device)
        graupelncv = torch.zeros((nyall,nxall)).to(device)
        sr  = torch.zeros((nyall,nxall)).to(device)
        h_diabatic = torch.zeros((nzall,nyall,nxall)).to(device)
        qv_diabatic = torch.zeros((nzall,nyall,nxall)).to(device)
        qc_diabatic = torch.zeros((nzall,nyall,nxall)).to(device)

        gcx = torch.tensor([0., 1.1111111e-3, 7.4074074e-4, 3.7037037e-4, 0.])
        fcx = torch.tensor([0., 5.5555557e-3, 3.7037039e-3, 1.8518519e-3, 0.])

        mub[jte-1,:] = mub[jte-2,:]
        mub[:,ite-1] = mub[:,ite-2]
        mub[jts-1,:] = mub[jts,:]
        mub[:,its-1] = mub[:,its]



        ### (3) MAIN TIME LOOP - 1600 steps x dt -------------------------
        for t_big_step in range(1,1601,1):     # 1600 steps x 60 s = 26.7 h
            # bdy input (refresh the lateral boundary arrays every bdy_interval steps)
            with torch.no_grad():
                if t_big_step%bdy_interval == 1:
                    it_bdy = int(t_big_step/bdy_interval)
                    u_bxs_input = wrfbdy["U_BXS"]
                    u_bxe_input = wrfbdy["U_BXE"]
                    u_bys_input = wrfbdy["U_BYS"]
                    u_bye_input = wrfbdy["U_BYE"]
                    u_btxs_input = wrfbdy["U_BTXS"]
                    u_btxe_input = wrfbdy["U_BTXE"]
                    u_btys_input = wrfbdy["U_BTYS"]
                    u_btye_input = wrfbdy["U_BTYE"]

                    u_bxs = torch.zeros((5,nzall,nyall)).to(device)
                    u_bxs[:,0:nzfull-1,5:nyall-6] = torch.tensor(u_bxs_input[it_bdy,:,:,:].data)
                    u_bxs = u_bxs.permute(1,2,0)

                    u_bxe = torch.zeros((5,nzall,nyall)).to(device)
                    u_bxe[:,0:nzfull-1,5:nyall-6] = torch.tensor(u_bxe_input[it_bdy,:,:,:].data)
                    u_bxe = u_bxe.permute(1,2,0)

                    u_bys = torch.zeros((5,nzall,nxall)).to(device)
                    u_bys[:,0:nzall-1,5:nxall-5] = torch.tensor(u_bys_input[it_bdy,:,:,:].data)
                    u_bys = u_bys.permute(1,0,2)

                    u_bye = torch.zeros((5,nzall,nxall)).to(device)
                    u_bye[:,0:nzall-1,5:nxall-5] = torch.tensor(u_bye_input[it_bdy,:,:,:].data)
                    u_bye = u_bye.permute(1,0,2)

                    u_btxs = torch.zeros((5,nzall,nyall)).to(device)
                    u_btxs[:,0:nzfull-1,5:nyall-6] = torch.tensor(u_btxs_input[it_bdy,:,:,:].data)
                    u_btxs = u_btxs.permute(1,2,0)

                    u_btxe = torch.zeros((5,nzall,nyall)).to(device)
                    u_btxe[:,0:nzfull-1,5:nyall-6] = torch.tensor(u_btxe_input[it_bdy,:,:,:].data)
                    u_btxe = u_btxe.permute(1,2,0)

                    u_btys = torch.zeros((5,nzall,nxall)).to(device)
                    u_btys[:,0:nzall-1,5:nxall-5] = torch.tensor(u_btys_input[it_bdy,:,:,:].data)
                    u_btys = u_btys.permute(1,0,2)

                    u_btye = torch.zeros((5,nzall,nxall)).to(device)
                    u_btye[:,0:nzall-1,5:nxall-5] = torch.tensor(u_btye_input[it_bdy,:,:,:].data)
                    u_btye = u_btye.permute(1,0,2)

                    v_bxs_input = wrfbdy["V_BXS"]
                    v_bxe_input = wrfbdy["V_BXE"]
                    v_bys_input = wrfbdy["V_BYS"]
                    v_bye_input = wrfbdy["V_BYE"]
                    v_btxs_input = wrfbdy["V_BTXS"]
                    v_btxe_input = wrfbdy["V_BTXE"]
                    v_btys_input = wrfbdy["V_BTYS"]
                    v_btye_input = wrfbdy["V_BTYE"]

                    v_bxs = torch.zeros((5,nzall,nyall)).to(device)
                    v_bxs[:,0:nzall-1,5:nyall-5] = torch.tensor(v_bxs_input[it_bdy,:,:,:].data)
                    v_bxs = v_bxs.permute(1,2,0)

                    v_bxe = torch.zeros((5,nzall,nyall)).to(device)
                    v_bxe[:,0:nzall-1,5:nyall-5] = torch.tensor(v_bxe_input[it_bdy,:,:,:].data)
                    v_bxe = v_bxe.permute(1,2,0)

                    v_bys = torch.zeros((5,nzall,nxall)).to(device)
                    v_bys[:,0:nzall-1,5:nxall-6] = torch.tensor(v_bys_input[it_bdy,:,:,:].data)
                    v_bys = v_bys.permute(1,0,2)

                    v_bye = torch.zeros((5,nzall,nxall)).to(device)
                    v_bye[:,0:nzall-1,5:nxall-6] = torch.tensor(v_bye_input[it_bdy,:,:,:].data)
                    v_bye = v_bye.permute(1,0,2)

                    v_btxs = torch.zeros((5,nzall,nyall)).to(device)
                    v_btxs[:,0:nzall-1,5:nyall-5] = torch.tensor(v_btxs_input[it_bdy,:,:,:].data)
                    v_btxs = v_btxs.permute(1,2,0)

                    v_btxe = torch.zeros((5,nzall,nyall)).to(device)
                    v_btxe[:,0:nzall-1,5:nyall-5] = torch.tensor(v_btxe_input[it_bdy,:,:,:].data)
                    v_btxe = v_btxe.permute(1,2,0)

                    v_btys = torch.zeros((5,nzall,nxall)).to(device)
                    v_btys[:,0:nzall-1,5:nxall-6] = torch.tensor(v_btys_input[it_bdy,:,:,:].data)
                    v_btys = v_btys.permute(1,0,2)

                    v_btye = torch.zeros((5,nzall,nxall)).to(device)
                    v_btye[:,0:nzall-1,5:nxall-6] = torch.tensor(v_btye_input[it_bdy,:,:,:].data)
                    v_btye = v_btye.permute(1,0,2)

                    ph_bxs_input = wrfbdy["PH_BXS"]
                    ph_bxe_input = wrfbdy["PH_BXE"]
                    ph_bys_input = wrfbdy["PH_BYS"]
                    ph_bye_input = wrfbdy["PH_BYE"]
                    ph_btxs_input = wrfbdy["PH_BTXS"]
                    ph_btxe_input = wrfbdy["PH_BTXE"]
                    ph_btys_input = wrfbdy["PH_BTYS"]
                    ph_btye_input = wrfbdy["PH_BTYE"]

                    ph_bxs = torch.zeros((5,nzall,nyall)).to(device)
                    ph_bxs[:,0:nzall,5:nyall-6] = torch.tensor(ph_bxs_input[it_bdy,:,:,:].data)
                    ph_bxs = ph_bxs.permute(1,2,0)

                    ph_bxe = torch.zeros((5,nzall,nyall)).to(device)
                    ph_bxe[:,0:nzall,5:nyall-6] = torch.tensor(ph_bxe_input[it_bdy,:,:,:].data)
                    ph_bxe = ph_bxe.permute(1,2,0)

                    ph_bys = torch.zeros((5,nzall,nxall)).to(device)
                    ph_bys[:,0:nzall,5:nxall-6] = torch.tensor(ph_bys_input[it_bdy,:,:,:].data)
                    ph_bys = ph_bys.permute(1,0,2)

                    ph_bye = torch.zeros((5,nzall,nxall)).to(device)
                    ph_bye[:,0:nzall,5:nxall-6] = torch.tensor(ph_bye_input[it_bdy,:,:,:].data)
                    ph_bye = ph_bye.permute(1,0,2)

                    ph_btxs = torch.zeros((5,nzall,nyall)).to(device)
                    ph_btxs[:,0:nzall,5:nyall-6] = torch.tensor(ph_btxs_input[it_bdy,:,:,:].data)
                    ph_btxs = ph_btxs.permute(1,2,0)

                    ph_btxe = torch.zeros((5,nzall,nyall)).to(device)
                    ph_btxe[:,0:nzall,5:nyall-6] = torch.tensor(ph_btxe_input[it_bdy,:,:,:].data)
                    ph_btxe = ph_btxe.permute(1,2,0)

                    ph_btys = torch.zeros((5,nzall,nxall)).to(device)
                    ph_btys[:,0:nzall,5:nxall-6] = torch.tensor(ph_btys_input[it_bdy,:,:,:].data)
                    ph_btys = ph_btys.permute(1,0,2)

                    ph_btye = torch.zeros((5,nzall,nxall)).to(device)
                    ph_btye[:,0:nzall,5:nxall-6] = torch.tensor(ph_btye_input[it_bdy,:,:,:].data)
                    ph_btye = ph_btye.permute(1,0,2)

                    t_bxs_input = wrfbdy["T_BXS"]
                    t_bxe_input = wrfbdy["T_BXE"]
                    t_bys_input = wrfbdy["T_BYS"]
                    t_bye_input = wrfbdy["T_BYE"]
                    t_btxs_input = wrfbdy["T_BTXS"]
                    t_btxe_input = wrfbdy["T_BTXE"]
                    t_btys_input = wrfbdy["T_BTYS"]
                    t_btye_input = wrfbdy["T_BTYE"]

                    t_bxs = torch.zeros((5,nzall,nyall)).to(device)
                    t_bxs[:,0:nzall-1,5:nyall-6] = torch.tensor(t_bxs_input[it_bdy,:,:,:].data)
                    t_bxs = t_bxs.permute(1,2,0)

                    t_bxe = torch.zeros((5,nzall,nyall)).to(device)
                    t_bxe[:,0:nzall-1,5:nyall-6] = torch.tensor(t_bxe_input[it_bdy,:,:,:].data)
                    t_bxe = t_bxe.permute(1,2,0)

                    t_bys = torch.zeros((5,nzall,nxall)).to(device)
                    t_bys[:,0:nzall-1,5:nxall-6] = torch.tensor(t_bys_input[it_bdy,:,:,:].data)
                    t_bys = t_bys.permute(1,0,2)

                    t_bye = torch.zeros((5,nzall,nxall)).to(device)
                    t_bye[:,0:nzall-1,5:nxall-6] = torch.tensor(t_bye_input[it_bdy,:,:,:].data)
                    t_bye = t_bye.permute(1,0,2)

                    t_btxs = torch.zeros((5,nzall,nyall)).to(device)
                    t_btxs[:,0:nzall-1,5:nyall-6] = torch.tensor(t_btxs_input[it_bdy,:,:,:].data)
                    t_btxs = t_btxs.permute(1,2,0)

                    t_btxe = torch.zeros((5,nzall,nyall)).to(device)
                    t_btxe[:,0:nzall-1,5:nyall-6] = torch.tensor(t_btxe_input[it_bdy,:,:,:].data)
                    t_btxe = t_btxe.permute(1,2,0)

                    t_btys = torch.zeros((5,nzall,nxall)).to(device)
                    t_btys[:,0:nzall-1,5:nxall-6] = torch.tensor(t_btys_input[it_bdy,:,:,:].data)
                    t_btys = t_btys.permute(1,0,2)

                    t_btye = torch.zeros((5,nzall,nxall)).to(device)
                    t_btye[:,0:nzall-1,5:nxall-6] = torch.tensor(t_btye_input[it_bdy,:,:,:].data)
                    t_btye = t_btye.permute(1,0,2)

                    mu_bxs_input = wrfbdy["MU_BXS"]
                    mu_bxe_input = wrfbdy["MU_BXE"]
                    mu_bys_input = wrfbdy["MU_BYS"]
                    mu_bye_input = wrfbdy["MU_BYE"]
                    mu_btxs_input = wrfbdy["MU_BTXS"]
                    mu_btxe_input = wrfbdy["MU_BTXE"]
                    mu_btys_input = wrfbdy["MU_BTYS"]
                    mu_btye_input = wrfbdy["MU_BTYE"]

                    mu_bxs = torch.zeros((5,1,nyall)).to(device)
                    mu_bxs[:,0,5:nyall-6] = torch.tensor(mu_bxs_input[it_bdy,:,:].data)
                    mu_bxs = mu_bxs.permute(1,2,0)

                    mu_bxe = torch.zeros((5,1,nyall)).to(device)
                    mu_bxe[:,0,5:nyall-6] = torch.tensor(mu_bxe_input[it_bdy,:,:].data)
                    mu_bxe = mu_bxe.permute(1,2,0)

                    mu_bys = torch.zeros((5,1,nxall)).to(device)
                    mu_bys[:,0,5:nxall-6] = torch.tensor(mu_bys_input[it_bdy,:,:].data)
                    mu_bys = mu_bys.permute(1,0,2)

                    mu_bye = torch.zeros((5,1,nxall)).to(device)
                    mu_bye[:,0,5:nxall-6] = torch.tensor(mu_bye_input[it_bdy,:,:].data)
                    mu_bye = mu_bye.permute(1,0,2)

                    mu_btxs = torch.zeros((5,1,nyall)).to(device)
                    mu_btxs[:,0,5:nyall-6] = torch.tensor(mu_btxs_input[it_bdy,:,:].data)
                    mu_btxs = mu_btxs.permute(1,2,0)

                    mu_btxe = torch.zeros((5,1,nyall)).to(device)
                    mu_btxe[:,0,5:nyall-6] = torch.tensor(mu_btxe_input[it_bdy,:,:].data)
                    mu_btxe = mu_btxe.permute(1,2,0)

                    mu_btys = torch.zeros((5,1,nxall)).to(device)
                    mu_btys[:,0,5:nxall-6] = torch.tensor(mu_btys_input[it_bdy,:,:].data)
                    mu_btys = mu_btys.permute(1,0,2)

                    mu_btye = torch.zeros((5,1,nxall)).to(device)
                    mu_btye[:,0,5:nxall-6] = torch.tensor(mu_btye_input[it_bdy,:,:].data)
                    mu_btye = mu_btye.permute(1,0,2)

                    qv_bxs_input = wrfbdy["QVAPOR_BXS"]
                    qv_bxe_input = wrfbdy["QVAPOR_BXE"]
                    qv_bys_input = wrfbdy["QVAPOR_BYS"]
                    qv_bye_input = wrfbdy["QVAPOR_BYE"]
                    qv_btxs_input = wrfbdy["QVAPOR_BTXS"]
                    qv_btxe_input = wrfbdy["QVAPOR_BTXE"]
                    qv_btys_input = wrfbdy["QVAPOR_BTYS"]
                    qv_btye_input = wrfbdy["QVAPOR_BTYE"]

                    qv_bxs = torch.zeros((5,nzall,nyall)).to(device)
                    qv_bxs[:,0:nzall-1,5:nyall-6] = torch.tensor(qv_bxs_input[it_bdy,:,:,:].data)
                    qv_bxs = qv_bxs.permute(1,2,0)

                    qv_bxe = torch.zeros((5,nzall,nyall)).to(device)
                    qv_bxe[:,0:nzall-1,5:nyall-6] = torch.tensor(qv_bxe_input[it_bdy,:,:,:].data)
                    qv_bxe = qv_bxe.permute(1,2,0)

                    qv_bys = torch.zeros((5,nzall,nxall)).to(device)
                    qv_bys[:,0:nzall-1,5:nxall-6] = torch.tensor(qv_bys_input[it_bdy,:,:,:].data)
                    qv_bys = qv_bys.permute(1,0,2)

                    qv_bye = torch.zeros((5,nzall,nxall)).to(device)
                    qv_bye[:,0:nzall-1,5:nxall-6] = torch.tensor(qv_bye_input[it_bdy,:,:,:].data)
                    qv_bye = qv_bye.permute(1,0,2)

                    qv_btxs = torch.zeros((5,nzall,nyall)).to(device)
                    qv_btxs[:,0:nzall-1,5:nyall-6] = torch.tensor(qv_btxs_input[it_bdy,:,:,:].data)
                    qv_btxs = qv_btxs.permute(1,2,0)

                    qv_btxe = torch.zeros((5,nzall,nyall)).to(device)
                    qv_btxe[:,0:nzall-1,5:nyall-6] = torch.tensor(qv_btxe_input[it_bdy,:,:,:].data)
                    qv_btxe = qv_btxe.permute(1,2,0)

                    qv_btys = torch.zeros((5,nzall,nxall)).to(device)
                    qv_btys[:,0:nzall-1,5:nxall-6] = torch.tensor(qv_btys_input[it_bdy,:,:,:].data)
                    qv_btys = qv_btys.permute(1,0,2)

                    qv_btye = torch.zeros((5,nzall,nxall)).to(device)
                    qv_btye[:,0:nzall-1,5:nxall-6] = torch.tensor(qv_btye_input[it_bdy,:,:,:].data)
                    qv_btye = qv_btye.permute(1,0,2)
            ##### end load bdy #####
            # Runge_Kutta_loop
            print("========t_big_step ",t_big_step,"=========")
            #if t_big_step == 18:
            #    sys.exit(1)
            moist_last_step = moist + 0.0
            with torch.no_grad():
                dtm = dt
                dts = dt/soundsteps
                dtbc = dt*(t_big_step - it_bdy * bdy_interval)  ### 注意


                ### (4) Runge-Kutta 3rd-order substeps ---------------------------
                for rk_step in range(1,4,1):  # 3rd-order Runge_Kutta_loop begin
                    if rk_step == 1:
                        dt_rk = dtm/3.
                        dts_rk = dt_rk
                        number_of_small_timesteps = 1
                    elif rk_step == 2:
                        dt_rk = 0.5*dtm
                        dts_rk = dts
                        number_of_small_timesteps = int(soundsteps/2)
                    else:
                        dt_rk = dtm
                        dts_rk = dts
                        number_of_small_timesteps = soundsteps
                    # rk_step_prep
                    print('rk_step')
                    print(rk_step)
                    print("rk start:",p[2, 160, 12], u[2, 160, 12], v[2, 160, 12], t[2, 160, 12], w[2, 160, 12])
                    print("000",p[2,160,11:13])
                    # --- RK-step preparation: masses / coupled momentum / alt / php ---
                    mut = calculate_full(mut, mub, mu,
                                        ids, ide, jds, jde, kds, kde,
                                        ims, ime, jms, jme, kms, kme,
                                        its, ite, jts, jte, kts, kte)
                    muu, muv = calc_mu_uv(mu, mub, muu, muv,
                                        ids, ide, jds, jde, kds, kde,
                                        ims, ime, jms, jme, kms, kme,
                                        its, ite, jts, jte, kts, kte)
                    ru, rv, rw = couple_momentum(muu, ru, u, msfu,
                                                muv, rv, v, msfv, msfvx_inv,
                                                mut, rw, w, msft,
                                                c1h, c2h, c1f, c2f,
                                                ids, ide, jds, jde, kds, kde,
                                                ims, ime, jms, jme, kms, kme,
                                                its, ite, jts, jte, kts, kte)
                    ### plot uv print("pres ru_tend: ", ru_tend[21,444,579])
                    maxmoist0, maxmoistindex0 = torch.max(moist[P_QG,:,200:400,200:400],dim=0)
                    maxmoist1, maxmoistindex1 = torch.max(maxmoist0,dim=0)
                    maxmoist2, maxmoistindex2 = torch.max(maxmoist1,dim=0)
                    index2 = maxmoistindex2
                    index1 = maxmoistindex1[maxmoistindex2]
                    index0 = maxmoistindex0[index1,index2]

                    ww = calc_ww_cp(u, v, mu, mub, muu, muv, c1h, c2h, ww,
                                    rdx, rdy, msftx, msfty,
                                    msfux, msfuy, msfvx, msfvx_inv,
                                    msfvy, dnw,
                                    ids, ide, jds, jde, kds, kde,
                                    ims, ime, jms, jme, kms, kme,
                                    its, ite, jts, jte, kts, kte)
                    cqu, cuv, cqw = calc_cq(moist, cqu, cqv, cqw, n_moist,
                                            ids, ide, jds, jde, kds, kde,
                                            ims, ime, jms, jme, kms, kme,
                                            its, ite, jts, jte, kts, kte)
                    alt = calc_alt(alt, al, alb,
                                ids, ide, jds, jde, kds, kde,
                                ims, ime, jms, jme, kms, kme,
                                its, ite, jts, jte, kts, kte)
                    print("calc alt ", alt[2, 160, 12], al[2, 160, 12], alb[2, 160, 12])
                    php = calc_php(php, ph, phb,
                                ids, ide, jds, jde, kds, kde,
                                ims, ime, jms, jme, kms, kme,
                                its, ite, jts, jte, kts, kte)
                    # rk_phys_bc_dry_1
                    # --- physical BCs on the coupled state ---
                    ru = set_physical_bc3d(ru, 'u',
                                        ids,ide, jds,jde, kds,kde,
                                        ims,ime, jms,jme, kms,kme,
                                        ips,ipe, jps,jpe, kps,kpe,
                                        its,ite, jts,jte, kts,kte )
                    rv = set_physical_bc3d(rv, 'v',
                                        ids,ide, jds,jde, kds,kde,
                                        ims,ime, jms,jme, kms,kme,
                                        ips,ipe, jps,jpe, kps,kpe,
                                        its,ite, jts,jte, kts,kte )


                    rw = set_physical_bc3d(rw, 'w',
                                        ids,ide, jds,jde, kds,kde,
                                        ims,ime, jms,jme, kms,kme,
                                        ips,ipe, jps,jpe, kps,kpe,
                                        its,ite, jts,jte, kts,kte )
                    ww = set_physical_bc3d(ww, 'w',
                                        ids,ide, jds,jde, kds,kde,
                                        ims,ime, jms,jme, kms,kme,
                                        ips,ipe, jps,jpe, kps,kpe,
                                        its,ite, jts,jte, kts,kte )
                    php = set_physical_bc3d(php, 'w',
                                        ids,ide, jds,jde, kds,kde,
                                        ims,ime, jms,jme, kms,kme,
                                        ips,ipe, jps,jpe, kps,kpe,
                                        its,ite, jts,jte, kts,kte )
                    alt = set_physical_bc3d(alt, 't',
                                        ids,ide, jds,jde, kds,kde,
                                        ims,ime, jms,jme, kms,kme,
                                        ips,ipe, jps,jpe, kps,kpe,
                                        its,ite, jts,jte, kts,kte )
                    p = set_physical_bc3d(p, 'p',
                                        ids,ide, jds,jde, kds,kde,
                                        ims,ime, jms,jme, kms,kme,
                                        ips,ipe, jps,jpe, kps,kpe,
                                        its,ite, jts,jte, kts,kte )
                    muu = set_physical_bc2d(muu, 'u',
                                        ids,ide, jds,jde,
                                        ims,ime, jms,jme,
                                        ips,ipe, jps,jpe,
                                        its,ite, jts,jte )
                    muv = set_physical_bc2d(muv, 'v',
                                        ids,ide, jds,jde,
                                        ims,ime, jms,jme,
                                        ips,ipe, jps,jpe,
                                        its,ite, jts,jte )
                    mut = set_physical_bc2d(mut, 't',
                                        ids,ide, jds,jde,
                                        ims,ime, jms,jme,
                                        ips,ipe, jps,jpe,
                                        its,ite, jts,jte )
                    al = set_physical_bc3d(al, 'p',
                                        ids,ide, jds,jde, kds,kde,
                                        ims,ime, jms,jme, kms,kme,
                                        ips,ipe, jps,jpe, kps,kpe,
                                        its,ite, jts,jte, kts,kte )
                    ### ph_2
                    ph = set_physical_bc3d(ph, 'w',
                                        ids,ide, jds,jde, kds,kde,
                                        ims,ime, jms,jme, kms,kme,
                                        ips,ipe, jps,jpe, kps,kpe,
                                        its,ite, jts,jte, kts,kte )
                    if rk_step == 1:
                        # first_rk_step_part1
                        ru_tendf[:,:,:] = 0.
                        rv_tendf[:,:,:] = 0.
                        rw_tendf[:,:,:] = 0.
                        ph_tendf[:,:,:] = 0.
                        t_tendf[:,:,:] = 0.
                        tke_tend[:,:,:] = 0.
                        mu_tendf[:,:] = 0.
                        moist_tend[:,:,:,:] = 0.
                        scalar_tend[:,:,:,:] = 0.
                        th_phy, p_phy, pi_phy, t_phy, rho, u_phy, v_phy, z_at_w, dz8w, \
                            z, p8w, t8w, p_hyd_w, p_hyd = phy_prep(mut, muu, muv,
                                                                c1h, c2h, c1f, c2f,
                                                                u, v, p, pb, alt, ph,
                                                                phb, t, moist, n_moist,
                                                                rho, th_phy, p_phy , pi_phy ,
                                                                u_phy, v_phy, p8w, t_phy, t8w,
                                                                z, z_at_w, dz8w,
                                                                p_hyd, p_hyd_w, dnw,
                                                                fnm, fnp, znw, p_top,
                                                                ids, ide, jds, jde, kds, kde,
                                                                ims, ime, jms, jme, kms, kme,
                                                                its, ite, jts, jte, kts, kte)
                        ### pbl_driver ###

                        ##################

                        # first_rk_step_part2
                        rublten,rvblten,rthblten,rqvblten,rqcblten,rqiblren,scalar_tend = \
                            calculate_phy_tend(c1h,c2h,
                                            mut,muu,muv,pi_phy,
                                            rublten,rvblten,rthblten,
                                            rqvblten,rqcblten,rqiblten,
                                            scalar, scalar_tend, n_scalar,
                                            ids,ide, jds,jde, kds,kde,
                                            ims,ime, jms,jme, kms,kme,
                                            its,ite, jts,jte, kts,kte)
                        rdzw, rdz, zx, zy, z = compute_diff_metrics(ph, phb, z, rdz, rdzw,
                                                                    zx, zy, rdx, rdy,
                                                                    ids, ide, jds, jde, kds, kde,
                                                                    ims, ime, jms, jme, kms, kme,
                                                                    its, ite, jts, jte, kts, kte)
                        rdzw = set_physical_bc3d(rdzw, 'w',
                                        ids,ide, jds,jde, kds,kde,
                                        ims,ime, jms,jme, kms,kme,
                                        ips,ipe, jps,jpe, kps,kpe,
                                        its,ite, jts,jte, kts,kte )
                        rdz = set_physical_bc3d(rdz, 'w',
                                        ids,ide, jds,jde, kds,kde,
                                        ims,ime, jms,jme, kms,kme,
                                        ips,ipe, jps,jpe, kps,kpe,
                                        its,ite, jts,jte, kts,kte )
                        z = set_physical_bc3d(z, 'w',
                                        ids,ide, jds,jde, kds,kde,
                                        ims,ime, jms,jme, kms,kme,
                                        ips,ipe, jps,jpe, kps,kpe,
                                        its,ite, jts,jte, kts,kte )
                        zx = set_physical_bc3d(zx, 'e',
                                        ids,ide, jds,jde, kds,kde,
                                        ims,ime, jms,jme, kms,kme,
                                        ips,ipe, jps,jpe, kps,kpe,
                                        its,ite, jts,jte, kts,kte )
                        zy = set_physical_bc3d(zy, 'f',
                                        ids,ide, jds,jde, kds,kde,
                                        ims,ime, jms,jme, kms,kme,
                                        ips,ipe, jps,jpe, kps,kpe,
                                        its,ite, jts,jte, kts,kte )
                        defor11,defor12,defor13,defor22,defor23,defor33,div = \
                            cal_deform_and_div(u, v, w, div,
                                            defor11, defor22, defor33,
                                            defor12, defor13, defor23,
                                            u_base, v_base, msfux, msfuy,
                                            msfvx, msfvy, msftx, msfty,
                                            rdx, rdy, dn, dnw, rdz, rdzw,
                                            fnm, fnp, cf1, cf2, cf3, zx, zy,
                                            ids, ide, jds, jde, kds, kde,
                                            ims, ime, jms, jme, kms, kme,
                                            its, ite, jts, jte, kts, kte)
                        BN2,xkmh,xkmv,xkhh,xkhv = calculate_km_kh(dt,
                                                                dampcoef, zdamp, damp_opt,
                                                                xkmh, xkmv, xkhh, xkhv,
                                                                BN2, khdif, kvdif, div,
                                                                defor11, defor22, defor33,
                                                                defor12, defor13, defor23,
                                                                tke, p8w, t8w, th_phy, t_phy, p_phy, moist,
                                                                dn, dnw, dx, dy, rdz, rdzw, 0,
                                                                n_moist, cf1, cf2, cf3, 0,
                                                                0.1,
                                                                msftx, msfty,
                                                                zx, zy,
                                                                ids, ide, jds, jde, kds, kde,
                                                                ims, ime, jms, jme, kms, kme,
                                                                its, ite, jts, jte, kts, kte)
                        # phy_bc
                        rublten = set_physical_bc3d(rublten, 't',
                                        ids,ide, jds,jde, kds,kde,
                                        ims,ime, jms,jme, kms,kme,
                                        ips,ipe, jps,jpe, kps,kpe,
                                        its,ite, jts,jte, kts,kte )
                        rvblten = set_physical_bc3d(rvblten, 't',
                                        ids,ide, jds,jde, kds,kde,
                                        ims,ime, jms,jme, kms,kme,
                                        ips,ipe, jps,jpe, kps,kpe,
                                        its,ite, jts,jte, kts,kte )
                        xkmh = set_physical_bc3d(xkmh, 't',
                                        ids,ide, jds,jde, kds,kde,
                                        ims,ime, jms,jme, kms,kme,
                                        ips,ipe, jps,jpe, kps,kpe,
                                        its,ite, jts,jte, kts,kte )
                        xkhh = set_physical_bc3d(xkhh, 't',
                                        ids,ide, jds,jde, kds,kde,
                                        ims,ime, jms,jme, kms,kme,
                                        ips,ipe, jps,jpe, kps,kpe,
                                        its,ite, jts,jte, kts,kte )
                        # update_phy_ten
                        t_tendf, ru_tendf, rv_tendf, moist_tend =      \
                            update_phy_ten(ph_tendf,t_tendf,ru_tendf,rv_tendf,moist_tend,
                                        scalar_tend,mu_tendf,
                                        rthblten, rublten, rvblten,
                                        rqvblten, rqcblten, rqiblten,
                                        n_moist,n_scalar,rk_step,adv_moist_cond,
                                        ids, ide, jds, jde, kds, kde,
                                        ims, ime, jms, jme, kms, kme,
                                        its, ite, jts, jte, kts, kte)
                    # end if rk_step_is_one
                    # rk_tendency
                    ru_tend[:,:,:] = 0.
                    rv_tend[:,:,:] = 0.
                    rw_tend[:,:,:] = 0.
                    t_tend[:,:,:] = 0.
                    ph_tend[:,:,:] = 0.
                    u_save[:,:,:] = 0.
                    v_save[:,:,:] = 0.
                    w_save[:,:,:] = 0.
                    ph_save[:,:,:] = 0.
                    t_save[:,:,:] = 0.
                    mu_tend[:,:] = 0.
                    mu_save[:,:] = 0.
                    mu_tend_e = mu_tend.unsqueeze(0)
                    mu_e = mu.unsqueeze(0)

                    # --- advection tendencies (u, v, w, theta) ---
                    ru_tend = advect_u(u, u , ru_tend, ru, rv, ww,
                                    c1h, c2h,
                                    mut, time_step,
                                    msfux, msfuy, msfvx, msfvy,
                                    msftx, msfty,
                                    fnm, fnp, rdx, rdy, rdnw,
                                    ids, ide, jds, jde, kds, kde,
                                    ims, ime, jms, jme, kms, kme,
                                    its, ite, jts, jte, kts, kte)
                    rv_tend = advect_v(v, v , rv_tend, ru, rv, ww,
                                    c1h, c2h,
                                    mut, time_step,
                                    msfux, msfuy, msfvx, msfvy,
                                    msftx, msfty,
                                    fnm, fnp, rdx, rdy, rdnw,
                                    ids, ide, jds, jde, kds, kde,
                                    ims, ime, jms, jme, kms, kme,
                                    its, ite, jts, jte, kts, kte )
                    print("advect tend:",ru_tend[2, 160, 12], rv_tend[2, 160, 12])
                    rw_tend = advect_w(w, w, rw_tend, ru, rv, ww,
                                    c1h, c2h,
                                    mut, time_step,
                                    msfux, msfuy, msfvx, msfvy,
                                    msftx, msfty,
                                    fnm, fnp, rdx, rdy, rdn,
                                    ids, ide, jds, jde, kds, kde,
                                    ims, ime, jms, jme, kms, kme,
                                    its, ite, jts, jte, kts, kte)
                    t_tend = advect_scalar(t, t, t_tend, ru, rv, ww,
                                        c1h, c2h,
                                        mut, time_step,
                                        msfux, msfuy, msfvx, msfvy,
                                        msftx, msfty, fnm, fnp,
                                        rdx, rdy, rdnw,
                                        ids, ide, jds, jde, kds, kde,
                                        ims, ime, jms, jme, kms, kme,
                                        its, ite, jts, jte, kts, kte)

                    # --- pressure-gradient / buoyancy / coriolis / curvature tendencies ---
                    ph_tend = rhs_ph(ph_tend, u, v, ww, ph, ph, phb, w,
                                    mut, muu, muv,
                                    c1f, c2f,
                                    fnm, fnp,
                                    rdnw, cfn, cfn1, rdx, rdy,
                                    msfux, msfuy, msfvx,
                                    msfvx_inv, msfvy,
                                    msftx, msfty,
                                    non_hydrostatic,
                                    ids, ide, jds, jde, kds, kde,
                                    ims, ime, jms, jme, kms, kme,
                                    its, ite, jts, jte, kts, kte)
                    ru_tend, rv_tend = horizontal_pressure_gradient(ru_tend,rv_tend,
                                                                    ph,alt,p,pb,al,php,cqu,cqv,
                                                                    muu,muv,mu,c1h,c2h,fnm,fnp,rdnw,
                                                                    cf1,cf2,cf3,cfn,cfn1,
                                                                    rdx,rdy,msfux,msfuy,
                                                                    msfvx,msfvy,msftx,msfty,
                                                                    non_hydrostatic, top_lid,
                                                                    ids, ide, jds, jde, kds, kde,
                                                                    ims, ime, jms, jme, kms, kme,
                                                                    its, ite, jts, jte, kts, kte)
                    print("pressure tend:",ru_tend[2, 160, 12], rv_tend[2, 160, 12])
                    maxmoist0, maxmoistindex0 = torch.max(ru_tend[:,:,:],dim=0)
                    maxmoist1, maxmoistindex1 = torch.max(maxmoist0,dim=0)
                    maxmoist2, maxmoistindex2 = torch.max(maxmoist1,dim=0)
                    index2 = maxmoistindex2
                    index1 = maxmoistindex1[maxmoistindex2]
                    index0 = maxmoistindex0[index1,index2]
                    rw_tend, cqw = pg_buoy_w(rw_tend, p, cqw, mu, mub,
                                            c1f, c2f,
                                            rdnw, rdn, g, msftx, msfty,
                                            ids, ide, jds, jde, kds, kde,
                                            ims, ime, jms, jme, kms, kme,
                                            its, ite, jts, jte, kts, kte)
                    # w_damp = 0

                    ru_tend, rv_tend, rw_tend = coriolis(ru, rv, rw,
                                                        ru_tend,  rv_tend,  rw_tend,
                                                        msftx, msfty, msfux, msfuy,
                                                        msfvx, msfvy,
                                                        f, e, sina, cosa, fnm, fnp,
                                                        ids, ide, jds, jde, kds, kde,
                                                        ims, ime, jms, jme, kms, kme,
                                                        its, ite, jts, jte, kts, kte)
                    print("coriolis tend:",ru_tend[2, 160, 12], rv_tend[2, 160, 12])
                    ru_tend, rv_tend, rw_tend = curvature(ru, rv, rw, u, v, w, ru_tend, rv_tend, rw_tend,
                                                        msfux, msfuy, msfvx, msfvy, msftx, msfty,
                                                        clat, fnm, fnp, rdx, rdy,
                                                        ids, ide, jds, jde, kds, kde,
                                                        ims, ime, jms, jme, kms, kme,
                                                        its, ite, jts, jte, kts, kte )
                    print("curvature tend:",ru_tend[2, 160, 12], rv_tend[2, 160, 12])
                    # horizontal_diffusion opt=1
                    if rk_step == 1:
                        # --- horizontal + vertical diffusion (first RK step) ---
                        ru_tendf = horizontal_diffusion('u', u, ru_tendf, mut,
                                                        c1h, c2h,
                                                        msfux, msfuy, msfvx, msfvx_inv,
                                                        msfvy,msftx, msfty,
                                                        khdif, xkmh, rdx, rdy,
                                                        ids, ide, jds, jde, kds, kde,
                                                        ims, ime, jms, jme, kms, kme,
                                                        its, ite, jts, jte, kts, kte)
                        rv_tendf = horizontal_diffusion('v', v, rv_tendf, mut,
                                                        c1h, c2h,
                                                        msfux, msfuy, msfvx, msfvx_inv,
                                                        msfvy,msftx, msfty,
                                                        khdif, xkmh, rdx, rdy,
                                                        ids, ide, jds, jde, kds, kde,
                                                        ims, ime, jms, jme, kms, kme,
                                                        its, ite, jts, jte, kts, kte)

                        rw_tendf = horizontal_diffusion('w', w, rw_tendf, mut,
                                                        c1f, c2f,
                                                        msfux, msfuy, msfvx, msfvx_inv,
                                                        msfvy,msftx, msfty,
                                                        khdif, xkmh, rdx, rdy,
                                                        ids, ide, jds, jde, kds, kde,
                                                        ims, ime, jms, jme, kms, kme,
                                                        its, ite, jts, jte, kts, kte)
                        khdq = 3.* khdif
                        t_tendf = horizontal_diffusion_3dmp( 'm', t, t_tendf, mut,
                                                            c1h, c2h, t_init,
                                                            msfux, msfuy, msfvx, msfvx_inv,
                                                            msfvy, msftx, msfty,
                                                            khdq , xkhh, rdx, rdy,
                                                            ids, ide, jds, jde, kds, kde,
                                                            ims, ime, jms, jme, kms, kme,
                                                            its, ite, jts, jte, kts, kte)
                        ### if bl_pbl_physics == 0, add here ###
                        ru_tendf = vertical_diffusion_u( u, ru_tendf,
                                                        u_base, c1h, c2h,
                                                        alt, muu, rdn, rdnw, kvdif,
                                                        ids, ide, jds, jde, kds, kde,
                                                        ims, ime, jms, jme, kms, kme,
                                                        its, ite, jts, jte, kts, kte )
                        rv_tendf = vertical_diffusion_v( v, rv_tendf,
                                                        v_base, c1h, c2h,
                                                        alt, muv, rdn, rdnw, kvdif,
                                                        ids, ide, jds, jde, kds, kde,
                                                        ims, ime, jms, jme, kms, kme,
                                                        its, ite, jts, jte, kts, kte )
                        rw_tendf = vertical_diffusion( 'w', w, rw_tendf,
                                                    c1f, c2f,
                                                    alt, mut, rdn, rdnw, kvdif,
                                                    ids, ide, jds, jde, kds, kde,
                                                    ims, ime, jms, jme, kms, kme,
                                                    its, ite, jts, jte, kts, kte )
                        kvdq = 3. * kvdif
                        t_tendf = vertical_diffusion_3dmp( t, t_tendf, t_init,
                                                        c1h, c2h,
                                                        alt, mut, rdn, rdnw, kvdq ,
                                                        ids, ide, jds, jde, kds, kde,
                                                        ims, ime, jms, jme, kms, kme,
                                                        its, ite, jts, jte, kts, kte )
                    ########################################
                    if rk_step == 1:
                        # relax_bdy_dry
                        # --- lateral boundary tendencies: relaxation ---
                        u_save = relax_bdytend(ru, u_save,
                                                u_bxs,u_bxe,u_bys,u_bye,u_btxs,u_btxe,u_btys,u_btye,
                                                'u', spec_bdy_width, spec_zone, relax_zone,
                                                dtbc, fcx, gcx,
                                                ids,ide, jds,jde, kds,kde,
                                                ims,ime, jms,jme, kms,kme,
                                                ips,ipe, jps,jpe, kps,kpe,
                                                its,ite, jts,jte, kts,kte)
                        v_save = relax_bdytend(rv, v_save,
                                                v_bxs,v_bxe,v_bys,v_bye,v_btxs,v_btxe,v_btys,v_btye,
                                                'v', spec_bdy_width, spec_zone, relax_zone,
                                                dtbc, fcx, gcx,
                                                ids,ide, jds,jde, kds,kde,
                                                ims,ime, jms,jme, kms,kme,
                                                ips,ipe, jps,jpe, kps,kpe,
                                                its,ite, jts,jte, kts,kte )

                        i_start = max(its-1, ids)
                        i_end = min(ite+1, ide-1)
                        j_start = max(jts-1, jds)
                        j_end = min(jte+1, jde-1)
                        rfield = mass_weight(ph, mut , rfield , c1f, c2f,
                                            ids,ide, jds,jde, kds,kde,
                                            ims,ime, jms,jme, kms,kme,
                                            its-1,ite+1 , jts-1,jte+1 ,
                                            kts,kte,
                                            i_start,i_end, j_start,j_end, kts,kte)
                        ph_save = relax_bdytend(rfield, ph_save,
                                                ph_bxs,ph_bxe,ph_bys,ph_bye, ph_btxs,ph_btxe,ph_btys,ph_btye,
                                                'h', spec_bdy_width, spec_zone, relax_zone,
                                                dtbc, fcx, gcx,
                                                ids,ide, jds,jde, kds,kde,
                                                ims,ime, jms,jme, kms,kme,
                                                ips,ipe, jps,jpe, kps,kpe,
                                                its,ite, jts,jte, kts,kte)
                        rfield = mass_weight(t, mut , rfield , c1h, c2h,
                                            ids,ide, jds,jde, kds,kde,
                                            ims,ime, jms,jme, kms,kme,
                                            its-1,ite+1 , jts-1,jte+1 ,
                                            kts,kte,
                                            i_start,i_end, j_start,j_end, kts,kte) ### 原程序fortran为kte-1
                        t_save = relax_bdytend(rfield, t_save,
                                                t_bxs,t_bxe,t_bys,t_bye, t_btxs,t_btxe,t_btys,t_btye,
                                                't', spec_bdy_width, spec_zone, relax_zone,
                                                dtbc, fcx, gcx,
                                                ids,ide, jds,jde, kds,kde,
                                                ims,ime, jms,jme, kms,kme,
                                                ips,ipe, jps,jpe, kps,kpe,
                                                its,ite, jts,jte, kts,kte)
                        mu_tend_e = relax_bdytend(mu_e, mu_tend_e,
                                                mu_bxs,mu_bxe,mu_bys,mu_bye, mu_btxs,mu_btxe,mu_btys,mu_btye,
                                                'm', spec_bdy_width, spec_zone, relax_zone,
                                                dtbc, fcx, gcx,
                                                ids,ide, jds,jde, 0, 1,
                                                ims,ime, jms,jme, 0, 1,
                                                ips,ipe, jps,jpe, 0, 1,
                                                its,ite, jts,jte, 0, 1)
                        mu_tend = mu_tend_e[0,:,:] + 0.
                        # rk_addtend_dry
                    # --- accumulate dry tendencies ---
                    ru_tend, rv_tend, rw_tend, ph_tend, t_tend, mu_tend = \
                        rk_addtend_dry(ru_tend, rv_tend, rw_tend, ph_tend, t_tend,
                                    ru_tendf, rv_tendf, rw_tendf, ph_tendf, t_tendf,
                                    u_save, v_save, w_save, ph_save, t_save,
                                    mu_tend, mu_tendf, rk_step, c1h, c2h,
                                    h_diabatic, mut, msftx, msfty, msfux, msfuy,
                                    msfvx, msfvx_inv, msfvy,
                                    ids,ide, jds,jde, kds,kde,
                                    ims,ime, jms,jme, kms,kme,
                                    ips,ipe, jps,jpe, kps,kpe,
                                    its,ite, jts,jte, kts,kte)
                    # spec_bdy_dry

                    # --- lateral boundary tendencies: specified ---
                    ru_tend = spec_bdytend(ru_tend,
                                        u_bxs,u_bxe,u_bys,u_bye, u_btxs,u_btxe,u_btys,u_btye,
                                        'u', spec_bdy_width, spec_zone,
                                        ids,ide, jds,jde, kds,kde,
                                        ims,ime, jms,jme, kms,kme,
                                        ips,ipe, jps,jpe, kps,kpe,
                                        its,ite, jts,jte, kts,kte)
                    rv_tend = spec_bdytend(rv_tend,
                                        v_bxs,v_bxe,v_bys,v_bye, v_btxs,v_btxe,v_btys,v_btye,
                                        'v', spec_bdy_width, spec_zone,
                                        ids,ide, jds,jde, kds,kde,
                                        ims,ime, jms,jme, kms,kme,
                                        ips,ipe, jps,jpe, kps,kpe,
                                        its,ite, jts,jte, kts,kte)
                    ph_tend = spec_bdytend(ph_tend,
                                        ph_bxs,ph_bxe,ph_bys,ph_bye, ph_btxs,ph_btxe,ph_btys,ph_btye,
                                        'h', spec_bdy_width, spec_zone,
                                        ids,ide, jds,jde, kds,kde,
                                        ims,ime, jms,jme, kms,kme,
                                        ips,ipe, jps,jpe, kps,kpe,
                                        its,ite, jts,jte, kts,kte )
                    t_tend = spec_bdytend(t_tend,
                                        t_bxs,t_bxe,t_bys,t_bye, t_btxs,t_btxe,t_btys,t_btye,
                                        't', spec_bdy_width, spec_zone,
                                        ids,ide, jds,jde, kds,kde,
                                        ims,ime, jms,jme, kms,kme,
                                        ips,ipe, jps,jpe, kps,kpe,
                                        its,ite, jts,jte, kts,kte )
                    mu_tend_e = spec_bdytend(mu_tend_e,
                                            mu_bxs,mu_bxe,mu_bys,mu_bye, mu_btxs,mu_btxe,mu_btys,mu_btye,
                                            'm', spec_bdy_width, spec_zone,
                                            ids,ide, jds,jde, 0, 1,
                                            ims,ime, jms,jme, 0, 1,
                                            ips,ipe, jps,jpe, 0, 1,
                                            its,ite, jts,jte, 0, 1)
                    mu_tend = mu_tend_e[0,:,:] + 0.
                    # small step prep
                    # --- small (acoustic) step setup ---
                    u_1, v_1, w_1, t_1, ph_1, u_save, v_save, w_save, t_save, ph_save, \
                    u, v, w, t, ph, c2a, ww1, mu_1, mu, muus, muvs, muts, mudf, mu_save = \
                        small_step_prep(u_1, u, v_1, v, w_1, w,
                                        t_1, t, ph_1, ph,
                                        mub, mu_1, mu, muu, muus,
                                        muv, muvs, mut, muts, mudf,
                                        c1h, c2h, c1f, c2f,
                                        c3h, c4h, c3f, c4f,
                                        u_save, v_save, w_save, t_save, ph_save, mu_save,
                                        ww, ww1, c2a, pb, p, alt,
                                        msfux, msfuy, msfvx, msfvx_inv, msfvy, msftx, msfty,
                                        rdx, rdy, rk_step,
                                        ids,ide, jds,jde, kds,kde,
                                        ims,ime, jms,jme, kms,kme,
                                        its,ite, jts,jte, kts,kte)
                    al, p, pm1 = calc_p_rho(al, p, ph,
                                            alt, t, t_save, c2a, pm1,
                                            mu, muts,
                                            c1h, c2h, c1f, c2f,
                                            c3h, c4h, c3f, c4f,
                                            znu, t0,
                                            rdnw, dnw, smdiv,
                                            non_hydrostatic,0,
                                            ids, ide, jds, jde, kds, kde,
                                            ims, ime, jms, jme, kms, kme,
                                            its, ite, jts, jte, kts,kte)
                    has_nan = torch.any(torch.isnan(p))
                    a, alpha, gamma = calc_coef_w(a,alpha,gamma,
                                                mut,
                                                c1h, c2h, c1f, c2f,
                                                c3h, c4h, c3f, c4f,
                                                cqw,
                                                rdn, rdnw, c2a,
                                                dts_rk, g, epssm, top_lid,
                                                ids,ide, jds,jde, kds,kde,
                                                ims,ime, jms,jme, kms,kme,
                                                its,ite, jts,jte, kts,kte)
                    # set_phys_bc2_tim
                    ru_tend = set_physical_bc3d(ru_tend, 'u',
                                                ids,ide, jds,jde, kds,kde,
                                                ims,ime, jms,jme, kms,kme,
                                                ips,ipe, jps,jpe, kps,kpe,
                                                its,ite, jts,jte, kts,kte)
                    rv_tend = set_physical_bc3d(rv_tend, 'v',
                                                ids,ide, jds,jde, kds,kde,
                                                ims,ime, jms,jme, kms,kme,
                                                ips,ipe, jps,jpe, kps,kpe,
                                                its,ite, jts,jte, kts,kte)
                    ph = set_physical_bc3d(ph, 'w',
                                        ids,ide, jds,jde, kds,kde,
                                        ims,ime, jms,jme, kms,kme,
                                        ips,ipe, jps,jpe, kps,kpe,
                                        its,ite, jts,jte, kts,kte)
                    al = set_physical_bc3d(al, 'p',
                                        ids,ide, jds,jde, kds,kde,
                                        ims,ime, jms,jme, kms,kme,
                                        ips,ipe, jps,jpe, kps,kpe,
                                        its,ite, jts,jte, kts,kte)
                    p = set_physical_bc3d(p, 'p',
                                        ids,ide, jds,jde, kds,kde,
                                        ims,ime, jms,jme, kms,kme,
                                        ips,ipe, jps,jpe, kps,kpe,
                                        its,ite, jts,jte, kts,kte)
                    t_1 = set_physical_bc3d(t_1, 't',
                                            ids,ide, jds,jde, kds,kde,
                                            ims,ime, jms,jme, kms,kme,
                                            ips,ipe, jps,jpe, kps,kpe,
                                            its,ite, jts,jte, kts,kte)
                    t_save = set_physical_bc3d(t_save, 't',
                                            ids,ide, jds,jde, kds,kde,
                                            ims,ime, jms,jme, kms,kme,
                                            ips,ipe, jps,jpe, kps,kpe,
                                            its,ite, jts,jte, kts,kte)
                    mu_1 = set_physical_bc2d(mu_1, 't',
                                            ids,ide, jds,jde,
                                            ims,ime, jms,jme,
                                            ips,ipe, jps,jpe,
                                            its,ite, jts,jte )
                    mu = set_physical_bc2d(mu, 't',
                                        ids,ide, jds,jde,
                                        ims,ime, jms,jme,
                                        ips,ipe, jps,jpe,
                                        its,ite, jts,jte )
                    mudf = set_physical_bc2d(mudf, 't',
                                            ids,ide, jds,jde,
                                            ims,ime, jms,jme,
                                            ips,ipe, jps,jpe,
                                            its,ite, jts,jte )
                    # small_steps

                    ### (5) Small (acoustic) time steps ------------------------------
                    for iteration in range(1,number_of_small_timesteps+1):
                        u, v = advance_uv(u, ru_tend, v, rv_tend,
                                        p, pb,
                                        ph, php, alt, al, mu,
                                        muu, cqu, muv, cqv, mudf,
                                        c1h, c2h, c1f, c2f,
                                        c3h, c4h, c3f, c4f,
                                        msfux, msfuy, msfvx,
                                        msfvx_inv, msfvy,
                                        rdx, rdy, dts_rk,
                                        cf1, cf2, cf3, fnm, fnp,
                                        emdiv,
                                        rdnw, spec_zone,
                                        non_hydrostatic, top_lid,
                                        ids, ide, jds, jde, kds, kde,
                                        ims, ime, jms, jme, kms, kme,
                                        its, ite, jts, jte, kts, kte)
                        u = spec_bdyupdate(u, ru_tend, dts_rk,
                                        'u', spec_zone,
                                        ids,ide, jds,jde, kds,kde,
                                        ims,ime, jms,jme, kms,kme,
                                        ips,ipe, jps,jpe, kps,kpe,
                                        its,ite, jts,jte, kts,kte)
                        v = spec_bdyupdate(v, rv_tend, dts_rk,
                                        'v', spec_zone,
                                        ids,ide, jds,jde, kds,kde,
                                        ims,ime, jms,jme, kms,kme,
                                        ips,ipe, jps,jpe, kps,kpe,
                                        its,ite, jts,jte, kts,kte)

                        ww,ww1,t,t_2save,ru_m,rv_m,ww_m,muave,muts,mudf,mu =  \
                            advance_mu_t(ww, ww1, u, u_save, v, v_save,
                                        mu, mut, muave, muts, muu, muv, mudf,
                                        c1h, c2h, c1f, c2f,
                                        c3h, c4h, c3f, c4f,
                                        ru_m, rv_m, ww_m, t, t_save,
                                        t_2save, t_tend, mu_tend,
                                        rdx, rdy, dts_rk, epssm,
                                        dnw, fnm, fnp, rdnw,
                                        msfux, msfuy, msfvx, msfvx_inv,
                                        msfvy, msftx, msfty,
                                        iteration,
                                        ids, ide, jds, jde, kds, kde,
                                        ims, ime, jms, jme, kms, kme,
                                        its, ite, jts, jte, kts, kte)
                        t = spec_bdyupdate(t, t_tend, dts_rk,
                                        't', spec_zone,
                                        ids,ide, jds,jde, kds,kde,
                                        ims,ime, jms,jme, kms,kme,
                                        ips,ipe, jps,jpe, kps,kpe,
                                        its,ite, jts,jte, kts,kte)
                        mu = spec_bdyupdate(mu, mu_tend, dts_rk,
                                            'm', spec_zone,
                                            ids,ide, jds,jde, 0, 0,
                                            ims,ime, jms,jme, 0, 0,
                                            ips,ipe, jps,jpe, 0, 0,
                                            its,ite, jts,jte, 0, 0,)
                        muts = spec_bdyupdate(muts, mu_tend, dts_rk,
                                            'm', spec_zone,
                                            ids,ide, jds,jde, 0, 0,
                                            ims,ime, jms,jme, 0, 0,
                                            ips,ipe, jps,jpe, 0, 0,
                                            its,ite, jts,jte, 0, 0)
                        t_2save,w,ph = advance_w(w, rw_tend, ww, w_save, u, v,      #### t_2save or t_2ave???
                                                mu, mut, muave, muts,
                                                c1h, c2h, c1f, c2f,
                                                c3h, c4h, c3f, c4f,
                                                t_2save, t, t_save,
                                                ph, ph_save, phb, ph_tend,
                                                ht, c2a, cqw, alt, alb,
                                                a, alpha, gamma,
                                                rdx, rdy, dts_rk, t0, epssm,
                                                dnw, fnm, fnp, rdnw, rdn,
                                                cf1, cf2, cf3, msftx, msfty,
                                                top_lid,
                                                ids,ide, jds,jde, kds,kde,
                                                ims,ime, jms,jme, kms,kme,
                                                its,ite, jts,jte, kts,kte)
                        ru_m, rv_m, ww_m = sumflux(u, v, ww,
                                                u_save, v_save, ww1,
                                                muu, muv,
                                                c1h, c2h, c1f, c2f,
                                                c3h, c4h, c3f, c4f,
                                                ru_m, rv_m, ww_m, epssm,
                                                msfux, msfuy, msfvx, msfvx_inv, msfvy,
                                                iteration , number_of_small_timesteps,
                                                ids,ide, jds,jde, kds,kde,
                                                ims,ime, jms,jme, kms,kme,
                                                its,ite, jts,jte, kts,kte)
                        ph = spec_bdyupdate_ph(ph_save, ph,
                                            ph_tend, mu_tend, muts,
                                            c1f, c2f, dts_rk,
                                            'h', spec_zone,
                                            ids,ide, jds,jde, kds,kde,
                                            ims,ime, jms,jme, kms,kme,
                                            ips,ipe, jps,jpe, kps,kpe,
                                            its,ite, jts,jte, kts,kte)
                        w = zero_grad_bdy(w,
                                        'w', spec_zone,
                                        ids,ide, jds,jde, kds,kde,
                                        ims,ime, jms,jme, kms,kme,
                                        ips,ipe, jps,jpe, kps,kpe,
                                        its,ite, jts,jte, kts,kte)
                        al, p, pm1 = calc_p_rho(al, p, ph,
                                                alt, t, t_save, c2a, pm1,
                                                mu, muts,
                                                c1h, c2h, c1f, c2f,
                                                c3h, c4h, c3f, c4f,
                                                znu, t0,
                                                rdnw, dnw, smdiv,
                                                non_hydrostatic,iteration,
                                                ids, ide, jds, jde, kds, kde,
                                                ims, ime, jms, jme, kms, kme,
                                                its,ite, jts,jte, kts,kte)
                        ph = set_physical_bc3d(ph, 'w',
                                            ids,ide, jds,jde, kds,kde,
                                            ims,ime, jms,jme, kms,kme,
                                            ips,ipe, jps,jpe, kps,kpe,
                                            its,ite, jts,jte, kts,kte)
                        al = set_physical_bc3d(al, 'p',
                                            ids,ide, jds,jde, kds,kde,
                                            ims,ime, jms,jme, kms,kme,
                                            ips,ipe, jps,jpe, kps,kpe,
                                            its,ite, jts,jte, kts,kte)
                        p = set_physical_bc3d(p, 'p',
                                            ids,ide, jds,jde, kds,kde,
                                            ims,ime, jms,jme, kms,kme,
                                            ips,ipe, jps,jpe, kps,kpe,
                                            its,ite, jts,jte, kts,kte)
                        muts = set_physical_bc2d(muts, 't',
                                                ids,ide, jds,jde,
                                                ims,ime, jms,jme,
                                                ips,ipe, jps,jpe,
                                                its,ite, jts,jte )
                        mu = set_physical_bc2d(mu, 't',
                                            ids,ide, jds,jde,
                                            ims,ime, jms,jme,
                                            ips,ipe, jps,jpe,
                                            its,ite, jts,jte )
                        mudf = set_physical_bc2d(mudf, 't',
                                                ids,ide, jds,jde,
                                                ims,ime, jms,jme,
                                                ips,ipe, jps,jpe,
                                                its,ite, jts,jte )
                        print("444:",p[2, 160, 12], u[2, 160, 12], v[2, 160, 12], t[2, 160, 12], w[2, 160, 12])
                    # end do small steps
                    # change time-perturbation variables back to full perturbation variables.first get updated mu at u and v points
                    muus, muvs = calc_mu_uv_1(muts, muus, muvs,
                                            ids, ide, jds, jde, kds, kde,
                                            ims, ime, jms, jme, kms, kme,
                                            its, ite, jts, jte, kts, kte)
                    #muv_e[kds:kde-1, j_start:j_endv, i_start:i_end] )
                    u, v, w, ph, ww, t, mu = small_step_finish(u, u_1, v, v_1, w, w_1,
                                                            t, t_1, ph, ph_1, ww, ww1,
                                                            mu, mu_1,
                                                            mut, muts, muu, muus, muv, muvs,
                                                            c1h, c2h, c1f, c2f,
                                                            c3h, c4h, c3f, c4f,
                                                            u_save, v_save, w_save,
                                                            t_save, ph_save, mu_save,
                                                            msfux, msfuy, msfvx, msfvy,
                                                            msftx, msfty,
                                                            h_diabatic,
                                                            number_of_small_timesteps,dts_rk,
                                                            rk_step, rk_order,
                                                            ids,ide, jds,jde, kds,kde,
                                                            ims,ime, jms,jme, kms,kme,
                                                            its,ite, jts,jte, kts,kte)

                    if rk_step == rk_order:
                        ru_m = set_physical_bc3d(ru_m, 'u',
                                                ids,ide, jds,jde, kds,kde,
                                                ims,ime, jms,jme, kms,kme,
                                                ips,ipe, jps,jpe, kps,kpe,
                                                its,ite, jts,jte, kts,kte)
                        rv_m = set_physical_bc3d(rv_m, 'v',
                                                ids,ide, jds,jde, kds,kde,
                                                ims,ime, jms,jme, kms,kme,
                                                ips,ipe, jps,jpe, kps,kpe,
                                                its,ite, jts,jte, kts,kte)
                        ww_m = set_physical_bc3d(ww_m, 'w',
                                                ids,ide, jds,jde, kds,kde,
                                                ims,ime, jms,jme, kms,kme,
                                                ips,ipe, jps,jpe, kps,kpe,
                                                its,ite, jts,jte, kts,kte)
                        mut = set_physical_bc2d(mut, 't',
                                                ids,ide, jds,jde,
                                                ims,ime, jms,jme,
                                                ips,ipe, jps,jpe,
                                                its,ite, jts,jte )
                        muts = set_physical_bc2d(muts, 't',
                                                ids,ide, jds,jde,
                                                ims,ime, jms,jme,
                                                ips,ipe, jps,jpe,
                                                its,ite, jts,jte )
                    # small step finish, moist_adv_opt = 1 ORIGINAL = 0
                    # moist_scalar_advance
                    # for im in range(1,7):
                    # --- moisture + scalar species: advection & RK update ---
                    moist_tend, advect_tend, \
                        h_tendency, z_tendency = rk_scalar_tend(1, 6, False,
                                                                rk_step, dt_rk,
                                                                ru_m, rv_m, ww_m, muts, mub, mu_1,
                                                                c1h, c2h, alt,
                                                                moist_old, moist,
                                                                moist_tend, advect_tend,
                                                                h_tendency, z_tendency,
                                                                RQVFTEN,
                                                                qv_base, True, fnm, fnp,
                                                                msfux, msfuy, msfvx, msfvx_inv,
                                                                msfvy, msftx, msfty,
                                                                rdx, rdy, rdn, rdnw,
                                                                khdif, kvdif, xkhh,
                                                                diff_6th_opt, diff_6th_factor,
                                                                moist_adv_opt,
                                                                ids, ide, jds, jde, kds, kde,
                                                                ims, ime, jms, jme, kms, kme,
                                                                its, ite, jts, jte, kts, kte)

                    if rk_step == 1:
                        moist_tend[P_QV,:,:,:] = relax_bdy_scalar(moist_tend[P_QV,:,:,:],
                                                                moist[P_QV,:,:,:], mut, c1h, c2h,
                                                                qv_bxs,qv_bxe,qv_bys,qv_bye,
                                                                qv_btxs,qv_btxe,qv_btys,qv_btye,
                                                                spec_bdy_width, spec_zone, relax_zone,
                                                                dtbc, fcx, gcx,
                                                                ids,ide, jds,jde, kds,kde,
                                                                ims,ime, jms,jme, kms,kme,
                                                                ips,ipe, jps,jpe, kps,kpe,
                                                                its, ite, jts, jte, kts, kte)
                        moist_tend[P_QV,:,:,:] = spec_bdy_scalar(moist_tend[P_QV,:,:,:],
                                                                qv_bxs,qv_bxe,qv_bys,qv_bye,
                                                                qv_btxs,qv_btxe,qv_btys,qv_btye,
                                                                spec_bdy_width, spec_zone,
                                                                ids,ide, jds,jde, kds,kde,
                                                                ims,ime, jms,jme, kms,kme,
                                                                ips,ipe, jps,jpe, kps,kpe,
                                                                its, ite, jts, jte, kts, kte)

                    moist_old, moist = rk_update_scalar(1, 6,
                                                        moist_old, moist, moist_tend,
                                                        advect_tend, h_tendency, z_tendency,
                                                        msftx, msfty, c1h, c2h,
                                                        mu_1, mu, mub,
                                                        rk_step, dt_rk, spec_zone,
                                                        False,
                                                        ids, ide, jds, jde, kds, kde,
                                                        ims, ime, jms, jme, kms, kme,
                                                        its, ite, jts, jte, kts, kte)
                    for im in range(2,7):
                        moist[im,:,:,:] = flow_dep_bdy(moist[im,:,:,:],
                                                    ru_m,rv_m,1,
                                                    ids,ide, jds,jde, kds,kde,
                                                    ims,ime, jms,jme, kms,kme,
                                                    ips,ipe, jps,jpe, kps,kpe,
                                                    its,ite, jts,jte, kts,kte )

                    # other scalar species num_3d_s
                    scalar_tend, advect_tend_sca, \
                        h_tendency, z_tendency = rk_scalar_tend(1, 2, False,
                                                                rk_step, dt_rk,
                                                                ru_m, rv_m, ww_m, muts, mub, mu_1,
                                                                c1h, c2h, alt,
                                                                scalar_old, scalar,
                                                                scalar_tend, advect_tend_sca,
                                                                h_tendency, z_tendency,
                                                                RQVFTEN,
                                                                qv_base, False, fnm, fnp,
                                                                msfux, msfuy, msfvx, msfvx_inv,
                                                                msfvy, msftx, msfty,
                                                                rdx, rdy, rdn, rdnw,
                                                                khdif, kvdif, xkhh,
                                                                diff_6th_opt, diff_6th_factor,
                                                                scalar_adv_opt,
                                                                ids, ide, jds, jde, kds, kde,
                                                                ims, ime, jms, jme, kms, kme,
                                                                its, ite, jts, jte, kts, kte)
                    # rk_update_scalar
                    scalar_old, scalar = rk_update_scalar(1, 2,
                                                        scalar_old, scalar, scalar_tend,
                                                        advect_tend_sca, h_tendency, z_tendency,
                                                        msftx, msfty, c1h, c2h,
                                                        mu_1, mu, mub,
                                                        rk_step, dt_rk, spec_zone,
                                                        False,
                                                        ids, ide, jds, jde, kds, kde,
                                                        ims, ime, jms, jme, kms, kme,
                                                        its, ite, jts, jte, kts, kte)
                    # update pressure and density
                    # --- diagnose pressure / density / geopotential ---
                    al, p, ph = calc_p_rho_phi(moist, n_moist, hypsometric_opt,
                                            al, alb, mu, muts,
                                            c1h, c2h, c3h, c4h, c3f, c4f,
                                            ph, phb, p, pb,
                                            t, p0, t0, p_top, znu, znw, dnw, rdnw,
                                            rdn, True,
                                            ids, ide, jds, jde, kds, kde,
                                            ims, ime, jms, jme, kms, kme,
                                            its, ite, jts, jte, kts, kte)
                    has_nan = torch.any(torch.isnan(p))
                    nan_mask = torch.isnan(p)
                    nan_indices = nan_mask.nonzero()
                    if has_nan:
                        print("in nan p: True")
                        print(nan_indices)
                        sys.exit(1)
                    # rk_phys_bc_dry_2
                    if rk_step < rk_order:
                        u = set_physical_bc3d(u, 'U',
                                            ids,ide, jds,jde, kds,kde,
                                            ims,ime, jms,jme, kms,kme,
                                            ips,ipe, jps,jpe, kps,kpe,
                                            its,ite, jts,jte, kts,kte )
                        v = set_physical_bc3d(v, 'V',
                                            ids,ide, jds,jde, kds,kde,
                                            ims,ime, jms,jme, kms,kme,
                                            ips,ipe, jps,jpe, kps,kpe,
                                            its,ite, jts,jte, kts,kte )
                        w = set_physical_bc3d(w, 'w',
                                            ids,ide, jds,jde, kds,kde,
                                            ims,ime, jms,jme, kms,kme,
                                            ips,ipe, jps,jpe, kps,kpe,
                                            its,ite, jts,jte, kts,kte )
                        t = set_physical_bc3d(t, 'p',
                                            ids,ide, jds,jde, kds,kde,
                                            ims,ime, jms,jme, kms,kme,
                                            ips,ipe, jps,jpe, kps,kpe,
                                            its,ite, jts,jte, kts,kte )
                        ph = set_physical_bc3d(ph, 'w',
                                            ids,ide, jds,jde, kds,kde,
                                            ims,ime, jms,jme, kms,kme,
                                            ips,ipe, jps,jpe, kps,kpe,
                                            its,ite, jts,jte, kts,kte )
                        mu = set_physical_bc2d(mu, 't',
                                            ids,ide, jds,jde,
                                            ims,ime, jms,jme,
                                            ips,ipe, jps,jpe,
                                            its,ite, jts,jte )
                        # moistrure, scalar bc set
                        for im in range(1,7):
                            moist[im,:,:,:] = set_physical_bc3d(moist[im,:,:,:], 'p',
                                                                ids,ide, jds,jde, kds,kde,
                                                                ims,ime, jms,jme, kms,kme,
                                                                ips,ipe, jps,jpe, kps,kpe,
                                                                its,ite, jts,jte, kts,kte )
                        for im in range(1,3):
                            scalar[im,:,:,:] = set_physical_bc3d(scalar[im,:,:,:], 'p',
                                                                ids,ide, jds,jde, kds,kde,
                                                                ims,ime, jms,jme, kms,kme,
                                                                ips,ipe, jps,jpe, kps,kpe,
                                                                its,ite, jts,jte, kts,kte )
                # END DO Runge_Kutta_loop
                # phy prep 2
                rublten, rvblten, rthblten, rqvblten, rqcblten, rqiblten = \
                                        phy_prep_part2(mut,muu,muv,
                                                        c1h, c2h, c1f, c2f,
                                                        rthblten, rublten, rvblten,
                                                        rqvblten, rqcblten, rqiblten,
                                                        ids, ide, jds, jde, kds, kde,
                                                        ims, ime, jms, jme, kms, kme,
                                                        its, ite, jts, jte, kts, kte)
            # start ML part
                # t_diff = t - t_last_step
            # t = t_last_step + t_diff
                # --- physics: prepare moisture, run WSM6, finalize ---
                rho, th_phy, pi_phy, p_phy, z, z_at_w, dz8w, p8w, \
                    h_diabatic, qv_diabatic, qc_diabatic,   \
                    t, t_1 = moist_physics_prep_em(t, t_1, t0, rho, al, alb,        ### 注意t加了300？
                                                    p, p8w, p0, pb, ph, phb,
                                                    th_phy, pi_phy, p_phy,
                                                    z, z_at_w, dz8w,
                                                    dtm,h_diabatic,
                                                    moist[P_QV,:,:,:],qv_diabatic,
                                                    moist[P_QC,:,:,:],qc_diabatic,
                                                    fnm, fnp,
                                                    ids+1,ide-1, jds+1,jde-1, kds,kde,
                                                    ims,ime, jms,jme, kms,kme,
                                                    its+1,ite-1, jts+1,jte-1, kts,kte)


                ### (6) Physics: WSM6 microphysics -------------------------------
                th_phy, moist[P_QV,:,:,:], moist[P_QC,:,:,:], moist[P_QR,:,:,:], moist[P_QI,:,:,:], moist[P_QS,:,:,:], moist[P_QG,:,:,:], rainnc,  \
                    rainncv, snownc, snowncv, graupelnc, graupelncv, sr, refl_10cm = \
                        microphysics_driver(th_phy, rho, pi_phy, p_phy, ht,
                                            dz8w, p8w, dtm, dx, dy,
                                            moist[P_QV,:,:,:], moist[P_QC,:,:,:], moist[P_QR,:,:,:],
                                            moist[P_QI,:,:,:], moist[P_QS,:,:,:], moist[P_QG,:,:,:],
                                            rainnc ,rainncv,snownc ,snowncv,sr,
                                            refl_10cm,1,1, # diagflag, do_radar_ref
                                            graupelnc ,graupelncv,
                                            0,#has_reqc,
                                            0,#has_reqi,
                                            0,#has_reqs,
                                            0,#re_cloud,
                                            0,#re_ice,
                                            0,#re_snow,
                                            ids+1,ide-1, jds+1,jde-1, kds,kde,
                                            ims,ime, jms,jme, kms,kme,
                                            its+1,ite-1, jts+1,jte-1, kts,kte )
            with torch.no_grad():
                t, t_1, th_phy, h_diabatic, qv_diabatic, qc_diabatic =  \
                    moist_physics_finish_em(t, t_1, t0, muts,
                                    th_phy, h_diabatic, dtm,
                                    moist[P_QV,:,:,:],qv_diabatic,
                                    moist[P_QC,:,:,:],qc_diabatic,
                                    ids+1,ide-1, jds+1,jde-1, kds,kde,
                                    ims,ime, jms,jme, kms,kme,
                                    its+1,ite-1, jts+1,jte-1, kts,kte)

                al, p, ph = calc_p_rho_phi(moist, n_moist, hypsometric_opt,
                                        al, alb, mu, muts,
                                        c1h, c2h, c3h, c4h, c3f, c4f,
                                        ph, phb, p, pb,
                                        t, p0, t0, p_top, znu, znw, dnw, rdnw,
                                        rdn, True,
                                        ids, ide, jds, jde, kds, kde,
                                        ims, ime, jms, jme, kms, kme,
                                        its, ite, jts, jte, kts, kte)

            has_nan = torch.any(torch.isnan(p))
            nan_mask = torch.isnan(p)
            nan_indices = nan_mask.nonzero()
            if has_nan:
                print("in nan p 2: True")
                print(nan_indices)
                sys.exit(1)
            # set_phys_bc_dry2
            u_1 = set_physical_bc3d(u_1, 'U',
                                ids,ide, jds,jde, kds,kde,
                                ims,ime, jms,jme, kms,kme,
                                ips,ipe, jps,jpe, kps,kpe,
                                its,ite, jts,jte, kts,kte )
            u = set_physical_bc3d(u, 'U',
                                ids,ide, jds,jde, kds,kde,
                                ims,ime, jms,jme, kms,kme,
                                ips,ipe, jps,jpe, kps,kpe,
                                its,ite, jts,jte, kts,kte )
            v_1 = set_physical_bc3d(v_1, 'V',
                                ids,ide, jds,jde, kds,kde,
                                ims,ime, jms,jme, kms,kme,
                                ips,ipe, jps,jpe, kps,kpe,
                                its,ite, jts,jte, kts,kte )
            v = set_physical_bc3d(v, 'V',
                                ids,ide, jds,jde, kds,kde,
                                ims,ime, jms,jme, kms,kme,
                                ips,ipe, jps,jpe, kps,kpe,
                                its,ite, jts,jte, kts,kte )
            w_1 = set_physical_bc3d(w_1, 'w',
                                ids,ide, jds,jde, kds,kde,
                                ims,ime, jms,jme, kms,kme,
                                ips,ipe, jps,jpe, kps,kpe,
                                its,ite, jts,jte, kts,kte )
            w = set_physical_bc3d(w, 'w',
                                ids,ide, jds,jde, kds,kde,
                                ims,ime, jms,jme, kms,kme,
                                ips,ipe, jps,jpe, kps,kpe,
                                its,ite, jts,jte, kts,kte )
            t_1 = set_physical_bc3d(t_1, 'p',
                                ids,ide, jds,jde, kds,kde,
                                ims,ime, jms,jme, kms,kme,
                                ips,ipe, jps,jpe, kps,kpe,
                                its,ite, jts,jte, kts,kte )
            t = set_physical_bc3d(t, 'p',
                                ids,ide, jds,jde, kds,kde,
                                ims,ime, jms,jme, kms,kme,
                                ips,ipe, jps,jpe, kps,kpe,
                                its,ite, jts,jte, kts,kte )
            ph_1 = set_physical_bc3d(ph_1, 'w',
                                ids,ide, jds,jde, kds,kde,
                                ims,ime, jms,jme, kms,kme,
                                ips,ipe, jps,jpe, kps,kpe,
                                its,ite, jts,jte, kts,kte )
            ph = set_physical_bc3d(ph, 'w',
                                ids,ide, jds,jde, kds,kde,
                                ims,ime, jms,jme, kms,kme,
                                ips,ipe, jps,jpe, kps,kpe,
                                its,ite, jts,jte, kts,kte )
            mu_1 = set_physical_bc2d(mu_1, 't',
                                ids,ide, jds,jde,
                                ims,ime, jms,jme,
                                ips,ipe, jps,jpe,
                                its,ite, jts,jte )
            mu = set_physical_bc2d(mu, 't',
                                ids,ide, jds,jde,
                                ims,ime, jms,jme,
                                ips,ipe, jps,jpe,
                                its,ite, jts,jte )

            tke_1 = set_physical_bc3d(tke_1, 'p',
                                    ids,ide, jds,jde, kds,kde,
                                    ims,ime, jms,jme, kms,kme,
                                    ips,ipe, jps,jpe, kps,kpe,
                                    its,ite, jts,jte, kts,kte )
            tke = set_physical_bc3d(tke, 'p',
                                    ids,ide, jds,jde, kds,kde,
                                    ims,ime, jms,jme, kms,kme,
                                    ips,ipe, jps,jpe, kps,kpe,
                                    its,ite, jts,jte, kts,kte )
            for im in range(1,7):
                moist[im,:,:,:] = set_physical_bc3d(moist[im,:,:,:], 'p',
                                                    ids,ide, jds,jde, kds,kde,
                                                    ims,ime, jms,jme, kms,kme,
                                                    ips,ipe, jps,jpe, kps,kpe,
                                                    its,ite, jts,jte, kts,kte )
            for im in range(1,3):
                scalar[im,:,:,:] = set_physical_bc3d(scalar[im,:,:,:], 'p',
                                                    ids,ide, jds,jde, kds,kde,
                                                    ims,ime, jms,jme, kms,kme,
                                                    ips,ipe, jps,jpe, kps,kpe,
                                                    its,ite, jts,jte, kts,kte )
            # spec_bdy_final

            ### (7) Post-RK boundary / state finalization --------------------
            # --- final specified boundary + surface w (end of step) ---
            u = spec_bdy_final(u, muus, c1h, c2h, msfuy,
                            u_bxs, u_bxe, u_bys, u_bye,
                            u_btxs, u_btxe, u_btys, u_btye,
                            'u', spec_bdy_width, spec_zone, dtbc,
                            ids,ide, jds,jde, kds,kde,
                            ims,ime, jms,jme, kms,kme,
                            ips,ipe, jps,jpe, kps,kpe,
                            its,ite, jts,jte, kts,kte)
            v = spec_bdy_final(v, muvs, c1h, c2h, msfvx,
                            v_bxs, v_bxe, v_bys, v_bye,
                            v_btxs, v_btxe, v_btys, v_btye,
                            'v', spec_bdy_width, spec_zone, dtbc,
                            ids,ide, jds,jde, kds,kde,
                            ims,ime, jms,jme, kms,kme,
                            ips,ipe, jps,jpe, kps,kpe,
                            its,ite, jts,jte, kts,kte)
            t = spec_bdy_final(t, muts, c1h, c2h, msfty,
                            t_bxs, t_bxe, t_bys, t_bye,
                            t_btxs, t_btxe, t_btys, t_btye,
                            't', spec_bdy_width, spec_zone, dtbc,
                            ids,ide, jds,jde, kds,kde,
                            ims,ime, jms,jme, kms,kme,
                            ips,ipe, jps,jpe, kps,kpe,
                            its,ite, jts,jte, kts,kte)

            ph = spec_bdy_final(ph, muts, c1h, c2h, msfty,
                            ph_bxs, ph_bxe, ph_bys, ph_bye,
                            ph_btxs, ph_btxe, ph_btys, ph_btye,
                            'h', spec_bdy_width, spec_zone, dtbc,
                            ids,ide, jds,jde, kds,kde,
                            ims,ime, jms,jme, kms,kme,
                            ips,ipe, jps,jpe, kps,kpe,
                            its,ite, jts,jte, kts,kte)
            moist[P_QV,:,:,:] = spec_bdy_final(moist[P_QV,:,:,:], muts, c1h, c2h, msfty,
                                            qv_bxs, qv_bxe, qv_bys, qv_bye,
                                            qv_btxs, qv_btxe, qv_btys, qv_btye,
                                            't', spec_bdy_width, spec_zone, dtbc,
                                            ids,ide, jds,jde, kds,kde,
                                            ims,ime, jms,jme, kms,kme,
                                            ips,ipe, jps,jpe, kps,kpe,
                                            its,ite, jts,jte, kts,kte)
            # reset surface w for consistency
            w = set_w_surface(znw, False,
                            w, ht, u, v, cf1, cf2, cf3, rdx, rdy,
                            msftx, msfty,
                            ids, ide, jds, jde, kds, kde,
                            ims, ime, jms, jme, kms, kme,
                            its, ite, jts, jte, kts, kte)

        # ---- final state: return the key fields ----
        return {
            "u": u,
            "v": v,
            "w": w,
            "t": t,
            "ph": ph,
            "mu": mu,
            "moist": moist,
        }


def main():
    """Run the 9 km WRF case end to end."""
    WrfSolver().solve()

