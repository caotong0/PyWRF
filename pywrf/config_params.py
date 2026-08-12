"""PyWRF run configuration and physical constants.

Single source of truth for everything the WRF core needs: device selection,
the case / domain decomposition, moisture species indices, physical constants
and the WSM6 microphysics parameters.

NOTE on style: these are deliberately module-level globals (flat namespace),
not a dataclass. The numerical kernels in :mod:`pywrf.wrf_dynamics`,
:mod:`pywrf.wrf_physics` and :mod:`pywrf.wrf_boundary` reference them by bare
name (``g``, ``pi``, ``nzall``, ``device``, ...) hundreds of times, and the
modules import them with ``from pywrf.config_params import *``. Keeping them
as globals preserves that contract exactly. Underscore-prefixed names (e.g.
``_PYWRF_GPU``) are not exported by the star import.
"""

import os
import torch

# --- Device selection -------------------------------------------------------
# The WRF core runs on GPU. Set the PYWRF_GPU environment variable to pick a
# device, e.g.  PYWRF_GPU=2  (the setup this code was originally written for).
# Falls back to CPU when CUDA is unavailable.
_PYWRF_GPU = int(os.environ.get("PYWRF_GPU", "0"))
torch.set_default_device(_PYWRF_GPU)
if torch.cuda.is_available():
    torch.cuda.set_device(_PYWRF_GPU)
device = torch.device(f"cuda:{_PYWRF_GPU}" if torch.cuda.is_available() else "cpu")

# --- config_flags (namelist-style runtime options) --------------------------
hypsometric_opt = 1
w_damp = 0
diff_opt = 1
non_hydrostatic = True
rk_order = 3
diff_6th_opt = 0
diff_6th_factor = 0.25
moist_adv_opt = 0
scalar_adv_opt = 0
dampcoef = 0.2
zdamp = 5000.
damp_opt = 3
dx = 9000.
dy = 9000.
adv_moist_cond = True
time_step = 60
spec_bdy_width = 5
spec_zone = 1
relax_zone = 4
top_lid = False

# --- Run / timing -----------------------------------------------------------
# Shipped 9 km case: 6 h, 60 s main step, boundary arrays refreshed every 320
# steps. (run_secs / dtsteps are documentation-consistent; the solver time loop
# currently uses a fixed step count.)
run_secs = 21600        # 6 hours
dt = 60
dtsteps = 1200          # 21600 / 18
bdy_interval = 320      # boundary update interval (steps)
nx = 230
ny = 230
nz = 41
dxdy = 9000
rdx = 1. / 9000
rdy = 1. / 9000
soundsteps = 4

# --- Domain / tile decomposition (WRF index sets, 1-based Fortran ports) ----
ids = 5
ide = 235
jds = 5
jde = 235
kds = 0
kde = 41
ims = 0
ime = 240
jms = 0
jme = 240
kms = 0
kme = 41
its = 5
ite = 235
jts = 5
jte = 235
kts = 0
kte = 41
ips = 5
ipe = 235
jps = 5
jpe = 235
kps = 0
kpe = 41

# --- Full / allocated grid dimensions ----------------------------------------
nxfull = 240
nyfull = 240
nzfull = 41
nxall = 240
nyall = 240
nzall = 41

# --- Moisture / scalar species indices ---------------------------------------
P_QV = 1
P_QS = 5
P_QI = 4
P_QC = 2
P_QG = 6
P_QR = 3
n_moist = 7
n_scalar = 3
P_QNC = 0
P_QNI = 1
P_QNR = 2

# --- Diffusion / damping coefficients ----------------------------------------
smdiv = 0.1
emdiv = 0.01
kvdif = 0.
khdif = 0.

# --- Physical constants --------------------------------------------------------
r_d = 287.0
p1000mb = 100000.0
t00 = 300.0
cvpm = -0.7142857
A = 50.
p00 = 100000.0
p0 = 100000.0
t0 = 300.0
p_top = 1000.0
g = 9.810000
rcp = 0.2857143

SVP1 = 0.611199975013732910
SVP2 = 17.670000076293945312
SVP3 = 29.649999618530273438
SVPT0 = 273.149993896484375
t0c = 273.149993896484375
EP_2 = 0.621750414371490479
XLV = 2500000.

prandtl = 1. / 3.
c_s = 0.2500000
Cp = 1004.500
r_v = 461.6000

cpovcv = 1.40000
reradius = 1.5698588e-7
epssm = 0.1
rvovrd = 1.608362

Cpv = 1846.400
ep_1 = 0.608362436294555664
ep_2 = 0.621750414371490479
epsilon = 1.0000000e-15
XLS = 2850000.
XLF = 350000.0
rhoair0 = 1.280000
rhowater = 1000.000
cliq = 4190.000
cice = 2106.000
psat = 610.7800

# --- WSM6 single-moment microphysics parameters --------------------------------
dtcldcr = 120.
n0r = 8.0e6
avtr = 841.9
bvtr = 0.8
r0 = 0.8e-5
peaut = .55
xncr = 3.0e8
xmyu = 1.718e-5
avts = 11.72
bvts = 0.41
n0smax = 1.0e11
lamdarmax = 8.0e4
lamdasmax = 1.0e5
dicon = 11.9
dimax = 500.0e-6
n0s = 2.0e6
alpha = 0.12
pfrz1 = 100.
pfrz2 = 0.66
qcrmin = 1.0e-9
eacrc = 1.0
dens = 100.0
qs0 = 6.0e-4

qc0 = 5.0265482E-04
qck1 = 6.773895
pidnc = 523.5988
bvtr1 = 1.800000
bvtr2 = 2.900000
bvtr3 = 3.800000
bvtr4 = 4.800000
g1pbr = 0.9312252
g3pbr = 4.690783
g4pbr = 17.81730
g5pbro2 = 1.826591
pvtr = 2500.064
eacrr = 1.000000
pacrr = 2.4813369E+10
bvtr6 = 6.800000
g6pbr = 495.4590
precr1 = 3.9207076E+07
precr2 = 8.2585357E+08
roqimax = 8.1250037E-05
bvts1 = 1.410000
bvts2 = 2.705000
bvts3 = 3.410000
bvts4 = 4.410000
g1pbs = 0.8866756
g3pbs = 3.011565
g4pbs = 10.26533
n0g = 4000000.0
avtg = 330.0000
bvtg = 0.8000000
deng = 500.0000
lamdagmax = 60000.00
g5pbso2 = 1.550299
pvts = 20.05161
pacrs = 5.5442112E+07
precs1 = 5200000.0
precs2 = 1.8681944E+07
pidn0r = 2.5132743E+10
pidn0s = 6.2831853E+08
xlv1 = 2343.600
pacrc = 5.5442112E+07
pi = 3.141593
bvtg1 = 1.800000
bvtg2 = 2.900000
bvtg3 = 3.800000
bvtg4 = 4.800000
g1pbg = 0.9312252
g3pbg = 4.690783
g4pbg = 17.81730
g5pbgo2 = 1.826591
pvtg = 979.9514
pacrg = 4.8630548E+09
precg1 = 1.9603538E+07
precg2 = 2.5852333E+08
pidn0g = 6.2831857E+09
rslopermax = 1.2500000E-05
rslopesmax = 9.9999997E-06
rslopegmax = 1.6666667E-05
rsloperbmax = 1.1954404E-04
rslopesbmax = 8.9125093E-03
rslopegbmax = 1.5048006E-04
rsloper2max = 1.5624999E-10
rslopes2max = 9.9999994E-11
rslopeg2max = 2.7777777E-10
rsloper3max = 1.9531248E-15
rslopes3max = 9.9999990E-16
rslopeg3max = 4.6296297E-15
