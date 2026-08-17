# PyWRF

**English** | [中文（简体）](README.zh.md)

**PyWRF** is the official code repository for the following paper:

> Chu, H., Cao, Y., Wang, R., Xu, J., Chen, L., Fan, X., Chen, S., Guo, J., &
> Wu, J. (2026). *A Novel ML–NWP Coupled Optimization Approach on a
> Reconstructed WRF Model*. Journal of Meteorological Research, 40(3), 653–668.

It is a pure **Python / PyTorch** reimplementation of the core of the
[WRF](https://github.com/NCAR/WRF) model — dynamical core, physics and lateral
boundary conditions — written as fully tensorized code that is **differentiable**
(autograd-friendly). It reconstructs WRF's thermodynamic framework together with
the WSM6 microphysics, and is intended for AI-for-weather research: a
physics-based core that can be embedded in / supervised by machine-learning
models, finetuned, or differentiated through.

> ⚠️ **Research prototype.** This is an independent reimplementation of WRF
> *algorithms*, not a drop-in replacement for the Fortran WRF model, and it is
> not affiliated with or endorsed by NCAR/NOAA. See
> [Known limitations](#known-limitations).

---

## Features

- **Dynamical core** — RK3 time integration, non-hydrostatic, terrain-following
  coordinates, momentum/heat/moisture advection, full and perturbation pressure,
  map factors, damping options.
- **Physics** — WSM6 single-moment microphysics (cloud/rain/ice/snow/graupel)
  with moist-adjustment hooks. The other WRF schemes (radiation, surface, PBL,
  cumulus) are out of scope for this release — see
  [Known limitations](#known-limitations).
- **Boundary conditions** — specified (relaxation + specified zones) and
  flow-dependent lateral boundary updates, mass-weighting.
- **Differentiable** — every operation is a PyTorch tensor op, so the whole
  core runs under `torch.autograd`. This is what enables the coupled ML–NWP
  optimization framework of the paper (observation-guided bias calibration
  trained through the integration with truncated backpropagation).
- **Matches a real case** — the shipped config reproduces the domain of a real
  9 km WRF run (230×230×41 grid points), with the real case data in `data/`.

## Project structure

```
PyWRF/
├── pywrf/
│   ├── config_params.py      # run/domain configuration, constants, device
│   ├── solver.py             # WrfSolver: orchestrates one full integration
│   ├── wrf_dynamics.py       # dynamical core (RK steps, advection, tendencies)
│   ├── wrf_physics.py        # physics (WSM6 microphysics, PBL, diffusion)
│   ├── wrf_boundary.py       # lateral boundary condition updates
│   ├── __init__.py           # exposes WrfSolver / main
│   └── __main__.py           # enables `python -m pywrf`
├── examples/
│   └── run_wrf_9km.py        # thin CLI entry point (calls pywrf.solver.main)
├── data/                     # real 9 km WRF case data (shipped with the repo)
├── pyproject.toml
└── LICENSE                   # MIT
```

## Requirements

- Python ≥ 3.8
- [PyTorch](https://pytorch.org/) ≥ 1.13 (CUDA build recommended; GPU required
  for practical runs)
- `numpy`, `xarray`, `netCDF4`, `matplotlib`

```bash
pip install -e .
```

## Usage

### 1. WRF input data

The model integrates a WRF state from NetCDF files produced by a real WRF run.
The **real 9 km case data ships with this repository** under `data/` (see
`data/README.md`):

| File | Description |
| ---- | ----------- |
| `wrf_inout_step3_<RUN>` | 3D input state (u, v, mu, mub, ...) |
| `wrfbdy_d01_<RUN>`      | lateral boundary tendencies (`U_BXS`, `T_BXS`, ...) |
| `wrfout_d01_<RUN>`      | reference output (used for `w`, etc.) |

The files are already in place, so you can run directly. To use a different run,
point the solver at their location:

```bash
export PYWRF_DATA_DIR=/path/to/data
export PYWRF_RUN_NAME=2024020106     # file-date suffix, default: 2024020106
```

### 2. Select the GPU

The device is configured centrally and defaults to GPU 0. For the setup this
code was originally written for, use GPU 2:

```bash
export PYWRF_GPU=2
```

If CUDA is unavailable the code falls back to CPU (very slow for practical runs).

### 3. Run

Any of the following are equivalent:

```bash
python examples/run_wrf_9km.py     # run from the repo root, no install needed
pywrf                              # after `pip install -e .` (console script)
python -m pywrf                    # after `pip install -e .`
```

The solver (`pywrf.solver.WrfSolver.solve`) loads the input/boundary/reference
files and integrates the configured case. It prints per-step progress and
returns the final key fields (`u`, `v`, `w`, `t`, `ph`, `mu`, `moist`).

## Configuration

- **Device** — `PYWRF_GPU` (default `0`). Defined once in
  `pywrf/config_params.py` and reused by all `pywrf` modules.
- **Case / domain** — edit `pywrf/config_params.py` (grid sizes, time step,
  `config_flags`, physics constants). The shipped values match a 9 km case with
  a 230×230×41 interior on a 240×240×41 grid.

## Attribution

**This repository is the official code of the following paper — please cite it
when you use PyWRF in your work:**

- Chu, H., Cao, Y., Wang, R., Xu, J., Chen, L., Fan, X., Chen, S., Guo, J., &
  Wu, J. (2026). *A Novel ML–NWP Coupled Optimization Approach on a
  Reconstructed WRF Model*. Journal of Meteorological Research, 40(3), 653–668.
  https://doi.org/10.1007/s13351-026-5154-1

PyWRF is an independent Python reimplementation of the numerical schemes of the
WRF model. For reference, the original WRF model is developed by NCAR, NOAA and
collaborators:

- Skamarock, W. C., et al. (2019). *A Description of the Advanced Research WRF
  Model Version 4*, NCAR/TN-556+STR.
- Hong, S.-Y., & Lim, J.-O. J. (2006). *The WRF single-moment 6-class
  microphysics scheme (WSM6)*. J. Korean Meteor. Soc., 42, 129–151.

PyWRF is not affiliated with or endorsed by NCAR, and is distributed under the
MIT license.

## Contributors

The entire PyWRF codebase was contributed by **Hai Chu (储海)**, the first
author of the paper above.

## Known limitations

- **Physics scope** — this release reconstructs WRF's *thermodynamic* framework
  and the WSM6 single-moment microphysics. The rest of the WRF physics suite
  (radiation, surface-layer, PBL and cumulus schemes) is not implemented; the
  PBL / diffusion hooks in the code are placeholders.
- **Validation scope** — as described in the paper, the model and the coupled
  ML–NWP framework were trained and evaluated on **8 mesoscale precipitation
  events over East China**. Accuracy on other regions, seasons or event types is
  not established.
- **Research prototype** — PyWRF is a differentiable reimplementation for
  ML–NWP research, not a drop-in replacement for the compiled Fortran WRF model:
  computational performance is far below the Fortran reference and the code is
  not operationally validated.
- **ML coupling not shipped** — the single-column neural-network coupling and
  the truncated-backpropagation training strategy described in the paper are the
  research application; this repository ships the standalone differentiable
  core.
- **Boundary data coverage** — the shipped `wrfbdy` holds 2 boundary time levels
  (6 h apart), so the time loop runs 720 steps (12 h); a longer run needs more
  boundary levels.
- **Single case** — the config hardcodes one 9 km single-domain case; other
  domains require editing `config_params.py`.
- **Machine-specific settings removed** — hardcoded GPU indices from the
  original working copy were replaced by the `PYWRF_GPU` variable.
- **fp16 variant** — a float16 variant of the physics module was present as a
  compiled `*.pyc` in the original archive but its source is not included.

## License

[MIT](LICENSE) © 2026 caotong0.
