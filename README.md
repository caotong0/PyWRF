# PyWRF

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

---

# PyWRF（中文版）

**PyWRF** 是以下论文的官方代码仓库：

> Chu, H., Cao, Y., Wang, R., Xu, J., Chen, L., Fan, X., Chen, S., Guo, J., &
> Wu, J. (2026). *A Novel ML–NWP Coupled Optimization Approach on a
> Reconstructed WRF Model*. Journal of Meteorological Research, 40(3), 653–668.

**PyWRF** 以纯 **Python / PyTorch** 重新实现了
[WRF](https://github.com/NCAR/WRF)（Weather Research and Forecasting，
天气研究与预报模式）的核心算法——涵盖动力核、物理过程和侧边界条件，
全部运算均为张量化操作，因此整体**可微**（兼容 autograd）。它实现了 WRF
的热力学框架与 WSM6 微物理，面向"人工智能+气象"研究：一个基于物理、
可嵌入机器学习模型、可微调、可端到端反向传播的动力核心。

> ⚠️ **研究原型。** 这是对 WRF *算法*的独立重实现，并非 Fortran WRF 模式
> 的替代品，且与 NCAR/NOAA 无任何隶属或背书关系。参见[已知限制](#已知限制-1)。

## 功能特性

- **动力核** — RK3 时间积分、非静力平衡、地形跟随坐标、动量/热/水汽平流、
  全气压与扰动气压、地图投影因子、阻尼选项。
- **物理过程** — WSM6 单矩微物理方案（云/雨/冰/雪/霰）及湿调整接口。
  其余 WRF 物理方案（辐射、陆面、边界层、积云）不在本版范围内，
  详见[已知限制](#已知限制-1)。
- **边界条件** — 指定侧边界（松弛区 + 指定区）与流依赖更新、质量加权。
- **可微** — 所有运算均为 PyTorch 张量运算，整个动力核可在 `torch.autograd`
  下运行。这也正是论文中耦合 ML–NWP 优化框架（通过截断反传在积分过程中
  训练、实现观测引导的偏差订正）所依赖的基础。
- **贴合真实个例** — 内置配置对应真实 9 km WRF 个例的区域设置
  （230×230×41 格点），真实个例数据位于 `data/` 目录。

## 目录结构

```
PyWRF/
├── pywrf/
│   ├── config_params.py      # 运行/区域配置、常数、设备选择
│   ├── solver.py             # WrfSolver：编排一次完整的模式积分
│   ├── wrf_dynamics.py       # 动力核（RK 步、平流、倾向项）
│   ├── wrf_physics.py        # 物理过程（WSM6 微物理、PBL、扩散）
│   ├── wrf_boundary.py       # 侧边界条件更新
│   ├── __init__.py           # 暴露 WrfSolver / main
│   └── __main__.py           # 支持 `python -m pywrf`
├── examples/
│   └── run_wrf_9km.py        # 精简 CLI 入口（调用 pywrf.solver.main）
├── data/                     # 真实 9 km 个例数据（随仓库发布）
├── pyproject.toml
└── LICENSE                   # MIT
```

## 依赖

- Python ≥ 3.8
- [PyTorch](https://pytorch.org/) ≥ 1.13（建议 CUDA 版本；实际运行需要 GPU）
- `numpy`、`xarray`、`netCDF4`、`matplotlib`

```bash
pip install -e .
```

## 使用方法

### 1. WRF 输入数据

求解器从真实 WRF 运行生成的 NetCDF 文件读取初始场并积分。**真实 9 km 个例
数据已随本仓库一并提供**，位于 `data/` 目录（见 `data/README.md`）：

| 文件 | 说明 |
| ---- | ---- |
| `wrf_inout_step3_<RUN>` | 三维输入场（u, v, mu, mub, ...） |
| `wrfbdy_d01_<RUN>`      | 侧边界倾向（`U_BXS`、`T_BXS`、...） |
| `wrfout_d01_<RUN>`      | 参考输出（用于读取 `w` 等） |

文件已就位，可直接运行。如需运行其它个例，可通过环境变量指定路径：

```bash
export PYWRF_DATA_DIR=/path/to/data
export PYWRF_RUN_NAME=2024020106     # 文件日期后缀，默认：2024020106
```

### 2. 选择 GPU

设备在 `pywrf/config_params.py` 中统一配置，默认使用 GPU 0；原研究环境使用
GPU 2：

```bash
export PYWRF_GPU=2
```

若无 CUDA，代码会回退到 CPU（实际运行会很慢）。

### 3. 运行

以下三种方式等价：

```bash
python examples/run_wrf_9km.py     # 在仓库根目录直接运行，无需安装
pywrf                              # 安装后使用命令行脚本（pip install -e .）
python -m pywrf                    # 安装后使用模块入口
```

求解器（`pywrf.solver.WrfSolver.solve`）加载输入/边界/参考文件，并对配置的
个例进行积分，逐步打印进度，最终返回关键场（`u`、`v`、`w`、`t`、`ph`、
`mu`、`moist`）。

## 配置

- **设备** — `PYWRF_GPU`（默认 `0`）。在 `pywrf/config_params.py` 中统一定义，
  所有 `pywrf` 模块共用。
- **个例 / 区域** — 编辑 `pywrf/config_params.py`（网格大小、时间步长、
  `config_flags`、物理常数）。当前配置对应 9 km 个例：240×240×41 网格，
  内部 230×230×41。

## 引用与致谢

**本仓库是以下论文的官方代码——在使用 PyWRF 时请引用它：**

- Chu, H., Cao, Y., Wang, R., Xu, J., Chen, L., Fan, X., Chen, S., Guo, J., &
  Wu, J. (2026). *A Novel ML–NWP Coupled Optimization Approach on a
  Reconstructed WRF Model*. Journal of Meteorological Research, 40(3), 653–668.
  https://doi.org/10.1007/s13351-026-5154-1

PyWRF 是对 WRF 模式数值方案的独立 Python 重实现。原始 WRF 模式由 NCAR、
NOAA 及合作者开发，相关文献如下：

- Skamarock, W. C., et al. (2019). *A Description of the Advanced Research WRF
  Model Version 4*, NCAR/TN-556+STR.
- Hong, S.-Y., & Lim, J.-O. J. (2006). *The WRF single-moment 6-class
  microphysics scheme (WSM6)*. J. Korean Meteor. Soc., 42, 129–151.

PyWRF 与 NCAR 无隶属或背书关系，以 MIT 许可证发布。

## 已知限制

- **物理范围** — 本版实现了 WRF 的*热力学*框架与 WSM6 单矩微物理；其余 WRF
  物理方案（辐射、陆面、边界层、积云）未实现，代码中的边界层/扩散仅保留
  占位接口。
- **验证范围** — 如论文所述，模型及耦合 ML–NWP 框架仅在中国东部 **8 次中尺度
  降水过程**上训练和评估，对其他区域、季节或事件类型的精度尚未验证。
- **研究原型** — PyWRF 是面向 ML–NWP 研究的可微重实现，并非官方 Fortran 版
  WRF 的替代品：计算性能远低于 Fortran 参考实现，也未经业务化验证。
- **ML 耦合未随仓库发布** — 论文中的单柱神经网络耦合与截断反传训练策略
  属于研究应用；本仓库发布的是独立的可微动力核心。
- **边界数据覆盖** — 随附的 `wrfbdy` 只有 2 个边界时次（相隔 6 小时），
  因此主时间循环限定为 720 步（12 小时）；如需更长运行，需要更多边界时次。
- **单一配置** — 仅内置一个 9 km 单域个例；运行其它区域需修改
  `config_params.py`。
- **已去除机器相关设置** — 原始工作代码中硬编码的 GPU 编号已统一为
  `PYWRF_GPU` 环境变量。
- **fp16 变体** — 原始压缩包中仅有物理模块 fp16 变体的编译产物（`*.pyc`），
  其源代码未包含在本仓库中。

## 许可证

[MIT](LICENSE) © 2026 caotong0。
