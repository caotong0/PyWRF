# PyWRF

**PyWRF** is a pure **Python / PyTorch** reimplementation of the core of the
[WRF](https://github.com/NCAR/WRF) (Weather Research and Forecasting) model —
dynamical core, physics and lateral boundary conditions — written as fully
tensorized code that is **differentiable** (autograd-friendly). It is intended
for AI-for-weather research: a physics-based dynamical core that can be
embedded in / supervised by machine-learning models, finetuned, or differentiated
through.

> ⚠️ **Research prototype.** This is an independent reimplementation of WRF
> *algorithms*, not a drop-in replacement for the Fortran WRF model, and it is
> not affiliated with or endorsed by NCAR/NOAA. See
> [Known limitations](#known-limitations).

---

## Features

- **Dynamical core** — RK3 time integration, non-hydrostatic, terrain-following
  coordinates, momentum/heat/moisture advection, full and perturbation pressure,
  map factors, damping options.
- **Physics** — WSM6 single-moment microphysics (cloud/rain/ice/snow/graupel),
  falling-rain / slope processes, PBL & diffusion, moist adjustment hooks.
- **Boundary conditions** — specified (relaxation + specified zones) and
  flow-dependent lateral boundary updates, mass-weighting.
- **Differentiable** — every operation is a PyTorch tensor op, so the whole
  core can run under `torch.autograd` and be tuned as a layer in a neural
  weather model.
- **Matches a real case** — the shipped config reproduces the domain of a real
  9 km WRF run (230×230×41 grid points).

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
├── data/                     # WRF NetCDF input files go here (not shipped)
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

### 1. Get WRF input data

The model integrates a WRF state from NetCDF files produced by a real WRF run.
These files are **not** bundled with this repository (see `data/README.md`):

| File | Description |
| ---- | ----------- |
| `wrf_inout_step3_<RUN>` | 3D input state (u, v, mu, mub, ...) |
| `wrfbdy_d01_<RUN>`      | lateral boundary tendencies (`U_BXS`, `T_BXS`, ...) |
| `wrfout_d01_<RUN>`      | reference output (used for `w`, etc.) |

Place them under `data/` (default) or point the driver at their location:

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

This code accompanies the following paper:

- Chu, H., Cao, Y., Wang, R., Xu, J., Chen, L., Fan, X., Chen, S., Guo, J., &
  Wu, J. (2026). *A Novel ML–NWP Coupled Optimization Approach on a
  Reconstructed WRF Model*. Journal of Meteorological Research, 40(3), 653–668.
  https://doi.org/10.1007/s13351-026-5154-1

PyWRF is an independent Python reimplementation of the numerical schemes of the
WRF model. The original WRF model is developed by NCAR, NOAA and collaborators:

- Skamarock, W. C., et al. (2019). *A Description of the Advanced Research WRF
  Model Version 4*, NCAR/TN-556+STR.
- Hong, S.-Y., & Lim, J.-O. J. (2006). *The WRF single-moment 6-class
  microphysics scheme (WSM6)*. J. Korean Meteor. Soc., 42, 129–151.

When you use PyWRF in a publication, please cite the paper above and the
original WRF documentation. PyWRF is not affiliated with or endorsed by NCAR,
and is distributed under the MIT license.

## Known limitations

- **Research prototype** — not a drop-in WRF replacement; performance is far
  below the compiled Fortran WRF model and accuracy is not guaranteed.
- **Single case** — the config hardcodes one 9 km single-domain case; other
  domains require editing `config_params.py`.
- **Machine-specific settings removed** — hardcoded GPU indices from the
  original working copy were replaced by the `PYWRF_GPU` variable.
- **fp16 variant** — a float16 variant of the physics module was present as a
  compiled `*.pyc` in the original archive but its source is not included.
- **Renamed / restructured files** — the original working filenames
  (`wrf_*_fuctions.py`, `read_wrfinput_writenc_noai_new_9km_1.py`) were renamed
  for this release (`wrf_*.py`), and the former monolithic driver script was
  split into the `pywrf.solver` module (`WrfSolver`) plus a thin CLI entry
  point (`examples/run_wrf_9km.py`).

## License

[MIT](LICENSE) © 2026 caoyuan.

---

# PyWRF（中文版）

**PyWRF** 是 [WRF](https://github.com/NCAR/WRF)（Weather Research and
Forecasting，天气预报研究与模式）核心算法的一个纯 **Python / PyTorch**
重实现——包含动力核、物理过程和侧边界条件——所有运算均为张量化操作，
因此整体是**可微的**（兼容 autograd）。项目面向“人工智能+气象”研究：
一个基于物理、可被嵌入机器学习模型、可微调、可端到端反向传播的动力核心。

> ⚠️ **研究原型。** 这是对 WRF *算法*的独立重实现，并非 Fortran WRF 模式
> 的替代品，且与 NCAR/NOAA 无任何隶属或背书关系。参见[已知限制](#已知限制-1)。

## 功能特性

- **动力核** — RK3 时间积分、非静力平衡、地形跟随坐标、动量/热/水汽平流、
  全扰动气压、地图投影因子、阻尼选项。
- **物理过程** — WSM6 单矩微物理方案（云/雨/冰/雪/霰）、降水下落与斜率过程、
  边界层与扩散、湿调整接口。
- **边界条件** — 指定区（松弛 + 指定区）与流依赖的侧边界更新、质量加权。
- **可微** — 所有运算均为 PyTorch 张量运算，整个动力核可在 `torch.autograd`
  下运行，可作为神经天气模型的一层参与训练。
- **贴合真实个例** — 内置配置复现了一个真实 9 km WRF 运行的区域
  （230×230×41 格点）。

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
├── data/                     # WRF NetCDF 输入文件放在这里（未随仓库发布）
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

### 1. 准备 WRF 输入数据

模式从真实 WRF 运行产生的 NetCDF 文件读取初始场并积分。这些文件
**不随本仓库发布**（见 `data/README.md`）：

| 文件 | 说明 |
| ---- | ---- |
| `wrf_inout_step3_<RUN>` | 三维输入场（u, v, mu, mub, ...） |
| `wrfbdy_d01_<RUN>`      | 侧边界倾向（`U_BXS`、`T_BXS`、...） |
| `wrfout_d01_<RUN>`      | 参考输出（用于读取 `w` 等） |

默认放在 `data/` 目录，或通过环境变量指定路径：

```bash
export PYWRF_DATA_DIR=/path/to/data
export PYWRF_RUN_NAME=2024020106     # 文件日期后缀，默认：2024020106
```

### 2. 选择 GPU

设备选择集中在 `pywrf/config_params.py`，默认 GPU 0。原研究环境使用 GPU 2：

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

求解器（`pywrf.solver.WrfSolver.solve`）加载输入/边界/参考文件并积分所配置
的个例，逐步打印进度，并返回最终关键场（`u`、`v`、`w`、`t`、`ph`、`mu`、
`moist`）。

## 配置

- **设备** — `PYWRF_GPU`（默认 `0`）。在 `pywrf/config_params.py` 中统一定义，
  所有 `pywrf` 模块共用。
- **个例 / 区域** — 编辑 `pywrf/config_params.py`（网格大小、时间步长、
  `config_flags`、物理常数）。当前配置对应 9 km 个例：240×240×41 网格，
  内部 230×230×41。

## 引用与致谢

本代码配套以下论文：

- Chu, H., Cao, Y., Wang, R., Xu, J., Chen, L., Fan, X., Chen, S., Guo, J., &
  Wu, J. (2026). *A Novel ML–NWP Coupled Optimization Approach on a
  Reconstructed WRF Model*. Journal of Meteorological Research, 40(3), 653–668.
  https://doi.org/10.1007/s13351-026-5154-1

PyWRF 是对 WRF 模式数值方案的独立 Python 重实现。WRF 模式由 NCAR、NOAA 及
合作者开发：

- Skamarock, W. C., et al. (2019). *A Description of the Advanced Research WRF
  Model Version 4*, NCAR/TN-556+STR.
- Hong, S.-Y., & Lim, J.-O. J. (2006). *The WRF single-moment 6-class
  microphysics scheme (WSM6)*. J. Korean Meteor. Soc., 42, 129–151.

在论文中使用 PyWRF 时，请引用上面的论文及上述 WRF 文档。PyWRF 与 NCAR 无隶属或
背书关系，以 MIT 许可证发布。

## 已知限制

- **研究原型** — 并非 WRF 模式的替代品，性能远低于编译版 Fortran WRF，
  精度不作保证。
- **单一配置** — 仅内置一个 9 km 单域个例；其他区域需修改 `config_params.py`。
- **已去除机器相关设置** — 原始工作代码中硬编码的 GPU 编号已替换为
  `PYWRF_GPU` 变量。
- **fp16 变体** — 原始压缩包中仅有物理模块 fp16 变体的编译产物（`*.pyc`），
  其源代码未包含在本仓库中。
- **文件名已重命名 / 结构重构** — 原始工作文件名（`wrf_*_fuctions.py`、
  `read_wrfinput_writenc_noai_new_9km_1.py`）在本版本中已重命名
  （`wrf_*.py`），且原单体驱动脚本已拆分为 `pywrf.solver` 模块
  （`WrfSolver`）+ 精简 CLI 入口（`examples/run_wrf_9km.py`）。

## 许可证

[MIT](LICENSE) © 2026 caoyuan。
