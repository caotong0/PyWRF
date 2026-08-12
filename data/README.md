# data/

This directory is where the WRF NetCDF input files go. They are **not**
bundled with the repository (each is tens of MB), so place them here yourself
or point the driver at them with `PYWRF_DATA_DIR` / `PYWRF_RUN_NAME`.

## Files needed

| File | Description |
| ---- | ----------- |
| `wrf_inout_step3_<RUN>` | 3D input state (u, v, mu, mub, ...) |
| `wrfbdy_d01_<RUN>`      | lateral boundary tendencies (`U_BXS`, `T_BXS`, ...) |
| `wrfout_d01_<RUN>`      | reference output (used for `w`, etc.) |

`<RUN>` is the date suffix, default `2024020106` for the shipped 9 km case
(configurable via the `PYWRF_RUN_NAME` env var).

## How to obtain them

These files are produced by running the real WRF model for the case and dumping
the relevant intermediate / output NetCDF fields (`U`, `V`, `MU`, `MUB`,
`PH`, `PHB`, `MAPFAC_*`, moisture, boundary tendency arrays such as
`U_BXS`, `T_BXS`, ...). For a quick test you can generate them by:

1. Setting up and running [WRF](https://github.com/NCAR/WRF) for a small case,
   with boundary-update output enabled;
2. Exporting the needed variables from `wrfinput_d01` / `wrfbdy_d01` /
   `wrfout_d01` (e.g. with `ncks` or xarray) into the three files above.

The original working copy of this project ran against a 9 km, 230×230×41
single-domain case; the shipped `pywrf/config_params.py` matches that domain.

---

# data/

本目录用于存放 WRF NetCDF 输入文件。这些文件**不随仓库发布**（每个数十 MB），
请自行放入，或用环境变量 `PYWRF_DATA_DIR` / `PYWRF_RUN_NAME` 指定位置。

## 所需文件

| 文件 | 说明 |
| ---- | ---- |
| `wrf_inout_step3_<RUN>` | 三维输入场（u, v, mu, mub, ...） |
| `wrfbdy_d01_<RUN>`      | 侧边界倾向（`U_BXS`、`T_BXS`、...） |
| `wrfout_d01_<RUN>`      | 参考输出（用于读取 `w` 等） |

`<RUN>` 为日期后缀，内置 9 km 个例默认 `2024020106`（可用 `PYWRF_RUN_NAME`
环境变量修改）。

## 如何获取

这些文件由真实 WRF 模式运行个例后，从相关中间/输出 NetCDF 场
（`U`、`V`、`MU`、`MUB`、`PH`、`PHB`、`MAPFAC_*`、水汽量以及
`U_BXS`、`T_BXS` 等边界倾向数组）导出得到。快速测试的步骤：

1. 配置并运行 [WRF](https://github.com/NCAR/WRF)（小个例即可），开启
   boundary-update 输出；
2. 用 `ncks` 或 xarray 从 `wrfinput_d01` / `wrfbdy_d01` / `wrfout_d01` 中
   导出所需变量，写入上面三个文件。

项目原始工作版本针对 9 km、230×230×41 单域个例运行，内置的
`pywrf/config_params.py` 与该区域一致。
