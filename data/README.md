# data/

This directory ships the **real 9 km WRF case data** used to develop PyWRF
(start time 2024-02-01 06:00 UTC, 230×230×41 grid points, WRF v3.9.1.1 real
preprocessor output):

| File | Description | Size |
| ---- | ----------- | ---- |
| `wrf_inout_step3_2024020106` | 3D input state (u, v, mu, mub, ...) | 55 MB |
| `wrfbdy_d01_2024020106`      | lateral boundary tendencies (`U_BXS`, ...) | 12 MB |
| `wrfout_d01_2024020106`      | reference output (used for `w`, etc.) | 54 MB |

## Running with this data

Just run the solver from the repo root — the files are already in place:

```bash
python examples/run_wrf_9km.py
# or after `pip install -e .`:  pywrf   /   python -m pywrf
```

To point at a different run, override `PYWRF_DATA_DIR` / `PYWRF_RUN_NAME`.

## ⚠️ Boundary data coverage

`wrfbdy_d01_2024020106` holds **2 boundary time levels, 6 h apart** — 06:00 and
12:00 UTC (its `Times` variable: 2024-02-01_06:00:00 / 12:00:00). The solver
loads a boundary state every `bdy_interval = 360` steps (= 6 h) and indexes
`it_bdy = t_big_step // 360`: level 0 at step 1 covers 06:00→12:00, level 1 at
step 361 covers 12:00→18:00. The time loop is therefore **720 steps = 12 h**
(06:00 → 18:00). A longer run would need a 3rd boundary level (18:00).

---

# data/

本目录随仓库发布用于开发 PyWRF 的**真实 9 km 个例数据**（起始时刻
2024-02-01 06:00 UTC，230×230×41 格点，WRF v3.9.1.1 real 预处理输出）：

| 文件 | 说明 | 大小 |
| ---- | ---- | ---- |
| `wrf_inout_step3_2024020106` | 三维输入场（u, v, mu, mub, ...） | 55 MB |
| `wrfbdy_d01_2024020106`      | 侧边界倾向（`U_BXS`、...） | 12 MB |
| `wrfout_d01_2024020106`      | 参考输出（用于读取 `w` 等） | 54 MB |

## 用这套数据运行

文件已就位，直接在仓库根目录运行即可：

```bash
python examples/run_wrf_9km.py
# 或安装后： pywrf   /   python -m pywrf
```

如需切换其它个例，可用 `PYWRF_DATA_DIR` / `PYWRF_RUN_NAME` 覆盖。

## ⚠️ 边界数据覆盖范围

`wrfbdy_d01_2024020106` 有 **2 个边界时次，相隔 6 小时**——06:00 与 12:00
UTC（其 `Times` 变量为 2024-02-01_06:00:00 / 12:00:00）。求解器每
`bdy_interval = 360` 步（=6 小时）加载一次新边界，按
`it_bdy = t_big_step // 360` 索引：第 1 步加载时次 0（覆盖 06:00→12:00），
第 361 步加载时次 1（覆盖 12:00→18:00）。因此**时间循环为 720 步 = 12 小时**
（06:00 → 18:00）。更长的运行需要第 3 个边界时次（18:00）。
