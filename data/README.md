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

`wrfbdy_d01_2024020106` contains **2 boundary-update time levels** (0 and 1).
The solver loads a new boundary state every `bdy_interval = 320` steps and
indexes `it_bdy = t_big_step // 320` (steps 1, 321, 641, 961, 1281 → `it_bdy` =
0, 1, 2, 3, 4). With only 2 levels available, indexing past step 640 would hit
the end of the boundary data. The solver's time loop is therefore set to
**640 steps** (≈ 10.7 h at dt = 60 s), which uses both boundary times exactly
(`it_bdy` = 0 at step 1, 1 at step 321). A longer run needs a `wrfbdy` file
with 5 boundary levels.

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

`wrfbdy_d01_2024020106` 只有 **2 个边界更新时次**（0 和 1）。求解器每
`bdy_interval = 320` 步加载一次新边界，按 `it_bdy = t_big_step // 320`
索引；超过第 640 步会超出边界数据末尾。因此求解器的**时间循环已设为
640 步**（dt=60s 下约 10.7 小时），恰好用完两个边界时次（第 1 步
`it_bdy`=0、第 321 步 `it_bdy`=1）。更长的运行需要 5 个边界时次的
`wrfbdy` 文件。
