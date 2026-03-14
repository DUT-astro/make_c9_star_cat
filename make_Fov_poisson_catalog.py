#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全焦面随机恒星星表生成器 (Poisson-like on ALL chips)
在所有 CCD (1~30) 上生成随机分布的恒星；
利用 inverse_target_location 自动处理每个芯片的几何位置；
生成单一的大星表 CSV。

说明：
这里采用的是“每片 CCD 固定总星数 N，在芯片范围内独立均匀随机撒点”。
从点过程角度看，这等价于“二维齐次泊松过程在给定总数 N 条件下”的空间分布。
"""
import os
import argparse, csv, sys, random
from pathlib import Path

# 限制底层数学库线程，防止并行运行时过载
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

# 引入位置计算库
import inverse_target_location as inv

# 尝试引入检查库获取芯片尺寸，如果不存在则使用默认值
try:
    import target_location_check as tlc
except Exception:
    tlc = None


def get_chip_size_default():
    if tlc is not None:
        try:
            _xt, _yt, x0, y0, *_ = tlc.ccdParam()
            return int(x0), int(y0)
        except Exception:
            pass
    # CSST 默认芯片尺寸 (9k x 9k)
    return 9216, 9232


def linmap(lo, hi, t):  # t ∈ [0,1]
    return lo + (hi - lo) * t


def main():
    ap = argparse.ArgumentParser(description="生成覆盖全焦面(30个CCD)的随机模拟星表")

    # 几何与随机采样参数
    ap.add_argument("--nstar-per-chip", type=int, default=10000,
                    help="每个芯片生成的恒星总数")
    ap.add_argument("--edge", type=float, default=1000,
                    help="距芯片边缘的安全像素边距")
    ap.add_argument("--ra0", type=float, default=90.0,
                    help="焦面中心指向 RA (deg)")
    ap.add_argument("--dec0", type=float, default=20.0,
                    help="焦面中心指向 DEC (deg)")
    ap.add_argument("--theta", type=float, default=-120.0,
                    help="旋转角 (deg)")
    ap.add_argument("--seed", type=int, default=12345,
                    help="随机数种子")
    ap.add_argument("--out", type=str, default="stars_poisson_all_chips.csv")

    # —— 模拟物理参数范围 (仍然使用芯片内归一化位置做映射) ——
    ap.add_argument("--app-g",      type=float, nargs=2, default=(7.0, 24.0),    metavar=("MIN", "MAX"))
    ap.add_argument("--teff-log10", type=float, nargs=2, default=(3.50, 4.20),   metavar=("MIN", "MAX"))
    ap.add_argument("--grav",       type=float, nargs=2, default=(0.0, 5.0),     metavar=("MIN", "MAX"))
    ap.add_argument("--z-met",      type=float, nargs=2, default=(-2.5, 0.5),    metavar=("MIN", "MAX"))
    ap.add_argument("--AV",         type=float, nargs=2, default=(0.0, 2.0),     metavar=("MIN", "MAX"))
    ap.add_argument("--DM",         type=float, nargs=2, default=(5.0, 20.0),    metavar=("MIN", "MAX"))
    ap.add_argument("--mass",       type=float, nargs=2, default=(0.1, 5.0),     metavar=("MIN", "MAX"))
    ap.add_argument("--pmra",       type=float, nargs=2, default=(-50.0, 50.0),  metavar=("MIN", "MAX"))
    ap.add_argument("--pmdec",      type=float, nargs=2, default=(-50.0, 50.0),  metavar=("MIN", "MAX"))
    ap.add_argument("--RV",         type=float, nargs=2, default=(-200.0, 200.0), metavar=("MIN", "MAX"))
    ap.add_argument("--parallax",   type=float, nargs=2, default=(0.0, 5.0),     metavar=("MIN", "MAX"))

    args = ap.parse_args()

    if args.nstar_per_chip <= 0:
        raise ValueError("--nstar-per-chip 必须是正整数")

    # 1. 准备单个芯片的可采样范围 (所有芯片通用)
    chip_w, chip_h = get_chip_size_default()
    x0, x1 = args.edge, chip_w - 1 - args.edge
    y0, y1 = args.edge, chip_h - 1 - args.edge
    if not (x0 < x1 and y0 < y1):
        raise ValueError("edge 过大，导致芯片内没有可用采样区域")

    cols = ["RA", "DEC", "app_sdss_g", "teff", "grav", "z_met", "AV", "DM", "mass", "pmra", "pmdec", "RV", "parallax"]

    # 2. 确定芯片数量
    # 优先从 inverse 库读取，保证一致性
    try:
        total_chips = inv.NCOL * inv.NROW
    except AttributeError:
        total_chips = 30  # Fallback

    rng = random.Random(args.seed)

    print(f"Start generating random catalog for {total_chips} chips...")
    print(f"Stars per chip: {args.nstar_per_chip}")
    print(f"Total stars expected: {args.nstar_per_chip * total_chips}")
    print(f"Random seed: {args.seed}")

    total_rows = 0

    # 3. 打开文件并开始循环生成
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()

        # 遍历所有芯片 ID (1 到 30)
        for chip_id in range(1, total_chips + 1):

            # 进度提示
            sys.stdout.write(f"\rProcessing Chip {chip_id}/{total_chips} ... ")
            sys.stdout.flush()

            for _ in range(args.nstar_per_chip):
                # 在芯片矩形区域内独立均匀随机撒点：
                # 这就是二维齐次泊松过程在给定总星数条件下的空间分布
                x = rng.uniform(x0, x1)
                y = rng.uniform(y0, y1)

                # 归一化到 [0, 1]，以尽量保持原脚本中各物理量的生成逻辑
                u = (x - x0) / (x1 - x0)
                v = (y - y0) / (y1 - y0)
                uv = 0.5 * (u + v)

                # === 核心调用 ===
                # 传入 chip_id，inverse 库会自动查找该芯片相对于中心的偏移
                # 从而计算出正确的 RA, DEC
                ra, dec = inv.ccd_to_radec(
                    chip_id, float(x), float(y),
                    float(args.ra0), float(args.dec0), float(args.theta)
                )

                row = {
                    "RA":         f"{ra:.9f}",
                    "DEC":        f"{dec:.9f}",
                    # 物理参数映射：保留原脚本风格，但自变量改为随机位置对应的 u, v
                    "app_sdss_g": f"{linmap(*args.app_g,      u):.6f}",
                    "teff":       f"{linmap(*args.teff_log10, u):.9f}",
                    "grav":       f"{linmap(*args.grav,       v):.6f}",
                    "z_met":      f"{linmap(*args.z_met,      u):.6f}",
                    "AV":         f"{linmap(*args.AV,         v):.6f}",
                    "DM":         f"{linmap(*args.DM,        uv):.6f}",
                    "mass":       f"{linmap(*args.mass,       v):.6f}",
                    "pmra":       f"{linmap(*args.pmra,      uv):.6f}",
                    "pmdec":      f"{linmap(*args.pmdec,     uv):.6f}",
                    "RV":         f"{linmap(*args.RV,        uv):.6f}",
                    "parallax":   f"{linmap(*args.parallax,   u):.6f}",
                }
                w.writerow(row)
                total_rows += 1

    print(f"\n[OK] Done! Wrote {total_rows} rows to {args.out}")


if __name__ == "__main__":
    main()
