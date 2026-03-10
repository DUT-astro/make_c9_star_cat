#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_galaxy_patch_catalog.py

从已有的 CSST Cycle-9 星系 bundle HDF5 星表中有放回抽样，生成一个新的天区 CSV 星表。
随后可用你已经验证可用的 csv_to_c9_galaxies_h5.py 再转换回 C9 兼容的 bundle HDF5。

本版修正点：
1) 将错误的波段名 wf/wh/wi 改为（并兼容别名到）GU/GV/GI。
2) 在输出 CSV 中保留输入 HDF5 里的其它列：
   - 1 维 dataset：按原列名原样输出；
   - 2 维 dataset：
       shear            -> shear_g1, shear_g2
       ellipticity_true -> ellipticity_true_e1, ellipticity_true_e2
       coeff            -> coeff_0 ... coeff_19
       其它 2 维列      -> <name>_0, <name>_1, ...
3) 增加坐标合法性检查：
   - 0 <= ra_center < 360
   - -90 <= dec_center <= 90
   - 0 < radius_deg <= 180
   - n_gal > 0
4) 仍然保持：
   - 有放回抽样；
   - 在球面圆帽内面积均匀随机散布；
   - shear_g1/g2 = 0, kappa = 0, detA = 1；
   - 随机旋转椭率；
   - size / bulgemass / diskmass 做 G1（乘性 lognormal）扰动；
   - 其余列原样保留。

注意：
- 本脚本输出的是 CSV。
- 是否“完全保留所有列”到最终 HDF5，还取决于后续的 csv_to_c9_galaxies_h5.py 是否会继续写入这些额外列。
  本脚本已经把它们保留到 CSV 里了。
"""

import argparse
import glob
import os
import sys
from typing import Dict, List, Tuple

import h5py
import numpy as np


# -----------------------------
# 输出所需的“核心列”
# 这些列需要与 csv_to_c9_galaxies_h5.py 兼容
# -----------------------------
REQ_BASE_COLS = [
    "ra",
    "dec",
    "redshift",
    "mag_csst_nuv",
    "mag_csst_u",
    "mag_csst_g",
    "mag_csst_r",
    "mag_csst_i",
    "mag_csst_z",
    "mag_csst_y",
    "mag_csst_GU",
    "mag_csst_GV",
    "mag_csst_GI",
    "shear_g1",
    "shear_g2",
    "kappa",
    "detA",
    "ellipticity_true_e1",
    "ellipticity_true_e2",
    "bulgemass",
    "diskmass",
    "size",
    "type",
    "veldisp",
    "qsoindex",
]
COEFF_COLS = [f"coeff_{i}" for i in range(20)]
REQUIRED_OUTPUT_COLS = REQ_BASE_COLS + COEFF_COLS

# 某些旧命名兼容到新命名
ALIAS_TO_CANONICAL = {
    "mag_csst_wf": "mag_csst_GU",
    "mag_csst_wh": "mag_csst_GV",
    "mag_csst_wi": "mag_csst_GI",
}

INT_COLS = {"type", "qsoindex"}


# -----------------------------
# Utilities: spherical geometry
# -----------------------------
def check_args_validity(ra_center_deg: float, dec_center_deg: float, radius_deg: float, n_gal: int) -> None:
    if not (0.0 <= ra_center_deg < 360.0):
        raise ValueError(f"ra_center must satisfy 0 <= ra_center < 360, got {ra_center_deg}")
    if not (-90.0 <= dec_center_deg <= 90.0):
        raise ValueError(f"dec_center must satisfy -90 <= dec_center <= 90, got {dec_center_deg}")
    if not (0.0 < radius_deg <= 180.0):
        raise ValueError(f"radius_deg must satisfy 0 < radius_deg <= 180, got {radius_deg}")
    if n_gal <= 0:
        raise ValueError(f"n_gal must be > 0, got {n_gal}")


def _unit_vec_from_radec_deg(ra_deg: float, dec_deg: float) -> np.ndarray:
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    return np.array(
        [np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)],
        dtype=float,
    )


def _orthonormal_basis_from_center(center_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = center_vec / np.linalg.norm(center_vec)
    a = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(np.dot(a, w)) > 0.95:
        a = np.array([1.0, 0.0, 0.0], dtype=float)
    u = np.cross(a, w)
    u = u / np.linalg.norm(u)
    v = np.cross(w, u)
    v = v / np.linalg.norm(v)
    return u, v, w


def sample_uniform_spherical_cap(
    rng: np.random.Generator,
    ra_center_deg: float,
    dec_center_deg: float,
    radius_deg: float,
    n: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    在球面圆帽内面积均匀采样。
    """
    R = np.deg2rad(radius_deg)
    cosR = np.cos(R)

    u = rng.uniform(cosR, 1.0, size=n)
    theta = np.arccos(u)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)

    center_vec = _unit_vec_from_radec_deg(ra_center_deg, dec_center_deg)
    e1, e2, e3 = _orthonormal_basis_from_center(center_vec)

    sin_theta = np.sin(theta)
    dirs = (
        u[:, None] * e3[None, :]
        + sin_theta[:, None] * (
            np.cos(phi)[:, None] * e1[None, :] + np.sin(phi)[:, None] * e2[None, :]
        )
    )

    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    ra = np.rad2deg(np.arctan2(y, x)) % 360.0
    dec = np.rad2deg(np.arcsin(np.clip(z, -1.0, 1.0)))
    return ra, dec


# -----------------------------
# Ellipticity rotation
# -----------------------------
def rotate_e1e2(
    rng: np.random.Generator,
    e1: np.ndarray,
    e2: np.ndarray,
    rot_max_deg: float = 180.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    随机旋转椭率：
      e' = e * exp(i 2phi)
    其中 phi ~ Uniform(0, rot_max_deg)
    """
    if rot_max_deg <= 0:
        return e1.copy(), e2.copy()

    phi = rng.uniform(0.0, np.deg2rad(rot_max_deg), size=e1.shape[0])
    c = np.cos(2.0 * phi)
    s = np.sin(2.0 * phi)
    e1p = e1 * c - e2 * s
    e2p = e1 * s + e2 * c
    return e1p, e2p


# -----------------------------
# Lognormal multiplicative noise
# -----------------------------
def lognormal_mul(
    rng: np.random.Generator,
    x: np.ndarray,
    sigma_ln: float,
) -> np.ndarray:
    if sigma_ln <= 0:
        return x.copy()
    return x * np.exp(rng.normal(0.0, sigma_ln, size=x.shape[0]))


# -----------------------------
# HDF5 reading / flattening
# -----------------------------
def discover_hids_in_file(h5: h5py.File) -> List[str]:
    if "galaxies" not in h5:
        raise KeyError("HDF5 file has no top group 'galaxies'")
    hids = []
    for k in h5["galaxies"].keys():
        obj = h5["galaxies"][k]
        if isinstance(obj, h5py.Group) and "ra" in obj:
            hids.append(k)
    return hids


def canonical_name(name: str) -> str:
    return ALIAS_TO_CANONICAL.get(name, name)


def flatten_group_schema(g: h5py.Group) -> List[str]:
    """
    只查看 schema，不读全量数据。
    """
    cols = []
    for name, ds in g.items():
        if not isinstance(ds, h5py.Dataset):
            continue
        out_name = canonical_name(name)

        shape = ds.shape
        if len(shape) == 1:
            cols.append(out_name)
        elif len(shape) == 2:
            ncol = shape[1]
            if out_name == "shear" and ncol == 2:
                cols.extend(["shear_g1", "shear_g2"])
            elif out_name == "ellipticity_true" and ncol == 2:
                cols.extend(["ellipticity_true_e1", "ellipticity_true_e2"])
            elif out_name == "coeff":
                cols.extend([f"coeff_{i}" for i in range(ncol)])
            else:
                cols.extend([f"{out_name}_{i}" for i in range(ncol)])
        else:
            raise ValueError(f"Unsupported dataset ndim={len(shape)} for /{g.name}/{name}, shape={shape}")
    return cols


def read_one_hid_group(h5: h5py.File, hid: str) -> Dict[str, np.ndarray]:
    """
    读取 /galaxies/<hid>，并展开为 CSV 列。
    """
    g = h5["galaxies"][hid]
    out: Dict[str, np.ndarray] = {}

    nrow = None

    for name, ds in g.items():
        if not isinstance(ds, h5py.Dataset):
            continue

        arr = ds[...]
        out_name = canonical_name(name)

        if arr.ndim == 1:
            out[out_name] = arr
            if nrow is None:
                nrow = arr.shape[0]
        elif arr.ndim == 2:
            if nrow is None:
                nrow = arr.shape[0]
            if arr.shape[0] != nrow:
                raise ValueError(f"Row count mismatch in /galaxies/{hid}/{name}: {arr.shape[0]} vs {nrow}")

            if out_name == "shear" and arr.shape[1] == 2:
                out["shear_g1"] = arr[:, 0]
                out["shear_g2"] = arr[:, 1]
            elif out_name == "ellipticity_true" and arr.shape[1] == 2:
                out["ellipticity_true_e1"] = arr[:, 0]
                out["ellipticity_true_e2"] = arr[:, 1]
            elif out_name == "coeff":
                for i in range(arr.shape[1]):
                    out[f"coeff_{i}"] = arr[:, i]
            else:
                for i in range(arr.shape[1]):
                    out[f"{out_name}_{i}"] = arr[:, i]
        else:
            raise ValueError(f"Unsupported dataset ndim={arr.ndim} for /galaxies/{hid}/{name}, shape={arr.shape}")

    # 核心列合法性检查：这些列必须能从输入 HDF5 中得到
    missing = [c for c in REQUIRED_OUTPUT_COLS if c not in out]
    if missing:
        raise KeyError(f"Missing required columns in /galaxies/{hid}: {missing}")

    return out


def default_array_for_missing(col: str, n: int) -> np.ndarray:
    """
    如果某个额外列在某些 HID 里缺失，用默认值补。
    对核心列，不应该走到这里；因为核心列缺失会直接报错。
    """
    if col in INT_COLS:
        return np.full(n, -1, dtype=np.int64)
    return np.full(n, np.nan, dtype=np.float64)


def discover_union_schema(files: List[str]) -> List[str]:
    """
    扫描所有输入文件，得到所有列的并集。
    """
    union = set()

    for fp in files:
        with h5py.File(fp, "r") as h5:
            for hid in discover_hids_in_file(h5):
                g = h5["galaxies"][hid]
                cols = flatten_group_schema(g)
                union.update(cols)

    # 核心列必须在 union 中
    missing = [c for c in REQUIRED_OUTPUT_COLS if c not in union]
    if missing:
        raise KeyError(f"Input HDF5 schema does not provide required columns: {missing}")

    # 输出顺序：先核心列，再其它列（按名字排序）
    extras = sorted(c for c in union if c not in REQUIRED_OUTPUT_COLS)
    return REQUIRED_OUTPUT_COLS + extras


def append_group_to_pool(
    pool_lists: Dict[str, List[np.ndarray]],
    chunk: Dict[str, np.ndarray],
    all_columns: List[str],
) -> None:
    """
    把一个 HID 的数据 append 到总池中。
    对于本 HID 缺失的额外列，用默认值补。
    """
    one_key = next(iter(chunk.keys()))
    n = chunk[one_key].shape[0]

    for c in all_columns:
        if c in chunk:
            pool_lists[c].append(np.asarray(chunk[c]))
        else:
            pool_lists[c].append(default_array_for_missing(c, n))


def concat_pool(pool_lists: Dict[str, List[np.ndarray]], all_columns: List[str]) -> Dict[str, np.ndarray]:
    pool = {}
    for c in all_columns:
        arr = np.concatenate(pool_lists[c], axis=0) if len(pool_lists[c]) > 1 else np.asarray(pool_lists[c][0])
        if c in INT_COLS:
            pool[c] = arr.astype(np.int64)
        else:
            pool[c] = arr
    return pool


def load_pool_from_h5(files: List[str]) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    全量读入到内存（按你的要求，先不处理内存问题）。
    """
    all_columns = discover_union_schema(files)
    pool_lists: Dict[str, List[np.ndarray]] = {c: [] for c in all_columns}

    for fp in files:
        if not os.path.exists(fp):
            raise FileNotFoundError(fp)
        with h5py.File(fp, "r") as h5:
            hids = discover_hids_in_file(h5)
            for hid in hids:
                try:
                    chunk = read_one_hid_group(h5, hid)
                except Exception as e:
                    raise RuntimeError(f"Failed reading {fp} HID={hid}: {e}") from e
                append_group_to_pool(pool_lists, chunk, all_columns)

    pool = concat_pool(pool_lists, all_columns)
    return pool, all_columns


# -----------------------------
# CSV writing
# -----------------------------
def write_csv(path: str, data: Dict[str, np.ndarray], columns: List[str]) -> None:
    n = data[columns[0]].shape[0]

    arrs = []
    fmt = []
    for c in columns:
        if c not in data:
            raise KeyError(f"Missing column in output: {c}")
        if data[c].shape[0] != n:
            raise ValueError(f"Column length mismatch for {c}: {data[c].shape[0]} vs {n}")

        if c in INT_COLS:
            arrs.append(data[c].astype(np.int64))
            fmt.append("%d")
        else:
            arrs.append(data[c].astype(np.float64))
            fmt.append("%.10g")

    mat = np.column_stack(arrs)
    header = ",".join(columns)
    np.savetxt(path, mat, delimiter=",", header=header, comments="", fmt=fmt)


# -----------------------------
# Main
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sample galaxies with replacement from existing Cycle-9 bundle HDF5 and create a new patch CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--in-h5",
        required=True,
        help="Input HDF5 bundle glob or comma-separated list. Example: '/path/galaxies_C6_bundle*.h5'",
    )
    p.add_argument("--out-csv", required=True, help="Output CSV filename")
    p.add_argument("--ra-center", type=float, required=True, help="Patch center RA in deg")
    p.add_argument("--dec-center", type=float, required=True, help="Patch center Dec in deg")
    p.add_argument("--radius-deg", type=float, default=0.6, help="Patch radius in deg")
    p.add_argument(
        "--n-gal",
        type=int,
        default=10000,
        help="Total number of galaxies in the patch (你的‘密度’定义：该半径内总数)",
    )
    p.add_argument("--seed", type=int, default=12345, help="Random seed for reproducibility")

    # 椭率旋转
    p.add_argument(
        "--rot-max-deg",
        type=float,
        default=180.0,
        help="Max random rotation angle in degrees; phi ~ Uniform(0, rot_max_deg)",
    )

    # G1 乘性 lognormal 扰动
    p.add_argument(
        "--sigma-ln-size",
        type=float,
        default=0.10,
        help="Lognormal sigma (ln-space) for size multiplicative perturbation",
    )
    p.add_argument(
        "--sigma-ln-bulge",
        type=float,
        default=0.20,
        help="Lognormal sigma (ln-space) for bulgemass multiplicative perturbation",
    )
    p.add_argument(
        "--sigma-ln-disk",
        type=float,
        default=0.20,
        help="Lognormal sigma (ln-space) for diskmass multiplicative perturbation",
    )

    p.add_argument(
        "--include-qso",
        action="store_true",
        help="If set, allow sampling rows with qsoindex != -1. Otherwise re-draw until qsoindex == -1.",
    )
    return p.parse_args()


def expand_inputs(in_h5: str) -> List[str]:
    parts = []
    for token in in_h5.split(","):
        token = token.strip()
        if not token:
            continue
        if any(ch in token for ch in ["*", "?", "["]):
            parts.extend(sorted(glob.glob(token)))
        else:
            parts.append(token)

    seen = set()
    out = []
    for f in parts:
        if f not in seen:
            out.append(f)
            seen.add(f)
    return out


def sample_indices_with_replacement(rng: np.random.Generator, n_pool: int, n_out: int) -> np.ndarray:
    return rng.integers(0, n_pool, size=n_out, endpoint=False)


def main() -> None:
    args = parse_args()
    check_args_validity(args.ra_center, args.dec_center, args.radius_deg, args.n_gal)
    rng = np.random.default_rng(args.seed)

    files = expand_inputs(args.in_h5)
    if len(files) == 0:
        raise RuntimeError(f"No input files matched: {args.in_h5}")

    print(f"[Info] Input files = {len(files)}")
    print(f"[Info] Output CSV  = {args.out_csv}")
    print(f"[Info] Patch center (RA,Dec)=({args.ra_center},{args.dec_center}) deg | radius={args.radius_deg} deg")
    print(f"[Info] N_gal = {args.n_gal} | seed={args.seed}")

    pool, all_columns = load_pool_from_h5(files)
    n_pool = pool["ra"].shape[0]
    if n_pool == 0:
        raise RuntimeError("Empty pool loaded from inputs.")
    print(f"[Info] Loaded pool size = {n_pool}")
    print(f"[Info] Total output columns = {len(all_columns)}")

    # 有放回抽样；若不允许 qso，则拒绝采样 qsoindex != -1 的条目
    out_idx = np.empty(args.n_gal, dtype=np.int64)
    filled = 0
    tries = 0
    while filled < args.n_gal:
        need = args.n_gal - filled
        idx = sample_indices_with_replacement(rng, n_pool, need)

        if args.include_qso:
            out_idx[filled:filled + need] = idx
            filled += need
        else:
            ok = (pool["qsoindex"][idx] == -1)
            keep = idx[ok]
            n_keep = keep.size
            if n_keep > 0:
                take = min(n_keep, need)
                out_idx[filled:filled + take] = keep[:take]
                filled += take

        tries += 1
        if tries > 10000:
            raise RuntimeError("Too many rejection-sampling iterations; maybe your pool has no qsoindex==-1 rows?")

    # 构造输出
    out: Dict[str, np.ndarray] = {}
    for c in all_columns:
        out[c] = pool[c][out_idx].copy()

    # 新坐标：球面圆帽内面积均匀
    ra_new, dec_new = sample_uniform_spherical_cap(
        rng=rng,
        ra_center_deg=args.ra_center,
        dec_center_deg=args.dec_center,
        radius_deg=args.radius_deg,
        n=args.n_gal,
    )
    out["ra"] = ra_new
    out["dec"] = dec_new

    # E：透镜量全部置零；detA 设为 1
    out["shear_g1"] = np.zeros(args.n_gal, dtype=float)
    out["shear_g2"] = np.zeros(args.n_gal, dtype=float)
    out["kappa"] = np.zeros(args.n_gal, dtype=float)
    out["detA"] = np.ones(args.n_gal, dtype=float)

    # F：随机旋转本征椭率
    e1 = out["ellipticity_true_e1"].astype(float)
    e2 = out["ellipticity_true_e2"].astype(float)
    e1p, e2p = rotate_e1e2(rng, e1, e2, rot_max_deg=args.rot_max_deg)
    out["ellipticity_true_e1"] = e1p
    out["ellipticity_true_e2"] = e2p

    # G：G1 乘性 lognormal 扰动
    out["size"] = lognormal_mul(rng, out["size"].astype(float), args.sigma_ln_size)
    out["bulgemass"] = lognormal_mul(rng, out["bulgemass"].astype(float), args.sigma_ln_bulge)
    out["diskmass"] = lognormal_mul(rng, out["diskmass"].astype(float), args.sigma_ln_disk)

    # 写 CSV：核心列在前，其它保留列在后
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    write_csv(args.out_csv, out, all_columns)

    print(f"[Done] Wrote {args.n_gal} rows to {args.out_csv}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[Error] {e}", file=sys.stderr)
        sys.exit(1)
