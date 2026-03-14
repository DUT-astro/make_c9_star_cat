#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_galaxy_chip_catalog_scaled.py

在 make_galaxy_patch_catalog_scaled.py 的基础上做最小改动：
- 保留原有：
  * 从 Cycle-9 bundle HDF5 有放回抽样；
  * 保留/展开额外列；
  * qso 过滤；
  * 椭率随机旋转；
  * size / bulgemass / diskmass 的 lognormal 扰动；
  * size / coeff 的全局缩放；
  * shear_g1/g2=0, kappa=0, detA=1。
- 只把“位置生成”改成：
  * 以每片 chip 为单位生成；
  * 每片 chip 固定输出同样的 N；
  * 在 chip 的局部 pixel 坐标中生成；
  * 以简洁的局部聚合模型增强 close pairs：
      - 概率 (1-p_clust): 背景均匀落点；
      - 概率 p_clust    : 以已有对象为锚点，在其附近按 size 相关尺度生成。
  * 最后用整焦平面 WCS 把 global pixel 坐标映回 RA/Dec。

注意：
- 这里直接把 size 近似当作一个“用于局部聚合尺度”的 HLR proxy，
  再通过 pix_scale 转成 pixel；这用于位置生成，不改动后续真正的成像模型。
- 边界不做补偿：若近邻候选点落到 chip 外，则本次回退为该 chip 内均匀落点。
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

ALIAS_TO_CANONICAL = {
    "mag_csst_wf": "mag_csst_GU",
    "mag_csst_wh": "mag_csst_GV",
    "mag_csst_wi": "mag_csst_GI",
}
INT_COLS = {"type", "qsoindex"}


# -----------------------------
# 参数检查
# -----------------------------
def check_args_validity(args: argparse.Namespace) -> None:
    if not (0.0 <= args.ra_center < 360.0):
        raise ValueError(f"ra_center must satisfy 0 <= ra_center < 360, got {args.ra_center}")
    if not (-90.0 <= args.dec_center <= 90.0):
        raise ValueError(f"dec_center must satisfy -90 <= dec_center <= 90, got {args.dec_center}")
    if args.n_per_chip <= 0:
        raise ValueError(f"n_per_chip must be > 0, got {args.n_per_chip}")
    if args.size_scale <= 0:
        raise ValueError(f"size_scale must be > 0, got {args.size_scale}")
    if args.coeff_scale <= 0:
        raise ValueError(f"coeff_scale must be > 0, got {args.coeff_scale}")
    if not (0.0 <= args.p_clust <= 1.0):
        raise ValueError(f"p_clust must be in [0,1], got {args.p_clust}")
    if args.sep_scale <= 0:
        raise ValueError(f"sep_scale must be > 0, got {args.sep_scale}")
    if args.pix_scale <= 0:
        raise ValueError(f"pix_scale must be > 0, got {args.pix_scale}")
    if args.size_to_radius_factor <= 0:
        raise ValueError(f"size_to_radius_factor must be > 0, got {args.size_to_radius_factor}")
    if args.min_radius_pix <= 0:
        raise ValueError(f"min_radius_pix must be > 0, got {args.min_radius_pix}")


# -----------------------------
# 椭率旋转
# -----------------------------
def rotate_e1e2(
    rng: np.random.Generator,
    e1: np.ndarray,
    e2: np.ndarray,
    rot_max_deg: float = 180.0,
) -> Tuple[np.ndarray, np.ndarray]:
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
def lognormal_mul(rng: np.random.Generator, x: np.ndarray, sigma_ln: float) -> np.ndarray:
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

    missing = [c for c in REQUIRED_OUTPUT_COLS if c not in out]
    if missing:
        raise KeyError(f"Missing required columns in /galaxies/{hid}: {missing}")
    return out


def default_array_for_missing(col: str, n: int) -> np.ndarray:
    if col in INT_COLS:
        return np.full(n, -1, dtype=np.int64)
    return np.full(n, np.nan, dtype=np.float64)


def discover_union_schema(files: List[str]) -> List[str]:
    union = set()
    for fp in files:
        with h5py.File(fp, "r") as h5:
            for hid in discover_hids_in_file(h5):
                union.update(flatten_group_schema(h5["galaxies"][hid]))

    missing = [c for c in REQUIRED_OUTPUT_COLS if c not in union]
    if missing:
        raise KeyError(f"Input HDF5 schema does not provide required columns: {missing}")

    extras = sorted(c for c in union if c not in REQUIRED_OUTPUT_COLS)
    return REQUIRED_OUTPUT_COLS + extras


def append_group_to_pool(
    pool_lists: Dict[str, List[np.ndarray]],
    chunk: Dict[str, np.ndarray],
    all_columns: List[str],
) -> None:
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
    all_columns = discover_union_schema(files)
    pool_lists: Dict[str, List[np.ndarray]] = {c: [] for c in all_columns}

    for fp in files:
        if not os.path.exists(fp):
            raise FileNotFoundError(fp)
        with h5py.File(fp, "r") as h5:
            for hid in discover_hids_in_file(h5):
                try:
                    chunk = read_one_hid_group(h5, hid)
                except Exception as e:
                    raise RuntimeError(f"Failed reading {fp} HID={hid}: {e}") from e
                append_group_to_pool(pool_lists, chunk, all_columns)

    return concat_pool(pool_lists, all_columns), all_columns


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
# Chip geometry / WCS
# 直接继承 target_location_check.py 的焦平面定义
# -----------------------------
def ccd_param() -> Tuple[int, int, int, int, Tuple[int, int], int, int, int]:
    xt, yt = 59516, 49752
    x0, y0 = 9216, 9232
    xgap, ygap = (534, 1309), 898
    xnchip, ynchip = 6, 5
    return xt, yt, x0, y0, xgap, ygap, xnchip, ynchip


def get_tan_wcs_params(ra: float, dec: float, img_rot_deg: float, pix_scale: float) -> Tuple[float, float, float, float, float, float]:
    """
    与 target_location_check.py 中 getTanWCS 保持同一仿射定义，
    这里只返回参数，不依赖 galsim。
    """
    img_rot_rad = np.deg2rad(img_rot_deg)
    dudx = -np.cos(img_rot_rad) * pix_scale
    dudy = +np.sin(img_rot_rad) * pix_scale
    dvdx = -np.sin(img_rot_rad) * pix_scale
    dvdy = -np.cos(img_rot_rad) * pix_scale
    return ra, dec, dudx, dudy, dvdx, dvdy


def fp_pixel_to_world_deg(
    x: float,
    y: float,
    ra_center_deg: float,
    dec_center_deg: float,
    dudx: float,
    dudy: float,
    dvdx: float,
    dvdy: float,
) -> Tuple[float, float]:
    """
    把焦平面 global pixel 坐标映回天空坐标，采用 TAN(gonomonic) 逆投影。

    其中 u,v 的单位是 arcsec：
        u = dudx * x + dudy * y
        v = dvdx * x + dvdy * y
    然后将 (u,v) 视为切平面坐标，围绕 (ra_center, dec_center) 做 gnomonic inverse。
    """
    u_arcsec = dudx * x + dudy * y
    v_arcsec = dvdx * x + dvdy * y

    x_tan = np.deg2rad(u_arcsec / 3600.0)
    y_tan = np.deg2rad(v_arcsec / 3600.0)

    ra0 = np.deg2rad(ra_center_deg)
    dec0 = np.deg2rad(dec_center_deg)

    rho = np.hypot(x_tan, y_tan)
    if rho == 0.0:
        return ra_center_deg % 360.0, dec_center_deg

    c = np.arctan(rho)
    sin_c = np.sin(c)
    cos_c = np.cos(c)

    dec = np.arcsin(cos_c * np.sin(dec0) + (y_tan * sin_c * np.cos(dec0) / rho))
    ra = ra0 + np.arctan2(
        x_tan * sin_c,
        rho * np.cos(dec0) * cos_c - y_tan * np.sin(dec0) * sin_c,
    )

    return np.rad2deg(ra) % 360.0, np.rad2deg(dec)


def get_chip_lim(chip_id: int) -> Tuple[float, float, float, float]:
    xt, yt, x0, y0, gx, gy, xnchip, ynchip = ccd_param()
    gx1, gx2 = gx

    row_id = ((chip_id - 1) % 5) + 1
    col_id = 6 - ((chip_id - 1) // 5)

    xrem = 2 * (col_id - 1) - (xnchip - 1)
    xcen = (x0 // 2 + gx1 // 2) * xrem
    if chip_id >= 26 or chip_id == 21:
        xcen = (x0 // 2 + gx1 // 2) * xrem - (gx2 - gx1)
    if chip_id <= 5 or chip_id == 10:
        xcen = (x0 // 2 + gx1 // 2) * xrem + (gx2 - gx1)
    nx0 = xcen - x0 // 2 + 1
    nx1 = xcen + x0 // 2

    yrem = (row_id - 1) - ynchip // 2
    ycen = (y0 + gy) * yrem
    ny0 = ycen - y0 // 2 + 1
    ny1 = ycen + y0 // 2

    return float(nx0 - 1), float(nx1 - 1), float(ny0 - 1), float(ny1 - 1)


# -----------------------------
# Position generation in chip-local pixel coordinates
# -----------------------------
def effective_radius_pix(
    size_arcsec: np.ndarray,
    pix_scale: float,
    size_to_radius_factor: float,
    min_radius_pix: float,
) -> np.ndarray:
    r = size_to_radius_factor * np.asarray(size_arcsec, dtype=float) / pix_scale
    return np.maximum(r, min_radius_pix)


def generate_positions_one_chip(
    rng: np.random.Generator,
    chip_id: int,
    n_obj: int,
    size_arcsec: np.ndarray,
    p_clust: float,
    sep_scale: float,
    sep_sigma_ln: float,
    pix_scale: float,
    size_to_radius_factor: float,
    min_radius_pix: float,
) -> Tuple[np.ndarray, np.ndarray]:
    x0, x1, y0, y1 = get_chip_lim(chip_id)
    width = x1 - x0
    height = y1 - y0

    r_pix = effective_radius_pix(size_arcsec, pix_scale, size_to_radius_factor, min_radius_pix)
    x = np.empty(n_obj, dtype=float)
    y = np.empty(n_obj, dtype=float)

    for i in range(n_obj):
        use_cluster = (i > 0) and (rng.random() < p_clust)
        if use_cluster:
            j = int(rng.integers(0, i))
            base_sep = sep_scale * (r_pix[i] + r_pix[j])
            if sep_sigma_ln > 0:
                base_sep *= float(np.exp(rng.normal(0.0, sep_sigma_ln)))
            theta = rng.uniform(0.0, 2.0 * np.pi)
            x_try = x[j] + base_sep * np.cos(theta)
            y_try = y[j] + base_sep * np.sin(theta)

            # 不做边界补偿：若甩出 chip，则直接回退为均匀背景落点
            if x0 <= x_try <= x1 and y0 <= y_try <= y1:
                x[i] = x_try
                y[i] = y_try
                continue

        x[i] = rng.uniform(x0, x0 + width)
        y[i] = rng.uniform(y0, y0 + height)

    return x, y


# -----------------------------
# Main helpers
# -----------------------------
def parse_chip_ids(text: str) -> List[int]:
    ids = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        cid = int(part)
        if not (1 <= cid <= 30):
            raise ValueError(f"chip id must be in [1,30], got {cid}")
        ids.append(cid)
    if not ids:
        raise ValueError("chip_ids is empty")
    if len(set(ids)) != len(ids):
        raise ValueError(f"chip_ids contains duplicates: {ids}")
    return ids


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


def sample_output_indices(
    rng: np.random.Generator,
    pool: Dict[str, np.ndarray],
    n_out: int,
    include_qso: bool,
) -> np.ndarray:
    n_pool = pool["ra"].shape[0]
    out_idx = np.empty(n_out, dtype=np.int64)
    filled = 0
    tries = 0
    while filled < n_out:
        need = n_out - filled
        idx = sample_indices_with_replacement(rng, n_pool, need)
        if include_qso:
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
    return out_idx


# -----------------------------
# Main
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sample galaxies with replacement from Cycle-9 HDF5 and generate equal-N catalogs per chip.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--in-h5", required=True,
                   help="Input HDF5 bundle glob or comma-separated list. Example: '/path/galaxies_C6_bundle*.h5'")
    p.add_argument("--out-csv", required=True, help="Output CSV filename")
    p.add_argument("--ra-center", type=float, required=True,
                   help="Focal-plane pointing center RA in deg")
    p.add_argument("--dec-center", type=float, required=True,
                   help="Focal-plane pointing center Dec in deg")
    p.add_argument("--image-rot", type=float, default=-113.4333,
                   help="Camera orientation in deg; keep consistent with your simulation setup")
    p.add_argument("--pix-scale", type=float, default=0.074,
                   help="Pixel scale in arcsec/pixel")
    p.add_argument("--chip-ids", type=str,
                   default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30",
                   help="Comma-separated chip ids to populate")
    p.add_argument("--n-per-chip", type=int, default=1000,
                   help="Number of galaxy centers per chip")
    p.add_argument("--seed", type=int, default=12345, help="Random seed")

    # 局部聚合参数
    p.add_argument("--p-clust", type=float, default=0.35,
                   help="Probability that a new object is generated near an existing object in the same chip")
    p.add_argument("--sep-scale", type=float, default=1.0,
                   help="Cluster separation scale in units of (r_i + r_j)")
    p.add_argument("--sep-sigma-ln", type=float, default=0.35,
                   help="Lognormal scatter for clustered separation")
    p.add_argument("--size-to-radius-factor", type=float, default=1.0,
                   help="Convert size to effective placement radius: r_pix = factor * size / pix_scale")
    p.add_argument("--min-radius-pix", type=float, default=0.5,
                   help="Minimum effective radius in pixel for placement scale")

    # 椭率旋转
    p.add_argument("--rot-max-deg", type=float, default=180.0,
                   help="Max random rotation angle in degrees; phi ~ Uniform(0, rot_max_deg)")

    # G1 乘性 lognormal 扰动
    p.add_argument("--sigma-ln-size", type=float, default=0.10,
                   help="Lognormal sigma (ln-space) for size multiplicative perturbation")
    p.add_argument("--size-scale", type=float, default=1.0,
                   help="Global multiplicative scale factor applied to size")
    p.add_argument("--sigma-ln-bulge", type=float, default=0.20,
                   help="Lognormal sigma (ln-space) for bulgemass multiplicative perturbation")
    p.add_argument("--sigma-ln-disk", type=float, default=0.20,
                   help="Lognormal sigma (ln-space) for diskmass multiplicative perturbation")
    p.add_argument("--coeff-scale", type=float, default=1.0,
                   help="Global multiplicative scale factor applied to coeff_0...coeff_19")
    p.add_argument("--include-qso", action="store_true",
                   help="If set, allow sampling rows with qsoindex != -1. Otherwise re-draw until qsoindex == -1.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    check_args_validity(args)
    rng = np.random.default_rng(args.seed)

    chip_ids = parse_chip_ids(args.chip_ids)
    files = expand_inputs(args.in_h5)
    if len(files) == 0:
        raise RuntimeError(f"No input files matched: {args.in_h5}")

    print(f"[Info] Input files       = {len(files)}")
    print(f"[Info] Output CSV        = {args.out_csv}")
    print(f"[Info] Pointing center   = ({args.ra_center}, {args.dec_center}) deg")
    print(f"[Info] image_rot         = {args.image_rot} deg")
    print(f"[Info] chip_ids          = {chip_ids}")
    print(f"[Info] N per chip        = {args.n_per_chip}")
    print(f"[Info] p_clust           = {args.p_clust}")
    print(f"[Info] sep_scale         = {args.sep_scale}")
    print(f"[Info] size_scale        = {args.size_scale}")
    print(f"[Info] coeff_scale       = {args.coeff_scale}")

    pool, all_columns = load_pool_from_h5(files)
    n_pool = pool["ra"].shape[0]
    if n_pool == 0:
        raise RuntimeError("Empty pool loaded from inputs.")
    print(f"[Info] Loaded pool size  = {n_pool}")
    print(f"[Info] Output columns    = {len(all_columns)}")

    n_total = len(chip_ids) * args.n_per_chip
    out_idx = sample_output_indices(rng, pool, n_total, args.include_qso)

    out: Dict[str, np.ndarray] = {}
    for c in all_columns:
        out[c] = pool[c][out_idx].copy()

    # 先做属性扰动，使位置生成直接使用扰动后的 size
    e1 = out["ellipticity_true_e1"].astype(float)
    e2 = out["ellipticity_true_e2"].astype(float)
    e1p, e2p = rotate_e1e2(rng, e1, e2, rot_max_deg=args.rot_max_deg)
    out["ellipticity_true_e1"] = e1p
    out["ellipticity_true_e2"] = e2p

    out["size"] = lognormal_mul(rng, out["size"].astype(float), args.sigma_ln_size)
    out["bulgemass"] = lognormal_mul(rng, out["bulgemass"].astype(float), args.sigma_ln_bulge)
    out["diskmass"] = lognormal_mul(rng, out["diskmass"].astype(float), args.sigma_ln_disk)
    out["size"] *= args.size_scale
    for c in COEFF_COLS:
        out[c] = out[c].astype(float) * args.coeff_scale

    # 透镜量重置
    out["shear_g1"] = np.zeros(n_total, dtype=float)
    out["shear_g2"] = np.zeros(n_total, dtype=float)
    out["kappa"] = np.zeros(n_total, dtype=float)
    out["detA"] = np.ones(n_total, dtype=float)

    # 逐 chip 生成 pixel 坐标，再映回 RA/Dec
    ra0, dec0, dudx, dudy, dvdx, dvdy = get_tan_wcs_params(
        args.ra_center, args.dec_center, args.image_rot, args.pix_scale
    )
    ra_new = np.empty(n_total, dtype=float)
    dec_new = np.empty(n_total, dtype=float)

    for ichip, chip_id in enumerate(chip_ids):
        i0 = ichip * args.n_per_chip
        i1 = i0 + args.n_per_chip

        xg, yg = generate_positions_one_chip(
            rng=rng,
            chip_id=chip_id,
            n_obj=args.n_per_chip,
            size_arcsec=out["size"][i0:i1].astype(float),
            p_clust=args.p_clust,
            sep_scale=args.sep_scale,
            sep_sigma_ln=args.sep_sigma_ln,
            pix_scale=args.pix_scale,
            size_to_radius_factor=args.size_to_radius_factor,
            min_radius_pix=args.min_radius_pix,
        )

        for k in range(args.n_per_chip):
            ra_deg, dec_deg = fp_pixel_to_world_deg(
                float(xg[k]), float(yg[k]), ra0, dec0, dudx, dudy, dvdx, dvdy
            )
            ra_new[i0 + k] = ra_deg
            dec_new[i0 + k] = dec_deg

    out["ra"] = ra_new
    out["dec"] = dec_new

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    write_csv(args.out_csv, out, all_columns)
    print(f"[Done] Wrote {n_total} rows to {args.out_csv}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[Error] {e}", file=sys.stderr)
        sys.exit(1)
