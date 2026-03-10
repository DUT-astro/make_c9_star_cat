#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 将上一个脚本生成的 CSV 转为 C9 风格 HDF5，组织为 /galaxies/<HID>
import argparse, os
import numpy as np
import pandas as pd
import h5py
import healpy as hp

NSIDE = 128  # 与仿真读取侧保持一致

def get_bundleIndex(healpixID_ring, bundleOrder=4, healpixOrder=7):
    # ring->nest->下采样->ring
    shift = 2*(healpixOrder - bundleOrder)
    nside_bundle = 2**bundleOrder
    nside_healpix = 2**healpixOrder
    healpixID_nest = hp.ring2nest(nside_healpix, healpixID_ring)
    bundleID_nest = (healpixID_nest >> shift)
    bundleID_ring = hp.nest2ring(nside_bundle, bundleID_nest)
    return bundleID_ring

def ensure_group(parent, path: str):
    """
    创建并返回 path 对应的 group。
    - 如果 path 以 '/' 开头：从文件根开始逐级 require_group
    - 否则：从 parent（可以是 File 或 Group）开始逐级 require_group
    """
    if not isinstance(parent, (h5py.File, h5py.Group)):
        raise TypeError("parent must be an h5py.File or h5py.Group")

    if path.startswith("/"):
        grp = parent.file  # 绝对路径：回到根
        parts = [p for p in path.split("/") if p]
    else:
        grp = parent       # 相对路径：基于父 group
        parts = [p for p in path.split("/") if p]

    for p in parts:
        grp = grp.require_group(p)
    return grp

def write_group(group: h5py.Group, name: str, data, dtype=None):
    if name in group:
        del group[name]
    arr = np.asarray(data, dtype=dtype) if dtype is not None else np.asarray(data)
    return group.create_dataset(name, data=arr, compression="gzip", shuffle=True, fletcher32=True)

def main():
    ap = argparse.ArgumentParser(description="CSV → CSST C9 Galaxy HDF5")
    ap.add_argument("--csv", required=True, help="输入 CSV")
    ap.add_argument("--outdir", required=True, help="输出目录")
    ap.add_argument("--nside", type=int, default=NSIDE, help="HEALPix NSIDE（默认 128）")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.csv)

    req = [
        "ra","dec","redshift",
        "mag_csst_u","mag_csst_g","mag_csst_r","mag_csst_i","mag_csst_z","mag_csst_y","mag_csst_nuv",
        "mag_csst_GU","mag_csst_GV","mag_csst_GI",
        "size","ellipticity_true_e1","ellipticity_true_e2",
        "shear_g1","shear_g2","kappa","detA",
        "bulgemass","diskmass","type","veldisp","qsoindex"
    ]
    for c in req + [f"coeff_{j}" for j in range(20)]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    theta = np.deg2rad(90.0 - df["dec"].values)
    phi   = np.deg2rad(df["ra"].values)
    hid = hp.ang2pix(args.nside, theta, phi, nest=False)
    df["_HID"] = hid
    df["_BUNDLE"] = get_bundleIndex(hid)

    for bundle_id, df_b in df.groupby("_BUNDLE"):
        fn = os.path.join(args.outdir, f"galaxies_C6_bundle{bundle_id:06d}.h5")
        with h5py.File(fn, "w") as h5f:
            g_gal = ensure_group(h5f, "/galaxies")
            for hid_val, d in df_b.groupby("_HID"):
                g_hid = ensure_group(g_gal, f"{hid_val}")
                shear = np.vstack([d["shear_g1"].values, d["shear_g2"].values]).T.astype(np.float32)
                ell   = np.vstack([d["ellipticity_true_e1"].values, d["ellipticity_true_e2"].values]).T.astype(np.float32)
                coeff = d[[f"coeff_{j}" for j in range(20)]].values.astype(np.float32)

                write_group(g_hid, "ra", d["ra"].values.astype(np.float64))
                write_group(g_hid, "dec", d["dec"].values.astype(np.float64))
                write_group(g_hid, "redshift", d["redshift"].values.astype(np.float64))
                for band in ["u","g","r","i","z","y","nuv","GU","GV","GI"]:
                    col = f"mag_csst_{band}"
                    write_group(g_hid, col, d[col].values.astype(np.float32))
                write_group(g_hid, "size", d["size"].values.astype(np.float32))
                write_group(g_hid, "ellipticity_true", ell, dtype=np.float32)
                write_group(g_hid, "shear", shear, dtype=np.float32)
                write_group(g_hid, "kappa", d["kappa"].values.astype(np.float32))
                write_group(g_hid, "detA", d["detA"].values.astype(np.float32))
                write_group(g_hid, "bulgemass", d["bulgemass"].values.astype(np.float32))
                write_group(g_hid, "diskmass", d["diskmass"].values.astype(np.float32))
                write_group(g_hid, "type", d["type"].values.astype(np.int32))
                write_group(g_hid, "veldisp", d["veldisp"].values.astype(np.float32))
                write_group(g_hid, "qsoindex", d["qsoindex"].values.astype(np.int32))
                write_group(g_hid, "coeff", coeff, dtype=np.float32)
        print(f"Wrote bundle file: {fn}")

    print("Done.")

if __name__ == "__main__":
    main()

