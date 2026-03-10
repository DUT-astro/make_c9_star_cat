Making star catalogue

## 1) Download Gaia DR3 cone-search FITS

```bash
python gaia-cone-download-v2.py \
  --ra 265.172083 \
  --dec -53.673056 \
  --radius 0.1 \
  --out ngc-6397-test.fits
```

### Optional proxy

Use `--proxy` (e.g. `socks5://127.0.0.1:1080`) or set one of
`socks_proxy/all_proxy/http_proxy/https_proxy` in the environment.

Example with proxy:

```bash
python gaia-cone-download-v2.py \
  --ra 265.172083 \
  --dec -53.673056 \
  --radius 0.1 \
  --out ngc-6397-test.fits \
  --proxy http://localhost:10809
```

Sample output (truncated):

```
[INFO] Configuring proxy settings from: http://localhost:10809
[INFO] TAP URL: https://gea.esac.esa.int/tap-server/tap
[INFO] Center (deg): RA=265.17208300, Dec=-53.67305600, R=0.1 deg
[INFO] G < 21.0
INFO: Query finished. [astroquery.utils.tap.core]
[INFO] Rows: 25518
[INFO] Saved: ngc-6397-test.fits
...
[INFO] Completeness plot: ngc-6397-test.fits.completeness.png
```

## 2) Download HST-HUGS globular cluster catalog

Index page:

`https://archive.stsci.edu/hlsps/hugs/`

Example file:

```bash
wget https://archive.stsci.edu/hlsps/hugs/ngc6397/hlsp_hugs_hst_wfc3-uvis-acs-wfc_ngc6397_multi_v1_catalog-meth1.txt
```

Convert to HDF5:

```bash
python hst-hugs.py \
  --in_txt ./hlsp_hugs_hst_wfc3-uvis-acs-wfc_ngc6397_multi_v1_catalog-meth1.txt \
  --out_h5 hst_ngc6397.h5
```

## 3) Make CSST C9 catalog from Gaia FITS

```bash
python gaia_fits_to_c9h5_v2.py \
  --in_fits ./ngc-6397-test.fits \
  --out_h5 ngc-6397-test.h5
```

## 4) Make C9 catalog from HST-HUGS `.txt`

```bash
python hst-hugs.py \
  --in_txt ./hlsp_hugs_hst_wfc3-uvis-acs-wfc_ngc6397_multi_v1_catalog-meth1.txt \
  --out_h5 hst_ngc6397.h5
```

## 5) Make artificial star gird:
```
python make_Fov_grid_catalog.py \
  --ra0 90 --dec0 20 --theta 126.060924 \
   --nx 40 --ny 40 \
--app-g 16 22  --out stars_grid_40_40_90_20_ccd13_minimal.csv
Start generating grid for 30 chips...
Grid per chip: 40x40 = 1600 stars
Total stars expected: 48000
Processing Chip 30/30 ... 
[OK] Done! Wrote 48000 rows to stars_grid_40_40_90_20_ccd13_minimal.csv
```
### Convert to hdf5
```
python csv_to_c9h5.py --csv stars_grid_40_40_90_20_ccd13_minimal.csv --out ./stars_grid_40_40_90_20_ccd13_minimal.h5      
[OK] Wrote 624000 rows into 13 HEALPix groups in ./stars_grid_40_40_90_20_ccd13_minimal.h5
```

## 6) Make simi-artificial galaxy catalogue:
```
python make_galaxy_patch_catalog.py \
  --in-h5 "/sharefiles1/CSSTSim_C9_o/galaxycat_C9/cat2CSSTSim_bundle-50sqDeg/galaxies_C6*.h5" \
  --out-csv ./ga_test_01.csv \
  --ra-center 90 \
  --dec-center 20 \
  --radius-deg 0.6 \
  --n-gal 200000 \
  --seed 20260310
[Info] Input files = 38
[Info] Output CSV  = ./ga_test_01.csv
[Info] Patch center (RA,Dec)=(90.0,20.0) deg | radius=0.6 deg
[Info] N_gal = 200000 | seed=20260310
[Info] Loaded pool size = 127488185
[Info] Total output columns = 49
[Done] Wrote 200000 rows to ./ga_test_01.csv
```
*
```
 python csv_to_c9_galaxies_h5.py --csv ./ga_test_01.csv  --out ./ga_test_01.h5       
Wrote bundle file: ./ga_test_01.h5/galaxies_C6_bundle000944.h5
Wrote bundle file: ./ga_test_01.h5/galaxies_C6_bundle001007.h5
Wrote bundle file: ./ga_test_01.h5/galaxies_C6_bundle001008.h5
Wrote bundle file: ./ga_test_01.h5/galaxies_C6_bundle001072.h5
Done.
```

