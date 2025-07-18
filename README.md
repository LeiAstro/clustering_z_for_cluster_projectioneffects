# Clustering_z method for Cluster Projection Effects

*Efficient estimation of line‑of‑sight projection / contamination around galaxy clusters via cross‑correlations with a spectroscopic reference sample.*

This repository implements a **clustering redshift ($w\_{ur}$) stacking pipeline** tailored to measuring **projection effects in optical cluster catalogs** (e.g. redMaPPer). The code:

* slices the cluster (member) sample into **narrow tomographic bins** inside **wide stacking bins**;
* cross‑correlates (projected) unknown objects with a **spectroscopic reference sample** and its random catalog;
* produces **jackknife (JK) resampled pair counts / weights** for robust covariance estimation;
* supports **linear or logarithmic physical radial bins** for the angular aperture conversion;
* chooses between **spatial K‑means JK** and **leave‑one‑cluster‑out JK** adaptively.

---

## 1. Scientific Background

The clustering redshift technique measures the excess surface density of spectroscopic reference galaxies around an “unknown” (photometric) sample as a function of the reference redshift. For cluster projection studies, the “unknown” set are **cluster member galaxies (or stacks of members)** and the signal traces structures physically / accidentally aligned along the line of sight. Comparing w\_{ur}(z) across richness / radius / cluster redshift bins quantifies projection contamination.

Key observable here:

$$
w_{ur}(z_j) \propto \frac{D_u R_r - D_u D_r}{R_r D_r}
$$

(Exact estimator depends on normalization; this code stores the weighted pair counts *wDuDr* and *wDuRr* plus raw counts *Nr*, *Rr* so you can form preferred estimators later.)

---

## 2. Repository Layout (core)

| Path                                           | Description                                                           |
| ---------------------------------------------- | --------------------------------------------------------------------- |
| `calc_wur_inputs.yml`                          | YAML configuration (paths, cosmology, binning, JK settings).          |
| `calc_wur.py`                                  | Loads config, builds bins, performs pair counts, writes HDF5 outputs. |
| `README.md`                                    | This document.                                                        |
| `environment.yml` / `requirements.txt` *(add)* | Reproducible software environment (recommended; see below).           |
| `LICENSE` *(add)*                              | Choose a license (e.g. MIT / BSD 3‑Clause).                           |

*(Rename the main script to something concise like `run_wur.py` for clarity.)*

---

## 3. Requirements

| Package           | Notes                                                         |
| ----------------- | ------------------------------------------------------------- |
| Python ≥ 3.8      | Tested with 3.8.                                              |
| `numpy`           | Vector math & arrays.                                         |
| `astropy`         | Cosmology distances, FITS I/O.                                |
| `h5py`            | Output container.                                             |
| `PyYAML`          | Config parsing.                                               |
| `joblib`          | Parallel processing.                                          |
| `treecorr`        | K‑means sky partition for spatial jackknife (NField.k‑means). |
| `Corrfunc`        | Fast angular pair counts (`DDtheta_mocks`).                   |
| (optional) `tqdm` | Progress bars (currently commented out).                      |

**Install (example with conda + conda‑forge):**

```bash
conda create -n wur_env python=3.11 numpy astropy h5py pyyaml joblib treecorr corrfunc tqdm
conda activate wur_env
```

Or add a `requirements.txt` and use `pip install -r requirements.txt`.

---

## 4. Configuration (`calc_wur_inputs.yml`)

All runtime parameters are centralized. Below is a concise map:

| Key                                                     | Meaning                                      | Units / Allowed             | Notes                                                            |
| ------------------------------------------------------- | -------------------------------------------- | --------------------------- | ---------------------------------------------------------------- |
| `cluster_mem_path`, `cluster_mem_fname`                 | Cluster member FITS table path/name          | —                           | redMaPPer (or similar) membership catalog.                       |
| `richness_min`, `richness_max`                          | Richness (λ) cuts                            | dimensionless               | Set to `null` (or omit) to disable.                              |
| `reference_path`, `reference_gname`                     | Reference galaxy sample (spectroscopic)      | —                           | Must contain RA, DEC, Z columns.                                 |
| `reference_rname`                                       | Reference random catalog                     | —                           | Must match angular & redshift selection of reference galaxies.   |
| `npatches`                                              | Desired JK spatial patches                   | int                         | Used only if clusters > 120; otherwise leave‑one‑cluster‑out JK. |
| `H0`, `Om0`                                             | Cosmology parameters                         | km s⁻¹ Mpc⁻¹, dimensionless | Flat ΛCDM assumed.                                               |
| `wide_dz`, `zwide_start`, `zwide_end`, `num_zwide_bins` | Wide stacking redshift binning for cluster z | —                           | Wide bins define iteration loops.                                |
| `narrow_tomo_dz`                                        | Narrow (photometric) tomographic slice width | Δz                          | Used inside each wide bin.                                       |
| `spec_dz`, `zspec_min`, `zspec_max`                     | Spectroscopic reference bin edges            | Δz                          | Builds `z_j` bins for w\_{ur}(z).                                |
| `rmin`, `rmax`, `num_rbins`, `spacing`                  | Physical (proper) radial bin edges           | Mpc                         | `spacing: linear` or `log`.                                      |
| `N_CPUs`                                                | Max processes to launch                      | int                         | Automatically capped by available cores.                         |

**Tip:** For *log* radial bins, ensure `rmin > 0`.

---

## 5. Running the Pipeline

1. **Edit** `calc_wur_inputs.yml` with correct filesystem paths and desired binning.

2. **Execute**:

   ```bash
   python run_wur.py
   ```

3. **Outputs**: For each wide z\_i interval you get an HDF5 file:

   ```
   wDuDr_wDuRr_zj{spec_dz}_zi{narrow_tomo_dz}_{JKMETHOD}_{zmin_i}_{zmax_i}.h5
   ```

   (Example: `wDuDr_wDuRr_zj0.003_zi0.002_spatial_0.15_0.18.h5`)

---

## 6. Output File Structure (HDF5)

| Dataset           | Shape                     | Description                                                                      |
| ----------------- | ------------------------- | -------------------------------------------------------------------------------- |
| `rbins`           | (N\_r+1,)                 | Physical radial bin edges (Mpc).                                                 |
| `dr_bins`         | (N\_r,)                   | Bin widths Δr.                                                                   |
| `rbins_cen`       | (N\_r,)                   | Bin centers.                                                                     |
| `z_i_bins`        | (N\_z\_i+1,)              | Edges of narrow photometric cluster bins within the current wide bin.            |
| `z_j_bins`        | (N\_z\_j+1,)              | Edges of spectroscopic reference (micro) bins.                                   |
| `wDuDr`           | (N\_z\_i, N\_z\_j, N\_JK) | Weighted D\_u–D\_r pair sums per patch.                                          |
| `wDuRr`           | (N\_z\_i, N\_z\_j, N\_JK) | Weighted D\_u–R\_r pair sums per patch.                                          |
| `Nr`              | (N\_z\_i, N\_z\_j, N\_JK) | Number of reference galaxies in bin (after JK exclusion) used for normalization. |
| `Rr`              | (N\_z\_i, N\_z\_j, N\_JK) | Number of reference randoms in bin (after JK exclusion).                         |
| `jk_patch_labels` | (N\_JK,)                  | Integer labels (0…Npatches−1) or cluster IDs (leave‑one‑out).                    |

You can form your estimator and covariance as:

```python
# Example: simple ratio-style clustering-redshift signal
signal = (wDuDr - wDuRr) / wDuRr
# Build a jackknife covariance across JK patches
```

(Choose the estimator consistent with literature you compare against.)

---

## 7. Jackknife Strategy

* **Spatial JK (K‑means on the sphere)**: Activated automatically if the number of **unique clusters > 120**. Uses combined RA/Dec of (members + reference + randoms) to define spatial patches.
* **Leave-one-cluster-out JK**: For smaller samples, each cluster index defines a JK “patch”.

**Potential Enhancements (roadmap)**

* Replace K‑means with an equal‑area HEALPix partition (deterministic, faster).
* Down-sample very large catalogs before K‑means to reduce memory.

---

## 8. Performance Notes

| Lever                | Recommendation                                                                                                                                      |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Parallelism**      | Currently: joblib processes × Corrfunc threads (`nthreads=4`). To avoid oversubscription on smaller nodes, tune `nthreads` to 1 or reduce `N_CPUs`. |
| **I/O**              | Large FITS reads happen once at startup. For huge catalogs, consider memory‑mapping & passing only views (see Issues / TODO).                       |
| **Radial bins**      | Log bins capture scale dependence better; set `spacing: "log"` in YAML.                                                                             |
| **Spectroscopic dz** | Smaller `spec_dz` gives finer structure in w\_{ur}(z) but increases noise & runtime.                                                                |
| **Narrow tomo dz**   | Must evenly tile the wide bin to avoid partial last bin (current code includes the final partial edge if not exact—monitor for off‑by‑one).         |

---

## 9. Visualization Example

See the Jupyter notebook example: $\tt {stacking_and_mcmc_example.ipynb}$. 
 
---


## 10. License

Choose a permissive license (MIT / BSD 3‑Clause recommended) and add a `LICENSE` file. Update this section accordingly.

---

