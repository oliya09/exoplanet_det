# import torch
# import torch.nn as nn

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class CNNModel(nn.Module):
#     def __init__(self):
#         super(CNNModel, self).__init__()
#         # ✅ conv1: 32 фильтра (так было в сохранённой модели)
#         self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2)
#         self.bn1 = nn.BatchNorm1d(32)
#         self.relu = nn.ReLU()
#         self.pool = nn.MaxPool1d(2)

#         # ✅ conv2: 64 фильтра (так было в сохранённой модели)
#         self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
#         self.bn2 = nn.BatchNorm1d(64)

#         # ✅ fc1 вход = 31808, выход = 64 (как в модели)
#         self.fc1 = nn.Linear(64 * 497, 64)  # 64*497=31808
#         self.fc2 = nn.Linear(64, 1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = self.relu(self.bn1(self.conv1(x)))
#         x = self.pool(x)
#         x = self.relu(self.bn2(self.conv2(x)))
#         x = self.pool(x)
#         x = x.view(x.size(0), -1)
#         x = self.relu(self.fc1(x))
#         x = self.sigmoid(self.fc2(x))
#         return x


# def load_fixed_state_dict(model, path):
#     """Загрузка модели с авто-исправлением ключей"""
#     state_dict = torch.load(path, map_location=DEVICE)
#     new_state_dict = {}
#     for k, v in state_dict.items():
#         if k.startswith("_orig_mod."):
#             new_state_dict[k[len("_orig_mod."):]] = v
#         else:
#             new_state_dict[k] = v
#     model.load_state_dict(new_state_dict, strict=False)
#     print("✅ Модель успешно загружена.")


# # ======== Тест загрузки ========
# if __name__ == "__main__":
#     model = CNNModel().to(DEVICE)
#     load_fixed_state_dict(model, "best_model.pth")
#     model.eval()
#     print("🚀 Модель готова к использованию.")







#----------------------------------------------------------------------------------------------------------------------#







# import lightkurve as lk
# import matplotlib.pyplot as plt
# import numpy as np

# tic = "TIC 7903477"
# search = lk.search_lightcurve(tic, mission="TESS")
# print("search results:", search)

# if len(search) == 0:
#     print("No LC found in TESS. Maybe synthetic fallback used.")
# else:
#     lc = search[0].download().remove_nans().normalize()
#     print("LC length:", len(lc.time))
#     plt.figure(figsize=(10,3))
#     plt.plot(lc.time.value, lc.flux.value, '.', ms=1)
#     plt.title(tic + " - raw LC")
#     plt.xlabel("time")
#     plt.ylabel("flux")
#     plt.show()

#     # BLS
#     from astropy.timeseries import BoxLeastSquares
#     t = lc.time.value
#     y = lc.flux.value
#     periods = np.linspace(0.5, 30, 2000)
#     bls = BoxLeastSquares(t, y)
#     res = bls.power(periods, 0.01)
#     best = np.nanargmax(res.power)
#     P = res.period[best]
#     t0 = res.transit_time[best]
#     depth = res.depth[best]
#     print("Best P, t0, depth:", P, t0, depth)

#     # Fold and plot
#     phase = ((t - t0 + 0.5*P) % P) / P - 0.5
#     idx = np.argsort(phase)
#     plt.figure(figsize=(6,3))
#     plt.plot(phase[idx], y[idx], '.', ms=2)
#     plt.xlim(-0.2,0.2)
#     plt.gca().invert_yaxis()
#     plt.title(f"Folded P={P:.4f} d, depth={depth:.4f}")
#     plt.show()














#------------------------------------------------------------------------------------------------------#



# #!/usr/bin/env python3
# import lightkurve as lk
# from astropy.timeseries import BoxLeastSquares
# import numpy as np
# import matplotlib.pyplot as plt

# # === 1. Скачиваем кривую блеска TESS ===
# target_id = "TIC 7903477"

# print(f"🔭 Downloading light curve for {target_id} ...")
# search = lk.search_lightcurve(target_id, mission="TESS")
# print(f"Found {len(search)} data products")

# # Загружаем первый доступный (например, SPOC сектор 14)
# lc = search[0].download()
# if lc is None:
#     raise RuntimeError("❌ Не удалось скачать кривую блеска!")

# print(f"✅ LC downloaded: {lc}")
# print(f"LC length: {len(lc.time.value)} points")

# # === 2. Нормализация и очистка ===
# lc = lc.remove_nans().normalize()
# t = lc.time.value
# y = lc.flux.value
# y = y / np.median(y)
# y = y - 1.0  # центрируем вокруг 0

# # === 3. Поиск транзитов (BLS) ===
# print("⚙️ Running BoxLeastSquares (BLS)...")
# periods = np.linspace(0.5, 30, 2000)  # 0.5–30 дней
# bls = BoxLeastSquares(t, y)
# res = bls.power(periods, 0.02)

# best = np.argmax(res.power)
# best_period = res.period[best]
# t0 = res.transit_time[best]
# depth = res.depth[best]

# print(f"⭐ Best period = {best_period:.4f} days")
# print(f"⭐ Transit depth = {depth*1e6:.1f} ppm")

# # === 4. Строим свёрнутую кривую (folded LC) ===
# phase = ((t - t0 + 0.5 * best_period) % best_period) / best_period - 0.5
# idx = np.argsort(phase)

# plt.figure(figsize=(10, 3))
# plt.plot(phase[idx], y[idx], ".", ms=2, alpha=0.6)
# plt.xlabel("Orbital phase")
# plt.ylabel("Relative flux")
# plt.title(f"TIC 7903477 — Folded LC (P={best_period:.3f} d)")
# plt.xlim(-0.2, 0.2)
# plt.gca().invert_yaxis()
# plt.tight_layout()
# plt.show()





























#------------------------------------------------------------------------------------------------------#

























#!/usr/bin/env python3
"""
analyze_tic_full.py
Comprehensive diagnostic for a TIC: BLS, folded plot, planet params, centroid,
odd/even & secondary checks, ML sanity tests.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk
from astropy.timeseries import BoxLeastSquares
from astroquery.mast import Catalogs
from astropy.constants import G, R_sun, M_sun, R_earth
from math import pi
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Import your classifier (adjust path if needed)
try:
    from pipeline.classifier import CNNClassifier
except Exception as e:
    CNNClassifier = None
    print("⚠️ Cant import CNNClassifier:", e)

# ----------------- Utility functions -----------------
def fetch_any_lc(tic, mission_preference=("TESS", "Kepler")):
    search = lk.search_lightcurve(tic, mission=mission_preference)
    if len(search) == 0:
        return None, search
    # choose the first available product that downloads
    for r in search:
        try:
            lc = r.download()
            if lc is not None:
                lc = lc.remove_nans().normalize()
                return lc, search
        except Exception:
            continue
    return None, search

def run_bls(lc, Pmin=0.5, Pmax=30.0, N=2000, duration_frac=0.02):
    t = lc.time.value
    y = lc.flux.value
    # center and small detrend
    y = y / np.nanmedian(y)
    y = y - 1.0
    periods = np.linspace(Pmin, Pmax, N)
    bls = BoxLeastSquares(t, y)
    res = bls.power(periods, duration_frac)
    ix = np.nanargmax(res.power)
    return res, res.period[ix], float(res.transit_time[ix]), float(res.duration[ix]), float(res.depth[ix])

def fold_and_plot(lc, period, t0, width_days=None, save=None):
    t = lc.time.value
    y = lc.flux.value
    phase = ((t - t0 + 0.5*period) % period) / period - 0.5
    idx = np.argsort(phase)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(t, y, '.', ms=1, alpha=0.6)
    plt.xlabel("Time")
    plt.ylabel("Normalized flux")
    plt.title("Raw LC")
    plt.subplot(1,2,2)
    plt.plot(phase[idx], y[idx], '.', ms=2, alpha=0.6)
    plt.xlim(-0.2, 0.2)
    plt.gca().invert_yaxis()
    plt.xlabel("Phase")
    plt.title(f"Folded (P={period:.4f} d)")
    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')
    plt.show()

def get_star_params_from_tic(tic):
    try:
        clean = tic.replace("TIC ", "").strip()
        catalog = Catalogs.query_object(f"TIC {clean}", catalog="TIC")
        if len(catalog) > 0:
            row = catalog[0]
            Teff = float(row['Teff']) if np.isfinite(row['Teff']) else None
            Rstar = float(row['rad']) if np.isfinite(row['rad']) else None
            Mstar = float(row['mass']) if np.isfinite(row['mass']) else None
            return {"T_star": Teff, "R_star": Rstar, "M_star": Mstar, "catalog_row": row}
    except Exception as e:
        print("WARN: MAST TIC query failed:", e)
    return {"T_star": None, "R_star": None, "M_star": None}

def estimate_planet_params(depth, R_star, T_star, M_star, period):
    """
    depth: absolute fractional depth (e.g. 0.001 = 1000 ppm)
    R_star in R_sun
    T_star in K
    M_star in M_sun
    period in days
    """
    if R_star is None:
        raise ValueError("R_star required to estimate radius")
    # Rp/Rs = sqrt(depth)
    Rp_Rs = np.sqrt(depth)
    Rs_m = R_star * R_sun.value
    Rp_m = Rp_Rs * Rs_m
    Rp_Rearth = Rp_m / R_earth.value

    # mass estimate empirical (power law); coarse
    M_p_Mearth = Rp_Rearth ** 2.7

    # semi-major axis (Kepler's third law)
    period_s = period * 86400.0
    Mstar_kg = (M_star if M_star is not None else 1.0) * M_sun.value
    a_m = ((G.value * Mstar_kg * period_s**2) / (4 * pi**2)) ** (1/3)
    a_AU = a_m / (1.495978707e11)

    # Equilibrium temperature for three albedos
    Teq = {}
    for tag, A in [("Gas giant (0.1)", 0.1), ("Rocky (0.3)", 0.3), ("Icy (0.7)", 0.7)]:
        if T_star is None or R_star is None:
            Teq[tag] = None
        else:
            Rstar_AU = R_star * (R_sun.value / 1.495978707e11)
            Teq_val = T_star * np.sqrt(Rstar_AU / (2*a_AU)) * (1 - A)**0.25
            Teq[tag] = round(float(Teq_val), 1)

    # density (approx) relative to Earth (using radius & mass estimate)
    V_rel = (Rp_Rearth)**3
    density_rel_earth = (M_p_Mearth / V_rel) if V_rel > 0 else None

    return {
        "R_p_Rearth": round(float(Rp_Rearth), 2),
        "M_p_Mearth": round(float(M_p_Mearth), 2),
        "Density_rel_Earth": round(float(density_rel_earth), 2) if density_rel_earth is not None else None,
        "Orbit_AU": round(float(a_AU), 4),
        "Teq": Teq
    }

def odd_even_test_simple(lc, period, t0, duration):
    # fold and compute median depth for even and odd transits
    t = lc.time.value
    f = lc.flux.value
    phases = ((t - t0 + 1e-8) / period)
    transit_nums = np.floor(phases).astype(int)
    fractional = (phases - transit_nums)
    in_transit = (fractional > 0.5 - (duration/period)/2) & (fractional < 0.5 + (duration/period)/2)
    if in_transit.sum() < 10:
        return "insufficient_data"
    transit_ids = transit_nums[in_transit]
    odd_mask = (transit_ids % 2 == 1)
    even_mask = ~odd_mask
    odd_med = 1 - np.median(f[in_transit][odd_mask]) if odd_mask.any() else None
    even_med = 1 - np.median(f[in_transit][even_mask]) if even_mask.any() else None
    if odd_med is None or even_med is None:
        return "insufficient_data"
    diff = abs(odd_med - even_med)
    return "consistent" if diff < 0.5 * max(odd_med, even_med, 1e-9) else "mismatch"

def secondary_eclipse_test_simple(lc, period, t0, duration):
    t = lc.time.value
    f = lc.flux.value
    phases = ((t - t0 + 1e-8) % period) / period
    # primary around phase 0.5 (we used that convention), secondary around 0.0
    prim_mask = np.abs(phases - 0.5) < (duration / period)
    sec_mask = np.abs(phases - 0.0) < (duration / period)
    if prim_mask.sum() == 0 or sec_mask.sum() == 0:
        return 0.0, 0.0
    prim = 1 - np.median(f[prim_mask])
    sec = abs(1 - np.median(f[sec_mask]))
    return float(sec), float(prim)

def centroid_shift_test(lc, period, t0, duration):
    # try to use centroid_col/row from lc.table if available
    tab = lc.table if hasattr(lc, "table") else None
    if tab is None:
        return "no_data"
    # try column names
    for col in ("centroid_col", "centroid_row", "mom_centr1", "mom_centr2", "psf_centr1"):
        if col in tab.colnames:
            # compute mean in-transit vs out-of-transit
            t = lc.time.value
            phases = ((t - t0 + 1e-8) % period) / period
            in_mask = np.abs(phases - 0.5) < (duration / period)
            if in_mask.sum() < 5:
                continue
            colvals = np.array(tab[col])
            med_in = np.nanmedian(colvals[in_mask])
            med_out = np.nanmedian(colvals[~in_mask])
            shift = abs(med_in - med_out)
            # threshold in pixels (0.1 pix ~ small)
            return "ok" if shift < 0.1 else f"shifted ({shift:.4f} pix)"
    return "no_data"

# ----------------- Main -----------------
if __name__ == "__main__":
    TIC = "TIC 7903477"
    print("TIC:", TIC)
    lc, search = fetch_any_lc(TIC)
    if lc is None:
        print("No LC found. Search summary:", search)
        raise SystemExit(1)
    print("LC length:", len(lc.time))

    # BLS
    res, period, t0, duration, depth = run_bls(lc, Pmin=0.5, Pmax=50.0, N=4000, duration_frac=0.02)
    print(f"Best P={period:.6f} d, t0={t0:.6f}, duration={duration:.6f} d, depth={depth:.6e} (fraction) => {depth*1e6:.1f} ppm")

    # Plot
    fold_and_plot(lc, period, t0, save="tic_fold.png")

    # Star params
    star = get_star_params_from_tic(TIC)
    print("Star params from MAST:", star)

    # Planet estimation
    try:
        planet = estimate_planet_params(depth, star["R_star"], star["T_star"], star["M_star"], period)
        print("Planet estimates:", planet)
    except Exception as e:
        print("Could not estimate planet params:", e)
        planet = {}

    # odd/even and secondary tests
    oe = odd_even_test_simple(lc, period, t0, duration)
    sec, prim = secondary_eclipse_test_simple(lc, period, t0, duration)
    centroid = centroid_shift_test(lc, period, t0, duration)
    print("Odd/Even:", oe)
    print("Secondary/Primary depths:", sec, prim)
    print("Centroid check:", centroid)

    # ML model predictions and sanity checks
    if CNNClassifier is None:
        print("ML classifier not available (import failed). Skip ML tests.")
    else:
        clf = CNNClassifier()
        if clf.model is None:
            print("ML model failed to load.")
        else:
            # prepare flux same way as pipeline (normalize/pad/truncate)
            flux = np.nan_to_num(np.array(lc.flux.value, dtype=np.float32), nan=1.0)
            # if lc longer than input, sample or crop — here we crop center
            LIN = clf.input_len if hasattr(clf, "input_len") else getattr(clf, "input_len", None)
            if LIN is None:
                LIN = 2000
            if len(flux) > LIN:
                # take center segment for stability
                start = len(flux)//2 - LIN//2
                flux_in = flux[start:start+LIN]
            else:
                flux_in = np.pad(flux, (0, max(0, LIN - len(flux))), mode="constant", constant_values=np.median(flux))

            # normalize similarly to training (if you know exact pipeline, apply same)
            flux_in = (flux_in - np.min(flux_in)) / (np.ptp(flux_in) + 1e-8)

            ml_score = clf.predict(flux_in)
            print(f"ML score on LC: {ml_score:.3f}")

            # Sanity checks
            noise = np.random.normal(np.mean(flux_in), np.std(flux_in), size=len(flux_in)).astype(np.float32)
            noise = (noise - np.min(noise)) / (np.ptp(noise) + 1e-8)
            synth_time = np.linspace(0, period*2, len(flux_in))
            # synthetic transit: depth ~ depth (from BLS)
            phase_synth = ((synth_time % period) / period)
            synth = np.ones_like(phase_synth)
            synth[np.abs(phase_synth - 0.5) < 0.02] -= depth  # approximate transit depth
            synth = (synth - np.min(synth)) / (np.ptp(synth) + 1e-8)

            print("Sanity: ML on random noise:", clf.predict(noise))
            print("Sanity: ML on synthetic transit:", clf.predict(synth))

    print("\nDone. saved folded plot as tic_fold.png")
