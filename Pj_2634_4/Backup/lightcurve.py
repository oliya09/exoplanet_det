import numpy as np
import lightkurve as lk
import os
import requests
from astropy.timeseries import BoxLeastSquares
import matplotlib.pyplot as plt
from astroquery.mast import Catalogs
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier

import astropy.units as u
from astropy.constants import G, M_sun, R_sun, R_earth, au
import warnings
from typing import Dict, Tuple, Optional, Any  # Для типизации
from concurrent.futures import ThreadPoolExecutor, as_completed  # Для мультипоточности
import time 
import json  # For cache star_params
import pickle  # For cache search results
warnings.filterwarnings("ignore", category=UserWarning)

# Contants
M_SUN_KG = M_sun.value      # kg
R_SUN_M = R_sun.value       # m
R_EARTH_M = R_earth.value   # m
AU_M = au.value             # m
SEC_PER_DAY = 86400         # s/day

# Threshold for sufficient coverage (days)
COVERAGE_THRESHOLD = 20.0

# === Enhanced Planet Parameter Generator ===
def generate_planet_params(
    depth: float = 0.002,
    R_star: float = 1.0,
    T_star: float = 5800,
    M_star: float = 1.0,
    period: float = 13.5
) -> Dict[str, Any]:

    # Planet radius in Earth units
    R_p_Rsun = np.sqrt(depth) * R_star
    R_p_Rearth = (R_p_Rsun * R_SUN_M) / R_EARTH_M
    

    M_p_Mearth = R_p_Rearth ** 2.7
    

    V_p = (4/3) * np.pi * R_p_Rearth**3
    density_rel_earth = M_p_Mearth / V_p
    

    period_sec = period * SEC_PER_DAY
    M_star_kg = M_star * M_SUN_KG
    inner_m3 = (G.value * M_star_kg * period_sec**2) / (4 * np.pi**2)
    a_m = inner_m3 ** (1/3)
    a_AU = a_m / AU_M
    
    # Equilibrium temperature for different albedos
    albedos = {"Gas Giant": 0.1, "Rocky": 0.3, "Icy": 0.7}
    Teq = {}
    R_star_AU = R_star * (R_SUN_M / AU_M)
    for planet_type, A in albedos.items():
        sqrt_term = np.sqrt(R_star_AU / (2 * a_AU))
        teq = T_star * sqrt_term * (1 - A)**0.25
        Teq[planet_type] = round(teq, 1)
    
    # Planet classification based on radius
    if R_p_Rearth < 1.8:
        planet_class = "Super-Earth"
    elif R_p_Rearth < 4:
        planet_class = "Mini-Neptune"
    elif R_p_Rearth < 10:
        planet_class = "Sub-Neptune"
    else:
        planet_class = "Gas Giant"
    
    return {
        "R_p_Rearth": round(R_p_Rearth, 2),
        "M_p_Mearth": round(M_p_Mearth, 2),
        "Density_rel_Earth": round(density_rel_earth, 2),
        "Orbit_AU": round(a_AU, 3),
        "Teq": Teq,
        "Class": planet_class
    }

# === NASA Exoplanet Archive TAP Query ===
def try_nasa_params(tic_id: str) -> Optional[Dict[str, Any]]:
    """Query NASA Exoplanet Archive for planet and stellar parameters."""
    url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    query = f"""
    SELECT pl_name, hostname, discoverymethod, pl_orbper, pl_rade, pl_bmasse, 
           st_teff, st_rad, st_mass 
    FROM pscomppars 
    WHERE hostname = '{tic_id}'
    """
    try:
        r = requests.get(url, params={"query": query, "format": "json"}, timeout=15)
        if r.status_code == 200 and len(r.json()) > 0:
            data = r.json()[0]
            print(f"[INFO] NASA TAP: Found {len(r.json())} records for {tic_id}")
            return {
                "T_star": data.get("st_teff"),
                "R_star": data.get("st_rad"),
                "M_star": data.get("st_mass", 1.0),
                "planet_data": {k: v for k, v in data.items() if k.startswith("pl_")}
            }
    except Exception as e:
        print(f"[WARN] NASA TAP failed: {e}")
    return None

# === ExoFOP Check ===
def try_exofop_data(tic_id: str) -> bool:
    """Check for TESS target data on ExoFOP."""
    clean_id = tic_id.replace("TIC ", "").strip()
    url = f"https://exofop.ipac.caltech.edu/tess/target.php?id={clean_id}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200 and len(r.text) > 5000:
            if any(keyword in r.text.lower() for keyword in ["toi", "planet", "tess magnitude"]):
                print(f"[INFO] ExoFOP: Data found for {tic_id}")
                return True
    except Exception as e:
        print(f"[WARN] ExoFOP request failed: {e}")
    return False

# === ESO HARPS ===
def try_eso_data(tic_id: str) -> Optional[Any]:
    """Query ESO archive for HARPS observations. Fixed index error."""
    try:
        from astroquery.eso import Eso
        eso = Eso()
        tbl = eso.query_instrument("HARPS", column_filters={"object": tic_id})
        if tbl is not None and len(tbl) > 0:
            print(f"[INFO] ESO HARPS: RV data found for {tic_id}")
            return tbl[0] if len(tbl) > 0 else None  # Safe access
    except ImportError:
        print("[WARN] astroquery.eso not available")
    except Exception as e:
        print(f"[WARN] ESO HARPS query failed: {e}")
    return None

# === Simbad Query ===
def try_simbad(tic_id: str) -> Optional[Any]:
    "Query Simbad for basic stellar info."
    try:
        result = Simbad.query_object(tic_id)
        if result is not None and len(result) > 0:
            print(f"[INFO] Simbad: Found data for {tic_id}")
            return result
    except Exception as e:
        print(f"[WARN] Simbad query failed: {e}")
    return None

# === Vizier Query ===
def try_vizier(tic_id: str) -> Optional[Any]:
    "Query Vizier catalogs for the object."
    try:
        v = Vizier(columns=["**"], row_limit=1)
        res = v.query_object(tic_id)
        if len(res) > 0:
            print(f"[INFO] Vizier: Match found for {tic_id}")
            return res
    except Exception as e:
        print(f"[WARN] Vizier query failed: {e}")
    return None

# === Gaia DR3 ===
def try_gaia_coords(tic_id: str) -> Optional[Any]:
    "Query Gaia DR3 using coordinates from MAST TIC."
    try:
        from astroquery.gaia import Gaia
        clean_id = tic_id.replace("TIC ", "").strip()
        tic_data = Catalogs.query_object(f"TIC {clean_id}", catalog="TIC")
        if len(tic_data) == 0:
            return None
        ra, dec = tic_data[0]["ra"], tic_data[0]["dec"]
        
        query = f"""
        SELECT TOP 1 source_id, ra, dec, phot_g_mean_mag, teff_gspphot
        FROM gaiadr3.gaia_source 
        WHERE 1=CONTAINS(POINT('ICRS', {ra}, {dec}), CIRCLE('ICRS', {ra}, {dec}, 0.001))
        """
        job = Gaia.launch_job_async(query)
        res = job.get_results()
        if len(res) > 0:
            print(f"[INFO] Gaia DR3: Cross-matched for {tic_id}")
            return res
    except ImportError:
        print("[WARN] astroquery.gaia not available")
    except Exception as e:
        print(f"[WARN] Gaia query failed: {e}")
    return None

# === MAST TIC Catalog ===
def get_star_params(tic_id: str) -> Dict[str, float]:
    "Fetch stellar parameters from MAST TIC (reliable source)."
    star_cache_file = f"cache/{tic_id.replace(' ', '_')}_star.json"  # Cache for star_params
    if os.path.exists(star_cache_file):
        try:
            with open(star_cache_file, 'r') as f:
                params = json.load(f)
            print(f"[INFO] Star params loaded from cache for {tic_id}")
            return params
        except Exception as e:
            print(f"[WARN] Star cache read error: {e}")

    try:
        clean_id = tic_id.replace("TIC ", "").strip()
        catalog_data = Catalogs.query_object(f"TIC {clean_id}", catalog="TIC")
        if len(catalog_data) > 0:
            row = catalog_data[0]
            # Prosses NaN/Inf
            t_star = row["Teff"] if np.isfinite(row["Teff"]) else 5800
            r_star = row["rad"] if np.isfinite(row["rad"]) else 1.0
            m_star = row["mass"] if np.isfinite(row["mass"]) else 1.0
            params = {"T_star": float(t_star), "R_star": float(r_star), "M_star": float(m_star)}
            with open(star_cache_file, 'w') as f:
                json.dump(params, f)
            print(f"[INFO] MAST TIC: Parameters fetched and cached for {tic_id} (T={t_star:.0f}K, R={r_star:.2f}, M={m_star:.2f})")
            return params
    except Exception as e:
        print(f"[WARN] MAST query failed: {e}")
    return {"T_star": 5800.0, "R_star": 1.0, "M_star": 1.0}

# === Cached Search Helper ===
def cached_search(tic_id: str, mission: str) -> lk.SearchResult:
    """Cached lightkurve search to avoid re-query."""
    cache_key = f"cache/search_{tic_id.replace(' ', '_')}_{mission}.pkl"
    os.makedirs("cache", exist_ok=True)
    if os.path.exists(cache_key):
        try:
            with open(cache_key, 'rb') as f:
                search = pickle.load(f)
            print(f"[INFO] Loaded cached search for {tic_id} ({mission})")
            return search
        except Exception as e:
            print(f"[WARN] Cache search read error: {e}")
    
    try:
        search = lk.search_lightcurve(tic_id, mission=mission)
        with open(cache_key, 'wb') as f:
            pickle.dump(search, f)
        print(f"[INFO] Cached new search for {tic_id} ({mission})")
        return search
    except Exception as e:
        print(f"[WARN] Search failed for {tic_id} ({mission}): {e}")
        return lk.SearchResult([])

# === Main Lightcurve and BLS Function ===
def get_lightcurve_and_bls(
    tic_id: str,
    missions: Tuple[str, ...] = ("TESS", "Kepler", "K2"),
    Pmin: float = 0.5,
    Pmax: float = 30.0,
    Nperiods: int = 800, #2000 -> 800 for speed
    dur_min: float = 0.01,
    dur_max: float = 0.3,
    use_cache: bool = True
) -> Tuple[Any, Any, Any, float, float, float, float]:

    start_total = time.time()
    os.makedirs("cache", exist_ok=True)
    os.makedirs("data/local_lightcurves", exist_ok=True)

    cache_file = f"cache/{tic_id.replace(' ', '_')}.fits"
    local_file = f"data/local_lightcurves/{tic_id}.npy"
    lc = None
    span = 0.0

    # 1. Cache load and check coverage (full/partial/none)
    full_cache = False
    partial_cache = False
    if use_cache and os.path.exists(cache_file):
        try:
            lc = lk.read(cache_file).remove_nans().normalize()
            span = np.ptp(lc.time.value)
            print(f"[INFO] LC loaded from cache for {tic_id}, span={span:.1f} days")
            if span >= COVERAGE_THRESHOLD:
                full_cache = True
                print(f"[INFO] Full cache hit — skipping fetches")
            else:
                partial_cache = True
                print(f"[INFO] Partial cache (span={span:.1f}d < {COVERAGE_THRESHOLD}d) — will augment")
        except Exception as e:
            print(f"[WARN] Cache read error: {e}")
            lc = None
            span = 0.0

    if full_cache:
        # Skip everything else
        pass
    else:
        # 2. Parallel check all external sources (metadata/planet data)
        start_sources = time.time()
        source_data = None  # To hold any found data (e.g., NASA planet_data)
        fetch_funcs = [try_nasa_params, try_exofop_data, try_eso_data, try_gaia_coords, try_simbad, try_vizier]
        with ThreadPoolExecutor(max_workers=min(6, len(fetch_funcs))) as executor:
            futures = {executor.submit(func, tic_id): func.__name__ for func in fetch_funcs}
            for future in as_completed(futures, timeout=20):  # Overall timeout 20s
                try:
                    data = future.result(timeout=5)  # Per future 5s
                    if data is not None:
                        source_name = futures[future]
                        print(f"[OK] Source {source_name} succeeded for {tic_id}")
                        if source_name == "try_nasa_params" and "planet_data" in data:
                            source_data = data  # Prioritize NASA for planet params
                            print(f"[INFO] Using NASA data as primary source")
                            break  # Stop on first good (NASA preferred)
                        elif source_name == "try_exofop_data":
                            print(f"[INFO] ExoFOP indicates data — prioritize TESS fetch")
                            break
                except Exception as e:
                    print(f"[WARN] {futures[future]} failed: {e}")
        print(f"[TIME] External sources check: {time.time() - start_sources:.2f}s")

        # If source found (e.g., NASA/ExoFOP), it signals data exists — proceed to missions confidently
        has_source_data = source_data is not None or any("succeeded" in log for log in ["NASA", "ExoFOP"])

        # 3. Parallel missions fetch with early stop on full coverage
        if lc is None or partial_cache or (partial_cache and not has_source_data):  # Augment if partial or no source
            start_download = time.time()
            mission_order = ["TESS", "K2", "Kepler"]  # Priority
            augmented = False


                


            def fetch_mission(mission):
                if mission not in missions:
                    return None, 0.0
                try:
                    search = cached_search(tic_id, mission)
                    if len(search) == 0:
                        print(f"[INFO] No data found for {mission}")
                        return None, 0.0
        # Remove cache
                    search.clear_download_cache()

                    lcs = search.download_all(flux_column="pdcsap_flux", timeout=60)
                    if lcs is None or len(lcs) == 0:
                        return None, 0.0
                    new_lc = lcs.stitch().remove_nans().normalize()
                    new_span = np.ptp(new_lc.time.value)
                    print(f"[INFO] {mission} fetched: span={new_span:.1f} days")
                    return new_lc, new_span
                except Exception as e:
                    print(f"[WARN] {mission} failed: {e}")
                    return None, 0.0
            




            with ThreadPoolExecutor(max_workers=len(mission_order)) as executor:
                futures = {executor.submit(fetch_mission, m): m for m in mission_order}
                fetched_lcs = {}  # {mission: (lc, span)}
                for future in as_completed(futures):
                    mission = futures[future]
                    new_lc, new_span = future.result()
                    if new_lc is not None:
                        fetched_lcs[mission] = (new_lc, new_span)

            # Now combine: Base from first, augment others if adds coverage
            if fetched_lcs:
                # Sort by priority (TESS first)
                sorted_missions = sorted(fetched_lcs, key=lambda m: mission_order.index(m))
                base_lc, base_span = fetched_lcs[sorted_missions[0]]
                lc = base_lc
                span = base_span
                print(f"[INFO] Base LC from {sorted_missions[0]} (span={span:.1f}d)")

                for mission in sorted_missions[1:]:
                    new_lc, new_span = fetched_lcs[mission]
                    if new_span + span < COVERAGE_THRESHOLD * 0.5:  # If adds < half threshold, skip
                        print(f"[INFO] {mission} adds little coverage — skipped")
                        continue
                    try:
                        temp_lc = lc.append(new_lc)
                        temp_span = np.ptp(temp_lc.time.value)
                        if temp_span > span + 0.1:  # Adds meaningful span
                            lc = temp_lc
                            span = temp_span
                            augmented = True
                            print(f"[INFO] Augmented with {mission}, new span={span:.1f}d")
                            if span >= COVERAGE_THRESHOLD:
                                print(f"[INFO] Full coverage achieved — no more fetches")
                                break
                        else:
                            print(f"[INFO] {mission} overlaps — skipped")
                    except Exception as e:
                        print(f"[WARN] Augment {mission} failed: {e}")

                if lc is not None:
                    lc.to_fits(cache_file)
                    print(f"[INFO] Cached combined LC for {tic_id} (span={span:.1f}d)")

            print(f"[TIME] Missions parallel fetch/augment: {time.time() - start_download:.2f}s")

        # If source_data from NASA, inject if possible (e.g., use pl_orbper as period hint, but for LC — no direct)
        if source_data:
            print(f"[INFO] NASA data used for params; LC from missions")

    # 4. Synthetic only if absolutely no data (last resort)
    if lc is None or span == 0:
        print(f"[WARN] No real data found — generating synthetic LC for {tic_id}")
        time_arr = np.linspace(0, 27.0, 1200)
        rand_period = np.random.uniform(5, 20)
        phase = (time_arr % rand_period) / rand_period
        transit_width = 0.01 / rand_period
        transit = np.where(np.abs(phase - 0.5) < transit_width / 2, 0.01, 0.0)
        flux = 1.0 - transit + np.random.normal(0, 0.0005, len(time_arr))
        lc = lk.LightCurve(time=time_arr, flux=flux).normalize()
        np.save(local_file, {"time": time_arr, "flux": flux})
        span = 27.0
        print(f"[INFO] Synthetic data saved to {local_file} (P≈{rand_period:.1f}d, span=27.0d)")

    # 5. Fetch stellar params (with NASA override if available)
    start_star = time.time()
    star_params = get_star_params(tic_id)
    if source_data and "planet_data" in source_data:
        star_params.update({"T_star": source_data.get("T_star", star_params["T_star"]),
                            "R_star": source_data.get("R_star", star_params["R_star"]),
                            "M_star": source_data.get("M_star", star_params["M_star"])})
        print(f"[INFO] Overrode star params with NASA data")
    print(f"[TIME] Star params fetch: {time.time() - start_star:.2f}s")

    # 6. BLS Analysis
    start_bls = time.time()
    time_val, flux_val = lc.time.value, lc.flux.value
    try:
        periods = np.linspace(Pmin, Pmax, Nperiods)
        durations = np.linspace(dur_min, dur_max, 5)
        bls = BoxLeastSquares(time_val, flux_val)
        result = bls.power(periods, durations)

        max_power = np.nanmax(result.power)
        if not np.isfinite(max_power) or max_power < 1e-6:
            print(f"[WARN] No significant BLS peak for {tic_id} (max_power={max_power})")
            default_period = 13.5
            default_depth = 0.01
            planet_params = generate_planet_params(
                depth=default_depth, 
                R_star=star_params["R_star"], 
                T_star=star_params["T_star"],
                M_star=star_params["M_star"],
                period=default_period
            )
            print(f"[INFO] Default params for {tic_id}: Star T={star_params['T_star']:.0f}K, R={star_params['R_star']:.2f}, M={star_params['M_star']:.2f}")
            print(f"Planet: R={planet_params['R_p_Rearth']:.2f}Re, Class={planet_params['Class']}")
            print(f"[TIME] BLS analysis: {time.time() - start_bls:.2f}s")
            print(f"[TIME] Total for {tic_id}: {time.time() - start_total:.2f}s")
            return lc, None, None, default_period, 0.0, 0.2, default_depth

        ix = np.nanargmax(result.power)
        period = float(result.period[ix])
        t0 = float(result.transit_time[ix])
        duration = float(result.duration[ix])
        depth = float(result.depth[ix])
        print(f"[INFO] BLS peak: P={period:.4f} days, depth={depth:.4f}")
        
        planet_params = generate_planet_params(
            depth=depth, 
            R_star=star_params["R_star"], 
            T_star=star_params["T_star"],
            M_star=star_params["M_star"],
            period=period
        )
        print(f"[SUCCESS] Planet params computed for {tic_id}")
        print(f"[INFO] Params for {tic_id}: Star T={star_params['T_star']:.0f}K, R={star_params['R_star']:.2f}, M={star_params['M_star']:.2f}")
        print(f"Planet: R={planet_params['R_p_Rearth']:.2f}Re, Class={planet_params['Class']}, Teq Rocky={planet_params['Teq']['Rocky']}K")
        print(f"[TIME] BLS analysis: {time.time() - start_bls:.2f}s")
        print(f"[TIME] Total for {tic_id}: {time.time() - start_total:.2f}s")
        return lc, bls, result, period, t0, duration, depth

    except Exception as e:
        print(f"[WARN] BLS error: {e}")
        default_period = 13.5
        default_depth = 0.01
        planet_params = generate_planet_params(
            depth=default_depth, 
            R_star=star_params["R_star"], 
            T_star=star_params["T_star"],
            M_star=star_params["M_star"],
            period=default_period
        )
        print(f"[INFO] Default params for {tic_id}: Star T={star_params['T_star']:.0f}K, R={star_params['R_star']:.2f}, M={star_params['M_star']:.2f}")
        print(f"Planet: R={planet_params['R_p_Rearth']:.2f}Re, Class={planet_params['Class']}")
        print(f"[TIME] BLS analysis: {time.time() - start_bls:.2f}s")
        print(f"[TIME] Total for {tic_id}: {time.time() - start_total:.2f}s")
        return lc, None, None, default_period, 0.0, 0.2, default_depth

# === Enhanced Plotting Function ===
def plot_lc_and_bls(lc: Any, bls_result: Optional[Any], tic_id: str, period: Optional[float] = None, depth: Optional[float] = None) -> None:
    """
    Plot lightcurve and BLS periodogram with enhancements.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Lightcurve
    time_val = lc.time.value
    flux_val = lc.flux.value
    ax1.plot(time_val, flux_val, "k.", markersize=1, alpha=0.7)
    if period:
        phase = ((time_val % period) - 0.5) / period + 0.5
        ax1_phase = ax1.twinx()
        ax1_phase.plot(phase, flux_val, "b-", alpha=0.5, linewidth=1)
        ax1_phase.set_ylabel("Flux (Phase Folded)")
    ax1.set_xlabel("Time [BJD - 2457000]")
    ax1.set_ylabel("Normalized Flux")
    ax1.set_title(f"Lightcurve: {tic_id}")
    ax1.grid(True, alpha=0.3)

    # BLS Periodogram
    if bls_result is not None:
        ax2.plot(bls_result.period, bls_result.power, "r-", linewidth=1)
        if period:
            ax2.axvline(period, color="g", linestyle="--", label=f"Best P={period:.3f}")
            ax2.legend()
        ax2.set_xlabel("Period [days]")
        ax2.set_ylabel("BLS Power")
    else:
        ax2.text(0.5, 0.5, "No BLS Result\n(Using Defaults)", ha="center", va="center", transform=ax2.transAxes)
    ax2.set_title(f"BLS Periodogram: {tic_id}")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Works with  ANY TIC ID
    tic_ids = ["TIC 150428135", "TIC 17361", "TIC 17362"]  # Example 
    for tid in tic_ids:
        lc, bls, result, period, t0, dur, depth = get_lightcurve_and_bls(tid)
        print(f"\n=== Results for {tid} ===")
        print(f"Transit: P={period:.3f}d, Depth={depth:.4f}")
     
        
        plot_lc_and_bls(lc, result, tid, period, depth)