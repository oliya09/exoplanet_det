# fetch.py (delete corrupt file before download)
import os
import time
import numpy as np
import lightkurve as lk
from typing import Tuple, Optional, Any, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from .constants import COVERAGE_THRESHOLD
from .utils import cached_search
from .catalog import try_nasa_params, try_exofop_data, try_eso_data, try_gaia_coords, try_simbad, try_vizier

def fetch_mission(tic_id: str, mission: str, missions: Tuple[str, ...]) -> Tuple[Optional[lk.LightCurve], float]:
    if mission not in missions:
        return None, 0.0
    try:
        search = cached_search(tic_id, mission)
        if len(search) == 0:
            print(f"[INFO] No data found for {mission}")
            return None, 0.0
        
        # Delete known corrupt file before download
        corrupt_path = os.path.expanduser("~/.lightkurve/cache/mastDownload/TESS/tess2018206045859-s0001-0000000150428135-0120-s/tess2018206045859-s0001-0000000150428135-0120-s_lc.fits")
        if os.path.exists(corrupt_path):
            os.remove(corrupt_path)
            print(f"[INFO] Removed corrupt file: {corrupt_path}")
        
        for attempt in range(3):  # Increased retries
            try:
                lcs = search.download_all(flux_column="pdcsap_flux", timeout=60, cache=False)  # Force no cache
                break
            except Exception as download_e:
                print(f"[WARN] Download attempt {attempt+1} failed for {mission}: {download_e}")
                if attempt == 2:
                    return None, 0.0
        if lcs is None or len(lcs) == 0:
            return None, 0.0
        new_lc = lcs.stitch().remove_nans().normalize()
        new_span = np.ptp(new_lc.time.value)
        print(f"[INFO] {mission} fetched: span={new_span:.1f} days")
        return new_lc, new_span
    except Exception as e:
        print(f"[WARN] {mission} failed: {e}")
        return None, 0.0

# Rest of fetch.py remains the same...

def get_lightcurve(
    tic_id: str,
    missions: Tuple[str, ...] = ("TESS", "Kepler", "K2"),
    use_cache: bool = True
) -> Tuple[Optional[lk.LightCurve], float, Optional[Dict[str, Any]]]:
    os.makedirs("cache", exist_ok=True)
    os.makedirs("data/local_lightcurves", exist_ok=True)

    cache_file = f"cache/{tic_id.replace(' ', '_')}.fits"
    lc = None
    span = 0.0

    # Cache load
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

    source_data = None
    if full_cache:
        return lc, span, source_data

    # Parallel sources check
    start_sources = time.time()
    fetch_funcs = [try_nasa_params, try_exofop_data, try_eso_data, try_gaia_coords, try_simbad, try_vizier]
    with ThreadPoolExecutor(max_workers=min(6, len(fetch_funcs))) as executor:
        futures = {executor.submit(func, tic_id): func.__name__ for func in fetch_funcs}
        for future in as_completed(futures, timeout=20):
            try:
                data = future.result(timeout=5)
                if data is not None:
                    source_name = futures[future]
                    print(f"[OK] Source {source_name} succeeded for {tic_id}")
                    if source_name == "try_nasa_params" and "planet_data" in data:
                        source_data = data
                        print(f"[INFO] Using NASA data as primary source")
                        break
                    elif source_name == "try_exofop_data":
                        print(f"[INFO] ExoFOP indicates data — prioritize TESS fetch")
                        break
            except Exception as e:
                print(f"[WARN] {futures[future]} failed: {e}")
    print(f"[TIME] External sources check: {time.time() - start_sources:.2f}s")

    has_source_data = source_data is not None

    # Parallel missions fetch
    if lc is None or partial_cache or (partial_cache and not has_source_data):
        start_download = time.time()
        mission_order = ["TESS", "K2", "Kepler"]
        with ThreadPoolExecutor(max_workers=len(mission_order)) as executor:
            futures = {executor.submit(fetch_mission, tic_id, m, missions): m for m in mission_order}
            fetched_lcs = {}
            for future in as_completed(futures):
                mission = futures[future]
                new_lc, new_span = future.result()
                if new_lc is not None:
                    fetched_lcs[mission] = (new_lc, new_span)

        # Combine
        if fetched_lcs:
            sorted_missions = sorted(fetched_lcs, key=lambda m: mission_order.index(m))
            base_lc, base_span = fetched_lcs[sorted_missions[0]]
            lc = base_lc
            span = base_span
            print(f"[INFO] Base LC from {sorted_missions[0]} (span={span:.1f}d)")

            for mission in sorted_missions[1:]:
                new_lc, new_span = fetched_lcs[mission]
                if new_span + span < COVERAGE_THRESHOLD * 0.5:
                    print(f"[INFO] {mission} adds little coverage — skipped")
                    continue
                try:
                    temp_lc = lc.append(new_lc)
                    temp_span = np.ptp(temp_lc.time.value)
                    if temp_span > span + 0.1:
                        lc = temp_lc
                        span = temp_span
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

    # Synthetic fallback
    if lc is None or span == 0:
        print(f"[WARN] No real data found — generating synthetic LC for {tic_id}")
        time_arr = np.linspace(0, 27.0, 1200)
        rand_period = np.random.uniform(5, 20)
        phase = (time_arr % rand_period) / rand_period
        transit_width = 0.01 / rand_period
        transit = np.where(np.abs(phase - 0.5) < transit_width / 2, 0.01, 0.0)
        flux = 1.0 - transit + np.random.normal(0, 0.0005, len(time_arr))
        lc = lk.LightCurve(time=time_arr, flux=flux).normalize()
        local_file = f"data/local_lightcurves/{tic_id}.npy"
        os.makedirs(os.path.dirname(local_file), exist_ok=True)
        np.save(local_file, {"time": time_arr, "flux": flux})
        span = 27.0
        print(f"[INFO] Synthetic data saved to {local_file} (P≈{rand_period:.1f}d, span=27.0d)")

    if source_data:
        print(f"[INFO] NASA data used for params; LC from missions")

    return lc, span, source_data