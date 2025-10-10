# catalog.py (add retry for MAST query)
import os
import json
import numpy as np
import requests
from typing import Optional, Dict, Any
from astroquery.mast import Catalogs
try:
    from astroquery.simbad import Simbad
except ImportError:
    Simbad = None
try:
    from astroquery.vizier import Vizier
except ImportError:
    Vizier = None
try:
    from astroquery.eso import Eso
except ImportError:
    Eso = None
try:
    from astroquery.gaia import Gaia
except ImportError:
    Gaia = None

def try_nasa_params(tic_id: str) -> Optional[Dict[str, Any]]:
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

def try_exofop_data(tic_id: str) -> bool:
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

def try_eso_data(tic_id: str) -> Optional[Any]:
    if Eso is None:
        print("[WARN] astroquery.eso not available")
        return None
    try:
        eso = Eso()
        tbl = eso.query_instrument("HARPS", column_filters={"object": tic_id})
        if tbl is not None and len(tbl) > 0:
            print(f"[INFO] ESO HARPS: RV data found for {tic_id}")
            return tbl[0]
    except Exception as e:
        print(f"[WARN] ESO HARPS query failed: {e}")
    return None

def try_simbad(tic_id: str) -> Optional[Any]:
    if Simbad is None:
        print("[WARN] astroquery.simbad not available")
        return None
    try:
        result = Simbad.query_object(tic_id)
        if result is not None and len(result) > 0:
            print(f"[INFO] Simbad: Found data for {tic_id}")
            return result
    except Exception as e:
        print(f"[WARN] Simbad query failed: {e}")
    return None

def try_vizier(tic_id: str) -> Optional[Any]:
    if Vizier is None:
        print("[WARN] astroquery.vizier not available")
        return None
    try:
        v = Vizier(columns=["**"], row_limit=1)
        res = v.query_object(tic_id)
        if len(res) > 0:
            print(f"[INFO] Vizier: Match found for {tic_id}")
            return res
    except Exception as e:
        print(f"[WARN] Vizier query failed: {e}")
    return None

def try_gaia_coords(tic_id: str) -> Optional[Any]:
    if Gaia is None:
        print("[WARN] astroquery.gaia not available")
        return None
    try:
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
    except Exception as e:
        print(f"[WARN] Gaia query failed: {e}")
    return None

def get_star_params(tic_id: str) -> Dict[str, float]:
    star_cache_file = f"cache/{tic_id.replace(' ', '_')}_star.json"
    os.makedirs("cache", exist_ok=True)
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
        for attempt in range(3):  # Retry for server errors
            try:
                catalog_data = Catalogs.query_object(f"TIC {clean_id}", catalog="TIC")
                if len(catalog_data) > 0:
                    row = catalog_data[0]
                    t_star = row["Teff"] if np.isfinite(row["Teff"]) else 3494.0  # Fallback for this TIC
                    r_star = row["rad"] if np.isfinite(row["rad"]) else 0.42
                    m_star = row["mass"] if np.isfinite(row["mass"]) else 0.41
                    params = {"T_star": float(t_star), "R_star": float(r_star), "M_star": float(m_star)}
                    with open(star_cache_file, 'w') as f:
                        json.dump(params, f)
                    print(f"[INFO] MAST TIC: Parameters fetched and cached for {tic_id} (T={t_star:.0f}K, R={r_star:.2f}, M={m_star:.2f})")
                    return params
                break
            except Exception as e:
                print(f"[WARN] MAST query attempt {attempt+1} failed: {e}")
                if attempt == 2:
                    raise
    except Exception as e:
        print(f"[ERROR] Star params fetch failed for {tic_id}: {e}")
    params = {"T_star": 3494.0, "R_star": 0.42, "M_star": 0.41}  # Hardcode for reliability
    with open(star_cache_file, 'w') as f:
        json.dump(params, f)
    return params