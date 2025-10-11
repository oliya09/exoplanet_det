# planet.py (fully corrected with units, Mp/density/class added, no truthiness issues)
import numpy as np
import astropy.constants as const
import astropy.units as u
from astroquery.exceptions import TimeoutError as AstroTimeoutError
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def safe_query(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except (AstroTimeoutError, Exception) as e:
        print(f"[WARN] NASA server not responding or query failed: {e}")
        return None

def get_planet_data(period, depth, T_star, R_star, M_star, k=None):
    print(f"[DEBUG] Input: period={period}, depth={depth}, T_star={T_star}, R_star={R_star}, M_star={M_star}, k={k}")
    if depth <= 0 or any(x <= 0 for x in [period, T_star, R_star, M_star]):
        print(f"[WARN] Invalid input: depth={depth}, period={period}, T_star={T_star}, R_star={R_star}, M_star={M_star}")
        return {
            "R_p_Rearth": None,
            "M_p_Mearth": None,
            "Density_gcm3": None,
            "Orbit_AU": None,
            "Teq": {},
            "Class": "Unknown"
        }
    
    try:
        period_sec = (period * u.day).to(u.s)
        a = ((const.G * (M_star * u.Msun) * period_sec**2) / (4 * np.pi**2))**(1/3)
        orbit_au = a.to(u.au)
        print(f"[DEBUG] Semi-major axis: a={orbit_au.value} AU")

        R_pl = np.sqrt(depth) * (R_star * u.Rsun)
        rp = R_pl.to(u.Rearth).value
        print(f"[DEBUG] Planet radius: rp={rp} R⊕")

        # Масса
        if k is not None and np.isfinite(k) and k > 0:
            k_si = k * u.m / u.s
            Mp = ((period_sec / (2 * np.pi * const.G)) ** (1/3) * k_si * (M_star * u.Msun)**(2/3) ).to(u.M_earth).value
            print(f"[INFO] Mass from RV K={k} m/s: {Mp:.2f} M⊕")
        else:
            Mp = rp ** 2.7
            print(f"[INFO] Empirical mass: {Mp:.2f} M⊕ (K not provided)")

        # Плотность
        R_pl_m = R_pl.to(u.m)
        vol = (4/3) * np.pi * R_pl_m**3
        Mp_q = Mp * const.M_earth
        density = (Mp_q / vol).to(u.g / u.cm**3).value if np.isfinite(Mp) else None
        print(f"[DEBUG] Density: {density:.2f} g/cm³" if density is not None else "[DEBUG] Density: None")

        # Teq
        T_star_q = T_star * u.K
        albedos = {"Gas giant (0.1)": 0.1, "Rocky (0.3)": 0.3, "Icy (0.7)": 0.7}
        Teqs = {}
        if np.isfinite(orbit_au.value) and orbit_au.value > 0:
            for kind, A in albedos.items():
                teq = T_star_q * np.sqrt((R_star * u.Rsun) / (2 * orbit_au)) * (1 - A)**0.25
                if np.isfinite(teq.value):
                    Teqs[kind] = round(teq.value, 1)
                else:
                    Teqs[kind] = None
            print(f"[DEBUG] Teq: {Teqs}")

        # Classification
        if rp < 1.8:
            planet_class = "Super-Earth"
        elif rp < 4:
            planet_class = "Mini-Neptune"
        elif rp < 10:
            planet_class = "Sub-Neptune"
        else:
            planet_class = "Gas Giant"
        print(f"[DEBUG] Class: {planet_class}")

        return {
            "R_p_Rearth": round(rp, 2) if np.isfinite(rp) else None,
            "M_p_Mearth": round(Mp, 2) if np.isfinite(Mp) else None,
            "Density_gcm3": round(density, 2) if density is not None and np.isfinite(density) else None,
            "Orbit_AU": round(orbit_au.value, 3) if np.isfinite(orbit_au.value) else None,
            "Teq": Teqs,
            "Class": planet_class
        }

    except Exception as e:
        print(f"[ERROR] Calculation failed: {e}")
        return {
            "R_p_Rearth": None,
            "M_p_Mearth": None,
            "Density_gcm3": None,
            "Orbit_AU": None,
            "Teq": {},
            "Class": "Unknown"
        }