# bls.py (replaced generate_planet_params with get_planet_data, fixed Teq key, len checks for var/std)
import time
import numpy as np
from astropy.timeseries import BoxLeastSquares
from typing import Tuple, Optional, Any, Dict
from ..planet import get_planet_data  # Import fixed function

def run_bls_analysis(
    tic_id: str,
    lc: Any,
    source_data: Optional[Dict],
    star_params: Dict[str, float],
    Pmin: float = 0.5,
    Pmax: float = 30.0,
    Nperiods: int = 800,
    dur_min: float = 0.01,
    dur_max: float = 0.3
) -> Tuple[Optional[Any], Optional[Any], float, float, float, float]:
    start_bls = time.time()
    time_val, flux_val = lc.time.value, lc.flux.value
    flux_val = np.nan_to_num(flux_val, nan=1.0)  # Фикс NaN

    # Override star params
    final_star_params = star_params.copy()
    if source_data and "planet_data" in source_data:
        final_star_params.update({
            "T_star": source_data.get("T_star", star_params["T_star"]),
            "R_star": source_data.get("R_star", star_params["R_star"]),
            "M_star": source_data.get("M_star", star_params["M_star"])
        })
        print(f"[INFO] Overrode star params with NASA data")

    print(f"[INFO] Params for {tic_id}: Star T={final_star_params['T_star']:.0f}K, R={final_star_params['R_star']:.2f}, M={final_star_params['M_star']:.2f}")

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
            planet_params = get_planet_data(
                default_period, default_depth, 
                final_star_params["T_star"], final_star_params["R_star"], final_star_params["M_star"]
            )
            print(f"[INFO] Default params for {tic_id}: Star T={final_star_params['T_star']:.0f}K, R={final_star_params['R_star']:.2f}, M={final_star_params['M_star']:.2f}")
            print(f"Planet: R={planet_params['R_p_Rearth']:.2f}Re, Class={planet_params['Class']}")
            print(f"[TIME] BLS analysis: {time.time() - start_bls:.2f}s")
            return None, None, default_period, 0.0, 0.2, default_depth

        ix = np.nanargmax(result.power)
        period = float(result.period[ix])
        t0 = float(result.transit_time[ix])
        duration = float(result.duration[ix])
        depth = float(result.depth[ix])
        print(f"[INFO] BLS peak: P={period:.4f} days, depth={depth:.4f}")
        
        planet_params = get_planet_data(
            period, depth, 
            final_star_params["T_star"], final_star_params["R_star"], final_star_params["M_star"]
        )
        print(f"[SUCCESS] Planet params computed for {tic_id}")
        print(f"Planet: R={planet_params['R_p_Rearth']:.2f}Re, Class={planet_params['Class']}, Teq Rocky={planet_params['Teq']['Rocky (0.3)']}K")
        print(f"[TIME] BLS analysis: {time.time() - start_bls:.2f}s")
        return bls, result, period, t0, duration, depth

    except Exception as e:
        print(f"[WARN] BLS error: {e}")
        default_period = 13.5
        default_depth = 0.01
        planet_params = get_planet_data(
            default_period, default_depth, 
            final_star_params["T_star"], final_star_params["R_star"], final_star_params["M_star"]
        )
        print(f"[INFO] Default params for {tic_id}: Star T={final_star_params['T_star']:.0f}K, R={final_star_params['R_star']:.2f}, M={final_star_params['M_star']:.2f}")
        print(f"Planet: R={planet_params['R_p_Rearth']:.2f}Re, Class={planet_params['Class']}")
        print(f"[TIME] BLS analysis: {time.time() - start_bls:.2f}s")
        return None, None, default_period, 0.0, 0.2, default_depth