import numpy as np
import lightkurve as lk
import os
from astropy.timeseries import BoxLeastSquares

def get_lightcurve_and_bls(
    tic_id, mission="TESS", Pmin=0.5, Pmax=30.0, Nperiods=5000,
    dur_min=0.01, dur_max=0.3
):
    """
    Загрузка кривой блеска и поиск транзита методом Box Least Squares.
    """
    os.makedirs("cache", exist_ok=True)
    cache_file = f"cache/{tic_id.replace(' ', '_')}.fits"

    # ✅ Исправленный способ чтения FITS
    if os.path.exists(cache_file):
        lc = lk.read(cache_file).remove_nans().normalize()
    else:
        search = lk.search_lightcurve(tic_id, mission=mission)
        if len(search) == 0:
            print(f"[WARN] No lightcurve found for {tic_id}")
            return None, None, None, None, None, None, None
        
        lc = search[0].download().remove_nans().normalize()
        lc.to_fits(cache_file)

    # Проверка на NaN
    time, flux = lc.time.value, lc.flux.value
    if np.all(np.isnan(flux)):
        print(f"[WARN] All flux values are NaN for {tic_id}")
        return None, None, None, None, None, None, None

    # BLS анализ
    periods = np.linspace(Pmin, Pmax, Nperiods)
    durations = np.linspace(dur_min, dur_max, 10)
    bls = BoxLeastSquares(time, flux)
    result = bls.power(periods, durations)

    power_max = np.nanmax(result.power)
    if np.isnan(power_max) or power_max == 0:
        print(f"[WARN] No significant BLS peak found for {tic_id}")
        return lc, bls, result, periods[0], time[0], durations[0], 0.0

    ix = int(np.nanargmax(result.power))
    return (
        lc, bls, result,
        float(result.period[ix]),
        float(result.transit_time[ix]),
        float(result.duration[ix]),
        float(result.depth[ix])
    )
