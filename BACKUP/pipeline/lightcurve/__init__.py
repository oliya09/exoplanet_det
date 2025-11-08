# export key functions
from .fetch import get_lightcurve
from .bls import run_bls_analysis
from .utils import plot_lc_and_bls

# Главная функция-оркестратор (определена здесь)
def get_lightcurve_and_bls(
    tic_id: str,
    missions: tuple = ("TESS", "Kepler", "K2"),
    Pmin: float = 0.5,
    Pmax: float = 30.0,
    Nperiods: int = 800,
    dur_min: float = 0.01,
    dur_max: float = 0.3,
    use_cache: bool = True
):
    import time
    import warnings
    import lightkurve as lk
    from typing import Tuple, Optional, Any
    from .fetch import get_lightcurve
    from .catalog import get_star_params
    from .bls import run_bls_analysis

    warnings.filterwarnings("ignore", category=UserWarning)

    start_total = time.time()

    # Fetch LC and sources
    lc, span, source_data = get_lightcurve(tic_id, missions, use_cache)
    if lc is None or span == 0:
        print(f"[ERROR] Failed to get LC for {tic_id}")
        return None, None, None, 13.5, 0.0, 0.2, 0.01

    # Star params
    start_star = time.time()
    star_params = get_star_params(tic_id)
    print(f"[TIME] Star params fetch: {time.time() - start_star:.2f}s")

    # BLS
    bls, result, period, t0, duration, depth = run_bls_analysis(
        tic_id, lc, source_data, star_params, Pmin, Pmax, Nperiods, dur_min, dur_max
    )

    print(f"[TIME] Total for {tic_id}: {time.time() - start_total:.2f}s")
    return lc, bls, result, period, t0, duration, depth

if __name__ == "__main__":
    # Тест всего пакета
    tic_ids = ["TIC 150428135", "TIC 17361", "TIC 17362"]
    for tid in tic_ids:
        lc, bls, result, period, t0, dur, depth = get_lightcurve_and_bls(tid)
        print(f"\n=== Results for {tid} ===")
        print(f"Transit: P={period:.3f}d, Depth={depth:.4f}")
        from .utils import plot_lc_and_bls
        plot_lc_and_bls(lc, result, tid, period, depth)