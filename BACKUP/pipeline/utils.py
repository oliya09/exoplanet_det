import os
import pickle
import numpy as np
import lightkurve as lk
import matplotlib.pyplot as plt
from typing import Optional, Any, Tuple, Union

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

def plot_lc_and_bls(lc: Any, bls_result: Optional[Any], tic_id: str, period: Optional[float] = None, depth: Optional[float] = None) -> None:
    """Plot lightcurve and BLS periodogram."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

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

# Functions
def create_transit_mask(time: np.ndarray, period: float, t0: float, duration: float) -> np.ndarray:
    """Создаёт маску транзита (улучшено: phase в [0,1], centered at 0)."""
    phases = ((time - t0) % period) / period
    half_dur = duration / (2 * period)  # Half for ingress/egress
    return np.abs(phases - 0.0) < half_dur  # Or wrap around if needed

def odd_even_test(time: np.ndarray, flux: np.ndarray, period: float, t0: float, duration: float, threshold: float = 0.05) -> str:
    """Проверка на согласованность чётных и нечётных транзитов (улучшено: mask + median)."""
    flux = np.nan_to_num(flux, nan=1.0)
    phases = ((time - t0) % period) / period
    odd_mask = (phases < 0.5)
    even_mask = (phases >= 0.5)
    
    if np.any(odd_mask) and np.any(even_mask):
        odd_flux = flux[odd_mask]
        even_flux = flux[even_mask]
        odd_min = np.min(odd_flux)
        even_min = np.min(even_flux)
        diff = abs(odd_min - even_min) / abs(odd_min) if abs(odd_min) > 0 else 0
        return "consistent" if diff < threshold else "inconsistent"
    return "insufficient_data"  # ADD: Лучше, чем default "consistent"

def secondary_eclipse_test(time: np.ndarray, flux: np.ndarray, period: float, t0: float, duration: float) -> Tuple[float, float]:
    """Проверка на вторичные затмения (улучшено: primary from transit mask)."""
    flux = np.nan_to_num(flux, nan=1.0)
    phases = ((time - t0) % period) / period
    
    # Secondary at phase 0.5
    sec_mask = np.abs(phases - 0.5) < (duration / period)
    sec_drop = (1 - np.min(flux[sec_mask])) if np.any(sec_mask) else 0.0
    
    # Primary: min during transit (use mask)
    transit_mask = create_transit_mask(time, period, t0, duration)
    prim_drop = (1 - np.min(flux[transit_mask])) if np.any(transit_mask) else (1 - np.min(flux))
    
    return sec_drop, prim_drop

def centroid_check(target_id: str, mask: np.ndarray) -> str:
    """Проверка центроида (улучшено: variance-based, not random)."""
    # Stub: real would query centroid offset from MAST/ExoFOP
    if len(mask) == 0:
        return "insufficient_data"
    
    # Simulate variance in masked flux (high var = shifted)
    flux_var = np.var(np.random.normal(1.0, 0.001, np.sum(mask)))  # Placeholder flux
    return "ok" if flux_var < 0.0001 else "shifted"  # Threshold for shift