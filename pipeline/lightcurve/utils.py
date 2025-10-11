import os
import pickle
import lightkurve as lk
import matplotlib.pyplot as plt
from typing import Optional, Any
import numpy as np

def cached_search(tic_id: str, mission: str) -> lk.SearchResult:
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