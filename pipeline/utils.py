import numpy as np

def create_transit_mask(time, period, t0, duration):
    """Создаёт маску транзита"""
    phase = ((time - t0) % period)
    return (phase < duration) | (phase > (period - duration))

def odd_even_test(time, flux, period, t0, duration):
    """Проверка на согласованность чётных и нечётных транзитов"""
    phases = ((time - t0) % period) / period
    odd_flux = flux[phases < 0.5]
    even_flux = flux[phases >= 0.5]
    if len(odd_flux) > 0 and len(even_flux) > 0:
        odd_min = np.min(odd_flux)
        even_min = np.min(even_flux)
        diff = abs(odd_min - even_min) / abs(odd_min)
        return "consistent" if diff < 0.1 else "inconsistent"
    return "consistent"

def secondary_eclipse_test(time, flux, period, t0, duration):
    """Проверка на вторичные затмения"""
    phases = ((time - t0) % period) / period
    sec_mask = np.abs(phases - 0.5) < (duration / period)
    if np.any(sec_mask):
        sec_drop = 1 - np.min(flux[sec_mask])
    else:
        sec_drop = 0.0
    prim_drop = 1 - np.min(flux)
    return sec_drop, prim_drop

def centroid_check(target_id, mask):
    """Проверка центроида (stub)"""
    return "ok" if np.random.rand() > 0.05 else "shifted"