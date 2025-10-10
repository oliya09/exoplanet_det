import numpy as np
import astropy.units as u
from astropy.constants import G, M_sun, R_sun, R_earth, au
from typing import Dict, Any

M_SUN_KG = M_sun.value
R_SUN_M = R_sun.value
R_EARTH_M = R_earth.value
AU_M = au.value
SEC_PER_DAY = 86400
COVERAGE_THRESHOLD = 20.0

def generate_planet_params(
    depth: float = 0.002,
    R_star: float = 1.0,
    T_star: float = 5800,
    M_star: float = 1.0,
    period: float = 13.5
) -> Dict[str, Any]:
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

    albedos = {"Gas Giant": 0.1, "Rocky": 0.3, "Icy": 0.7}
    Teq = {}
    R_star_AU = R_star * (R_SUN_M / AU_M)
    for planet_type, A in albedos.items():
        sqrt_term = np.sqrt(R_star_AU / (2 * a_AU))
        teq = T_star * sqrt_term * (1 - A)**0.25
        Teq[planet_type] = round(teq, 1)

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