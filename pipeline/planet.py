import numpy as np
import astropy.constants as c
import astropy.units as u

def get_planet_data(period, depth, T_star, R_star, M_star, k=None):
    """
    Расчёт радиуса, массы, плотности и равновесной температуры планеты.
    """
    if depth <= 0 or any(x <= 0 for x in [period, T_star, R_star, M_star]):
        return {"R_p_Rearth": None, "M_p_Mearth": None, "Density_gcm3": None, "Teq": {}, "Class": "Unknown"}
    
    period_sec = (period * u.day).to(u.s)
    a = ((c.G * (M_star * u.Msun) * period_sec**2) / (4 * np.pi**2))**(1/3)

    R_pl = np.sqrt(depth) * R_star * u.Rsun
    Mp = None
    if k is not None and k > 0:
        k_si = k * u.m / u.s
        Mp = ((M_star * u.Msun)**(2/3) * k_si * (period_sec / (2 * np.pi * c.G))**(1/3)).to(u.Mearth)

    density = None
    if Mp is not None:
        vol = (4/3) * np.pi * R_pl.to(u.m)**3
        density = (Mp / vol).to(u.g/u.cm**3)

    albedos = {"Gas giant (0.1)": 0.1, "Rocky (0.3)": 0.3, "Icy (0.7)": 0.7}
    Teqs = {}
    if a.value > 0:
        for kind, A in albedos.items():
            Teq = T_star * np.sqrt((R_star * u.Rsun)/(2 * a)) * (1 - A)**0.25
            Teqs[kind] = round(Teq.value, 1)
    
    rp = R_pl.to(u.Rearth).value
    planet_class = "Super-Earth" if rp < 1.8 else "Mini-Neptune" if rp < 4 else "Gas Giant"
    
    return {
        "R_p_Rearth": round(rp, 2),
        "M_p_Mearth": round(Mp.value, 2) if Mp else None,
        "Density_gcm3": round(density.value, 2) if density else None,
        "Teq": Teqs,
        "Class": planet_class
    }