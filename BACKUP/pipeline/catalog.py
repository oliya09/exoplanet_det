from astroquery.mast import Catalogs
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
from astroquery.gaia import Gaia
import astropy.units as u

def get_star_params(tic_id):
    """Gets temperature, radius and mass of star from its TIC ID"""
    try:
        tic_str = tic_id.replace("TIC", "").strip()
        data = Catalogs.query_object(f"TIC {tic_str}", catalog="TIC")
        if len(data) == 0:
            return None
        row = data[0]
        params = {"T_star": row["Teff"], "R_star": row["rad"], "M_star": row["mass"]}
        # P1: Cross с Gaia
        gaia_data = Gaia.query_object_async(f"TIC {tic_str}", radius=1*u.arcsec)
        params["cross_conf"] = 0.95 if len(gaia_data) > 0 else 0.5
        return params
    except Exception as e:
        print(f"[ERROR] Star params fetch failed for {tic_id}: {e}")
        return None

def get_rv_k(host_name):
    """Gets RV"""
    try:
        rec = NasaExoplanetArchive.query_criteria(
            table="pscomppars",
            select="pl_rvamp, hostname",
            where=f"hostname like '%{host_name}%'"
        )
        if not rec or len(rec) == 0:
            return None
        return float(rec[0]["pl_rvamp"])
    except Exception as e:
        print(f"[ERROR] RV query failed for {host_name}: {e}")
        return None

def get_hostname_from_tic(tic_id):
    """Mapping TIC -> hostname (stub)"""
    return f"TOI_{tic_id.split()[-1]}"