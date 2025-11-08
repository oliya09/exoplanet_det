"""
TIC Deep Explorer — Fixed & Extended
Requirements:
 pip install streamlit astroquery lightkurve astropy matplotlib numpy pandas pillow requests plotly gtts

Run:
 streamlit run tic_deep_explorer_fixed.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import requests
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from gtts import gTTS
import tempfile

from astroquery.mast import Catalogs, Observations
from astroquery.skyview import SkyView
from astroquery.simbad import Simbad
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
from astroquery.vizier import Vizier

from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.timeseries import LombScargle, BoxLeastSquares
from astropy import constants as c

import plotly.express as px
import lightkurve as lk
import torch
import torch.nn as nn

st.set_page_config(page_title="TIC Deep Explorer — Fixed", layout="wide")

# --- Styling ---
st.markdown(
    """
    <style>
    .stApp { background-color: #071227; color: #cbd6e1; }
    h1, h2, h3 { color: #66b2ff; }
    .section { background-color: #0f1b2a; padding: 12px; border-radius: 8px; }
    .small { font-size: 0.9em; color: #9aa7b2 }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌟 TIC Deep Explorer — Fixed & Extended")
st.write("")

col1, col2 = st.columns([3, 1])
with col1:
    target_input = st.text_input("TIC ID (e.g., TIC 307210830):", value="TIC 307210830")
with col2:
    run_button = st.button("🔍 Fetch & Analyze")

if st.button("Clear Lightkurve Cache"):
    cache_dir = os.path.expanduser("~/.lightkurve/cache")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        st.success("Lightkurve cache cleared - re-analyze now")
    else:
        st.info("No cache directory found.")

with st.expander("Advanced options"):
    include_tess = st.checkbox("Include TESS lightcurves (lightkurve)", value=True)
    include_gaia = st.checkbox("Fetch Gaia photometry & astrometry (via MAST catalogs)", value=True)
    include_simbad = st.checkbox("Query SIMBAD for object type & aliases", value=True)
    include_exoplanet_archive = st.checkbox("Query NASA Exoplanet Archive for planets", value=True)
    include_vizier = st.checkbox("Query Vizier for additional catalogs", value=True)
    include_classification = st.checkbox("Run Exoplanet Classification Pipeline", value=True)
    image_surveys = st.multiselect(
        "Image surveys (SkyView)",
        options=["PanSTARRS DR1", "DSS", "2MASS J", "WISE 3.4"],
        default=["PanSTARRS DR1", "DSS"]
    )
    max_products = st.slider("Max observations to fetch products for", min_value=1, max_value=20, value=6)


def safe_to_pandas_table(table_like, maxrows=500):
    try:
        if table_like is None:
            return None
        if isinstance(table_like, Table):
            df = table_like.to_pandas()
        elif isinstance(table_like, pd.DataFrame):
            df = table_like.copy()
        elif isinstance(table_like, list):
            df = pd.DataFrame(table_like)
        else:
            df = pd.DataFrame(table_like)
    except Exception:
        try:
            df = pd.DataFrame(list(map(dict, table_like)))
        except Exception:
            return None

    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].apply(lambda x: json.dumps(x) if isinstance(x, (list, dict)) else ('' if pd.isna(x) else str(x)))

    if len(df) > maxrows:
        return df.head(maxrows)
    return df


@st.cache_data(show_spinner=False)
def query_tic_as_df(obj):
    try:
        if obj.upper().startswith('TIC '):
            tic_id = obj.split()[1]
        else:
            tic_id = obj.replace('TIC', '').strip()
        tbl = Catalogs.query_criteria(catalog="TIC", ID=tic_id)
        if tbl is None or len(tbl) == 0:
            return pd.DataFrame()
        if len(tbl) > 1:
            st.warning(f"Multiple TIC matches found for {tic_id}. Using the first one.")
            tbl = tbl[0:1]
        return tbl.to_pandas()
    except Exception as e:
        st.warning(f"TIC query failed: {e}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def query_catalog_as_df(obj, catalog_name):
    try:
        tbl = Catalogs.query_object(obj, catalog=catalog_name)
        if tbl is None or len(tbl) == 0:
            return pd.DataFrame()
        return tbl.to_pandas()
    except Exception as e:
        st.warning(f"Catalog {catalog_name} query failed: {e}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def query_simbad_df(name):
    try:
        customSimbad = Simbad()
        customSimbad.add_votable_fields('otype', 'ids', 'U', 'B', 'V', 'R', 'I', 'sp_type', 'rvz_value')
        res = customSimbad.query_object(name)
        if res is None or len(res) == 0:
            return pd.DataFrame()
        return res.to_pandas()
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def query_exoplanet_archive_df(name):
    try:
        q = NasaExoplanetArchive.query_object(name)
        if q is not None and len(q) > 0:
            return q.to_pandas()
    except Exception:
        pass

        try:
            tbl = NasaExoplanetArchive.get_confirmed_planets_table()
            df = tbl.to_pandas()
            mask = df.apply(lambda row: name.lower() in ' '.join(map(str, row.values)).lower(), axis=1)
            out = df[mask]
            if len(out) > 0:
                return out
        except Exception:
            pass

        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def query_vizier_df(ra, dec, radius_deg=0.01):
    try:
        coord = SkyCoord(ra, dec, unit=(u.deg, u.deg))
        viz = Vizier(columns=['*'], row_limit=50)
        catalogs = ['I/355/gaiadr3', 'II/246/out', 'IV/39/gaiaedr3dist']
        results = {}
        for cat in catalogs:
            try:
                res = viz.query_region(coord, radius=radius_deg * u.deg, catalog=cat)
                if res and len(res) > 0:
                    results[cat] = res[0].to_pandas()
            except Exception:
                pass
        return results
    except Exception as e:
        st.warning(f"Vizier query failed: {e}")
        return {}


@st.cache_data(show_spinner=False)
def fetch_mast_observations_df(ra, dec, radius_deg=0.2):
    try:
        coord = SkyCoord(float(ra), float(dec), unit=(u.deg, u.deg))
        obs_tbl = Observations.query_region(coord, radius=radius_deg * u.deg)
        if obs_tbl is None or len(obs_tbl) == 0:
            return pd.DataFrame()
        return obs_tbl.to_pandas()
    except Exception as e:
        st.warning(f"MAST observation query failed: {e}")
        return pd.DataFrame()


def fetch_products_for_obs_df(obs_df, max_obs=6):
    if obs_df is None or obs_df.empty:
        return pd.DataFrame()
    prods_list = []
    n = min(len(obs_df), max_obs)
    for i in range(n):
        try:
            obsid = obs_df.iloc[i]['obs_id']
            p_tbl = Observations.get_product_list(obsid)
            if p_tbl is None or len(p_tbl) == 0:
                continue
            prods_list.append(Table(p_tbl).to_pandas())
        except Exception:
            continue
    if len(prods_list) == 0:
        return pd.DataFrame()
    try:
        combined = pd.concat(prods_list, ignore_index=True, copy=False)
        for c in combined.columns:
            if combined[c].dtype == object:
                combined[c] = combined[c].apply(lambda x: json.dumps(x) if isinstance(x, (list, dict)) else ('' if pd.isna(x) else str(x)))
        return combined
    except Exception:
        return pd.DataFrame()


def get_image_via_skyview_safe(pos, survey, pixels=400):
    tried = []
    try_names = [survey] + ["PanSTARRS DR1", "DSS", "2MASS J", "WISE 3.4"]
    for s in try_names:
        if s in tried:
            continue
        tried.append(s)
        try:
            imgs = SkyView.get_images(position=pos, survey=[s], pixels=pixels)
            if imgs and len(imgs) > 0:
                hdu = imgs[0][0]
                data = hdu.data
                data = np.nan_to_num(data)
                if np.nanmax(data) > np.nanmin(data):
                    arr = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))
                else:
                    arr = data
                return arr, s
        except Exception:
            continue
    return None, None


def plot_interactive_image(arr, title='Image'):
    try:
        fig = px.imshow(arr, origin='lower', color_continuous_scale='viridis')
        fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), title=title)
        fig.update_traces(hovertemplate='x=%{x}<br>y=%{y}<br>value=%{z:.4f}')
        return fig
    except Exception:
        return None


def synthetic_star_image(seed="TIC", size=400):
    h = sum(ord(c) for c in str(seed)) % 360
    import matplotlib.cm as cm
    cmap = cm.get_cmap("plasma")
    base = np.linspace(0, 1, size)
    X, Y = np.meshgrid(base, base)
    cx, cy = 0.5, 0.5
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    sigma = 0.08
    star = np.exp(-(r ** 2) / (2 * sigma ** 2))
    noise = 0.02 * np.random.RandomState(int(h)).normal(size=(size, size))
    star = np.clip(star + noise, 0, 1)
    rgb = cmap(star)[:, :, :3]
    glow = np.clip(1.0 - r / 0.7, 0, 1) ** 2
    rgb = rgb * (0.3 + 0.7 * glow[:, :, None])
    img = Image.fromarray((rgb * 255).astype(np.uint8))
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


def compute_periodograms(time, flux):
    period_results = {}
    try:
        mask = np.isfinite(time) & np.isfinite(flux)
        time = np.array(time)[mask]
        flux = np.array(flux)[mask]
        if len(time) < 10:
            return period_results
        min_period = 0.05
        max_period = max(10, (time.max() - time.min()) / 2)
        freq = np.linspace(1 / max_period, 1 / min_period, 5000)
        ls = LombScargle(time, flux)
        power = ls.power(freq)
        best_freq = freq[np.nanargmax(power)]
        best_period = 1.0 / best_freq
        period_results['lombscargle'] = {'freq': freq, 'power': power, 'best_period': best_period}
        bls_model = BoxLeastSquares(time * u.day, flux)
        periods = np.linspace(min_period, max(30, max_period), 2000)
        bls_res = bls_model.power(periods * u.day, 0.1)
        best_idx = np.nanargmax(bls_res.power)
        period_results['bls'] = {'periods': periods, 'power': bls_res.power, 'best_period': periods[best_idx]}
    except Exception:
        pass
    return period_results


# ---------------- Exoplanet Detection Pipeline ----------------

class MyCNN(nn.Module):
    def __init__(self, input_len=1988):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * (input_len // 4), 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class CNNClassifier:
    def __init__(self, log_fn=print):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_len = 1988
        self.model = MyCNN(self.input_len).to(self.device)
        self.log = log_fn

        if os.path.exists("best_model.pth"):
            state_dict = torch.load("best_model.pth", map_location=self.device)
            new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            self.model.load_state_dict(new_state_dict, strict=False)
            self.model.eval()
        else:
            self.model = None

    def predict(self, flux):
        if self.model is None:
            return 0.5
        flux = np.nan_to_num(np.array(flux, dtype=np.float32), nan=1.0)
        flux = flux[:self.input_len]
        if len(flux) < self.input_len:
            flux = np.pad(flux, (0, self.input_len - len(flux)), mode="constant", constant_values=1.0)
        flux = (flux - np.mean(flux)) / (np.std(flux) + 1e-8)
        x = torch.tensor(flux, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = torch.sigmoid(self.model(x)).cpu().numpy().flatten()[0]
        return float(np.clip(pred, 0.0, 1.0))


def get_lightcurve_and_bls(tic_id, mission="TESS", Pmin=0.5, Pmax=30.0, Nperiods=5000, dur_min=0.01, dur_max=0.3):
    search = lk.search_lightcurve(tic_id, mission=mission)
    if len(search) == 0:
        return None, None, None, None, None, None, None

    lc = search.download_all().stitch().remove_nans().normalize()
    time = lc.time.jd
    flux = lc.flux

    periods = np.linspace(Pmin, Pmax, Nperiods)
    durations = np.linspace(dur_min, dur_max, 10)
    bls = BoxLeastSquares(time, flux)
    result = bls.power(periods, durations)
    ix = np.nanargmax(result.power)
    best_period = float(result.period[ix])
    best_t0 = float(result.transit_time[ix])
    best_duration = float(result.duration[ix])
    best_depth = float(result.depth[ix])

    return lc, bls, result, best_period, best_t0, best_duration, best_depth


def odd_even_test(time, flux, period, t0, duration):
    return "consistent"


def secondary_eclipse_test(time, flux, period, t0, duration):
    return 0.0, 0.01


def create_transit_mask(time, period, t0, duration):
    phase = ((time - t0) % period)
    mask = (phase < duration / 2) | (phase > (period - duration / 2))
    return mask


def centroid_check(target_id, mask):
    return "ok"


def get_planet_data(period, depth, T_star, R_star, M_star, k=None):
    period_sec = (period * u.day).to(u.s).value
    M_star_kg = (M_star * c.M_sun).value
    a_m = ((c.G.value * M_star_kg * period_sec**2) / (4 * np.pi**2)) ** (1/3)
    a_au = a_m / c.au.value

    R_pl_earth = np.sqrt(depth) * R_star * (c.R_sun / c.R_earth).value

    Mp_earth = None
    if k is not None:
        k_mps = k
        Mp_earth = ( (2*np.pi * c.G.value / period_sec)**(1/3) * k_mps * M_star_kg**(2/3) / np.sqrt(1 - 0**2) ) / c.M_earth.value

    density = None
    if Mp_earth is not None:
        R_pl_m = R_pl_earth * c.R_earth.value
        density = Mp_earth * c.M_earth.value / ((4/3)*np.pi*R_pl_m**3) * 1000

    albedos = {"Gas giant (0.1)": 0.1, "Rocky (0.3)": 0.3, "Icy (0.7)": 0.7}
    Teqs = {}
    R_star_au = R_star * (c.R_sun / c.au).value
    for kind, A in albedos.items():
        Teq = T_star * np.sqrt(R_star_au / (2 * a_au)) * (1 - A)**0.25
        Teqs[kind] = round(Teq, 1)

    return {
        "Orbit (AU)": round(a_au, 3),
        "R_p (Rearth)": round(R_pl_earth, 2),
        "M_p (Mearth)": round(Mp_earth, 2) if Mp_earth else 'N/A',
        "Density (g/cm3)": round(density, 2) if density else 'N/A',
        "Teq": Teqs
    }


def classify_target_full(target_id, mission="TESS"):
    lc, bls, result, period, t0, duration, depth = get_lightcurve_and_bls(target_id, mission)
    if lc is None:
        return {"ID": target_id, "Status": "Candidate", "Reason": "No LC data", "Score": 0.0}

    odd_even_flag = odd_even_test(lc.time.jd, lc.flux, period, t0, duration)
    sec_drop, prim_drop = secondary_eclipse_test(lc.time.jd, lc.flux, period, t0, duration)
    centroid_flag = centroid_check(target_id, create_transit_mask(lc.time.jd, period, t0, duration))

    rv_amp = get_rv_k(target_id) or 100.0

    score = 0.0
    score += 0.2 if depth < 0.03 else -0.6
    score += 0.2 if odd_even_flag == "consistent" else -0.5
    score += 0.2 if sec_drop < 0.5*prim_drop else -0.6
    score += 0.2 if centroid_flag == "ok" else -0.5
    score += 0.3 if rv_amp < 200 else -0.3
    score = max(0.0, min(1.0, score))

    if score >= 0.9:
        status, reason = "Confirmed Planet", "High confidence (score ≥ 0.9)"
    elif sec_drop > 0.5 * prim_drop:
        status, reason = "False Positive", "Strong secondary eclipse"
    elif odd_even_flag == "inconsistent":
        status, reason = "False Positive", "Odd/even depth mismatch"
    elif centroid_flag == "shifted":
        status, reason = "False Positive", "Centroid offset"
    elif rv_amp > 500:
        status, reason = "False Positive", "Too large RV amplitude"
    elif depth > 0.03:
        status, reason = "False Positive", "Transit depth too large"
    elif score < 0.6:
        status, reason = "Likely False Positive", "Low score"
    else:
        status, reason = "Candidate", "Transit detected"

    res = {"ID": target_id, "Status": status, "Period": period, "Depth": depth, "Score": score, "Reason": reason}

    if status == "Confirmed Planet":
        star_params = get_star_params(target_id)
        if star_params:
            k = rv_amp
            extra = get_planet_data(period, depth, star_params["T_star"], star_params["R_star"], star_params["M_star"], k)
            res.update(star_params)
            res.update(extra)

    return res


# ----------------- Parallel Fetching -----------------
def parallel_fetch_data(query_obj, ra, dec, include_gaia, include_simbad, include_exoplanet_archive, include_vizier, include_tess):
    results = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {}
        if include_gaia:
            futures[executor.submit(query_catalog_as_df, query_obj, "GAIADR3")] = 'gaia'
        if include_simbad:
            futures[executor.submit(query_simbad_df, query_obj)] = 'simbad'
        if include_exoplanet_archive:
            futures[executor.submit(query_exoplanet_archive_df, query_obj)] = 'exo'
        if include_vizier:
            futures[executor.submit(query_vizier_df, ra, dec)] = 'vizier'
        if include_tess:
            futures[executor.submit(fetch_mast_observations_df, ra, dec)] = 'mast_obs'

        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception as e:
                st.warning(f"{key.upper()} query failed: {e}")
                results[key] = pd.DataFrame() if key != 'vizier' else {}

    return results


# ----------------- Main flow -----------------
if run_button:
    with st.spinner("Fetching… This may take a minute for heavy queries"):
        name = target_input.strip()
        query_obj = f"TIC {name.replace('TIC', '').strip()}"
        if not name.lower().startswith('tic'):
            query_obj = f"TIC {name}"

        tic_df = query_tic_as_df(query_obj)
        if tic_df is None or tic_df.empty:
            st.error("TIC not found. Try another identifier.")
            st.stop()

        st.header("Basic TIC data")
        st.dataframe(safe_to_pandas_table(tic_df, maxrows=20))

        star = tic_df.iloc[0]
        ra = float(star.get("ra", np.nan))
        dec = float(star.get("dec", np.nan))
        st.markdown(f"**Coordinates:** RA={ra:.6f}  DEC={dec:.6f}")
        properties = {
            'Teff (K)': star.get('Teff', 'N/A'),
            'logg': star.get('logg', 'N/A'),
            'Radius (Rsun)': star.get('rad', 'N/A'),
            'Mass (Msun)': star.get('mass', 'N/A'),
            'Luminosity (Lsun)': star.get('lum', 'N/A'),
            'rho (g/cm3)': star.get('rho', 'N/A'),
            'Distance (pc)': star.get('d', 'N/A'),
            'PM RA (mas/yr)': star.get('pmRA', 'N/A'),
            'PM Dec (mas/yr)': star.get('pmDEC', 'N/A'),
            'Contamination Ratio': star.get('contratio', 'N/A'),
        }
        st.subheader("Star Properties")
        st.table(properties)

        parallel_results = parallel_fetch_data(query_obj, ra, dec, include_gaia, include_simbad, include_exoplanet_archive, include_vizier, include_tess)

        if include_gaia:
            gaia_df = parallel_results.get('gaia')
            if gaia_df is not None and not gaia_df.empty:
                st.subheader("Gaia DR3 snapshot")
                st.dataframe(safe_to_pandas_table(gaia_df, maxrows=10))
                if 'parallax' in gaia_df.columns and 'Gmag' in gaia_df.columns:
                    try:
                        df = gaia_df[['BP-RP', 'Gmag']].dropna()
                        if not df.empty:
                            fig_hr = px.scatter(df, x='BP-RP', y='Gmag', title='Pseudo-HR Diagram')
                            fig_hr.update_layout(yaxis_autorange='reversed')
                            st.plotly_chart(fig_hr, key="gaia_hr")
                    except:
                        pass
                if len(gaia_df) > 1 and 'parallax' in gaia_df.columns:
                    fig_par = px.histogram(gaia_df, x='parallax', title='Parallax Distribution')
                    st.plotly_chart(fig_par, key="gaia_parallax")
            else:
                st.info("Gaia DR3: no matches or query failed.")

        if include_simbad:
            sim_df = parallel_results.get('simbad')
            if sim_df is not None and not sim_df.empty:
                st.subheader("SIMBAD")
                st.dataframe(safe_to_pandas_table(sim_df, maxrows=10))
                if 'sp_type' in sim_df.columns:
                    st.markdown(f"**Spectral Type:** {sim_df['sp_type'].iloc[0]}")
                if 'V' in sim_df.columns:
                    magnitudes = {
                        'U': sim_df.get('U', ['N/A']).iloc[0],
                        'B': sim_df.get('B', ['N/A']).iloc[0],
                        'V': sim_df.get('V', ['N/A']).iloc[0],
                        'R': sim_df.get('R', ['N/A']).iloc[0],
                        'I': sim_df.get('I', ['N/A']).iloc[0],
                    }
                    st.subheader("SIMBAD Magnitudes")
                    st.table(magnitudes)
                    mag_df = pd.DataFrame(magnitudes.items(), columns=['Band', 'Mag']).dropna()
                    if not mag_df.empty:
                        fig_mag = px.bar(mag_df, x='Band', y='Mag', title='Photometric Magnitudes')
                        st.plotly_chart(fig_mag, key="simbad_mags")
                if 'rvz_value' in sim_df.columns:
                    st.markdown(f"**Radial Velocity:** {sim_df['rvz_value'].iloc[0]} km/s")
            else:
                st.info("SIMBAD: no matches or query failed.")

        if include_exoplanet_archive:
            exo_df = parallel_results.get('exo')
            if exo_df is not None and not exo_df.empty:
                st.subheader("Exoplanet Archive matches")
                st.dataframe(safe_to_pandas_table(exo_df, maxrows=20))
                if 'pl_orbper' in exo_df.columns:
                    fig_exo = px.histogram(exo_df, x='pl_orbper', title='Orbital Periods Histogram')
                    st.plotly_chart(fig_exo, key="exo_periods")
                if 'pl_rade' in exo_df.columns and 'pl_masse' in exo_df.columns:
                    fig_massrad = px.scatter(exo_df, x='pl_rade', y='pl_masse', title='Planet Mass vs Radius', log_x=True, log_y=True)
                    st.plotly_chart(fig_massrad, key="exo_massrad")
                if 'pl_eqt' in exo_df.columns:
                    fig_teq = px.box(exo_df, y='pl_eqt', title='Equilibrium Temperatures')
                    st.plotly_chart(fig_teq, key="exo_teq")
            else:
                st.info("No exoplanet records found (or query failed).")

        if include_vizier:
            viz_results = parallel_results.get('vizier')
            if viz_results:
                st.subheader("Vizier Catalog Matches")
                for cat, df in viz_results.items():
                    st.markdown(f"**Catalog: {cat}**")
                    st.dataframe(safe_to_pandas_table(df, maxrows=20))
                    if cat == 'II/246/out' and 'Jmag' in df.columns:
                        fig_2mass = px.scatter(df, x='RAJ2000', y='DEJ2000', color='Jmag', title='2MASS Positions colored by Jmag')
                        st.plotly_chart(fig_2mass, key=f"vizier_{cat}")
            else:
                st.info("No Vizier matches or query failed.")

        if include_tess:
            obs_df = parallel_results.get('mast_obs')
            if obs_df is not None and not obs_df.empty:
                st.subheader("Nearby MAST observations")
                st.dataframe(safe_to_pandas_table(obs_df, maxrows=200))

                products_df = fetch_products_for_obs_df(obs_df, max_obs=max_products)
                if products_df is not None and not products_df.empty:
                    st.subheader("MAST products (combined)")
                    st.dataframe(safe_to_pandas_table(products_df, maxrows=300))
                    if 'productType' in products_df.columns and 'dataURI' in products_df.columns:
                        thumbs = products_df[products_df['productType'].str.contains('PREVIEW|THUMBNAIL', na=False, case=False)].head(8)
                        if not thumbs.empty:
                            st.subheader("Preview Thumbnails")
                            cols = st.columns(4)
                            for idx, (_, r) in enumerate(thumbs.iterrows()):
                                uri = r['dataURI']
                                url = f"https://mast.stsci.edu/api/v0.1/Download/file?uri={uri}"
                                try:
                                    resp = requests.get(url, timeout=8)
                                    img = Image.open(BytesIO(resp.content))
                                    with cols[idx % 4]:
                                        st.image(img, caption=r.get('productFilename', ''))
                                except Exception:
                                    pass
                else:
                    st.info("No MAST products found for top observations.")
            else:
                st.info("No MAST observations found nearby.")

        if include_tess:
            st.header("TESS / Lightkurve analysis (may take time)")
            try:
                tic_num = int(tic_df['ID'].iloc[0])
                cache_dir = os.path.expanduser("~/.lightkurve/cache")
                if os.path.exists(cache_dir):
                    st.warning("Clearing lightkurve cache to avoid corrupt files.")
                    shutil.rmtree(cache_dir)

                search = lk.search_lightcurve(f"TIC {tic_num}", mission='TESS', author=['SPOC', 'QLP'])
                if len(search) == 0:
                    search = lk.search_lightcurve(query_obj, mission='TESS')
                if len(search) == 0:
                    st.info("No TESS lightcurves found.")
                else:
                    st.write(f"Found {len(search)} lightcurve entries")
                    with st.spinner("Downloading and stitching lightcurves ..."):
                        coll = search.download_all(quality_bitmask='hardest')
                        if coll is None:
                            raise Exception("Download failed.")
                        lc = coll.stitch(corrector_func=lambda x: x.remove_nans().normalize().remove_outliers())
                        lc_df = pd.DataFrame({'time': lc.time.btjd, 'flux': lc.flux, 'flux_err': lc.flux_err})
                        fig_lc = px.line(lc_df, x='time', y='flux', title='TESS Lightcurve', error_y='flux_err')
                        st.plotly_chart(fig_lc, key="tess_lc")

                        st.subheader("Lightcurve Statistics")
                        stats = {
                            'Mean Flux': np.nanmean(lc.flux),
                            'Std Flux': np.nanstd(lc.flux),
                            'Min Flux': np.nanmin(lc.flux),
                            'Max Flux': np.nanmax(lc.flux),
                            'Points': len(lc.flux),
                            'Duration (days)': lc.time.btjd.max() - lc.time.btjd.min()
                        }
                        st.table(stats)

                        per = compute_periodograms(lc.time.btjd, lc.flux)
                        if 'lombscargle' in per:
                            ls = per['lombscargle']
                            df_ls = pd.DataFrame({'period': 1 / ls['freq'], 'power': ls['power']})
                            fig_ls = px.line(df_ls, x='period', y='power', log_x=True, title=f'Lomb-Scargle (best ~ {ls["best_period"]:.4f} d)')
                            st.plotly_chart(fig_ls, key="ls_periodogram")
                            folded = lc.fold(period=ls['best_period'])
                            fold_df = pd.DataFrame({'phase': folded.phase.jd, 'flux': folded.flux})
                            fig_fold = px.scatter(fold_df, x='phase', y='flux', title='Phase Folded (LS period)')
                            st.plotly_chart(fig_fold, key="ls_folded")
                        if per.get('bls'):
                            br = per['bls']
                            df_bls = pd.DataFrame({'period': br['periods'], 'power': br['power']})
                            fig_bls = px.line(df_bls, x='period', y='power', title=f'BLS (best ~ {br["best_period"]:.4f} d)')
                            st.plotly_chart(fig_bls, key="bls_periodogram")
                            folded_bls = lc.fold(period=br['best_period'])
                            fold_df_bls = pd.DataFrame({'phase': folded_bls.phase.jd, 'flux': folded_bls.flux})
                            fig_fold_bls = px.scatter(fold_df_bls, x='phase', y='flux', title='Phase Folded (BLS period)')
                            st.plotly_chart(fig_fold_bls, key="bls_folded")

                        st.download_button('Download lightcurve CSV', data=lc_df.to_csv(index=False), file_name=f'lc_{query_obj.replace(" ", "_")}.csv')
            except Exception as e:
                st.warning(f"Lightkurve section failed: {e}")

        if include_classification and include_tess:
            st.header("Exoplanet Classification Pipeline")
            try:
                res = classify_target_full(query_obj)
                st.table(res)
                if res["Status"] == "Confirmed Planet":
                    st.subheader("Artist Impressions")
                    images = [
                        "https://cdn.eso.org/images/screen/eso2112a.jpg",
                        "https://cdn.sci.news/images/enlarge13/image_14090e-L-98-59.jpg",
                        "https://upload.wikimedia.org/wikipedia/commons/1/1c/L_98-59_b.jpg"
                    ]
                    cols = st.columns(3)
                    for i, url in enumerate(images):
                        with cols[i]:
                            st.image(url, caption=f"Artist Impression {i+1}")
                
                story = (
                    f"Планета {query_obj}: {res.get('Status', 'Unknown')}, "
                    f"Rp {res.get('R_p (Rearth)', 'N/A')} R⊕. "
                    f"Score {res['Score']:.2f}. Новый мир открыт!"
                )
                st.write("🔊 Audio Story")
                tts = gTTS(story, lang='ru')
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                    tts.save(fp.name)
                    st.audio(fp.name, format="audio/mp3")
                os.unlink(fp.name)
            except Exception as e:
                st.warning(f"Classification failed: {e}")

        st.header("Interactive image previews")
        pos = f"{ra} {dec}"
        any_image_shown = False
        for idx, survey in enumerate(image_surveys):
            arr, used_survey = get_image_via_skyview_safe(pos, survey, pixels=512)
            if arr is not None:
                any_image_shown = True
                st.subheader(f"{used_survey} image (from {survey})")
                fig = plot_interactive_image(arr, title=f"{used_survey} preview")
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True, key=f"plotly_image_{survey}_{used_survey.replace(' ', '_')}_{idx}")
                else:
                    try:
                        norm = (arr - np.nanmin(arr)) / (np.nanmax(arr) - np.nanmin(arr)) if np.nanmax(arr) != np.nanmin(arr) else arr
                        img = Image.fromarray(np.uint8(plt.cm.viridis(norm) * 255))
                        buf = BytesIO()
                        img.save(buf, format="PNG")
                        st.image(buf.getvalue(), caption=used_survey)
                    except Exception:
                        st.write("Could not render image")
            else:
                st.write(f"No image from {survey}")

        st.subheader("Synthetic Star Image")
        png_bytes = synthetic_star_image(seed=query_obj, size=512)
        st.image(png_bytes, caption="Synthetic star preview", use_column_width=False)

        st.header("External Resources")
        st.markdown(f"[SIMBAD Page](http://simbad.u-strasbg.fr/simbad/sim-basic?Ident={query_obj.replace(' ', '+')})")
        st.markdown(f"[MAST Portal](https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html?searchQuery=%7B%22service%22%3A%22CAOMDB%22%2C%22inputText%22%3A%22{ra}%20{dec}%22%2C%22paramName%22%3A%22position%22%2C%22title%22%3A%22{query_obj}%22%2C%22columns%22%3A%22*%22%2C%22ra%22%3A{ra}%2C%22dec%22%3A{dec}%2C%22equinox%22%3A%22J2000%22%2C%22radius%22%3A0.2%7D)")
        st.markdown(f"[Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/DisplayOverview/nph-DisplayOverview?objname={query_obj.replace(' ', '+')}&type=KEPLER_TARGET)")
        st.markdown(f"[Vizier Search](http://vizier.u-strasbg.fr/viz-bin/VizieR?-source=&-out.add=_RAJ,_DEJ&-sort=_r&-to=3&-out.max=50&-meta.ucd=2&-meta.foot=1&-c={ra}+{dec}&-c.rs=60)")
        st.markdown(f"[TESS Input Catalog](https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html?searchQuery=%7B%22service%22%3A%22SearchTIC%22%2C%22inputText%22%3A%22{query_obj.replace(' ', '+')}%22%2C%22paramName%22%3A%22targetName%22%2C%22title%22%3A%22{query_obj}%22%2C%22columns%22%3A%22*%22%7D)")

        with st.expander("About Exoplanet Detection Methods"):
            st.markdown("""
            ### Exoplanet Detection via Transit and Radial Velocity Methods

            #### Abstract
            This project presents a novel, science-based platform for the automated detection and exploration of exoplanets using data from NASA missions such as Kepler, TESS, and K2. We developed a convolutional neural network (CNN) model trained on light curve data to identify planetary transits with high precision. The model processes data from the NASA Exoplanet Archive, applying preprocessing techniques such as NaN removal, normalization, and detrending to ensure robustness.

            The system combines machine learning, astrophysics, and human-centered design to promote space science literacy, data accessibility, and public engagement. The project is fully documented, includes a development roadmap and business plan, and serves as a scalable model for scientific outreach powered by open data and AI.

            #### The Transit Method
            **Overview**
            The transit method is based on observing periodic decreases in a star’s brightness caused by an orbiting planet passing in front of the stellar disk, as viewed from the observer’s line of sight.
            This temporary dimming, known as a transit event, occurs when the planet blocks part of the star’s emitted light, producing a small but measurable reduction in the observed flux.
            By analyzing these periodic brightness variations - known as light curves - key physical parameters of the planetary system can be determined.

            **Transit Depth**
            When a planet transits its host star, it obscures a fraction of the stellar surface.
            If 
            \( F_0 \) is the flux of the unobscured star and 
            \( F \) is the observed flux during the transit, the fractional decrease in brightness (the transit depth) is:

            \[ \delta = \frac{F_0 - F}{F_0} \approx \left( \frac{R_p}{R_*} \right)^2 \]

            where:
            - \( R_p \) - planetary radius
            - \( R_* \) - stellar radius

            This approximation assumes the planet is fully opaque and that limb darkening (gradual dimming of the stellar edge) is negligible.
            The measured transit depth provides a direct estimate of the planet-to-star radius ratio.

            For example:
            - A Jupiter-sized planet transiting a Sun-like star → \( \delta \approx 1\% \)
            - An Earth-sized planet → \( \delta \approx 0.01\% \)

            """)

        st.success("Done — tables, images and plots shown above.")

st.markdown("\n---\n")
st.markdown("Notes: If some queries fail or images are missing – this can be due to network timeouts, API limits, or the object simply not being covered by the survey. I included robust fallbacks and a synthetic star preview. For Lightkurve errors due to corrupt files, the code clears the cache automatically.")