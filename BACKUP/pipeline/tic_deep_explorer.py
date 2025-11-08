# File: tic_deep_explorer_fixed.py

import streamlit as st
import pandas as pd
import numpy as np

from utils import compute_periodograms, classify_target_full


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
import logging
import pickle  # Added for local caching

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

from classifier import CNNClassifier  # Import here to fix undefined error

# Logging setup
logging.basicConfig(filename='tic_explorer.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

st.set_page_config(page_title="TIC Deep Explorer — Fixed", layout="wide")

# --- Styling ---
st.markdown(
    """
    <style>
    .stApp { background-color: #071227; color: #cbd6e1; }
    h1, h2, h3 { color: #66b2ff; }
    .section { background-color: #0f1b2a; padding: 12px; border-radius: 8px; }
    .small { font-size: 0.9em; color: #9aa7b2 }
    .stTabs [data-testid="stTab"] { 
        background-color: #0f1b2a; 
        color: #cbd6e1; 
        border-radius: 8px 8px 0 0; 
        padding: 10px 20px; 
        font-weight: bold; 
    }
    .stTabs [aria-selected="true"] { 
        background-color: #66b2ff; 
        color: #071227; 
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌟 TIC Deep Explorer")
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
        logging.info("Lightkurve cache cleared by user")
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

# Local cache directory
cache_dir = "tic_cache"
os.makedirs(cache_dir, exist_ok=True)

def get_cache_file(query_obj, key):
    safe_name = query_obj.replace(" ", "_").replace("/", "_")
    return os.path.join(cache_dir, f"{safe_name}_{key}.pkl")

def load_from_cache(query_obj, key):
    file = get_cache_file(query_obj, key)
    if os.path.exists(file):
        with open(file, "rb") as f:
            return pickle.load(f)
    return None

def save_to_cache(query_obj, key, data):
    file = get_cache_file(query_obj, key)
    with open(file, "wb") as f:
        pickle.dump(data, f)

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
    except Exception as e:
        logging.error(f"Error converting to pandas: {e}")
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
    logging.info(f"Querying TIC for {obj}")
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
        logging.error(f"TIC query failed for {obj}: {e}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def query_catalog_as_df(obj, catalog_name):
    logging.info(f"Querying catalog {catalog_name} for {obj}")
    try:
        tbl = Catalogs.query_object(obj, catalog=catalog_name)
        if tbl is None or len(tbl) == 0:
            return pd.DataFrame()
        return tbl.to_pandas()
    except Exception as e:
        st.warning(f"Catalog {catalog_name} query failed: {e}")
        logging.error(f"Catalog {catalog_name} query failed for {obj}: {e}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def query_simbad_df(name):
    logging.info(f"Querying SIMBAD for {name}")
    try:
        customSimbad = Simbad()
        customSimbad.add_votable_fields('otype', 'ids', 'U', 'B', 'V', 'R', 'I', 'sp_type', 'rvz_radvel')
        res = customSimbad.query_object(name)
        if res is None or len(res) == 0:
            return pd.DataFrame()
        return res.to_pandas()
    except Exception as e:
        logging.error(f"SIMBAD query failed for {name}: {e}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def query_exoplanet_archive_df(name):
    logging.info(f"Querying Exoplanet Archive for {name}")
    try:
        q = NasaExoplanetArchive.query_object(name)
        if q is not None and len(q) > 0:
            return q.to_pandas()
    except Exception as e:
        logging.error(f"Exoplanet Archive object query failed for {name}: {e}")
        pass

        try:
            tbl = NasaExoplanetArchive.get_confirmed_planets_table()
            df = tbl.to_pandas()
            mask = df.apply(lambda row: name.lower() in ' '.join(map(str, row.values)).lower(), axis=1)
            out = df[mask]
            if len(out) > 0:
                return out
        except Exception as e2:
            logging.error(f"Exoplanet Archive confirmed planets query failed for {name}: {e2}")
            pass

        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def query_vizier_df(ra, dec, radius_deg=0.01):
    logging.info(f"Querying Vizier at RA={ra}, DEC={dec}")
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
            except Exception as e:
                logging.error(f"Vizier catalog {cat} query failed: {e}")
        return results
    except Exception as e:
        st.warning(f"Vizier query failed: {e}")
        logging.error(f"Vizier query failed: {e}")
        return {}


@st.cache_data(show_spinner=False)
def fetch_mast_observations_df(ra, dec, radius_deg=0.2):
    logging.info(f"Fetching MAST observations at RA={ra}, DEC={dec}")
    try:
        coord = SkyCoord(float(ra), float(dec), unit=(u.deg, u.deg))
        obs_tbl = Observations.query_region(coord, radius=radius_deg * u.deg)
        if obs_tbl is None or len(obs_tbl) == 0:
            return pd.DataFrame()
        return obs_tbl.to_pandas()
    except Exception as e:
        st.warning(f"MAST observation query failed: {e}")
        logging.error(f"MAST observation query failed: {e}")
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
        except Exception as e:
            logging.error(f"Fetching MAST products failed for obs_id {obsid}: {e}")
            continue
    if len(prods_list) == 0:
        return pd.DataFrame()
    try:
        combined = pd.concat(prods_list, ignore_index=True, copy=False)
        for c in combined.columns:
            if combined[c].dtype == object:
                combined[c] = combined[c].apply(lambda x: json.dumps(x) if isinstance(x, (list, dict)) else ('' if pd.isna(x) else str(x)))
        return combined
    except Exception as e:
        logging.error(f"Combining MAST products failed: {e}")
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
                logging.info(f"Fetched image from {s}")
                return arr, s
        except Exception as e:
            logging.error(f"SkyView failed for {s}: {e}")
            continue
    return None, None


def plot_interactive_image(arr, title='Image'):
    try:
        fig = px.imshow(arr, origin='lower', color_continuous_scale='viridis')
        fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), title=title)
        fig.update_traces(hovertemplate='x=%{x}<br>y=%{y}<br>value=%{z:.4f}')
        return fig
    except Exception as e:
        logging.error(f"Plot interactive image failed: {e}")
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


# ----------------- Parallel Fetching -----------------
def parallel_fetch_data(query_obj, ra, dec, include_gaia, include_simbad, include_exoplanet_archive, include_vizier, include_tess):
    results = {}
    with ThreadPoolExecutor(max_workers=8) as executor:  # Increased to 8 for speed
        futures = {}
        if include_gaia:
            cached = load_from_cache(query_obj, 'gaia')
            if cached is not None:
                results['gaia'] = cached
            else:
                futures[executor.submit(query_catalog_as_df, query_obj, "GAIADR3")] = 'gaia'
        if include_simbad:
            cached = load_from_cache(query_obj, 'simbad')
            if cached is not None:
                results['simbad'] = cached
            else:
                futures[executor.submit(query_simbad_df, query_obj)] = 'simbad'
        if include_exoplanet_archive:
            cached = load_from_cache(query_obj, 'exo')
            if cached is not None:
                results['exo'] = cached
            else:
                futures[executor.submit(query_exoplanet_archive_df, query_obj)] = 'exo'
        if include_vizier:
            cached = load_from_cache(query_obj, 'vizier')
            if cached is not None:
                results['vizier'] = cached
            else:
                futures[executor.submit(query_vizier_df, ra, dec)] = 'vizier'
        if include_tess:
            cached = load_from_cache(query_obj, 'mast_obs')
            if cached is not None:
                results['mast_obs'] = cached
            else:
                futures[executor.submit(fetch_mast_observations_df, ra, dec)] = 'mast_obs'

        for future in as_completed(futures):
            key = futures[future]
            try:
                data = future.result()
                results[key] = data
                save_to_cache(query_obj, key, data)
            except Exception as e:
                st.warning(f"{key.upper()} query failed: {e}")
                logging.error(f"{key.upper()} query failed: {e}")
                results[key] = pd.DataFrame() if key != 'vizier' else {}

    return results


def format_num(val, prec=2, fmt='f'):
    if isinstance(val, (int, float)):
        return f"{val:.{prec}{fmt}}"
    return str(val)


# ----------------- Main flow -----------------
if run_button:
    logging.info(f"Starting analysis for TIC ID: {target_input}")
    name = target_input.strip()
    query_obj = f"TIC {name.replace('TIC', '').strip()}"
    if not name.lower().startswith('tic'):
        query_obj = f"TIC {name}"

    progress_bar = st.progress(0)
    progress_text = st.empty()
    progress_text.text("Fetching TIC data... (0%)")

    # Placeholders for streaming updates
    left_col, right_col = st.columns([3, 1])
    with right_col:
        passport_placeholder = st.subheader("Planet Passport")
        passport_content = st.empty()
        passport_content.markdown("Fetching data...")

    with left_col:
        tabs = st.tabs(["Basic Info", "Catalogs", "Lightcurve", "Classification", "Images", "Resources"])
        basic_placeholder = tabs[0].empty()
        catalogs_placeholder = tabs[1].empty()
        lc_placeholder = tabs[2].empty()
        class_placeholder = tabs[3].empty()
        images_placeholder = tabs[4].empty()
        resources_placeholder = tabs[5].empty()

    # Step 1: Fetch TIC data
    tic_df = query_tic_as_df(query_obj)
    if tic_df is None or tic_df.empty:
        st.error("TIC not found. Try another identifier.")
        st.stop()
    progress_bar.progress(10)
    progress_text.text("TIC data fetched (10%)")

    star = tic_df.iloc[0]
    ra = float(star.get("ra", np.nan))
    dec = float(star.get("dec", np.nan))
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

    # Update basic info immediately
    with basic_placeholder.container():
        st.header("Basic TIC data")
        st.dataframe(safe_to_pandas_table(tic_df, maxrows=20))
        st.markdown(f"**Coordinates:** RA={ra:.6f}  DEC={dec:.6f}")
        st.subheader("Star Properties")
        st.table(properties)

    # Parallel fetch
    progress_text.text("Fetching parallel data... (20%)")
    parallel_results = parallel_fetch_data(query_obj, ra, dec, include_gaia, include_simbad, include_exoplanet_archive, include_vizier, include_tess)
    progress_bar.progress(30)
    progress_text.text("Parallel data fetched (30%)")

    exo_df = parallel_results.get('exo')

    # Update catalogs immediately
    with catalogs_placeholder.container():
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
                    except Exception as e:
                        logging.error(f"Gaia HR diagram failed: {e}")
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
                if 'rvz_radvel' in sim_df.columns:
                    st.markdown(f"**Radial Velocity:** {sim_df['rvz_radvel'].iloc[0]} km/s")
            else:
                st.info("SIMBAD: no matches or query failed.")

        if include_exoplanet_archive:
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

                progress_text.text("Fetching MAST products... (40%)")
                products_df = fetch_products_for_obs_df(obs_df, max_obs=min(max_products, 3))
                progress_bar.progress(50)
                progress_text.text("MAST products fetched (50%)")
                if products_df is not None and not products_df.empty:
                    st.subheader("MAST products (combined)")
                    st.dataframe(safe_to_pandas_table(products_df, maxrows=300))
                    if 'productType' in products_df.columns and 'dataURI' in products_df.columns:
                        thumbs = products_df[products_df['productType'].str.contains('PREVIEW|THUMBNAIL', na=False, case=False)].head(4)
                        if not thumbs.empty:
                            st.subheader("Preview Thumbnails")
                            cols = st.columns(4)
                            for idx, (_, r) in enumerate(thumbs.iterrows()):
                                uri = r['dataURI']
                                url = f"https://mast.stsci.edu/api/v0.1/Download/file?uri={uri}"
                                try:
                                    resp = requests.get(url, timeout=5)
                                    img = Image.open(BytesIO(resp.content))
                                    with cols[idx % 4]:
                                        st.image(img, caption=r.get('productFilename', ''))
                                except Exception as e:
                                    logging.error(f"Thumbnail fetch failed: {e}")
            else:
                st.info("No MAST observations found nearby.")

    lc = None
    with lc_placeholder.container():
        if include_tess:
            progress_text.text("Fetching lightcurve... (60%)")
            cache_dir = os.path.expanduser("~/.lightkurve/cache")
            try:
                tic_num = int(tic_df['ID'].iloc[0])
                search = lk.search_lightcurve(f"TIC {tic_num}", mission='TESS', author=['SPOC', 'QLP'])
                if len(search) == 0:
                    search = lk.search_lightcurve(query_obj, mission='TESS')
                if len(search) == 0:
                    st.info("No TESS lightcurves found.")
                else:
                    st.write(f"Found {len(search)} lightcurve entries")
                    with st.spinner("Downloading and stitching lightcurves ..."):
                        coll = None
                        for attempt in range(2):  # Retry once
                            try:
                                coll = search.download_all(quality_bitmask='hardest', flux_column='sap_flux')
                                logging.info(f"Lightcurve downloaded for {query_obj}")
                                break
                            except Exception as download_err:
                                if "corrupt" in str(download_err).lower() or "supported data product" in str(download_err).lower():
                                    if os.path.exists(cache_dir):
                                        shutil.rmtree(cache_dir)
                                        logging.warning(f"Corrupt file detected. Cleared cache and retrying download (attempt {attempt+1}/2).")
                                else:
                                    logging.error(f"Lightkurve download failed: {download_err}")
                                    raise download_err
                        if coll is None:
                            raise Exception("Download failed after retries.")
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
                logging.error(f"Lightkurve failed for {query_obj}: {e}")
            progress_bar.progress(70)
            progress_text.text("Lightcurve processed (70%)")

    res = None
    with class_placeholder.container():
        if include_classification and include_tess and lc is not None:
            progress_text.text("Classifying... (80%)")
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
                logging.error(f"Classification failed for {query_obj}: {e}")
                cnn = CNNClassifier()
                preds = cnn.predict(lc.flux.value if lc else None)  # Use LC if available
                res = {
                    "ID": query_obj,
                    "Status": "Candidate",
                    "Reason": "Classification failed, using ML estimates",
                    "Score": preds['confidence'],
                    "ML Score": preds['confidence'],
                    "Period": preds['period'],
                    "Depth": preds['depth'],
                    "R_p (Rearth)": preds['rp'],
                    "M_p (Mearth)": preds['mp'],
                    "Density (g/cm3)": preds['density'],
                    "Orbit (AU)": preds['orbit'],
                    "RV Amplitude (m/s)": preds['rv'],
                    "Teq": {
                        "Gas": preds['teq'],
                        "Rocky": preds['teq'] * 1.05,
                        "Icy": preds['teq'] * 0.95
                    },
                    "Class": preds['class']
                }
                st.table(res)  # Still display table with fallback data
            progress_bar.progress(85)
            progress_text.text("Classification done (85%)")

        # Visual model of differences + percent bar
        st.subheader("Visual Comparison to Earth and Sun")
        planet_r = res.get('R_p (Rearth)', 1) if res else 1
        planet_m = res.get('M_p (Mearth)', 1) if res else 1
        star_r = properties['Radius (Rsun)'] if isinstance(properties['Radius (Rsun)'], (int, float)) else 1
        star_m = properties['Mass (Msun)'] if isinstance(properties['Mass (Msun)'], (int, float)) else 1

        df_planet = pd.DataFrame({
            'Body': ['Earth', 'Planet'],
            'Radius (R⊕)': [1, planet_r],
            'Mass (M⊕)': [1, planet_m]
        })
        fig_p = px.bar(df_planet, x='Body', y=['Radius (R⊕)', 'Mass (M⊕)'], barmode='group', title='Planet Comparison')
        st.plotly_chart(fig_p)

        df_star = pd.DataFrame({
            'Body': ['Sun', 'Star'],
            'Radius (R☉)': [1, star_r],
            'Mass (M☉)': [1, star_m]
        })
        fig_s = px.bar(df_star, x='Body', y=['Radius (R☉)', 'Mass (M☉)'], barmode='group', title='Star Comparison')
        st.plotly_chart(fig_s)

        # Percent bar for differences
        radius_diff_planet = abs(planet_r - 1) / 1 * 100
        mass_diff_planet = abs(planet_m - 1) / 1 * 100
        radius_diff_star = abs(star_r - 1) / 1 * 100
        mass_diff_star = abs(star_m - 1) / 1 * 100

        st.subheader("Difference Percentages")
        st.progress(min(1.0, max(0.0, radius_diff_planet / 100)))
        st.caption(f"Planet Radius Difference: {radius_diff_planet:.2f}%")
        st.progress(min(1.0, max(0.0, mass_diff_planet / 100)))
        st.caption(f"Planet Mass Difference: {mass_diff_planet:.2f}%")
        st.progress(min(1.0, max(0.0, radius_diff_star / 100)))
        st.caption(f"Star Radius Difference: {radius_diff_star:.2f}%")
        st.progress(min(1.0, max(0.0, mass_diff_star / 100)))
        st.caption(f"Star Mass Difference: {mass_diff_star:.2f}%")

    with images_placeholder.container():
        progress_text.text("Fetching images... (90%)")
        pos = f"{ra} {dec}"
        any_image_shown = False
        for idx, survey in enumerate(image_surveys):
            arr, used_survey = get_image_via_skyview_safe(pos, survey, pixels=256)
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
                    except Exception as e:
                        logging.error(f"Image rendering failed: {e}")
                        st.write("Could not render image")
            else:
                st.write(f"No image from {survey}")

        st.subheader("Synthetic Star Image")
        png_bytes = synthetic_star_image(seed=query_obj, size=256)
        st.image(png_bytes, caption="Synthetic star preview", use_container_width=False)
        progress_bar.progress(95)
        progress_text.text("Images done (95%)")

    with resources_placeholder.container():
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

    # Update Planet Passport early with real data
    rp = 1.0
    mp = 1.0
    density = 'N/A'
    teq = {"Gas": 'N/A', "Rocky": 'N/A', "Icy": 'N/A'}
    class_type = 'Unknown'
    score = 0.0
    ml = 0.0
    odd_even = 'N/A'
    secondary = 'N/A'
    centroid = 'N/A'
    depth_reason = 'N/A'
    period = 'N/A'
    depth = 'N/A'
    orbit_au = 'N/A'
    rv_amp = 'N/A'
    status = 'Unknown'

    if exo_df is not None and not exo_df.empty:
        rp = exo_df.get('pl_rade', pd.Series([rp])).iloc[0]
        mp = exo_df.get('pl_masse', pd.Series([mp])).iloc[0]
        density = exo_df.get('pl_dens', pd.Series([density])).iloc[0]
        if 'pl_eqt' in exo_df.columns:
            eqt = exo_df['pl_eqt'].iloc[0]
            teq = {"Gas": eqt, "Rocky": eqt, "Icy": eqt}
        period = exo_df.get('pl_orbper', pd.Series([period])).iloc[0]
        class_type = 'Super-Earth' if rp > 1.5 else 'Earth-like'

    if res is not None:
        score = res.get('Score', score)
        ml = res.get('ML Score', ml)
        rp = res.get('R_p (Rearth)', rp)
        mp = res.get('M_p (Mearth)', mp)
        density = res.get('Density (g/cm3)', density)
        teq = res.get('Teq', teq)
        period = res.get('Period', period)
        depth = res.get('Depth', depth)
        orbit_au = res.get('Orbit (AU)', orbit_au)
        status = res.get('Status', status)
        rv_amp = res.get('RV Amplitude (m/s)', rv_amp)
        class_type = 'Super-Earth' if rp > 1.5 else 'Earth-like'
        if 'Reason' in res:
            if "odd/even" in res['Reason'].lower():
                odd_even = 'inconsistent'
            if "secondary" in res['Reason'].lower():
                secondary = 'strong'
            if "centroid" in res['Reason'].lower():
                centroid = 'shifted'
            if "depth" in res['Reason'].lower():
                depth_reason = 'unreasonable'
            else:
                odd_even = 'consistent'
                secondary = 'weak'
                centroid = 'ok'
                depth_reason = 'reasonable'

    teq_str = " | ".join([f"{k}={format_num(v,1)}K" for k, v in teq.items()])

    summary = f"""
Hybrid Score

{format_num(score,2)}
Rp: {format_num(rp,2)} R⊕

Class: {class_type}

Teq: {teq_str}

Mp

{format_num(mp,2)} M⊕
Density

{format_num(density,1)} g/cm³
Why? ML={format_num(ml,2)}; Odd/even={odd_even}; Secondary={secondary}; Centroid={centroid}; Depth={depth_reason}. → Общий score={format_num(score,2)}

Additional Data:
Status: {status}
Period (days): {format_num(period,2)}
Depth: {format_num(depth,4)}
Orbit (AU): {format_num(orbit_au,3)}
RV Amplitude (m/s): {format_num(rv_amp,1)}
    """
    passport_content.markdown(summary)

    progress_bar.progress(100)
    progress_text.text("Done (100%)")
    st.success("Done — tables, images and plots shown above.")
    logging.info(f"Analysis completed for {query_obj}")

st.markdown("\n---\n")
st.markdown("Notes: If some queries fail or images are missing – this can be due to network timeouts, API limits, or the object simply not being covered by the survey. I included robust fallbacks and a synthetic star preview. For Lightkurve errors due to corrupt files, the code clears the cache automatically and retries download.")