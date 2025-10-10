# demo.py (fixed cache clear with os.rmtree)
import sys
import os
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from gtts import gTTS  # pip install gtts для TTS
import tempfile  # For temp files
import shutil  # For rmtree

# --- Pipeline imports ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from pipeline.planet import get_planet_data
from pipeline.classifier import classify_target_full, CNNClassifier
from pipeline.catalog import get_star_params
from pipeline import get_lightcurve_and_bls


# 🚀 STREAMLIT UI

st.set_page_config(page_title="🚀 Exoplanet Hunter", layout="wide")
st.title("🚀 Mission Control: Exoplanet Hunter")

tic_id = st.text_input("TIC ID", "TIC 150428135")
analyze_btn = st.button("🔍 Analyze")

# Button to clear lightkurve cache
if st.button("Clear Lightkurve Cache"):
    cache_dir = os.path.expanduser("~/.lightkurve/cache")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        st.success("Lightkurve cache cleared - re-analyze now")
    else:
        st.info("No cache directory found.")

# --- Logging and progress ---
log_container = st.empty()
progress_bar = st.progress(0)
logs = []


def log(msg, step=None, total=None):
    """Вывод логов и обновление прогресса."""
    logs.append(msg)
    log_container.text("\n".join(logs))
    if step is not None and total is not None:
        progress_bar.progress(min(step / total, 1.0))



# ⚙️ Load ML Models

def load_models():
    """Загружаем все ML модели (один раз за сессию)."""
    st.write("🔄 Loading CNN model...")
    cnn_model = CNNClassifier(log_fn=st.write)
    st.write("✅ CNN model is loaded")
    return {"cnn": cnn_model}


# --- Load models once ---
if "models" not in st.session_state:
    with st.spinner("Loading models..."):
        st.session_state.models = load_models()


# Main func

def analyze_tic(tic_id):
    steps = 6
    step = 0
    res = None  
    lc = None   

    try:
        step += 1
        log(f"🔍 Starting analisys {tic_id}...", step, steps)

        # STEP 2: get LC и BLS (FIX: first!)
        step += 1
        log("⏳ Loading light curve and performing BLS...", step, steps)
        lc, bls, bls_result, period, t0, duration, depth = get_lightcurve_and_bls(tic_id)
        if lc is None:
            log("❌ Failed to retrieve light curve. Check the lightkurve cache or internet connection.", steps, steps)
            return None, None
        log(f"✅ LC loaded (span ~{np.ptp(lc.time.value):.1f}d)", step, steps)  # FIX: Debug лог

        # STEP 3: Classification (FIX: After fetch!)
        step += 1
        log("🔬 Classifying the signal using CNN...", step, steps)
        model = st.session_state.models["cnn"]
        res = classify_target_full(tic_id, lc, period, t0, duration, depth, model=model)
        log("📦 LC classified", step, steps)

        # STEP 4: Stellar parameters (with cache)
        step += 1
        cache_key = f"star_params_{tic_id}"
        if cache_key not in st.session_state:
            log("🔎 Fetching stellar parameters (MAST/TIC)...", step, steps)
            star_params = get_star_params(tic_id) or {"T_star": 3494, "R_star": 0.42, "M_star": 0.41}  # Hardcode fallback for this TIC
            star_params.pop("cross_conf", None)
            st.session_state[cache_key] = star_params
            log("🌍 Stellar parameters saved to cache", step, steps)
        else:
            star_params = st.session_state[cache_key]
            log("🌍 Stellar parameters loaded from cache", step, steps)

        # Get K from RV (optional, for mass estimation in get_planet_data)
        k = None  # Default
        try:
            from pipeline.lightcurve.catalog import try_nasa_params  # Import for K
            nasa_data = try_nasa_params(tic_id)
            if nasa_data and "planet_data" in nasa_data:
                k = nasa_data["planet_data"].get("pl_rv", None)  # RV semi-amplitude
                log(f"RV K: {k} m/s (from NASA)", step, steps)
        except Exception as e:
            log(f"[WARN] RV K fetch failed: {e}", step, steps)

        # STEP 5: Planet parameters
        step += 1
        log("🧮 Calculating planet passport...", step, steps)
        planet_passport = get_planet_data(
            res["Period"], res["Depth"], 
            star_params["T_star"], star_params["R_star"], star_params["M_star"], 
            k=k 
        )
        res.update(planet_passport)
        log("✅ Planet parametrs are calculated", step, steps)

        # STEP 6: Finalize
        step += 1
        log(f"🎉 Analysis of {tic_id} completed!", step, steps)
        return res, lc

    except Exception as e:
        log(f"❌ Error processing {tic_id}: {e}", steps, steps)
        return None, None

    finally:
        progress_bar.progress(1.0)


# 🚀 Button handling

if analyze_btn:
    res, lc = analyze_tic(tic_id)

    if res is not None and lc is not None:
        # --- Display ---
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 LC & BLS")
            fig, ax = plt.subplots()
            ax.plot(lc.time.value, lc.flux.value, color='blue', linewidth=1)
            ax.set_xlabel("Time (days)")
            ax.set_ylabel("Flux")
            st.pyplot(fig)

            if st.button("Explain LC"):
                st.write(f"🔴 Transit: phase 0, depth = {res['Depth']:.4f}")

        with col2:
            st.subheader("🪐 Planet Passport")
            st.metric("Hybrid Score", f"{res['Hybrid_score']:.2f}")
            
            # Обработка None в UI
            rp = res.get('R_p_Rearth', 'N/A')
            st.write(f"**Rp**: {rp} R⊕")
            
            cls = res.get('Class', 'Unknown')
            st.write(f"**Class**: {cls}")
            
            teq = res.get('Teq', {})
            st.write(
                f"**Teq**: Gas={teq.get('Gas giant (0.1)', '?')}K | "
                f"Rocky={teq.get('Rocky (0.3)', '?')}K | "
                f"Icy={teq.get('Icy (0.7)', '?')}K"
            )
            
            # Additional metrics (if available)
            if res.get('M_p_Mearth'):
                st.metric("Mp", f"{res['M_p_Mearth']} M⊕")
            if res.get('Density_gcm3'):
                st.metric("Density", f"{res['Density_gcm3']} g/cm³")
                
            st.write("**Why?** " + res.get("Explain", "—"))

        if st.button("🎤 Tell Story"):
            story = (
                f"Planet {tic_id}: {res.get('Class', 'Unknown')}, "
                f"Rp {res.get('R_p_Rearth', 'N/A')} R⊕. "
                f"Score {res['Hybrid_score']:.2f}. A new world discovered!"
            )
            st.write("🔊 Audio: " + story)
            
            # TTS (Google Text-to-Speech)
            try:
                tts = gTTS(story, lang='ru')
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                    tts.save(fp.name)
                    st.audio(fp.name, format="audio/mp3")
                os.unlink(fp.name)  # Clean up temp file
            except Exception as e:
                st.error(f"TTS failed: {e}")
                st.info("Audio playback: Use a valid TTS service URL in production.")
    else:
        st.error("Analysis failed. Check the TIC ID or clear the lightkurve cache (~/.lightkurve/cache).")