import sys
import os
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Добавляем родительскую директорию, чтобы импорты из pipeline работали
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from pipeline.planet import get_planet_data
from pipeline.classifier import classify_target_full
from pipeline.catalog import get_star_params
from pipeline.lightcurve import get_lightcurve_and_bls


st.title("🚀 Mission Control: Exoplanet Hunter")
tic_id = st.text_input("TIC ID", "TIC 150428135")

if st.button("🔍 Analyze"):
    try:
        lc, _, _, period, t0, duration, depth = get_lightcurve_and_bls(tic_id)

        if lc is None:
            st.error("Не удалось получить кривую блеска.")
        else:
            res = classify_target_full(tic_id, lc, period, t0, duration, depth)

            # Получаем параметры звезды
            star_params = get_star_params(tic_id) or {
                "T_star": 5000,
                "R_star": 1.0,
                "M_star": 1.0,
            }

            # Убираем неподдерживаемые аргументы
            if "cross_conf" in star_params:
                star_params.pop("cross_conf")

            # Вычисляем параметры планеты
            passport = get_planet_data(res["Period"], res["Depth"], **star_params)
            res.update(passport)

            # --- Отображение ---
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("LC & BLS")
                fig, ax = plt.subplots()
                time, flux = lc.time.value, lc.flux.value
                ax.plot(time, flux, color='blue', linewidth=1)
                ax.set_xlabel("Time (days)")
                ax.set_ylabel("Flux")
                st.pyplot(fig)

                if st.button("Explain LC"):
                    st.write(f"🔴 Транзит: phase 0, depth = {res['Depth']:.4f}")

            with col2:
                st.subheader("Planet Passport")
                st.metric("Hybrid Score", f"{res['Hybrid_score']:.2f}")
                st.write(f"**Rp**: {passport['R_p_Rearth']} R⊕")
                st.write(f"**Class**: {passport['Class']}")
                st.write(
                    f"**Teq**: "
                    f"Gas={passport['Teq'].get('Gas giant (0.1)', '?')}K | "
                    f"Rocky={passport['Teq'].get('Rocky (0.3)', '?')}K | "
                    f"Icy={passport['Teq'].get('Icy (0.7)', '?')}K"
                )
                st.write("**Why?** " + res.get("Explain", "—"))

            if st.button("🎤 Tell Story"):
                story = (
                    f"Планета {tic_id}: {passport['Class']}, "
                    f"Rp {passport['R_p_Rearth']} R⊕. "
                    f"Score {res['Hybrid_score']}. Новый мир открыт!"
                )
                st.write("🔊 Audio: " + story)

    except Exception as e:
        st.error(f"❌ Ошибка при обработке: {e}")
