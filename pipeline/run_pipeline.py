import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from .classifier import classify_target_full
from .catalog import get_star_params, get_rv_k, get_hostname_from_tic
from .planet import get_planet_data
from .lightcurve import get_lightcurve_and_bls
from .report import generate_report

def parallel_process_targets(ids, mission="TESS", max_workers=5):
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(get_lightcurve_and_bls, tid, mission): tid for tid in ids}
        for future in as_completed(futures):
            tid = futures[future]
            try:
                results[tid] = future.result()  # (lc, bls, result, period, t0, dur, depth)
            except Exception as e:
                print(f"{tid} failed: {e}")
                results[tid] = (None, None, None, None, None, None, None)
    return results

def run_pipeline(ids, mission="TESS", max_workers=5):
    """Основной конвейер анализа списка TIC ID"""
    confirmed, candidates, false_pos = [], [], []
    os.makedirs("reports", exist_ok=True)
    os.makedirs("cache", exist_ok=True)

    print("Fetching LCs in parallel...")
    all_lcs = parallel_process_targets(ids, mission, max_workers)

    for tid, (lc, bls, result, period, t0, duration, depth) in all_lcs.items():
        if lc is None:
            res = {"ID": tid, "Status": "No Data", "Reason": "No LC data", "Score": 0.0, "lc": None}
            candidates.append(res)
            continue

        res = classify_target_full(tid, lc, period, t0, duration, depth, mission)

        if res["Status"] == "Confirmed Planet":
            hostname = get_hostname_from_tic(tid)
            star_params = get_star_params(tid)
            if star_params:
                k = get_rv_k(hostname)
                extra = get_planet_data(res["Period"], res["Depth"],
                                        star_params["T_star"], star_params["R_star"], star_params["M_star"], k)
                res.update(star_params)
                res.update(extra)
                confirmed.append(res)
                generate_report(tid, res, lc)
            else:
                candidates.append(res)
        elif res["Status"] == "Candidate":
            candidates.append(res)
        else:
            false_pos.append(res)

    pd.DataFrame(confirmed).to_csv("confirmed_planets.csv", index=False)
    pd.DataFrame(candidates).to_csv("candidates.csv", index=False)
    pd.DataFrame(false_pos).to_csv("false_positives.csv", index=False)

    print(f"✅ Confirmed: {len(confirmed)}, 🪐 Candidates: {len(candidates)}, ❌ False Positives: {len(false_pos)}")
    print("📊 Отчёты в /reports/, метрики: python pipeline/metrics.py")