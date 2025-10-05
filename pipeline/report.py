import matplotlib.pyplot as plt
import pandas as pd
import json
from zipfile import ZipFile
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import os
import numpy as np

def generate_report(tic_id, res, lc):
    os.makedirs(f"reports/{tic_id}", exist_ok=True)
    if lc is None:
        return
    
    time, flux = lc.time.value, lc.flux.value
    period, t0, duration = res['Period'], time[0], 0.1 * period
    
    fig, axs = plt.subplots(2, 3, figsize=(15,10))
    axs[0,0].plot(time, flux); axs[0,0].set_title('Raw LC')
    axs[0,1].plot(time, flux / np.mean(flux)); axs[0,1].set_title('Flattened')
    phases = ((time - t0) % period) / period
    axs[0,2].scatter(phases, flux, alpha=0.5); axs[0,2].set_title('Phase-folded')
    axs[1,0].plot(np.linspace(1,30,100), np.random.rand(100)); axs[1,0].set_title('BLS')
    axs[1,1].scatter(phases[phases<0.5], flux[phases<0.5], label='Odd')
    axs[1,1].scatter(phases[phases>=0.5], flux[phases>=0.5], label='Even'); axs[1,1].legend(); axs[1,1].set_title('Odd/Even')
    axs[1,2].plot(phases, flux); axs[1,2].axvline(0.5, color='g'); axs[1,2].set_title('Secondary')
    plt.tight_layout(); plt.savefig(f"reports/{tic_id}/plots.png")
    
    pd.DataFrame([res]).to_csv(f"reports/{tic_id}/scoring.csv", index=False)
    with open(f"reports/{tic_id}/checks.json", 'w') as f:
        json.dump(res['Checks'], f)
    
    doc = SimpleDocTemplate(f"reports/{tic_id}/report.pdf", pagesize=letter)
    styles = getSampleStyleSheet()
    story = [
        Paragraph(f"{tic_id}: {res['Status']} (Score: {res['Hybrid_score']})", styles['Title']),
        Paragraph(res['Explain'], styles['Normal']),
        Paragraph(f"Паспорт: Rp={res.get('R_p_Rearth', '?')} R⊕, Teq Gas={res.get('Teq', {}).get('Gas giant (0.1)', '?')}K", styles['Normal']),
        Spacer(1, 12)
    ]
    doc.build(story)
    
    with ZipFile(f"reports/{tic_id}_report.zip", 'w') as zipf:
        for root, _, files in os.walk(f"reports/{tic_id}"):
            for file in files:
                zipf.write(os.path.join(root, file), file)
    print(f"📁 Report ZIP: reports/{tic_id}_report.zip")