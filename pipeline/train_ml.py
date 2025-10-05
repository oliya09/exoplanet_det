import numpy as np
import pandas as pd
import lightkurve as lk
import os
import tensorflow as tf  # Добавлено для GPU
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
from astroquery.mast import Catalogs
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Проверка и установка GPU
physical_devices = tf.config.list_physical_devices('GPU')
print("GPUs available:", len(physical_devices))
if physical_devices:
    tf.config.set_visible_devices(physical_devices[0], 'GPU')
    print("Using GPU for training")
else:
    print("Using CPU for training")

print("Fetching confirmed planets from NASA...")
confirmed = NasaExoplanetArchive.query_criteria(
    table="pscomppars",
    select="hostname, pl_name",
    where="st_teff > 0 AND pl_rade < 4 AND disc_facility like 'Transiting Exoplanet Survey Satellite (TESS)'"  # Правильный фильтр для TESS
)
planet_hosts = np.unique(confirmed["hostname"][:1])  # Limit 50 для теста
print(f"Planet hosts: {len(planet_hosts)}")

# Real non-planets: Known TESS FPs/EBs
non_planet_hosts = [
    "TIC 393818343", "TIC 356473034", "TIC 278825932", "TIC 144870384", "TIC 1000165289",
    "TIC 219134195", "TIC 261136679", "TIC 307210830", "TIC 350153977", "TIC 425933644"  # Из TESS FP lists
] * 5  # ~50 total

def fetch_lightcurve(target, length=800):
    cache_file = f"..cache/{target.replace(' ', '_')}.npy"
    if os.path.exists(cache_file):
        return np.load(cache_file)
    
    for mission in ["TESS", "Kepler"]:
        try:
            search = lk.search_lightcurve(target, mission=mission)
            if len(search) == 0:
                continue
            lc_collection = search[:2].download_all()
            if lc_collection is None:
                continue
            lc = lc_collection.stitch().remove_nans().normalize()
            flux = lc.flux.value
            
            if hasattr(flux, 'filled'):
                flux = flux.filled(1.0)
            flux = np.asarray(flux)
            if len(flux) < length:
                flux = np.pad(flux, (0, length - len(flux)), mode="constant", constant_values=1.0)
            else:
                flux = flux[:length]
            
            np.save(cache_file, flux)
            print(f"✅ Fetched {target} ({mission})")
            return flux
        except Exception as e:
            print(f"Failed {mission} for {target}: {e}")
            continue
    
    print(f"❌ No LC for {target}")
    return None

def augment_flux(flux, n_aug=3, noise_level=0.001):
    aug = []
    for _ in range(n_aug):
        noisy = flux + np.random.normal(0, noise_level, len(flux))
        shifted = np.roll(noisy, np.random.randint(-50, 50))
        aug.append(shifted)
    return aug

X, y = [], []
os.makedirs("cache", exist_ok=True)

print("Fetching planet LCs...")
valid_planets = 0
for i, host in enumerate(planet_hosts):
    print(f"[Planet {i+1}/{len(planet_hosts)}] {host}")
    flux = fetch_lightcurve(host)
    if flux is not None:
        X.append(flux)
        y.append(1)
        valid_planets += 1
        for aug_flux in augment_flux(flux):
            X.append(aug_flux)
            y.append(1)

print(f"Valid planets: {valid_planets}")

print("Fetching non-planet LCs...")
valid_non = 0
for i, host in enumerate(non_planet_hosts[:1]):  # Limit
    print(f"[Non-planet {i+1}/50] {host}")
    flux = fetch_lightcurve(host)
    if flux is not None:
        X.append(flux)
        y.append(0)
        valid_non += 1
        for aug_flux in augment_flux(flux):
            X.append(aug_flux)
            y.append(0)

print(f"Valid non-planets: {valid_non}")

if len(X) == 0:
    print("❌ No data collected! Check internet/queries.")
    exit(1)

X = np.array(X).reshape(-1, 2000, 1)
y = np.array(y)
print(f"Dataset: {X.shape[0]} samples (Planets: {sum(y)}, Non: {len(y)-sum(y)})")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Building & training CNN...")
cnn_model = Sequential([
    Conv1D(32, 5, activation='relu', input_shape=(2000, 1)),
    MaxPooling1D(2),
    Conv1D(64, 5, activation='relu'),
    MaxPooling1D(2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=3)
history = cnn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2,
                        callbacks=[early_stop], verbose=1)
cnn_loss, cnn_acc = cnn_model.evaluate(X_test, y_test, verbose=0)
cnn_auc = roc_auc_score(y_test, cnn_model.predict(X_test).flatten())
print(f"CNN: Acc={cnn_acc:.3f}, AUC={cnn_auc:.3f}")

print("Training RF...")
rf_model = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)
rf_model.fit(X_train.reshape(len(X_train), -1), y_train)
rf_pred = rf_model.predict_proba(X_test.reshape(len(X_test), -1))[:, 1]
rf_auc = roc_auc_score(y_test, rf_pred)
print(f"RF: AUC={rf_auc:.3f}")

cnn_model.save('cnn_model.h5')
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
print("Models saved: cnn_model.h5, rf_model.pkl")
print(classification_report(y_test, (cnn_model.predict(X_test) > 0.5).astype(int)))