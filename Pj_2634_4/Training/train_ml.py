import numpy as np
import lightkurve as lk
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
from astroquery.mast import Catalogs
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

warnings.filterwarnings("ignore", category=UserWarning)


# Utility functions


def resolve_tic_hosts(hosts):
    """Resolve all hosts to TIC IDs sequentially."""
    resolved = {}
    for host in hosts:
        try:
            tic_data = Catalogs.query_object(host, catalog="TIC")
            if len(tic_data) > 0:
                resolved[host] = f"TIC {int(tic_data['ID'][0])}"
                print(f"Resolved {host} → {resolved[host]}")
            else:
                resolved[host] = host
                print(f"No TIC for {host}, using raw name")
        except Exception as e:
            print(f"TIC query failed for {host}: {e}, using raw name")
            resolved[host] = host
    return resolved


def fetch_single_lightcurve(target, resolved_target):
    """Фетч одной LC с кэшем."""
    cache_file = f"cache/{target.replace(' ', '_')}.npy"
    os.makedirs("cache", exist_ok=True)
    if os.path.exists(cache_file):
        return np.load(cache_file)

    search_target = resolved_target
    for mission in ["TESS", "Kepler"]:
        try:
            search = lk.search_lightcurve(search_target, mission=mission)
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
            if len(flux) < 2000:
                flux = np.pad(flux, (0, 2000 - len(flux)), mode="constant", constant_values=1.0)
            else:
                flux = flux[:2000]
            flux = (flux - np.min(flux)) / (np.max(flux) - np.min(flux) + 1e-8)
            np.save(cache_file, flux)
            print(f"✅ Fetched {target} ({mission})")
            return flux
        except Exception as e:
            print(f"Failed {mission} for {search_target}: {e}")
            continue
    print(f"❌ No LC for {target}")
    return None


def fetch_lightcurves_parallel(hosts, resolved_dict, max_workers=4):
    """Параллельный фетч (ускорение IO)."""
    X = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_single_lightcurve, host, resolved_dict[host]): host for host in hosts}
        for future in as_completed(futures):
            flux = future.result()
            if flux is not None:
                X.append(flux)
    return np.array(X)


def augment_flux_advanced(flux, n_aug=10):
    """Аугментация данных."""
    aug = [flux]
    for _ in range(n_aug):
        noisy = flux + np.random.normal(0, 0.001, len(flux))
        shifted = np.roll(noisy, np.random.randint(-50, 50))
        scale = 1 + np.random.uniform(-0.05, 0.05)
        scaled = shifted * scale
        jitter = np.random.uniform(0.95, 1.05, len(flux))
        jittered = scaled * jitter
        aug.append(np.clip(jittered, 0, 1))
    return aug


# Dataset and model


class LightcurveDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 497, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 1)  # Without sigmoids

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x  # Without sigmoid


# Main training loop


if __name__ == '__main__':
    multiprocessing.freeze_support()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    print("Fetching confirmed planets from NASA...")
    confirmed = NasaExoplanetArchive.query_criteria(
        table="pscomppars",
        select="hostname, pl_name",
        where="st_teff > 0 AND pl_rade < 4 AND disc_facility like 'Transiting Exoplanet Survey Satellite (TESS)'"
    )
    planet_hosts = np.unique(confirmed["hostname"][:10])
    print(f"Planet hosts: {len(planet_hosts)}")

    non_planet_hosts = [
        "TIC 393818343", "TIC 356473034", "TIC 278825932", "TIC 144870384",
        "TIC 1000165289", "TIC 219134195", "TIC 261136679", "TIC 307210830",
        "TIC 350153977", "TIC 425933644"
    ][:10]

    print("Resolving TIC IDs...")
    planet_resolved = resolve_tic_hosts(planet_hosts)
    non_resolved = resolve_tic_hosts(non_planet_hosts)

    print("Fetching planet lightcurves...")
    planet_fluxes = fetch_lightcurves_parallel(planet_hosts, planet_resolved)
    X_planet, y_planet = [], []
    for flux in planet_fluxes:
        X_planet.extend(augment_flux_advanced(flux))
        y_planet.extend([1] * 11)

    print("Fetching non-planet lightcurves...")
    non_fluxes = fetch_lightcurves_parallel(non_planet_hosts, non_resolved)
    X_non, y_non = [], []
    for flux in non_fluxes:
        X_non.extend(augment_flux_advanced(flux))
        y_non.extend([0] * 11)

    X = np.array(X_planet + X_non).reshape(-1, 1, 2000)
    y = np.array(y_planet + y_non)
    print(f"Dataset: {X.shape[0]} samples (Planets: {sum(y)}, Non: {len(y)-sum(y)})")

    if len(X) < 50:
        print("❌ Too few data! Add more hosts or simulate.")
        exit(1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    batch_size = 64
    num_workers = 4
    prefetch_factor = 2 if num_workers > 0 else None
    persistent_workers = num_workers > 0
    train_loader = DataLoader(LightcurveDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              prefetch_factor=prefetch_factor,
                              persistent_workers=persistent_workers)
    val_loader = DataLoader(LightcurveDataset(X_test, y_test),
                            batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            prefetch_factor=prefetch_factor,
                            persistent_workers=persistent_workers)

    print("Building & training CNN...")
    model = CNNModel().to(device)
    if hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode='reduce-overhead')
        except Exception as e:
            print(f"torch.compile failed: {e}")

    criterion = nn.BCEWithLogitsLoss()  
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5)
    scaler = GradScaler(device.type)

    patience = 10
    best_val_loss = float('inf')
    epochs_no_improve = 0
    num_epochs = 50

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device, non_blocking=True), batch_y.to(device, non_blocking=True).unsqueeze(1)
            optimizer.zero_grad()
            with autocast(device_type=device.type):
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        val_preds, val_true = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device, non_blocking=True), batch_y.to(device, non_blocking=True).unsqueeze(1)
                with autocast(device_type=device.type):
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                probs = torch.sigmoid(outputs)  
                val_preds.extend(probs.cpu().numpy().flatten())
                val_true.extend(batch_y.cpu().numpy().flatten())

        val_loss /= len(val_loader)
        train_loss /= len(train_loader)
        val_auc = roc_auc_score(val_true, val_preds)
        val_acc = accuracy_score(val_true, [1 if p > 0.5 else 0 for p in val_preds])
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val AUC={val_auc:.4f}, Val Acc={val_acc:.4f}")

        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).unsqueeze(1)
            outputs = model(batch_x)
            probs = torch.sigmoid(outputs)
            all_preds.extend((probs > 0.5).float().cpu().numpy().flatten())
            all_true.extend(batch_y.cpu().numpy().flatten())

    print("\nFinal Classification Report:")
    print(classification_report(all_true, all_preds, target_names=['Non-Planet', 'Planet']))
    print(f"Final AUC: {roc_auc_score(all_true, all_preds):.4f}")
    print(f"Final Accuracy: {accuracy_score(all_true, all_preds):.4f}")
    print("✅ Training complete. Best model saved as 'best_model.pth'")
