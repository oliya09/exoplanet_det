# classificator.py
import numpy as np
import torch
import torch.nn as nn
import os
from typing import Dict, Any, Optional, Tuple
from .utils import odd_even_test, secondary_eclipse_test, create_transit_mask, centroid_check
from .catalog import get_rv_k, get_hostname_from_tic

CNN_MODEL_PATH = "exoplanet_det/best_model.pth"

# ----------------------- CNN model -----------------------
class MyCNN(nn.Module):
    def __init__(self, input_len: int = 1988):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * (input_len // 4), 64)  # <-- calculates automaticaly
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


# ----------------------- Classification class -----------------------
class CNNClassifier:
    def __init__(self, log_fn=print):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_len = 1988
        self.model = MyCNN(self.input_len).to(self.device)
        self.log = log_fn

        if os.path.exists(CNN_MODEL_PATH):
            try:
                state_dict = torch.load(CNN_MODEL_PATH, map_location=self.device)
                new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
                self.model.load_state_dict(new_state_dict, strict=False)
                self.model.eval()
                self.log(f"✅ CNN model successfully loaded: {CNN_MODEL_PATH}")
            except Exception as e:
                self.log(f"❌ Error loading model: {e}")
                self.model = None
        else:
            self.log("⚠️ Model file best_model.pth not found.")
            self.model = None

    # ----------------------- Predictions -----------------------
    def predict(self, flux: np.ndarray) -> float:
        if self.model is None:
            self.log("⚠️ Model not loaded, returning 0.5")
            return 0.5

        # preprocessing
        flux = np.nan_to_num(np.array(flux, dtype=np.float32), nan=1.0)
        flux = flux[:self.input_len]
        if len(flux) < self.input_len:
            flux = np.pad(flux, (0, self.input_len - len(flux)), mode="constant", constant_values=1.0)
        flux = (flux - np.mean(flux)) / (np.std(flux) + 1e-8)

        x = torch.tensor(flux, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = torch.sigmoid(self.model(x)).cpu().numpy().flatten()[0]
        return float(np.clip(pred, 0.0, 1.0))

    # ----------------------- Feature Extraction -----------------------
    def extract_features(self, lc: Any, period: float, t0: float, duration: float, depth: float) -> Tuple[np.ndarray, np.ndarray]:
        time, flux = lc.time.value, lc.flux.value
        flux = np.nan_to_num(flux, nan=1.0)
        if len(time) < 20:
            self.log("⚠️ Light curve too short.")
            return np.zeros(7), flux

        flux_rms = np.std(flux)
        phases = ((time - t0) % period) / period
        odd_mask = phases < 0.5
        even_mask = ~odd_mask
        odd_depth = 1 - np.median(flux[odd_mask]) if np.any(odd_mask) else depth
        even_depth = 1 - np.median(flux[even_mask]) if np.any(even_mask) else depth
        odd_even_diff = abs(odd_depth - even_depth) / depth if depth > 0 else 0

        sec_mask = np.abs(phases - 0.5) < (duration / period)
        secondary_depth = abs(1 - np.median(flux[sec_mask])) if np.any(sec_mask) else 0
        centroid_var = np.var(flux) / (np.mean(flux) + 1e-8)

        features = [depth, period, duration, flux_rms, odd_even_diff, secondary_depth, centroid_var]
        return np.array(features, dtype=float), flux

    # ----------------------- Classical Tests -----------------------
    def classical_scores(self, lc: Any, period: float, t0: float, duration: float, depth: float, target_id: str) -> Dict[str, Any]:
        time, flux = lc.time.value, lc.flux.value
        flux = np.nan_to_num(flux, nan=1.0)

        hostname = get_hostname_from_tic(target_id)
        rv_amp = get_rv_k(hostname) or 100.0

        transit_mask = create_transit_mask(time, period, t0, duration)
        odd_even_flag = odd_even_test(time, flux, period, t0, duration)
        sec_drop, prim_drop = secondary_eclipse_test(time, flux, period, t0, duration)
        centroid_flag = centroid_check(target_id, transit_mask)

        return {
            "odd_even": odd_even_flag,
            "odd_even_score": 1.0 if odd_even_flag == "consistent" else 0.0,
            "secondary": "weak" if sec_drop < 0.5 * prim_drop else "strong",
            "secondary_score": 1.0 if sec_drop < 0.5 * prim_drop else 0.0,
            "centroid": centroid_flag,
            "centroid_score": 1.0 if centroid_flag == "ok" else 0.0,
            "depth": "reasonable" if depth < 0.03 else "too_large",
            "depth_score": 1.0 if depth < 0.03 else 0.0,
            "rv": "ok" if rv_amp < 200 else "high",
            "rv_score": 1.0 if rv_amp < 200 else 0.0,
        }


# -----------------------  Main Classification Function -----------------------
def classify_target_full(target_id: str, lc: Any, period: float, t0: float, duration: float, depth: float,
                         mission: str = "TESS", model: Optional[CNNClassifier] = None) -> Dict[str, Any]:
    if lc is None or len(lc.flux) == 0 or np.isnan(lc.flux).all():
        return {"ID": target_id, "Status": "No Data", "Reason": "Empty LC", "Score": 0.0, "lc": None}
    if model is None:
        raise ValueError("CNN model not provided!")

    features, flux = model.extract_features(lc, period, t0, duration, depth)
    ml_score = model.predict(flux)
    checks = model.classical_scores(lc, period, t0, duration, depth, target_id)

    hybrid_score = (
        0.45 * ml_score +
        0.25 * checks["odd_even_score"] +
        0.20 * checks["secondary_score"] +
        0.10 * checks["depth_score"]
    )
    hybrid_score += np.random.normal(0, 0.02)
    hybrid_score = float(np.clip(hybrid_score, 0.0, 1.0))

    # Interpretation
    if hybrid_score >= 0.85:
        status, reason = "Confirmed Planet", "High confidence"
    elif hybrid_score >= 0.6:
        status, reason = "Candidate", "Moderate confidence"
    else:
        status, reason = "Likely False Positive", "Low confidence"

    explain = (
        f"ML={ml_score:.2f}; Odd/even={checks['odd_even']}; "
        f"Secondary={checks['secondary']}; Centroid={checks['centroid']}; "
        f"Depth={checks['depth']}. → Overall score={hybrid_score:.2f}"
    )

    return {
        "ID": target_id,
        "Status": status,
        "Period": float(period),
        "Depth": float(depth),
        "ML_score": round(ml_score, 2),
        "Hybrid_score": round(hybrid_score, 2),
        "Checks": checks,
        "Reason": reason,
        "Explain": explain,
        "lc": lc,
    }