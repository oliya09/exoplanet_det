import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import pickle

# ⚙️ Params
N_SAMPLES = 10000
LENGTH = 800
EPOCHS = 30
BATCH_SIZE = 64
np.random.seed(42)

# Generators of syntatic lightcurves
def generate_planet_curve():
    x = np.linspace(0, 2 * np.pi, LENGTH)
    flux = 1.0 + np.random.normal(0, 0.0005, LENGTH)
    start = np.random.randint(100, LENGTH - 100)
    duration = np.random.randint(10, 50)
    depth = np.random.uniform(0.002, 0.02)
    flux[start:start + duration] -= depth
    flux += 0.001 * np.sin(5 * x)
    return flux

def generate_non_planet_curve():
    flux = 1.0 + np.random.normal(0, 0.001, LENGTH)
    drift = np.linspace(0, np.random.uniform(-0.002, 0.002), LENGTH)
    flux += drift
    if np.random.rand() < 0.3:
        idx = np.random.randint(100, LENGTH - 50)
        flux[idx:idx+10] += np.random.uniform(-0.01, 0.01)
    return flux

# list generation
X, y = [], []
for _ in range(N_SAMPLES // 2):
    X.append(generate_planet_curve()); y.append(1)
    X.append(generate_non_planet_curve()); y.append(0)

X = np.array(X).reshape(-1, LENGTH, 1)
y = np.array(y)
print(f"Dataset: {X.shape[0]} samples ({sum(y)} planets, {len(y)-sum(y)} non-planets)")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# CNN Model
cnn_model = Sequential([
    Input(shape=(LENGTH,1)),
    Conv1D(32, 7, activation='relu', kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.3),

    Conv1D(64, 5, activation='relu', kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.3),

    Conv1D(128, 3, activation='relu', kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.4),

    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=l2(1e-4)),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
    ModelCheckpoint('best_cnn_model.keras', save_best_only=True)
]

# Training
cnn_model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# Evaluation
cnn_loss, cnn_acc = cnn_model.evaluate(X_test, y_test, verbose=0)
y_pred_prob = cnn_model.predict(X_test).flatten()

threshold = 0.3
y_pred = (y_pred_prob > threshold).astype(int)

cnn_auc = roc_auc_score(y_test, y_pred_prob)
print(f"\nCNN: Accuracy={cnn_acc:.3f}, AUC={cnn_auc:.3f}, Threshold={threshold}")

# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    n_jobs=-1,
    random_state=42
)
rf_model.fit(X_train.reshape(len(X_train), -1), y_train)
rf_pred = rf_model.predict_proba(X_test.reshape(len(X_test), -1))[:, 1]
rf_auc = roc_auc_score(y_test, rf_pred)
print(f"RF: AUC={rf_auc:.3f}")

# Save models
cnn_model.save('cnn_model_final.keras')
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

print("\nClassification Report (CNN):")
print(classification_report(y_test, y_pred))
