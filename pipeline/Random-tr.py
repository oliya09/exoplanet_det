import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pickle

# Параметры
N_SAMPLES = 1000   # Количество синтетических кривых
LENGTH = 800       # Длина каждой кривой
np.random.seed(42)

# ✅ Генератор "планетных" сигналов (транзит)
def generate_planet_curve():
    x = np.linspace(0, 2 * np.pi, LENGTH)
    flux = 1.0 + np.random.normal(0, 0.0005, LENGTH)
    # Вырезаем кусок — "транзит"
    start = np.random.randint(100, LENGTH - 100)
    duration = np.random.randint(10, 50)
    depth = np.random.uniform(0.002, 0.02)
    flux[start:start + duration] -= depth
    # Добавим лёгкую синусоиду (периодическая модуляция)
    flux += 0.001 * np.sin(5 * x)
    return flux

# ✅ Генератор "не планет" — просто шум и случайные тренды
def generate_non_planet_curve():
    flux = 1.0 + np.random.normal(0, 0.001, LENGTH)
    # Добавим случайный дрейф
    drift = np.linspace(0, np.random.uniform(-0.002, 0.002), LENGTH)
    flux += drift
    # Иногда — одиночные шумовые всплески
    if np.random.rand() < 0.3:
        idx = np.random.randint(100, LENGTH - 50)
        flux[idx:idx+10] += np.random.uniform(-0.01, 0.01)
    return flux

# ✅ Генерация синтетического набора
X, y = [], []
for _ in range(N_SAMPLES // 2):
    X.append(generate_planet_curve()); y.append(1)
    X.append(generate_non_planet_curve()); y.append(0)

X = np.array(X).reshape(-1, LENGTH, 1)
y = np.array(y)
print(f"Dataset: {X.shape[0]} samples ({sum(y)} planets, {len(y)-sum(y)} non-planets)")

# ✅ Разделение
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ✅ CNN
cnn_model = Sequential([
    Conv1D(32, 5, activation='relu', input_shape=(LENGTH, 1)),
    MaxPooling1D(2),
    Conv1D(64, 5, activation='relu'),
    MaxPooling1D(2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

cnn_model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2,
              callbacks=[early_stop], verbose=1)

cnn_loss, cnn_acc = cnn_model.evaluate(X_test, y_test, verbose=0)
cnn_auc = roc_auc_score(y_test, cnn_model.predict(X_test).flatten())
print(f"CNN: Acc={cnn_acc:.3f}, AUC={cnn_auc:.3f}")

# ✅ RF
rf_model = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)
rf_model.fit(X_train.reshape(len(X_train), -1), y_train)
rf_pred = rf_model.predict_proba(X_test.reshape(len(X_test), -1))[:, 1]
rf_auc = roc_auc_score(y_test, rf_pred)
print(f"RF: AUC={rf_auc:.3f}")

# ✅ Сохранение моделей
cnn_model.save('cnn_model.h5')
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

print(classification_report(y_test, (cnn_model.predict(X_test) > 0.5).astype(int)))
