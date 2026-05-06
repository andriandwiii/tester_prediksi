"""
=============================================================
  GrowSafe — Script Pelatihan Model Linear Regression
=============================================================
  Jalankan: python train_model.py
  Output  : regression_model.pkl
            scaler.pkl
            (keduanya tersimpan di folder models_ml/)
=============================================================
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ─────────────────────────────────────────────────────────────
# KONFIGURASI — ubah sesuai kebutuhan
# ─────────────────────────────────────────────────────────────
CSV_PATH    = "dataset_growsafe.csv"   # path ke dataset
OUTPUT_DIR  = "models_ml"              # folder output pkl
TEST_SIZE   = 0.2                      # 20% untuk testing
RANDOM_STATE= 42                       # seed agar hasil reproducible

FEATURE_COLS = [
    "suhu",
    "kelembaban",
    "total_led_menyala",
    "infected_area_percent",
]
TARGET_COL = "risiko_blackmold"


# ─────────────────────────────────────────────────────────────
# LANGKAH 0 — Persiapan folder output
# ─────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
print("=" * 60)
print("  GrowSafe — Pelatihan Model Linear Regression")
print("=" * 60)


# ─────────────────────────────────────────────────────────────
# LANGKAH 1 — Load Dataset
# ─────────────────────────────────────────────────────────────
print("\n[1/5] Memuat dataset...")

df = pd.read_csv(CSV_PATH)

print(f"      ✅ Dataset dimuat: {len(df)} baris, {len(df.columns)} kolom")
print(f"      Kolom: {df.columns.tolist()}")
print(f"\n      Statistik target ({TARGET_COL}):")
print(f"        Mean  : {df[TARGET_COL].mean():.4f}")
print(f"        Std   : {df[TARGET_COL].std():.4f}")
print(f"        Min   : {df[TARGET_COL].min():.4f}")
print(f"        Max   : {df[TARGET_COL].max():.4f}")

X = df[FEATURE_COLS]
y = df[TARGET_COL]

print(f"\n      Fitur (X): {FEATURE_COLS}")
print(f"      Target (y): {TARGET_COL}")


# ─────────────────────────────────────────────────────────────
# LANGKAH 2 — Train / Test Split (80 / 20)
# ─────────────────────────────────────────────────────────────
print("\n[2/5] Membagi data (train/test split)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

print(f"      ✅ Total data   : {len(df)} sampel")
print(f"         Training set : {len(X_train)} sampel ({int((1-TEST_SIZE)*100)}%)")
print(f"         Testing set  : {len(X_test)} sampel ({int(TEST_SIZE*100)}%)")
print(f"         random_state : {RANDOM_STATE}")


# ─────────────────────────────────────────────────────────────
# LANGKAH 3 — Fitting StandardScaler
#
# ⚠️  PENTING:
#   - scaler.fit_transform() hanya dipanggil pada X_TRAIN
#   - X_test hanya di-transform (BUKAN fit) agar tidak terjadi
#     data leakage (bocornya info test set ke proses training)
#   - Scaler menyimpan mean & std dari training set
# ─────────────────────────────────────────────────────────────
print("\n[3/5] Fitting StandardScaler & menyimpan scaler.pkl...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)  # ← fit + transform
X_test_scaled  = scaler.transform(X_test)        # ← transform saja (NO fit)

print("      ✅ Scaler berhasil di-fit pada data training")
print(f"\n      Parameter yang dipelajari scaler:")
print(f"      {'Fitur':<25} {'Mean (μ)':>12} {'Std Dev (σ)':>12}")
print(f"      {'-'*50}")
for i, col in enumerate(FEATURE_COLS):
    print(f"      {col:<25} {scaler.mean_[i]:>12.4f} {scaler.scale_[i]:>12.4f}")
print(f"\n      Jumlah sampel yang dilihat scaler: {int(scaler.n_samples_seen_)}")

# Simpan scaler ke pkl
scaler_path = os.path.join(OUTPUT_DIR, "scaler.pkl")
with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)
print(f"\n      💾 Tersimpan → {scaler_path}")


# ─────────────────────────────────────────────────────────────
# LANGKAH 4 — Melatih Linear Regression
# ─────────────────────────────────────────────────────────────
print("\n[4/5] Melatih model Linear Regression...")

model = LinearRegression()
model.fit(X_train_scaled, y_train)

print("      ✅ Model berhasil dilatih")
print(f"\n      Persamaan model yang dipelajari:")
print(f"      Intercept (β₀) : {model.intercept_:.6f}")
print(f"\n      {'Fitur':<25} {'Koefisien':>14} {'Interpretasi'}")
print(f"      {'-'*60}")
for i, col in enumerate(FEATURE_COLS):
    arah = "↑ risiko naik" if model.coef_[i] > 0 else "↓ risiko turun"
    print(f"      {col:<25} {model.coef_[i]:>+14.6f}   {arah}")

# Simpan model ke pkl
model_path = os.path.join(OUTPUT_DIR, "regression_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model, f)
print(f"\n      💾 Tersimpan → {model_path}")


# ─────────────────────────────────────────────────────────────
# LANGKAH 5 — Evaluasi Model
# ─────────────────────────────────────────────────────────────
print("\n[5/5] Evaluasi performa model...")

y_pred_test  = model.predict(X_test_scaled)
y_pred_train = model.predict(X_train_scaled)

# Clip ke [0, 100] seperti di prediction_service.py
y_pred_test_clipped = np.clip(y_pred_test, 0, 100)

r2_test   = r2_score(y_test,  y_pred_test)
r2_train  = r2_score(y_train, y_pred_train)
mse_test  = mean_squared_error(y_test, y_pred_test_clipped)
rmse_test = np.sqrt(mse_test)
mae_test  = mean_absolute_error(y_test, y_pred_test_clipped)

print(f"\n      ┌─────────────────────────────────────┐")
print(f"      │  Metrik Evaluasi                    │")
print(f"      ├─────────────────────────────────────┤")
print(f"      │  R² Test Set    : {r2_test:.4f}             │")
print(f"      │  R² Train Set   : {r2_train:.4f}             │")
print(f"      │  RMSE (test)    : {rmse_test:.4f}            │")
print(f"      │  MAE  (test)    : {mae_test:.4f}            │")
print(f"      │  MSE  (test)    : {mse_test:.4f}          │")
print(f"      └─────────────────────────────────────┘")

if r2_test >= 0.75:
    kualitas = "✅ Baik — model menangkap pola data dengan cukup akurat"
elif r2_test >= 0.5:
    kualitas = "⚠️  Sedang — model masih bisa ditingkatkan"
else:
    kualitas = "❌ Kurang — pertimbangkan fitur atau algoritma lain"
print(f"\n      Kualitas model: {kualitas}")

# Overfitting check
gap = r2_train - r2_test
if gap > 0.1:
    print(f"      ⚠️  Gap train-test = {gap:.4f} → kemungkinan overfitting")
else:
    print(f"      ✅ Gap train-test = {gap:.4f} → generalisasi baik")


# ─────────────────────────────────────────────────────────────
# RINGKASAN
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  SELESAI — File yang dihasilkan:")
print("=" * 60)
print(f"  📦 {model_path}")
print(f"       └─ model LinearRegression, {len(FEATURE_COLS)} fitur")
print(f"  ⚖️  {scaler_path}")
print(f"       └─ StandardScaler, difit dari {int(scaler.n_samples_seen_)} sampel training")
print(f"\n  Cara pakai di kode lain:")
print(f"  ─────────────────────────────────────")
print(f"  import pickle, numpy as np")
print(f"  with open('{model_path}','rb') as f: model  = pickle.load(f)")
print(f"  with open('{scaler_path}','rb') as f: scaler = pickle.load(f)")
print(f"")
print(f"  def predict(suhu, kelembaban, led, infeksi):")
print(f"      inp  = scaler.transform([[suhu, kelembaban, led, infeksi]])")
print(f"      risk = float(np.clip(model.predict(inp)[0], 0, 100))")
print(f"      kat  = 'Rendah' if risk<40 else ('Sedang' if risk<70 else 'Tinggi')")
print(f"      return risk, kat")
print("=" * 60)