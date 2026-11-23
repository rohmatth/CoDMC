# src/meta_learners.py — Hibah Kemdiktisaintek Part 3
"""
Meta-feature SEDERHANA: probabilitas OOF backbone (n_classes dim) +
confidence (margin top1-top2, entropy), TIDAK ada PHI atau local
complexity feature seperti di tesis.

Daftar isi:
  1. Meta-feature builder (view tunggal) : build_meta_features
  2. Kandidat meta-learner (justifikasi_meta) : META_LEARNER_NAMES, fit_meta_learner

DISIPLIN OOF: probabilitas yang dipakai sebagai meta-feature WAJIB berasal
dari prediksi out-of-fold (model tidak pernah melihat sampel itu saat
training), bukan dari model yang dievaluasi pada data trainingnya sendiri.

Lima kandidat meta-learner (empat aktif secara default + SVM arsip) diuji
SETARA lewat search budget yang sama (META_GRID_BUDGET di config.py),
tidak ada kandidat yang di-tuning lebih dalam dari kandidat lain.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from src.config import META_GRID_BUDGET, SEED

try:
    import xgboost as xgb
except ImportError:
    xgb = None
try:
    import lightgbm as lgb
except ImportError:
    lgb = None




# ══════════════════════════════════════════════════════════════════════════
# 1. META-FEATURE BUILDER (VIEW TUNGGAL)
# ══════════════════════════════════════════════════════════════════════════

def build_meta_features(oof_proba):
    """
    oof_proba: (N, n_classes) probabilitas OOF dari backbone terpilih.
    Return: (N, n_classes + 2) = proba + margin(top1-top2) + entropy.
    """
    sorted_proba = np.sort(oof_proba, axis=1)
    margin  = (sorted_proba[:, -1] - sorted_proba[:, -2]).reshape(-1, 1)
    entropy = -np.sum(oof_proba * np.log(oof_proba + 1e-12), axis=1, keepdims=True)
    return np.hstack([oof_proba, margin, entropy]).astype(np.float32)


# ── Tabel C: 5 kandidat meta-learner, search space SETARA ──────────────────────



# ══════════════════════════════════════════════════════════════════════════
# 2. KANDIDAT META-LEARNER (justifikasi_meta)
# ══════════════════════════════════════════════════════════════════════════

META_LEARNER_NAMES = ["LogisticRegression", "RandomForest", "XGBoost", "LightGBM"]
# "SVM" SENGAJA dikeluarkan dari default (keputusan lama, lihat memory project):
# O(n^2)-O(n^3), macet berjam-jam di dataset besar (SOR, ~100 ribu baris).
# Definisinya tetap ada di _PARAM_SPACES untuk arsip, panggil eksplisit lewat
# fit_meta_learner("SVM", ...) kalau benar-benar diperlukan di dataset kecil.

_PARAM_SPACES = {
    "LogisticRegression": {
        "estimator": lambda: LogisticRegression(max_iter=2000, random_state=SEED),
        "params": {"C": [0.01, 0.1, 1.0, 10.0, 100.0]},
    },
    "SVM": {
        "estimator": lambda: SVC(probability=True, random_state=SEED),
        "params": {"C": [0.1, 1.0, 10.0], "kernel": ["rbf", "linear"]},
    },
    "RandomForest": {
        "estimator": lambda: RandomForestClassifier(random_state=SEED, n_jobs=-1),
        "params": {"n_estimators": [100, 200, 400], "max_depth": [None, 5, 10, 20]},
    },
    "XGBoost": {
        "estimator": lambda: (xgb.XGBClassifier(random_state=SEED, eval_metric="mlogloss")
                               if xgb else None),
        "params": {"n_estimators": [100, 200, 400], "max_depth": [3, 5, 7],
                   "learning_rate": [0.01, 0.05, 0.1]},
    },
    "LightGBM": {
        "estimator": lambda: (lgb.LGBMClassifier(random_state=SEED, verbosity=-1)
                               if lgb else None),
        "params": {"n_estimators": [100, 200, 400], "max_depth": [-1, 5, 10],
                   "learning_rate": [0.01, 0.05, 0.1]},
    },
}


def fit_meta_learner(name, X_meta, y, seed=SEED, budget=META_GRID_BUDGET):
    """
    Fit satu kandidat meta-learner dengan RandomizedSearchCV, n_iter=budget
    SAMA untuk semua kandidat, cv internal kecil (3-fold) untuk memilih
    hyperparameter, terpisah dari OOF loop luar yang menghasilkan X_meta.
    """
    if name not in _PARAM_SPACES:
        raise ValueError(f"Meta-learner '{name}' tidak dikenal. Pilihan: {META_LEARNER_NAMES}")

    spec = _PARAM_SPACES[name]
    base_estimator = spec["estimator"]()
    if base_estimator is None:
        raise ImportError(f"Pustaka untuk '{name}' belum terinstall.")

    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    search = RandomizedSearchCV(
        base_estimator, spec["params"], n_iter=budget,
        cv=inner_cv, scoring="f1_macro", random_state=seed, n_jobs=-1,
    )
    search.fit(X_meta, y)
    return search.best_estimator_, search.best_params_
