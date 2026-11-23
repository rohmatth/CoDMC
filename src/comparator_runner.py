# src/comparator_runner.py — Hibah Kemdiktisaintek Part 3
"""
Latih 5 arsitektur eksternal (Abdelli, Soothar, Yang, Healy, Titouni) dan
5 baseline konvensional (LogisticRegression, RandomForest, SVM, CNN polos,
BiLSTM polos), di KEEMPAT dataset (SOR, OTDR, HardFailure, SoftFailure),
protokol IDENTIK dengan HDMC (n_folds=10, n_repeats=1, config.py
N_FOLDS_FINAL/N_REPEATS_FINAL), WAJIB supaya uji Wilcoxon berpasangan
(nb03/significance_tests.py) valid.

Daftar isi:
  1. Baseline konvensional (classical ML) : CLASSICAL_BASELINES, run_classical_baseline
  2. Baseline deep + eksternal (PyTorch)  : run_deep_model
  3. Orkestrasi: 10 pembanding, satu dataset : run_all_comparators_one_dataset

BEDA DARI PART 2 (perbaikan penting): run_deep_model()/run_classical_baseline()
di Part 2 cuma menyimpan SATU model final (refit ulang di seluruh data,
bukan model dari fold manapun). Di Part 3, SETIAP fold menyimpan modelnya
sendiri (state_dict untuk deep, joblib untuk classical) ke
results/models/comparators/<dataset>/<model_name>/fold_XX.*, plus OOF-style
prediksi val per fold ke results/predictions/. Model final (refit di
seluruh data) TETAP disimpan terpisah sebagai model "deployment-ready",
bukan pengganti model per fold.
"""
import os
import time
import json
import numpy as np
import torch
import joblib
import torch.nn as nn

from src.config import SEED, TAB_DIR, MDL_DIR, PRED_DIR, N_FOLDS_FINAL, N_REPEATS_FINAL
from src.hdmc_multiview import load_dataset
from src.dataset_loaders import get_repeated_folds, get_repeated_folds_grouped, scale_fold
from src.training_engine import set_seed, get_device, train_model, predict_proba, inference_timing_sweep, fold_seed
from src.metrics import compute_metrics, aggregate_fold_metrics
from src.comparator_architectures import EXTERNAL_REGISTRY, COMPARATOR_REGISTRY
from src.resampling_and_loss import apply_resampling
from src.dataset_loaders import get_repeated_folds_grouped, get_repeated_folds, scale_fold
import joblib
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import torch


# ══════════════════════════════════════════════════════════════════════════
# 1. BASELINE KONVENSIONAL (CLASSICAL ML)
# ══════════════════════════════════════════════════════════════════════════

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

CLASSICAL_BASELINES = {
    "Baseline-LogReg": lambda seed: LogisticRegression(max_iter=2000, random_state=seed),
    "Baseline-RandomForest": lambda seed: RandomForestClassifier(
        n_estimators=200, random_state=seed, n_jobs=-1),
    # PERBAIKAN: SVC(kernel="linear") TETAP pakai algoritma libsvm (SMO),
    # kompleksitas O(n^2)-O(n^3) TERLEPAS dari kernel yang dipilih, ganti ke
    # "linear" TIDAK menghindari masalah itu (klaim lama di sini salah).
    # LinearSVC pakai algoritma liblinear (~O(n)), jauh lebih layak untuk
    # dataset besar seperti SOR (~100 ribu baris). LinearSVC tidak native
    # punya predict_proba, dibungkus CalibratedClassifierCV supaya tetap
    # kompatibel dengan seluruh pipeline (butuh proba untuk metrik/plot).
    "Baseline-SVM": lambda seed: CalibratedClassifierCV(
        LinearSVC(random_state=seed, dual="auto", max_iter=2000), cv=3),
}


def run_classical_baseline(dataset_name, model_name, X, y, class_names, groups=None,
                            n_folds=N_FOLDS_FINAL, n_repeats=N_REPEATS_FINAL, seed=SEED, verbose=True):
    """
    CV untuk evaluasi LENGKAP (accuracy, macro-F1, bacc, mcc, precision/
    recall/F1 per kelas) + waktu training/inferensi, model SETIAP FOLD
    disimpan (joblib + scaler), plus training final di SELURUH data
    disimpan terpisah sebagai model deployment-ready.
    """
    if groups is not None:
        folds = get_repeated_folds_grouped(X, y, groups, n_splits=n_folds,
                                            n_repeats=n_repeats, seed=seed)
    else:
        folds = get_repeated_folds(X, y, n_splits=n_folds, n_repeats=n_repeats, seed=seed)
    folds = list(folds)

    model_dir = os.path.join(MDL_DIR, "comparators", dataset_name, model_name)
    pred_dir = os.path.join(PRED_DIR, dataset_name, model_name)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    fold_metrics, timing_list = [], []
    for split_idx, (train_idx, val_idx) in enumerate(folds):
        run_seed = fold_seed(0, split_idx, base_seed=seed)
        X_tr, X_va, scaler = scale_fold(X[train_idx], X[val_idx])
        y_tr, y_va = y[train_idx], y[val_idx]
        model = CLASSICAL_BASELINES[model_name](run_seed)

        t0 = time.perf_counter()
        model.fit(X_tr, y_tr)
        t_train_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        y_pred = model.predict(X_va)
        y_proba = model.predict_proba(X_va)
        t_infer_s = time.perf_counter() - t0

        metrics = compute_metrics(y_va, y_pred, class_names)
        fold_metrics.append(metrics)
        timing = {
            "t_train_s": t_train_s,
            "t_infer_total_s": t_infer_s,
            "t_infer_per_sample_ms": t_infer_s / max(len(X_va), 1) * 1000,
        }
        timing_list.append(timing)

        joblib.dump(model, os.path.join(model_dir, f"fold_{split_idx:02d}.joblib"))
        joblib.dump(scaler, os.path.join(model_dir, f"fold_{split_idx:02d}_scaler.joblib"))
        np.save(os.path.join(pred_dir, f"fold_{split_idx:02d}_val_proba.npy"), y_proba)
        np.save(os.path.join(pred_dir, f"fold_{split_idx:02d}_val_true.npy"), y_va)
        np.save(os.path.join(pred_dir, f"fold_{split_idx:02d}_val_idx.npy"), val_idx)
        with open(os.path.join(model_dir, f"fold_{split_idx:02d}_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        if verbose:
            print(f"  [{dataset_name}/{model_name}] fold {split_idx+1}/{len(folds)} "
                  f"macro_f1={metrics['macro_f1']:.4f} bacc={metrics['bacc']:.4f} "
                  f"mcc={metrics['mcc']:.4f} t_train={t_train_s:.2f}s")

    agg = aggregate_fold_metrics(fold_metrics, timing_list=timing_list)

    # Training FINAL di seluruh data, disimpan TERPISAH sebagai deployment-ready
    X_full_scaled, _, scaler_full = scale_fold(X, X[:1])
    model_final = CLASSICAL_BASELINES[model_name](seed)
    model_final.fit(X_full_scaled, y)
    joblib.dump(model_final, os.path.join(model_dir, "final_full.joblib"))
    joblib.dump(scaler_full, os.path.join(model_dir, "final_full_scaler.joblib"))
    if verbose:
        print(f"  [{dataset_name}/{model_name}] {len(folds)} model per fold + 1 model final "
              f"tersimpan di {model_dir}")

    agg["dataset"] = dataset_name
    agg["model_name"] = model_name
    agg["class_names"] = class_names
    agg["accuracy_per_fold"] = [m["accuracy"] for m in fold_metrics]
    agg["macro_f1_per_fold"] = [m["macro_f1"] for m in fold_metrics]
    agg["bacc_per_fold"] = [m["bacc"] for m in fold_metrics]
    agg["mcc_per_fold"] = [m["mcc"] for m in fold_metrics]
    return agg


# ══════════════════════════════════════════════════════════════════════════
# 2. BASELINE DEEP + EKSTERNAL (PYTORCH)
# ══════════════════════════════════════════════════════════════════════════

def run_deep_model(dataset_name, model_name, model_registry, X, y, class_names,
                    groups=None, n_folds=N_FOLDS_FINAL, n_repeats=N_REPEATS_FINAL,
                    seed=SEED, verbose=True, max_epochs=None, patience=None,
                    loss_type="CE", resampler="None"):
    """
    CV untuk evaluasi LENGKAP + waktu training/inferensi, model SETIAP FOLD
    disimpan (state_dict), plus training final di seluruh data (90/10
    internal split cuma untuk early stopping) disimpan terpisah sebagai
    model deployment-ready dengan inference timing sweep lengkap.

    max_epochs/patience: default None -> pakai MAX_EPOCHS/PATIENCE penuh
    (rigor protokol final). HANYA di-override untuk smoke test.
    """
    n_classes = len(class_names)

    if groups is not None:
        folds = get_repeated_folds_grouped(X, y, groups, n_splits=n_folds,
                                            n_repeats=n_repeats, seed=seed)
    else:
        folds = get_repeated_folds(X, y, n_splits=n_folds, n_repeats=n_repeats, seed=seed)
    folds = list(folds)

    model_dir = os.path.join(MDL_DIR, "comparators", dataset_name, model_name)
    pred_dir = os.path.join(PRED_DIR, dataset_name, model_name)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    fold_metrics, timing_list = [], []
    for split_idx, (train_idx, val_idx) in enumerate(folds):
        run_seed = fold_seed(0, split_idx, base_seed=seed)
        X_tr_raw, X_va_raw = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]
        X_tr, X_va, _ = scale_fold(X_tr_raw, X_va_raw)
        X_tr, y_tr = apply_resampling(X_tr, y_tr, method=resampler, seed=run_seed, n_classes=n_classes)

        model_cls = model_registry[model_name]
        model = model_cls(X.shape[1], n_classes)

        train_kwargs = {}
        if max_epochs is not None:
            train_kwargs["max_epochs"] = max_epochs
        if patience is not None:
            train_kwargs["patience"] = patience

        model, va_f1, va_preds, va_trues, va_probas, history, t_train_s = train_model(
            model, X_tr, y_tr, X_va, y_va, loss_type=loss_type, n_classes=n_classes,
            seed=run_seed, verbose=False, **train_kwargs,
        )
        metrics = compute_metrics(va_trues, va_preds, class_names)
        fold_metrics.append(metrics)
        timing = {"t_train_s": t_train_s}
        timing.update(inference_timing_sweep(model, X_va))
        timing_list.append(timing)

        torch.save(model.state_dict(), os.path.join(model_dir, f"fold_{split_idx:02d}.pt"))
        np.save(os.path.join(pred_dir, f"fold_{split_idx:02d}_val_proba.npy"), va_probas)
        np.save(os.path.join(pred_dir, f"fold_{split_idx:02d}_val_true.npy"), va_trues)
        np.save(os.path.join(pred_dir, f"fold_{split_idx:02d}_val_idx.npy"), val_idx)
        with open(os.path.join(model_dir, f"fold_{split_idx:02d}_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        with open(os.path.join(model_dir, f"fold_{split_idx:02d}_history.json"), "w") as f:
            json.dump(history, f)

        if verbose:
            print(f"  [{dataset_name}/{model_name}] fold {split_idx+1}/{len(folds)} "
                  f"macro_f1={metrics['macro_f1']:.4f} bacc={metrics['bacc']:.4f} "
                  f"mcc={metrics['mcc']:.4f} t_train={t_train_s:.1f}s")

    agg = aggregate_fold_metrics(fold_metrics, timing_list=timing_list)

    # Training FINAL di seluruh data (90/10 internal split cuma untuk early stopping)
    set_seed(seed)
    n = len(y)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    cut = int(n * 0.9)
    tr_idx, va_idx = perm[:cut], perm[cut:]
    X_tr_raw, X_va_raw = X[tr_idx], X[va_idx]
    y_tr_raw, y_va_raw = y[tr_idx], y[va_idx]
    X_tr, X_va, _ = scale_fold(X_tr_raw, X_va_raw)
    X_tr, y_tr = apply_resampling(X_tr, y_tr_raw, method=resampler, seed=seed, n_classes=n_classes)

    model_cls = model_registry[model_name]
    model_final = model_cls(X.shape[1], n_classes)
    model_final, _, _, _, _, _, _ = train_model(
        model_final, X_tr, y_tr, X_va, y_va_raw,
        loss_type=loss_type, n_classes=n_classes, seed=seed, verbose=False,
    )
    timing_final = inference_timing_sweep(model_final, X_va, batch_sizes=[1, 32, 256])
    agg.update(timing_final)
    torch.save(model_final.state_dict(), os.path.join(model_dir, "final_full.pt"))
    if verbose:
        print(f"  [{dataset_name}/{model_name}] {len(folds)} model per fold + 1 model final "
              f"tersimpan di {model_dir}")
        print(f"  [{dataset_name}/{model_name}] inference (bs=1): "
              f"{timing_final.get('t_infer_bs1_per_sample_ms', float('nan')):.4f} ms/sample")

    agg["dataset"] = dataset_name
    agg["model_name"] = model_name
    agg["class_names"] = class_names
    agg["accuracy_per_fold"] = [m["accuracy"] for m in fold_metrics]
    agg["macro_f1_per_fold"] = [m["macro_f1"] for m in fold_metrics]
    agg["bacc_per_fold"] = [m["bacc"] for m in fold_metrics]
    agg["mcc_per_fold"] = [m["mcc"] for m in fold_metrics]
    return agg


# ══════════════════════════════════════════════════════════════════════════
# 3. ORKESTRASI: 10 PEMBANDING, SATU DATASET
# ══════════════════════════════════════════════════════════════════════════

DEEP_BASELINE_NAMES = ["Baseline-CNNOnly", "Baseline-BiLSTMOnly"]
EXTERNAL_NAMES = ["Abdelli-BiGRU", "Soothar-CNNLSTM", "Yang-IMCBiLSTM", "Healy-GRUClassifier"]


def run_all_comparators_one_dataset(dataset_name, n_folds=N_FOLDS_FINAL, n_repeats=N_REPEATS_FINAL,
                                     seed=SEED, verbose=True, deep_baseline_names=None):
    """
    Jalankan 5 eksternal + 5 baseline di SATU dataset, protokol final
    (n_folds=10, n_repeats=1 default), IDENTIK dengan protokol HDMC.
    Hasil ringkasan disimpan ke results/tables/comparator_<dataset>.json,
    model+prediksi per fold tersimpan penuh (lihat run_deep_model/
    run_classical_baseline).

    deep_baseline_names: default None -> pakai DEEP_BASELINE_NAMES
    (["Baseline-CNNOnly", "Baseline-BiLSTMOnly"]). Override eksplisit
    (mis. ["Baseline-CNNOnly", "Baseline-BiGRUOnly"]) untuk dataset yang
    baseline rekurennya ingin disesuaikan dengan backbone HDMC saat ini
    (CNN-BiGRU), tanpa mengubah default global.
    """
    X, y, class_names, groups = load_dataset(dataset_name)
    hasil = {}
    deep_names = deep_baseline_names if deep_baseline_names is not None else DEEP_BASELINE_NAMES

    if verbose:
        print(f"\n{'='*70}\n5 EKSTERNAL + 5 BASELINE, dataset={dataset_name}\n"
              f"n_folds={n_folds}, n_repeats={n_repeats} (protokol final, sama dengan HDMC)\n{'='*70}")

    external_configs = {
        "Abdelli-BiGRU": {"loss_type": "CE", "resampler": "None"},
        "Soothar-CNNLSTM": {"loss_type": "CE", "resampler": "None"},
        "Yang-IMCBiLSTM": {"loss_type": "CE", "resampler": "None"},
        "Healy-GRUClassifier": {"loss_type": "CE", "resampler": "cVAE"}
    }

    for name in EXTERNAL_NAMES:
        config = external_configs[name]
        hasil[name] = run_deep_model(
            dataset_name, name, EXTERNAL_REGISTRY, X, y, class_names,
            groups=groups, n_folds=n_folds, n_repeats=n_repeats, seed=seed, verbose=verbose,
            loss_type=config["loss_type"], resampler=config["resampler"]
        )

    # Titouni: Meta-Ensemble (CNN -> XGBoost, SVM, RF -> Bagged XGBoost)
    hasil["Titouni-CNNExtractor"] = run_titouni_ensemble(
        dataset_name, "Titouni-CNNExtractor", EXTERNAL_REGISTRY["Titouni-CNNEnsemble"],
        X, y, class_names, groups=groups, n_folds=n_folds, n_repeats=n_repeats,
        seed=seed, verbose=verbose
    )

    # --- 2 baseline deep sederhana ---
    for name in deep_names:
        hasil[name] = run_deep_model(
            dataset_name, name, COMPARATOR_REGISTRY, X, y, class_names,
            groups=groups, n_folds=n_folds, n_repeats=n_repeats, seed=seed, verbose=verbose,
        )

    # --- 3 baseline klasik ---
    for name in CLASSICAL_BASELINES:
        hasil[name] = run_classical_baseline(
            dataset_name, name, X, y, class_names, groups=groups,
            n_folds=n_folds, n_repeats=n_repeats, seed=seed, verbose=verbose,
        )

    out_path = os.path.join(TAB_DIR, f"comparator_{dataset_name}.json")
    with open(out_path, "w") as f:
        json.dump(hasil, f, indent=2, default=str)
    if verbose:
        print(f"\n[{dataset_name}] SEMUA 10 PEMBANDING SELESAI, ringkasan: {out_path}")
        print(f"[{dataset_name}] Model+prediksi per fold tersimpan di "
              f"{os.path.join(MDL_DIR, 'comparators', dataset_name)} dan "
              f"{os.path.join(PRED_DIR, dataset_name)}")

    return hasil

# ══════════════════════════════════════════════════════════════════════════
# 4. TITOUNI META-ENSEMBLE
# ══════════════════════════════════════════════════════════════════════════

def run_titouni_ensemble(dataset_name, model_name, model_cls, X, y, class_names,
                         groups=None, n_folds=N_FOLDS_FINAL, n_repeats=N_REPEATS_FINAL,
                         seed=SEED, verbose=True):
    n_classes = len(class_names)

    if groups is not None:
        folds = get_repeated_folds_grouped(X, y, groups, n_splits=n_folds, n_repeats=n_repeats, seed=seed)
    else:
        folds = get_repeated_folds(X, y, n_splits=n_folds, n_repeats=n_repeats, seed=seed)
    folds = list(folds)

    model_dir = os.path.join(MDL_DIR, "comparators", dataset_name, model_name)
    pred_dir = os.path.join(PRED_DIR, dataset_name, model_name)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    fold_metrics, timing_list = [], []
    device = get_device()

    for split_idx, (train_idx, val_idx) in enumerate(folds):
        run_seed = fold_seed(0, split_idx, base_seed=seed)
        X_tr_raw, X_va_raw = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]
        X_tr, X_va, scaler = scale_fold(X_tr_raw, X_va_raw)

        # 1. Train CNN with WCE
        # max_epochs=5: Section IV-D paper Titouni ("Training was limited to
        # 5 epochs based on early experiments that showed rapid convergence").
        # Section otoritatif yang sama dipakai untuk arsitektur FC 128-64-32.
        cnn = model_cls(X.shape[1], n_classes).to(device)
        t0 = time.perf_counter()
        cnn, _, _, _, _, _, t_train_cnn = train_model(
            cnn, X_tr, y_tr, X_va, y_va, loss_type="WCE", n_classes=n_classes,
            seed=run_seed, verbose=False, max_epochs=5
        )

        # 2. Extract Features
        cnn.eval()
        with torch.no_grad():
            X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(device)
            X_va_t = torch.tensor(X_va, dtype=torch.float32).to(device)
            F_tr = cnn.forward_features(X_tr_t).cpu().numpy()
            F_va = cnn.forward_features(X_va_t).cpu().numpy()

        # 3. Train Base Classifiers
        # Parameter sesuai Section IV-D paper Titouni (Model Configuration and
        # Hyperparameter Justification): XGBoost 150 tree max 10 split, SVM RBF
        # box constraint 0.5 (~C=0.5 di scikit-learn), RandomForest 75 tree.
        xgb = XGBClassifier(n_estimators=150, max_depth=10, random_state=run_seed, eval_metric='logloss')
        svm = SVC(kernel='rbf', C=0.5, probability=True, random_state=run_seed)
        rf = RandomForestClassifier(n_estimators=75, random_state=run_seed)

        xgb.fit(F_tr, y_tr)
        svm.fit(F_tr, y_tr)
        rf.fit(F_tr, y_tr)

        P_xgb_tr = xgb.predict_proba(F_tr)
        P_svm_tr = svm.predict_proba(F_tr)
        P_rf_tr = rf.predict_proba(F_tr)
        M_tr = np.hstack([P_xgb_tr, P_svm_tr, P_rf_tr])

        P_xgb_va = xgb.predict_proba(F_va)
        P_svm_va = svm.predict_proba(F_va)
        P_rf_va = rf.predict_proba(F_va)
        M_va = np.hstack([P_xgb_va, P_svm_va, P_rf_va])

        # 4. Train Meta-Classifier
        meta = XGBClassifier(n_estimators=200, max_depth=5, random_state=run_seed, eval_metric='logloss')
        meta.fit(M_tr, y_tr)

        t_train_total = time.perf_counter() - t0

        t0_infer = time.perf_counter()
        y_proba = meta.predict_proba(M_va)
        y_pred = meta.predict(M_va)
        t_infer_total = time.perf_counter() - t0_infer

        metrics = compute_metrics(y_va, y_pred, class_names)
        fold_metrics.append(metrics)
        timing = {
            "t_train_s": t_train_total,
            "t_infer_total_s": t_infer_total,
            "t_infer_per_sample_ms": t_infer_total / max(len(X_va), 1) * 1000,
        }
        timing_list.append(timing)

        # Save models
        torch.save(cnn.state_dict(), os.path.join(model_dir, f"fold_{split_idx:02d}_cnn.pt"))
        joblib.dump({"xgb": xgb, "svm": svm, "rf": rf, "meta": meta, "scaler": scaler}, 
                    os.path.join(model_dir, f"fold_{split_idx:02d}_ensemble.joblib"))
        
        np.save(os.path.join(pred_dir, f"fold_{split_idx:02d}_val_proba.npy"), y_proba)
        np.save(os.path.join(pred_dir, f"fold_{split_idx:02d}_val_true.npy"), y_va)
        np.save(os.path.join(pred_dir, f"fold_{split_idx:02d}_val_idx.npy"), val_idx)
        with open(os.path.join(model_dir, f"fold_{split_idx:02d}_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        if verbose:
            print(f"  [{dataset_name}/{model_name}] fold {split_idx+1}/{len(folds)} "
                  f"macro_f1={metrics['macro_f1']:.4f} bacc={metrics['bacc']:.4f} "
                  f"mcc={metrics['mcc']:.4f} t_train={t_train_total:.1f}s")

    # Training FINAL
    set_seed(seed)
    n = len(y)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    cut = int(n * 0.9)
    tr_idx, va_idx = perm[:cut], perm[cut:]
    X_tr_raw, X_va_raw = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    X_tr, X_va, scaler_final = scale_fold(X_tr_raw, X_va_raw)
    
    cnn_final = model_cls(X.shape[1], n_classes).to(device)
    cnn_final, _, _, _, _, _, _ = train_model(
        cnn_final, X_tr, y_tr, X_va, y_va, loss_type="WCE", n_classes=n_classes,
        seed=seed, verbose=False, max_epochs=5  # Section IV-D, sama seperti per-fold
    )
    
    cnn_final.eval()
    with torch.no_grad():
        X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(device)
        F_tr = cnn_final.forward_features(X_tr_t).cpu().numpy()

    # Parameter sama seperti per-fold (Section IV-D), lihat catatan di atas.
    xgb_final = XGBClassifier(n_estimators=150, max_depth=10, random_state=seed, eval_metric='logloss')
    svm_final = SVC(kernel='rbf', C=0.5, probability=True, random_state=seed)
    rf_final = RandomForestClassifier(n_estimators=75, random_state=seed)

    xgb_final.fit(F_tr, y_tr)
    svm_final.fit(F_tr, y_tr)
    rf_final.fit(F_tr, y_tr)

    P_xgb_tr = xgb_final.predict_proba(F_tr)
    P_svm_tr = svm_final.predict_proba(F_tr)
    P_rf_tr = rf_final.predict_proba(F_tr)
    M_tr = np.hstack([P_xgb_tr, P_svm_tr, P_rf_tr])

    meta_final = XGBClassifier(n_estimators=200, max_depth=5, random_state=seed, eval_metric='logloss')
    meta_final.fit(M_tr, y_tr)

    torch.save(cnn_final.state_dict(), os.path.join(model_dir, "final_full_cnn.pt"))
    joblib.dump({"xgb": xgb_final, "svm": svm_final, "rf": rf_final, "meta": meta_final, "scaler": scaler_final}, 
                os.path.join(model_dir, "final_full_ensemble.joblib"))

    agg = aggregate_fold_metrics(fold_metrics, timing_list=timing_list)
    agg["dataset"] = dataset_name
    agg["model_name"] = model_name
    agg["class_names"] = class_names
    agg["accuracy_per_fold"] = [m["accuracy"] for m in fold_metrics]
    agg["macro_f1_per_fold"] = [m["macro_f1"] for m in fold_metrics]
    agg["bacc_per_fold"] = [m["bacc"] for m in fold_metrics]
    agg["mcc_per_fold"] = [m["mcc"] for m in fold_metrics]
    return agg
