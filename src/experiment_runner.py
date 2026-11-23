# src/experiment_runner.py — Hibah Kemdiktisaintek Part 3
"""
run_candidate() adalah jalur eksekusi training+evaluasi dasar, dipakai oleh
justification_runner.py (justifikasi backbone/resampler/loss/meta) DAN
oleh hdmc_multiview.py (lewat generate_oof_proba, yang memakai training
loop yang sama). Tidak ada percabangan khusus per model atau per dataset,
perbedaan murni dari parameter yang di-pass.

Daftar isi:
  1. Util index fold                 : _fold_repeat_index
  2. Training + evaluasi satu kandidat, artifact PENUH per fold : run_candidate
  3. Checkpoint IO (CSV ringkasan)   : append_checkpoint, read_checkpoint_csv
  4. Full Model (scaffolding, belum dipakai) : train_and_save_full_model, load_full_model

PENYIMPANAN ARTIFACT PENUH (poin wajib Part 3): tiap fold-repeat
menyimpan predictions.npz (y_true/y_pred/y_proba), history.json (learning
curve), timing.json, metrics.json (metrik lengkap: accuracy/bacc/mcc/
macro-F1/per-kelas), config.json, dan model_state.pt (state_dict, BUKAN
objek model utuh, lihat backbone_architectures.py untuk definisi arsitektur
saat load ulang). Semua bisa dibuka kapan saja tanpa retrain (nb04/nb05).
"""
import os
import json
import time
import numpy as np
import pandas as pd
import torch

from src.config import RAW_DIR, TAB_DIR, N_FOLDS_JUSTIFIKASI, N_REPEATS_JUSTIFIKASI, SEED, TIMING_N_REPEATS
from src.dataset_loaders import get_repeated_folds, get_repeated_folds_grouped, scale_fold
from src.resampling_and_loss import apply_resampling
from src.backbone_architectures import get_model
from src.training_engine import train_model, fold_seed, inference_timing_sweep, predict_proba
from src.metrics import compute_metrics, aggregate_fold_metrics


# ══════════════════════════════════════════════════════════════════════════
# 1. UTIL INDEX FOLD
# ══════════════════════════════════════════════════════════════════════════

def _fold_repeat_index(n_folds, split_idx):
    """RepeatedStratifiedKFold/StratifiedGroupKFold-manual-repeat mengiterasi
    repeat di luar, fold di dalam, format index split_idx identik untuk
    keduanya karena get_repeated_folds_grouped() menyusun list dengan pola
    yang sama."""
    return split_idx // n_folds, split_idx % n_folds


# ══════════════════════════════════════════════════════════════════════════
# 2. TRAINING + EVALUASI SATU KANDIDAT, ARTIFACT PENUH PER FOLD
# ══════════════════════════════════════════════════════════════════════════

def run_candidate(dataset_name, stage_name, candidate_name, model_name, X, y, class_names,
                   resampler="None", loss_type="CE",
                   groups=None, n_folds=N_FOLDS_JUSTIFIKASI, n_repeats=N_REPEATS_JUSTIFIKASI,
                   seed=SEED, save_timing=True, verbose=True,
                   max_epochs=None, patience=None):
    """
    dataset_name  : "sor" / "otdr" / "hardfailure" / "softfailure", sesuai
                    src/config.py DATASETS, dipakai untuk namespace folder.
    stage_name    : "justifikasi_backbone" / "justifikasi_resampler" /
                    "justifikasi_loss" / "justifikasi_meta" / "justifikasi_pemasangan"
                    (dulu disebut "tabelA"/"tabelB1"/"tabelB2"/"tabelC" di Part 2).
    candidate_name: nama tampil kandidat, dipakai sebagai nama folder artifact.
    model_name    : key di BACKBONE_REGISTRY (backbone_architectures.py).
    resampler     : key di RESAMPLER_NAMES (resampling_and_loss.py).
    loss_type     : key di LOSS_NAMES (resampling_and_loss.py).
    groups        : WAJIB diisi (array id file asal tiap baris) untuk SOR,
                    None untuk dataset lain. Kalau diisi, CV pakai
                    get_repeated_folds_grouped() (anti-bocor level file).
    max_epochs/patience: default None -> pakai MAX_EPOCHS_JUSTIFIKASI/
                    PATIENCE_JUSTIFIKASI (budget ringan, fungsi ini HANYA
                    dipakai jalur justifikasi, tidak pernah protokol final).
    """
    from src.config import MAX_EPOCHS_JUSTIFIKASI, PATIENCE_JUSTIFIKASI
    max_epochs = max_epochs if max_epochs is not None else MAX_EPOCHS_JUSTIFIKASI
    patience = patience if patience is not None else PATIENCE_JUSTIFIKASI

    if groups is not None:
        folds = get_repeated_folds_grouped(X, y, groups, n_splits=n_folds,
                                            n_repeats=n_repeats, seed=seed)
    else:
        folds = get_repeated_folds(X, y, n_splits=n_folds, n_repeats=n_repeats, seed=seed)

    n_classes = len(class_names)

    fold_metrics = []
    timing_list = []
    out_root = os.path.join(RAW_DIR, dataset_name, stage_name, candidate_name)
    os.makedirs(out_root, exist_ok=True)

    for split_idx, (train_idx, val_idx) in enumerate(folds):
        r_idx, f_idx = _fold_repeat_index(n_folds, split_idx)
        run_seed = fold_seed(r_idx, f_idx, base_seed=seed)

        X_tr_raw, X_va_raw = X[train_idx], X[val_idx]
        y_tr_raw, y_va_raw = y[train_idx], y[val_idx]

        X_tr_sc, X_va_sc, _ = scale_fold(X_tr_raw, X_va_raw)
        X_tr_res, y_tr_res = apply_resampling(X_tr_sc, y_tr_raw, method=resampler, seed=run_seed)

        model = get_model(model_name, in_features=X.shape[1], n_classes=n_classes)
        model, va_f1, va_preds, va_trues, va_probas, history, t_train_s = train_model(
            model, X_tr_res, y_tr_res, X_va_sc, y_va_raw,
            loss_type=loss_type, n_classes=n_classes, seed=run_seed, verbose=False,
            max_epochs=max_epochs, patience=patience,
        )

        metrics = compute_metrics(va_trues, va_preds, class_names)
        fold_metrics.append(metrics)

        timing = {"t_train_s": t_train_s}
        if save_timing:
            timing.update(inference_timing_sweep(model, X_va_sc, n_repeats=TIMING_N_REPEATS))
        timing_list.append(timing)

        fold_dir = os.path.join(out_root, f"repeat{r_idx}_fold{f_idx}")
        os.makedirs(fold_dir, exist_ok=True)
        np.savez(os.path.join(fold_dir, "predictions.npz"),
                  y_true=va_trues, y_pred=va_preds, y_proba=va_probas)
        with open(os.path.join(fold_dir, "history.json"), "w") as f:
            json.dump(history, f)
        with open(os.path.join(fold_dir, "timing.json"), "w") as f:
            json.dump(timing, f)
        with open(os.path.join(fold_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        with open(os.path.join(fold_dir, "config.json"), "w") as f:
            json.dump({
                "dataset": dataset_name, "stage": stage_name, "candidate": candidate_name,
                "model_name": model_name, "resampler": resampler, "loss_type": loss_type,
                "repeat_idx": r_idx, "fold_idx": f_idx, "seed": run_seed,
                "grouped_cv": groups is not None,
            }, f, indent=2)
        torch.save(model.state_dict(), os.path.join(fold_dir, "model_state.pt"))

        if verbose:
            print(f"  [{dataset_name}/{candidate_name}] repeat{r_idx} fold{f_idx} "
                  f"acc={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} "
                  f"bacc={metrics['bacc']:.4f} mcc={metrics['mcc']:.4f} "
                  f"t_train={t_train_s:.1f}s")

    agg = aggregate_fold_metrics(fold_metrics, timing_list=timing_list)
    agg["dataset"] = dataset_name
    agg["candidate"] = candidate_name
    agg["model_name"] = model_name
    agg["resampler"] = resampler
    agg["loss_type"] = loss_type
    agg["class_names"] = class_names
    # per-fold metric lists, dipakai significance_tests.py (Friedman/Wilcoxon/Cliff's delta)
    agg["accuracy_per_fold"] = [m["accuracy"] for m in fold_metrics]
    agg["macro_f1_per_fold"] = [m["macro_f1"] for m in fold_metrics]
    agg["bacc_per_fold"] = [m["bacc"] for m in fold_metrics]
    agg["mcc_per_fold"] = [m["mcc"] for m in fold_metrics]

    return agg


# ══════════════════════════════════════════════════════════════════════════
# 3. CHECKPOINT IO (CSV RINGKASAN)
# ══════════════════════════════════════════════════════════════════════════

def append_checkpoint(dataset_name, stage_name, agg):
    """
    Simpan/append ringkasan agregat kandidat ke
    results/tables/<dataset_name>_checkpoint_<stage_name>.csv. Dipanggil
    TIAP kandidat selesai, bukan tunggu semua kandidat di stage yang sama,
    supaya progres tidak hilang kalau kernel terhenti di tengah jalan.

    Baris checkpoint memuat metrik LENGKAP: accuracy, bacc, mcc, macro-F1
    (mean+-std), F1/precision/recall per kelas, waktu training, dan waktu
    inference di semua batch size, supaya bisa dibaca langsung tanpa buka
    results/raw/.
    """
    path = os.path.join(TAB_DIR, f"{dataset_name}_checkpoint_{stage_name}.csv")

    row = {
        "dataset": dataset_name,
        "candidate": agg["candidate"],
        "model_name": agg["model_name"],
        "resampler": agg["resampler"],
        "loss_type": agg["loss_type"],
        "accuracy_mean": agg["accuracy_mean"], "accuracy_std": agg["accuracy_std"],
        "macro_f1_mean": agg["macro_f1_mean"], "macro_f1_std": agg["macro_f1_std"],
        "bacc_mean": agg["bacc_mean"], "bacc_std": agg["bacc_std"],
        "mcc_mean": agg["mcc_mean"], "mcc_std": agg["mcc_std"],
        "n_observations": agg["n_observations"],
    }

    for cname, vals in agg["per_class"].items():
        safe = cname.replace(" ", "_")
        row[f"precision_{safe}"] = vals["precision_mean"]
        row[f"recall_{safe}"] = vals["recall_mean"]
        row[f"f1_{safe}"] = vals["f1_mean"]

    if "timing" in agg:
        for k, v in agg["timing"].items():
            row[k] = v

    df_new = pd.DataFrame([row])
    if os.path.exists(path):
        df_old = read_checkpoint_csv(path)
        df_old = df_old[df_old["candidate"] != row["candidate"]]  # replace kalau re-run
        df_new = pd.concat([df_old, df_new], ignore_index=True)
    df_new.to_csv(path, index=False)
    return path


def read_checkpoint_csv(path):
    """
    Baca checkpoint CSV dengan aman. `resampler="None"` (baseline tanpa
    resampling) adalah nilai VALID di kolom `candidate`/`resampler`, tapi
    pandas.read_csv() secara default membaca string literal "None" sebagai
    NaN, yang akan merusak lookup nama kandidat. SELALU pakai fungsi ini,
    bukan pd.read_csv() langsung, untuk file checkpoint_*.csv di project ini.
    """
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


# ══════════════════════════════════════════════════════════════════════════
# 4. FULL MODEL (SCAFFOLDING, BELUM DIPAKAI DI PART 3)
# ══════════════════════════════════════════════════════════════════════════
# Latih backbone+meta PENUH di seluruh data satu dataset (bukan CV fold),
# untuk skenario generalisasi/transfer lintas dataset. BELUM dipanggil di
# notebook manapun Part 3 (fokus Part 3 adalah 4 dataset independen, bukan
# transfer OTDR->lainnya seperti Part 1), dipertahankan sebagai scaffolding
# kalau dibutuhkan untuk pembahasan jurnal nanti.

def train_and_save_full_model(dataset_name, model_name, resampler, loss_type, meta_learner_name,
                               X, y, class_names, meta_grid_budget, seed=SEED,
                               holdout_frac=0.1, verbose=True):
    """
    holdout_frac: sisihkan sebagian kecil data cuma untuk early stopping
    internal (pilih epoch terbaik), BUKAN untuk melaporkan performa dataset
    ini, performa resmi tetap dari protokol final (CV berulang).
    """
    import joblib
    from sklearn.model_selection import train_test_split
    from src.meta_learners import build_meta_features, fit_meta_learner
    from src.config import MDL_DIR

    n_classes = len(class_names)

    X_tr, X_ho, y_tr, y_ho = train_test_split(
        X, y, test_size=holdout_frac, stratify=y, random_state=seed)
    X_tr_sc, X_ho_sc, scaler = scale_fold(X_tr, X_ho)
    X_tr_res, y_tr_res = apply_resampling(X_tr_sc, y_tr, method=resampler, seed=seed)

    model = get_model(model_name, in_features=X.shape[1], n_classes=n_classes)
    model, ho_f1, _, _, _, history, t_train_s = train_model(
        model, X_tr_res, y_tr_res, X_ho_sc, y_ho, loss_type=loss_type,
        n_classes=n_classes, seed=seed, verbose=verbose,
    )
    if verbose:
        print(f"Full Model [{dataset_name}] backbone selesai. "
              f"Holdout internal F1={ho_f1:.4f} (bukan angka pelaporan resmi).")

    X_full_sc = scaler.transform(X).astype(np.float32)
    proba_full = predict_proba(model, X_full_sc)
    X_meta_full = build_meta_features(proba_full)
    meta_model, meta_params = fit_meta_learner(
        meta_learner_name, X_meta_full, y, seed=seed, budget=meta_grid_budget)

    backbone_dir = os.path.join(MDL_DIR, "backbones", dataset_name)
    meta_dir = os.path.join(MDL_DIR, "meta", dataset_name)
    os.makedirs(backbone_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(backbone_dir, "full_model_backbone.pt"))
    joblib.dump(scaler, os.path.join(backbone_dir, "full_model_scaler.pkl"))
    joblib.dump(meta_model, os.path.join(meta_dir, "full_model_meta.pkl"))

    manifest = {
        "dataset": dataset_name, "model_name": model_name, "resampler": resampler,
        "loss_type": loss_type, "meta_learner_name": meta_learner_name,
        "meta_params": meta_params, "in_features": X.shape[1], "n_classes": n_classes,
        "class_names": class_names, "holdout_f1_internal": float(ho_f1), "seed": seed,
    }
    with open(os.path.join(backbone_dir, "full_model_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    return model, scaler, meta_model, manifest


def load_full_model(dataset_name, model_name, in_features, n_classes):
    """Load Full Model yang sudah disimpan train_and_save_full_model()."""
    import joblib
    from src.config import MDL_DIR

    backbone_dir = os.path.join(MDL_DIR, "backbones", dataset_name)
    meta_dir = os.path.join(MDL_DIR, "meta", dataset_name)

    model = get_model(model_name, in_features=in_features, n_classes=n_classes)
    model.load_state_dict(torch.load(
        os.path.join(backbone_dir, "full_model_backbone.pt"), map_location="cpu"))
    scaler = joblib.load(os.path.join(backbone_dir, "full_model_scaler.pkl"))
    meta_model = joblib.load(os.path.join(meta_dir, "full_model_meta.pkl"))
    with open(os.path.join(backbone_dir, "full_model_manifest.json")) as f:
        manifest = json.load(f)
    return model, scaler, meta_model, manifest
