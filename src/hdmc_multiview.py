# src/hdmc_multiview.py — Hibah Kemdiktisaintek Part 3, inti metode HDMC
"""
Inti Hybrid Deep Meta-Classification (HDMC) via skema Multi-View
Meta-Stacking: backbone CNN-BiGRU dilatih 3x dengan treatment berbeda
(SMOTETomek+Focal, ADASYN+WCE, None+CE), OOF probability + margin + entropy
tiap treatment digabung dengan disagreement (L1, KL) antar-pasangan jadi
satu meta-feature, diklasifikasikan Logistic Regression sebagai
meta-classifier. Kombinasi ini TERKUNCI (bukan hasil pencarian ulang di
Part 3), lihat instruksi proyek dan hasil justifikasi Part 2.

Daftar isi:
  1. Meta-feature builder (view tunggal + N view) : _entropy, _margin,
     build_multiview_meta_features_n
  2. Diagnostik treatment pair                     : justifikasi_treatment_pair,
     justifikasi_pemasangan_treatment (eksperimen ringan pasangan resampler+loss)
  3. OOF generator, SATU treatment, model+prediksi tersimpan per fold :
     generate_oof_proba_single
  4. Pipeline HDMC penuh, model+prediksi+metrik tersimpan lengkap :
     run_hdmc_pipeline

BEDA DARI PART 2 (perbaikan penting): generate_oof_proba() dan
run_multiview_meta_n() di Part 2 SAMA SEKALI TIDAK menyimpan model, OOF
proba, atau metrik lengkap ke disk, cuma mengembalikan angka macro-F1
rata-rata di memori. Ini akar masalah "tidak ada artifact untuk plot"
yang dialami di Part 2. Di Part 3, SETIAP fold (backbone 3 treatment +
meta-learner) menyimpan model (state_dict/joblib), OOF proba, dan metrik
lengkap ke results/models/ dan results/predictions/, bisa dipakai ulang
kapan saja tanpa retrain (nb04/nb05).
"""
import os
import json
import time
import numpy as np
import torch
import joblib

from src.config import (N_FOLDS_FINAL, N_REPEATS_FINAL, SEED, MDL_DIR, PRED_DIR, TAB_DIR,
                         MAX_EPOCHS_JUSTIFIKASI, PATIENCE_JUSTIFIKASI)
from src.dataset_loaders import get_repeated_folds, get_repeated_folds_grouped, scale_fold, load_otdr, load_failure_wide, load_sor_windowed
from src.resampling_and_loss import apply_resampling
from src.backbone_architectures import get_model
from src.training_engine import train_model, fold_seed, predict_proba, inference_timing_sweep
from src.meta_learners import META_LEARNER_NAMES, fit_meta_learner
from src.metrics import compute_metrics, aggregate_fold_metrics

# Treatment HDMC TERKUNCI, tidak dicari ulang di Part 3.
HDMC_BACKBONE = "CNN-BiGRU"
HDMC_TREATMENTS = [("SMOTETomek", "Focal"), ("None", "CE")]
HDMC_META_LEARNER = "LogisticRegression"


def load_dataset(dataset_name, run_outlier_analysis_flag=True):
    """
    Loader tunggal untuk keempat dataset, dipakai di seluruh Part 3
    (dulu tersebar di beberapa modul part2_*.py). Return X, y, class_names,
    groups (groups None kecuali SOR).

    run_outlier_analysis_flag=True (default): jalankan deteksi outlier
    OTOMATIS (lihat data_quality.py), hasil dicatat ke CSV untuk provenance
    metodologi. TIDAK ADA baris yang dihapus di sini, cuma dianalisis dan
    dicatat, supaya kalau nanti diputuskan untuk ditindaklanjuti, datanya
    sudah siap tanpa perlu deteksi ulang. Data yang dikembalikan SELALU
    data ASLI (belum ada baris yang dibuang atas nama "pembersihan otomatis").
    """
    from src.config import DATASETS, OTDR_FILE, HARD_FILE, SOFT_FILE

    cfg = DATASETS[dataset_name]
    class_names = cfg["class_names"]

    if dataset_name == "otdr":
        X, y, _ = load_otdr(OTDR_FILE, class_names)
        groups = None
    elif dataset_name == "hardfailure":
        X, y, _ = load_failure_wide(HARD_FILE, class_names, name="HardFailure")
        groups = None
    elif dataset_name == "softfailure":
        X, y, _ = load_failure_wide(SOFT_FILE, class_names, name="SoftFailure")
        groups = None
    elif dataset_name == "sor":
        X, y, groups = load_sor_windowed(class_names)
    else:
        raise ValueError(f"Dataset '{dataset_name}' tidak dikenal.")

    if run_outlier_analysis_flag:
        from src.data_quality import run_outlier_analysis
        X, y, groups, _ = run_outlier_analysis(X, y, class_names, dataset_name, groups=groups, verbose=False)

    return X, y, class_names, groups


def compute_ir(y):
    counts = np.bincount(y)
    counts = counts[counts > 0]
    return float(counts.max() / counts.min())


# ══════════════════════════════════════════════════════════════════════════
# 1. META-FEATURE BUILDER (VIEW TUNGGAL + N VIEW)
# ══════════════════════════════════════════════════════════════════════════

def _entropy(proba, eps=1e-9):
    return -np.sum(proba * np.log(proba + eps), axis=1)


def _margin(proba):
    sorted_proba = np.sort(proba, axis=1)
    return sorted_proba[:, -1] - sorted_proba[:, -2]


def build_multiview_meta_features_n(oof_proba_list):
    """
    Gabungkan N OOF view jadi satu meta-feature. Kolom per view:
    [proba, margin, entropy], ditambah disagreement PER PASANGAN
    (L1 dan KL) untuk semua kombinasi 2 dari N view.
    """
    blocks = []
    for proba in oof_proba_list:
        margin, entropy = _margin(proba), _entropy(proba)
        blocks.append(np.concatenate([proba, margin[:, None], entropy[:, None]], axis=1))

    eps = 1e-9
    pair_blocks = []
    for i in range(len(oof_proba_list)):
        for j in range(i + 1, len(oof_proba_list)):
            pa, pb = oof_proba_list[i], oof_proba_list[j]
            l1 = np.sum(np.abs(pa - pb), axis=1)
            kl = np.sum(pa * np.log((pa + eps) / (pb + eps)), axis=1)
            pair_blocks.append(l1[:, None])
            pair_blocks.append(kl[:, None])

    return np.concatenate(blocks + pair_blocks, axis=1)


# ══════════════════════════════════════════════════════════════════════════
# 2. DIAGNOSTIK TREATMENT PAIR (justifikasi_treatment_pair)
# ══════════════════════════════════════════════════════════════════════════

def justifikasi_treatment_pair(dataset_name, backbone=HDMC_BACKBONE, treatments=None,
                                n_folds=3, seed=SEED, verbose=True):
    """
    Diagnostik MURAH (OOF cepat, n_folds=3, tanpa training meta-learner)
    untuk cek apakah 3 treatment HDMC (SMOTETomek+Focal, ADASYN+WCE,
    None+CE) benar-benar punya disagreement tinggi antar pasangan,
    menjustifikasi kenapa kombinasi ini dipilih untuk multi-view (bukan
    kombinasi acak atau kombinasi yang saling redundan).

    Hasil disimpan ke results/tables/<dataset>_justifikasi_treatment_pair.json
    """
    treatments = treatments or HDMC_TREATMENTS
    X, y, class_names, groups = load_dataset(dataset_name)

    if verbose:
        print(f"\n{'='*70}\n[JUSTIFIKASI TREATMENT PAIR] {dataset_name}, backbone={backbone}")
        print(f"Treatments: {treatments}\n{'='*70}")

    oof_list, labels = [], []
    for resampler, loss_type in treatments:
        if verbose:
            print(f"\n--- OOF {resampler}+{loss_type} ---")
        oof, _ = generate_oof_proba_single(
            dataset_name, backbone, resampler, loss_type, X, y, class_names,
            groups=groups, n_folds=n_folds, seed=seed, verbose=verbose,
            save_artifacts=False,  # diagnostik murah, tidak perlu simpan model
            max_epochs=MAX_EPOCHS_JUSTIFIKASI, patience=PATIENCE_JUSTIFIKASI,
        )
        oof_list.append(oof)
        labels.append(f"{resampler}+{loss_type}")

    eps = 1e-9
    pair_stats = {}
    for i in range(len(oof_list)):
        for j in range(i + 1, len(oof_list)):
            l1 = float(np.mean(np.sum(np.abs(oof_list[i] - oof_list[j]), axis=1)))
            kl = float(np.mean(np.sum(
                oof_list[i] * np.log((oof_list[i] + eps) / (oof_list[j] + eps)), axis=1)))
            pair_name = f"{labels[i]}  vs  {labels[j]}"
            pair_stats[pair_name] = {"mean_l1": l1, "mean_kl": kl}
            if verbose:
                print(f"  {pair_name}\n    mean L1 disagreement = {l1:.4f}\n    mean KL divergence   = {kl:.4f}")

    sorted_pairs = sorted(pair_stats.items(), key=lambda kv: kv[1]["mean_l1"], reverse=True)
    if verbose:
        print(f"\nUrutan disagreement (L1) TERBESAR ke TERKECIL:")
        for name, stats in sorted_pairs:
            print(f"  {stats['mean_l1']:.4f}  <-  {name}")

    out_path = os.path.join(TAB_DIR, f"{dataset_name}_justifikasi_treatment_pair.json")
    with open(out_path, "w") as f:
        json.dump(pair_stats, f, indent=2)
    if verbose:
        print(f"Tersimpan: {out_path}")

    return pair_stats


def justifikasi_pemasangan_treatment(dataset_name, backbone=HDMC_BACKBONE, n_folds=3,
                                      seed=SEED, verbose=True):
    """
    Eksperimen RINGAN (n_folds=3, protokol sama seperti justifikasi lain,
    murah) yang menguji LANGSUNG apakah pasangan resampler+loss yang
    TERKUNCI di HDMC_TREATMENTS (SMOTETomek+Focal, ADASYN+WCE) memang
    kombinasi TERBAIK, bukan cuma "komponennya bagus di axis terpisah".

    BEDA dari justifikasi_resampler/justifikasi_loss (justification_runner.py):
    axis-wise di sana MENGUNCI satu parameter lalu menguji parameter lain
    (resampler diuji dengan loss=CE, loss diuji dengan resampler pemenang
    axis sebelumnya), jadi TIDAK PERNAH benar-benar melatih kombinasi
    "SMOTETomek+Focal" atau "ADASYN+WCE" sebagai satu pasangan utuh. Fungsi
    ini melatih pasangannya LANGSUNG, dengan backbone dikunci CNN-BiGRU
    (sama seperti HDMC final), supaya klaimnya benar-benar berdasar
    kombinasi yang sama persis dipakai di HDMC.

    Cakupan SENGAJA dibatasi (bukan 20 kombinasi resampler x loss penuh,
    supaya tetap ringan): SMOTETomek x {CE, WCE, Focal} dan ADASYN x
    {CE, WCE, Focal}, 6 kombinasi total. "None" (baseline) TIDAK diuji
    ulang di sini karena None+CE adalah baseline definisional (tanpa
    intervensi sama sekali), tidak ada "pasangan alternatif" yang relevan
    untuk dibandingkan.

    PENTING: fungsi ini TIDAK PERNAH mengubah HDMC_TREATMENTS secara
    otomatis. Kalau hasil eksperimen menunjukkan pasangan lain lebih baik
    dari yang terkunci, itu DITANDAI JELAS sebagai peringatan di output
    dan di JSON (key "cocok_dengan_terkunci": False), tapi treatment yang
    benar-benar dipakai HDMC tetap HDMC_TREATMENTS sampai ada keputusan
    SADAR dan MANUAL untuk mengubahnya (mengubah treatment berarti seluruh
    protokol final 10-fold harus diulang, itu bukan keputusan yang pas
    diambil otomatis dari eksperimen 3-fold).

    Hasil disimpan ke results/tables/<dataset>_justifikasi_pemasangan.csv
    dan results/tables/<dataset>_justifikasi_pemasangan_ringkasan.json.
    """
    from src.experiment_runner import run_candidate, append_checkpoint

    X, y, class_names, groups = load_dataset(dataset_name)
    stage_name = "justifikasi_pemasangan"

    kandidat_losses = ["CE", "WCE", "Focal"]
    resampler_terkunci = {"SMOTETomek": "Focal", "ADASYN": "WCE"}

    if verbose:
        print(f"\n{'='*70}\n[JUSTIFIKASI PEMASANGAN TREATMENT] {dataset_name}\n"
              f"Backbone dikunci: {backbone}. Menguji SMOTETomek x 3 loss, ADASYN x 3 loss.\n{'='*70}")

    hasil = {}
    for resampler in ["SMOTETomek", "ADASYN"]:
        hasil[resampler] = {}
        for loss_type in kandidat_losses:
            candidate_name = f"{resampler}+{loss_type}"
            agg = run_candidate(
                dataset_name=dataset_name, stage_name=stage_name, candidate_name=candidate_name,
                model_name=backbone, resampler=resampler, loss_type=loss_type,
                X=X, y=y, class_names=class_names, groups=groups,
                n_folds=n_folds, n_repeats=1, seed=seed, verbose=False,
            )
            append_checkpoint(dataset_name, stage_name, agg)
            hasil[resampler][loss_type] = agg["macro_f1_mean"]
            if verbose:
                tanda = " <- TERKUNCI di HDMC" if resampler_terkunci.get(resampler) == loss_type else ""
                print(f"  {candidate_name:22s} macro_f1={agg['macro_f1_mean']:.4f}{tanda}")

    ringkasan = {}
    peringatan = []
    for resampler, skor_per_loss in hasil.items():
        loss_terkunci = resampler_terkunci[resampler]
        loss_pemenang_eksperimen = max(skor_per_loss, key=skor_per_loss.get)
        cocok = (loss_pemenang_eksperimen == loss_terkunci)
        ringkasan[resampler] = {
            "loss_terkunci": loss_terkunci,
            "skor_loss_terkunci": skor_per_loss[loss_terkunci],
            "loss_pemenang_eksperimen": loss_pemenang_eksperimen,
            "skor_pemenang_eksperimen": skor_per_loss[loss_pemenang_eksperimen],
            "cocok_dengan_terkunci": cocok,
            "semua_skor": skor_per_loss,
        }
        if not cocok:
            selisih = skor_per_loss[loss_pemenang_eksperimen] - skor_per_loss[loss_terkunci]
            peringatan.append(
                f"[{dataset_name}] {resampler}: eksperimen ringan menunjukkan {loss_pemenang_eksperimen} "
                f"(macro_f1={skor_per_loss[loss_pemenang_eksperimen]:.4f}) LEBIH BAIK dari {loss_terkunci} "
                f"yang terkunci (macro_f1={skor_per_loss[loss_terkunci]:.4f}, selisih={selisih:+.4f}). "
                f"HDMC_TREATMENTS TIDAK diubah otomatis, ini cuma peringatan untuk keputusan manual."
            )

    out_csv = os.path.join(TAB_DIR, f"{dataset_name}_justifikasi_pemasangan.csv")
    import pandas as pd
    rows = [{"resampler": r, "loss": l, "macro_f1_mean": s}
            for r, per_loss in hasil.items() for l, s in per_loss.items()]
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    out_json = os.path.join(TAB_DIR, f"{dataset_name}_justifikasi_pemasangan_ringkasan.json")
    with open(out_json, "w") as f:
        json.dump({"dataset": dataset_name, "backbone": backbone, "ringkasan": ringkasan,
                    "peringatan": peringatan}, f, indent=2)

    if verbose:
        print(f"\nTersimpan: {out_csv}")
        print(f"Tersimpan: {out_json}")
        if peringatan:
            print(f"\n[PERINGATAN] {len(peringatan)} pasangan terkunci TIDAK COCOK dengan hasil eksperimen ringan:")
            for p in peringatan:
                print(f"  - {p}")
        else:
            print(f"\n[OK] Semua pasangan terkunci COCOK dengan hasil eksperimen ringan "
                  f"(loss terbaik untuk tiap resampler memang loss yang sudah dipakai di HDMC).")

    return ringkasan


# ══════════════════════════════════════════════════════════════════════════
# 3. OOF GENERATOR, SATU TREATMENT, MODEL+PREDIKSI TERSIMPAN PER FOLD
# ══════════════════════════════════════════════════════════════════════════

def generate_oof_proba_single(dataset_name, backbone_name, resampler, loss_type,
                               X, y, class_names, groups=None, n_folds=N_FOLDS_FINAL,
                               seed=SEED, verbose=True, save_artifacts=True,
                               treatment_label=None, max_epochs=None, patience=None):
    """
    Bangun probabilitas OOF dari SATU backbone+resampler+loss. SATU sweep
    K-fold (n_repeats=1, disiplin OOF standar: tiap sampel diprediksi TEPAT
    SEKALI oleh model yang tidak pernah melihatnya saat training itu).

    save_artifacts=True (default): SETIAP fold menyimpan model backbone
    (state_dict) ke results/models/backbones/<dataset>/<treatment_label>/
    fold_XX.pt, dan OOF proba+y_true+val_idx ke
    results/predictions/<dataset>/<treatment_label>/fold_XX_*.npy. Set
    False untuk pemakaian diagnostik murah (justifikasi_treatment_pair)
    yang tidak perlu artifact tersimpan.

    treatment_label: nama folder artifact, default "<resampler>_<loss_type>".

    max_epochs/patience: default None -> pakai MAX_EPOCHS/PATIENCE penuh
    (fungsi ini dipakai run_hdmc_pipeline, PROTOKOL FINAL, harus rigor
    penuh). justifikasi_treatment_pair() secara eksplisit mem-pass budget
    lebih ringan (MAX_EPOCHS_JUSTIFIKASI/PATIENCE_JUSTIFIKASI) karena
    tujuannya cuma diagnostik cepat, bukan protokol final.

    Return: (oof_proba, fold_timing_list). fold_timing_list dipakai
    run_hdmc_pipeline() untuk mencatat biaya waktu training per fold.
    """
    treatment_label = treatment_label or f"{resampler}_{loss_type}"
    n_classes = len(class_names)
    if groups is not None:
        folds = get_repeated_folds_grouped(X, y, groups, n_splits=n_folds, n_repeats=1, seed=seed)
    else:
        folds = get_repeated_folds(X, y, n_splits=n_folds, n_repeats=1, seed=seed)

    oof_proba = np.zeros((len(y), n_classes), dtype=np.float32)
    covered = np.zeros(len(y), dtype=bool)
    fold_timing = []

    if save_artifacts:
        backbone_dir = os.path.join(MDL_DIR, "backbones", dataset_name, treatment_label)
        pred_dir = os.path.join(PRED_DIR, dataset_name, treatment_label)
        os.makedirs(backbone_dir, exist_ok=True)
        os.makedirs(pred_dir, exist_ok=True)

    train_kwargs = {}
    if max_epochs is not None:
        train_kwargs["max_epochs"] = max_epochs
    if patience is not None:
        train_kwargs["patience"] = patience

    for split_idx, (train_idx, val_idx) in enumerate(folds):
        run_seed = fold_seed(0, split_idx, base_seed=seed)
        X_tr_raw, X_va_raw = X[train_idx], X[val_idx]
        y_tr_raw, y_va_raw = y[train_idx], y[val_idx]

        X_tr_sc, X_va_sc, _ = scale_fold(X_tr_raw, X_va_raw)
        X_tr_res, y_tr_res = apply_resampling(X_tr_sc, y_tr_raw, method=resampler, seed=run_seed)

        model = get_model(backbone_name, in_features=X.shape[1], n_classes=n_classes)
        t0 = time.perf_counter()
        model, _, _, _, _, history, t_train_s = train_model(
            model, X_tr_res, y_tr_res, X_va_sc, y_va_raw,
            loss_type=loss_type, n_classes=n_classes, seed=run_seed, verbose=False,
            **train_kwargs,
        )
        proba_va = predict_proba(model, X_va_sc)
        oof_proba[val_idx] = proba_va
        covered[val_idx] = True
        timing_fold = {"fold": split_idx, "t_train_s": t_train_s}
        if save_artifacts:
            # Sweep inference PENUH (batch 1/32/256) cuma dihitung kalau
            # save_artifacts=True (protokol final), supaya diagnostik murah
            # (justifikasi_treatment_pair, save_artifacts=False) tetap cepat.
            timing_fold.update(inference_timing_sweep(model, X_va_sc))
        fold_timing.append(timing_fold)

        if save_artifacts:
            torch.save(model.state_dict(),
                       os.path.join(backbone_dir, f"fold_{split_idx:02d}.pt"))
            np.save(os.path.join(pred_dir, f"fold_{split_idx:02d}_val_proba.npy"), proba_va)
            np.save(os.path.join(pred_dir, f"fold_{split_idx:02d}_val_true.npy"), y_va_raw)
            np.save(os.path.join(pred_dir, f"fold_{split_idx:02d}_val_idx.npy"), val_idx)
            with open(os.path.join(backbone_dir, f"fold_{split_idx:02d}_history.json"), "w") as f:
                json.dump(history, f)

        if verbose:
            print(f"  [OOF {dataset_name}/{treatment_label}] fold {split_idx+1}/{len(folds)} "
                  f"selesai (t_train={t_train_s:.1f}s)")

    if not covered.all():
        missing = int((~covered).sum())
        print(f"[PERINGATAN] {missing} sampel tidak ter-cover OOF (bisa terjadi kalau "
              f"n_splits diturunkan otomatis akibat guard kelas langka di CV grouped), "
              f"diisi probabilitas seragam 1/n_classes supaya tidak crash di meta-feature.")
        oof_proba[~covered] = 1.0 / n_classes

    return oof_proba, fold_timing


# ══════════════════════════════════════════════════════════════════════════
# 4. PIPELINE HDMC PENUH, MODEL+PREDIKSI+METRIK TERSIMPAN LENGKAP
# ══════════════════════════════════════════════════════════════════════════

def run_hdmc_pipeline(dataset_name, backbone=HDMC_BACKBONE, treatments=None,
                       meta_learner_name=HDMC_META_LEARNER,
                       n_folds=N_FOLDS_FINAL, n_repeats=N_REPEATS_FINAL,
                       seed=SEED, verbose=True, max_epochs=None, patience=None):
    """
    Pipeline HDMC PENUH (Multi-View Meta-Stacking): 3 backbone (satu per
    treatment) -> gabung OOF jadi meta-feature -> Logistic Regression
    sebagai meta-classifier. Protokol final (n_folds=10, n_repeats=1,
    default dari config.py), SAMA untuk keempat dataset dan seluruh
    pembanding (baseline+eksternal), supaya uji Wilcoxon berpasangan valid.

    max_epochs/patience: default None -> pakai MAX_EPOCHS/PATIENCE penuh
    (rigor protokol final). HANYA di-override untuk kebutuhan smoke test
    (_PRAFLIGHT_smoke_test.ipynb), TIDAK PERNAH di-override di nb02 asli.

    Model backbone (3 treatment x n_folds), OOF proba, model meta-learner
    (n_folds), dan metrik lengkap (accuracy/bacc/mcc/macro-F1/per-kelas)
    SEMUA tersimpan penuh per fold, tidak cuma model final atau angka
    rata-rata. Bisa dipakai ulang di nb04 (plot) dan nb05 (ablasi
    komponen) tanpa retrain.

    Return: dict ringkasan (macro_f1_mean, per-fold metrics, timing),
    juga disimpan ke results/tables/<dataset>_hdmc_final_metrics.json
    """
    treatments = treatments or HDMC_TREATMENTS
    X, y, class_names, groups = load_dataset(dataset_name)
    n_classes = len(class_names)

    if verbose:
        print(f"\n{'='*70}\n[HDMC PIPELINE] {dataset_name}, backbone={backbone}")
        print(f"Treatments: {treatments}, meta={meta_learner_name}")
        print(f"n_folds={n_folds}, n_repeats={n_repeats}\n{'='*70}")

    # --- Tahap 1: OOF tiap treatment (n_repeats=1, disiplin OOF), model+OOF tersimpan ---
    oof_list, oof_timing = [], {}
    for resampler, loss_type in treatments:
        label = f"{resampler}_{loss_type}"
        if verbose:
            print(f"\n--- Backbone treatment: {label} ---")
        t0 = time.perf_counter()
        oof, fold_timing = generate_oof_proba_single(
            dataset_name, backbone, resampler, loss_type, X, y, class_names,
            groups=groups, n_folds=n_folds, seed=seed, verbose=verbose,
            save_artifacts=True, treatment_label=label,
            max_epochs=max_epochs, patience=patience,
        )
        elapsed = time.perf_counter() - t0

        # Agregasi timing per treatment (mean+-std lintas fold), termasuk
        # inference sweep (batch 1/32/256) yang sekarang tersimpan per fold.
        timing_keys = sorted({k for t in fold_timing for k in t.keys() if k != "fold"})
        timing_agg = {}
        for k in timing_keys:
            vals = np.array([t[k] for t in fold_timing if k in t])
            timing_agg[f"{k}_mean"] = float(vals.mean())
            timing_agg[f"{k}_std"] = float(vals.std())
        oof_timing[label] = {"total_s": elapsed, "per_fold": fold_timing, "agg": timing_agg}
        oof_list.append(oof)
        if verbose:
            print(f"  [waktu total {label}] {elapsed:.1f}s ({elapsed/60:.1f} menit)")

    X_meta = build_multiview_meta_features_n(oof_list)

    # --- Tahap 2: meta-learner, evaluasi CV (n_folds x n_repeats), model+prediksi tersimpan per fold ---
    if groups is not None:
        meta_folds = get_repeated_folds_grouped(X_meta, y, groups, n_splits=n_folds,
                                                 n_repeats=n_repeats, seed=seed)
    else:
        meta_folds = get_repeated_folds(X_meta, y, n_splits=n_folds, n_repeats=n_repeats, seed=seed)
    meta_folds = list(meta_folds)

    meta_dir = os.path.join(MDL_DIR, "meta", dataset_name)
    meta_pred_dir = os.path.join(PRED_DIR, dataset_name, "meta")
    os.makedirs(meta_dir, exist_ok=True)
    os.makedirs(meta_pred_dir, exist_ok=True)

    fold_metrics, timing_list = [], []
    for split_idx, (train_idx, val_idx) in enumerate(meta_folds):
        X_tr, X_va = X_meta[train_idx], X_meta[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        t0 = time.perf_counter()
        meta_model, meta_params = fit_meta_learner(meta_learner_name, X_tr, y_tr, seed=seed)
        t_fit_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_proba_va = meta_model.predict_proba(X_va)
        y_pred_va = np.argmax(y_proba_va, axis=1)
        t_predict_s = time.perf_counter() - t1

        metrics = compute_metrics(y_va, y_pred_va, class_names)
        fold_metrics.append(metrics)
        timing = {
            "t_train_s": t_fit_s,
            "t_infer_total_s": t_predict_s,
            "t_infer_per_sample_ms": t_predict_s / max(len(X_va), 1) * 1000,
        }
        timing_list.append(timing)

        joblib.dump(meta_model, os.path.join(meta_dir, f"fold_{split_idx:02d}.joblib"))
        np.save(os.path.join(meta_pred_dir, f"fold_{split_idx:02d}_val_proba.npy"), y_proba_va)
        np.save(os.path.join(meta_pred_dir, f"fold_{split_idx:02d}_val_true.npy"), y_va)
        np.save(os.path.join(meta_pred_dir, f"fold_{split_idx:02d}_val_pred.npy"), y_pred_va)
        with open(os.path.join(meta_dir, f"fold_{split_idx:02d}_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        if verbose:
            print(f"  [HDMC meta/{dataset_name}] fold {split_idx+1}/{len(meta_folds)} "
                  f"macro_f1={metrics['macro_f1']:.4f} bacc={metrics['bacc']:.4f} "
                  f"mcc={metrics['mcc']:.4f}")

    agg = aggregate_fold_metrics(fold_metrics, timing_list=timing_list)
    agg["dataset"] = dataset_name
    agg["backbone"] = backbone
    agg["treatments"] = treatments
    agg["meta_learner"] = meta_learner_name
    agg["class_names"] = class_names
    agg["accuracy_per_fold"] = [m["accuracy"] for m in fold_metrics]
    agg["macro_f1_per_fold"] = [m["macro_f1"] for m in fold_metrics]
    agg["bacc_per_fold"] = [m["bacc"] for m in fold_metrics]
    agg["mcc_per_fold"] = [m["mcc"] for m in fold_metrics]
    agg["oof_timing_per_treatment"] = oof_timing

    out_path = os.path.join(TAB_DIR, f"{dataset_name}_hdmc_final_metrics.json")
    with open(out_path, "w") as f:
        json.dump(agg, f, indent=2, default=str)
    if verbose:
        print(f"\n[{dataset_name}] HDMC selesai. macro_f1_mean={agg['macro_f1_mean']:.4f} "
              f"(std={agg['macro_f1_std']:.4f}, n_observations={agg['n_observations']})")
        print(f"Ringkasan tersimpan: {out_path}")

    return agg
