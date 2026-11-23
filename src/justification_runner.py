# src/justification_runner.py — Hibah Kemdiktisaintek Part 3
"""
Orkestrasi justifikasi desain HDMC: kenapa CNN-BiLSTM (backbone), kenapa
resampler+loss terpilih, kenapa Logistic Regression (meta-learner). Dulu
disebut "Tabel A/B1/B2/C" di Part 2, direname di sini supaya jelas ini
alasan desain, bukan klaim performa utama.

Daftar isi:
  1. Ablasi satu axis (backbone/resampler/loss) : run_justification_axis
  2. Ablasi meta-learner (di atas OOF meta-feature) : run_meta_candidate, run_justification_meta
  3. Orkestrasi penuh, satu dataset             : run_justification_pipeline
  4. Sintesis pemilihan 3 treatment (baca ulang, TANPA training baru) : sintesis_pemilihan_treatment
  5. Sintesis backbone lintas dataset (baca ulang, TANPA training baru) : sintesis_backbone_lintas_dataset
  6. Diagnostik overfitting backbone (baca ulang checkpoint, TANPA training baru) : analisis_overfitting_backbone

PROTOKOL: n_folds=3, n_repeats=1 (N_FOLDS_JUSTIFIKASI/N_REPEATS_JUSTIFIKASI
di config.py), SENGAJA ringkas karena ini alasan desain, bukan klaim utama
yang dibandingkan ke Titouni/eksternal lain. Hasil TETAP disimpan lengkap
(model per fold, prediksi, metrik) lewat run_candidate(), bisa dipakai
untuk plot pendukung di nb04 kalau diperlukan.

Pemenang tiap axis dipilih OTOMATIS (significance_tests.py::auto_select_winner,
mean macro-F1 tertinggi), uji statistik (Friedman+Wilcoxon+Cliff's delta)
tetap dihitung dan disimpan untuk pelaporan, tidak memblokir alur.
"""
import os
import json
import time
import numpy as np

from src.config import TAB_DIR, N_FOLDS_JUSTIFIKASI, N_REPEATS_JUSTIFIKASI, SEED, RAW_DIR
from src.dataset_loaders import get_repeated_folds, get_repeated_folds_grouped
from src.experiment_runner import run_candidate, append_checkpoint, read_checkpoint_csv, _fold_repeat_index
from src.training_engine import fold_seed
from src.metrics import compute_metrics, aggregate_fold_metrics
from src.significance_tests import auto_select_winner


# ══════════════════════════════════════════════════════════════════════════
# 1. ABLASI SATU AXIS (BACKBONE/RESAMPLER/LOSS)
# ══════════════════════════════════════════════════════════════════════════

def run_justification_axis(dataset_name, stage_name, candidates, X, y, class_names,
                            model_name, resampler, loss_type, axis,
                            groups=None, n_folds=N_FOLDS_JUSTIFIKASI,
                            n_repeats=N_REPEATS_JUSTIFIKASI, seed=SEED, verbose=True,
                            save_timing=True, max_epochs=None, patience=None):
    """
    Jalankan SATU axis justifikasi (backbone/resampler/loss): loop semua
    `candidates` di parameter yang ditentukan `axis` ("model_name"/
    "resampler"/"loss_type"), dua parameter lain DIKUNCI ke nilai yang
    di-pass. Otomatis pilih pemenang di akhir, simpan hasil uji statistik.

    Dipakai seragam:
      justifikasi_backbone : axis="model_name", candidates=list(BACKBONE_REGISTRY)
      justifikasi_resampler: axis="resampler",  candidates=RESAMPLER_NAMES
      justifikasi_loss     : axis="loss_type",  candidates=LOSS_NAMES

    save_timing=True (default): tiap kandidat-fold jalankan inference_timing_sweep
    (5 repeat x 3 batch size), berguna untuk laporan waktu tapi menambah waktu
    eksekusi cukup besar, apalagi backbone axis ganti arsitektur tiap kandidat
    (cuDNN benchmark ulang). Set False untuk smoke test (kecepatan diutamakan,
    data timing tidak dipakai di sana).

    max_epochs/patience: default None -> pakai MAX_EPOCHS_JUSTIFIKASI/
    PATIENCE_JUSTIFIKASI (budget ringan, resmi lewat run_candidate). Bisa
    di-override eksplisit (mis. MAX_EPOCHS/PATIENCE penuh) untuk pengujian
    konvergensi backbone tanpa terpotong budget ringan.

    Return: (winner, results_by_candidate)
    """
    results = {}
    for cand in candidates:
        kwargs = {"model_name": model_name, "resampler": resampler, "loss_type": loss_type}
        kwargs[axis] = cand
        agg = run_candidate(
            dataset_name=dataset_name, stage_name=stage_name, candidate_name=cand,
            X=X, y=y, class_names=class_names, groups=groups,
            n_folds=n_folds, n_repeats=n_repeats, seed=seed, verbose=verbose,
            save_timing=save_timing, max_epochs=max_epochs, patience=patience,
            **kwargs,
        )
        append_checkpoint(dataset_name, stage_name, agg)
        results[cand] = agg

    winner, mean_scores, friedman_result, wilcoxon_df = auto_select_winner(results, verbose=verbose)
    if wilcoxon_df is not None:
        stat_path = os.path.join(TAB_DIR, f"{dataset_name}_statistical_test_{stage_name}.csv")
        wilcoxon_df.to_csv(stat_path, index=False)
        if verbose:
            print(f"Uji statistik tersimpan: {stat_path}")

    return winner, results


# ══════════════════════════════════════════════════════════════════════════
# 2. ABLASI META-LEARNER (DI ATAS OOF META-FEATURE)
# ══════════════════════════════════════════════════════════════════════════

def run_meta_candidate(dataset_name, candidate_name, X_meta, y, class_names,
                        groups=None, n_folds=N_FOLDS_JUSTIFIKASI, n_repeats=N_REPEATS_JUSTIFIKASI,
                        seed=SEED, meta_grid_budget=None, verbose=True):
    """
    Setara run_candidate(), tapi untuk justifikasi_meta (meta-learner di
    atas meta-feature OOF single-view), bukan backbone neural network.
    Struktur fold, artifact per fold, dan checkpoint format IDENTIK dengan
    run_candidate().
    """
    import joblib
    from src.meta_learners import fit_meta_learner
    from src.config import META_GRID_BUDGET

    budget = meta_grid_budget or META_GRID_BUDGET
    stage_name = "justifikasi_meta"

    if groups is not None:
        folds = get_repeated_folds_grouped(X_meta, y, groups, n_splits=n_folds,
                                            n_repeats=n_repeats, seed=seed)
    else:
        folds = get_repeated_folds(X_meta, y, n_splits=n_folds, n_repeats=n_repeats, seed=seed)

    fold_metrics, timing_list = [], []
    out_root = os.path.join(RAW_DIR, dataset_name, stage_name, candidate_name)
    os.makedirs(out_root, exist_ok=True)

    for split_idx, (train_idx, val_idx) in enumerate(folds):
        r_idx, f_idx = _fold_repeat_index(n_folds, split_idx)
        run_seed = fold_seed(r_idx, f_idx, base_seed=seed)

        X_tr, X_va = X_meta[train_idx], X_meta[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        t0 = time.perf_counter()
        model, params = fit_meta_learner(candidate_name, X_tr, y_tr, seed=run_seed, budget=budget)
        t_train_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_pred = model.predict(X_va)
        t_infer_s = time.perf_counter() - t1

        metrics = compute_metrics(y_va, y_pred, class_names)
        fold_metrics.append(metrics)
        timing = {
            "t_train_s": t_train_s,
            "t_infer_total_s": t_infer_s,
            "t_infer_per_sample_ms": t_infer_s / max(len(X_va), 1) * 1000,
        }
        timing_list.append(timing)

        fold_dir = os.path.join(out_root, f"repeat{r_idx}_fold{f_idx}")
        os.makedirs(fold_dir, exist_ok=True)
        np.savez(os.path.join(fold_dir, "predictions.npz"), y_true=y_va, y_pred=y_pred)
        with open(os.path.join(fold_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        with open(os.path.join(fold_dir, "timing.json"), "w") as f:
            json.dump(timing, f)
        with open(os.path.join(fold_dir, "config.json"), "w") as f:
            json.dump({
                "dataset": dataset_name, "stage": stage_name, "candidate": candidate_name,
                "meta_params": params, "repeat_idx": r_idx, "fold_idx": f_idx, "seed": run_seed,
            }, f, indent=2)
        joblib.dump(model, os.path.join(fold_dir, "meta_model.pkl"))

        if verbose:
            print(f"  [{dataset_name}/{stage_name}/{candidate_name}] repeat{r_idx} fold{f_idx} "
                  f"acc={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}")

    agg = aggregate_fold_metrics(fold_metrics, timing_list=timing_list)
    agg.update({
        "dataset": dataset_name, "candidate": candidate_name, "model_name": candidate_name,
        "resampler": "N/A", "loss_type": "N/A", "class_names": class_names,
    })
    agg["accuracy_per_fold"] = [m["accuracy"] for m in fold_metrics]
    agg["macro_f1_per_fold"] = [m["macro_f1"] for m in fold_metrics]
    agg["bacc_per_fold"] = [m["bacc"] for m in fold_metrics]
    agg["mcc_per_fold"] = [m["mcc"] for m in fold_metrics]
    return agg


def run_justification_meta(dataset_name, X_meta, y, class_names, groups=None,
                            n_folds=N_FOLDS_JUSTIFIKASI, n_repeats=N_REPEATS_JUSTIFIKASI,
                            seed=SEED, verbose=True):
    """justifikasi_meta penuh: loop META_LEARNER_NAMES, otomatis pilih pemenang."""
    from src.meta_learners import META_LEARNER_NAMES

    stage_name = "justifikasi_meta"
    results = {}
    for cand in META_LEARNER_NAMES:
        agg = run_meta_candidate(
            dataset_name, cand, X_meta, y, class_names, groups=groups,
            n_folds=n_folds, n_repeats=n_repeats, seed=seed, verbose=verbose,
        )
        append_checkpoint(dataset_name, stage_name, agg)
        results[cand] = agg

    winner, mean_scores, friedman_result, wilcoxon_df = auto_select_winner(results, verbose=verbose)
    if wilcoxon_df is not None:
        stat_path = os.path.join(TAB_DIR, f"{dataset_name}_statistical_test_{stage_name}.csv")
        wilcoxon_df.to_csv(stat_path, index=False)
        if verbose:
            print(f"Uji statistik tersimpan: {stat_path}")

    return winner, results


# ══════════════════════════════════════════════════════════════════════════
# 3. ORKESTRASI PENUH, SATU DATASET
# ══════════════════════════════════════════════════════════════════════════

def run_justification_pipeline(dataset_name, X, y, class_names, groups=None,
                                n_folds=N_FOLDS_JUSTIFIKASI, n_repeats=N_REPEATS_JUSTIFIKASI,
                                seed=SEED, verbose=True, save_timing=True):
    """
    Jalankan justifikasi_backbone -> justifikasi_resampler ->
    justifikasi_loss -> justifikasi_meta PENUH untuk SATU dataset, otomatis
    ujung ke ujung, TANPA input manual nama pemenang di titik manapun.

    Dipakai IDENTIK untuk keempat dataset (SOR, OTDR, HardFailure,
    SoftFailure). groups WAJIB diisi (array id file) untuk SOR, None untuk
    dataset lain.

    save_timing=True (default): teruskan ke run_justification_axis. Set
    False untuk smoke test (percepat signifikan, backbone axis ganti
    arsitektur 5x jadi tidak perlu cuDNN benchmark ulang tiap kali).

    Return: dict ringkasan pemenang tiap axis, juga disimpan ke
    results/tables/<dataset_name>_justifikasi_winners.json. Pemenang
    backbone+resampler+loss di sini adalah REFERENSI justifikasi desain,
    BUKAN otomatis dipakai sebagai treatment HDMC (treatment HDMC final
    memang CNN-BiLSTM + 3 treatment tetap, lihat hdmc_multiview.py dan
    instruksi proyek soal treatment terkunci).
    """
    from src.backbone_architectures import BACKBONE_REGISTRY
    from src.resampling_and_loss import RESAMPLER_NAMES, LOSS_NAMES
    from src.meta_learners import build_meta_features
    from src.experiment_runner import run_candidate as _rc  # noqa: F401 (dipakai run_justification_axis)
    from src.hdmc_multiview import generate_oof_proba_single

    sep = "=" * 70

    print(f"\n{sep}\n[JUSTIFIKASI BACKBONE] {dataset_name}\n{sep}")
    backbone_winner, results_backbone = run_justification_axis(
        dataset_name, "justifikasi_backbone", list(BACKBONE_REGISTRY), X, y, class_names,
        model_name=None, resampler="None", loss_type="CE", axis="model_name",
        groups=groups, n_folds=n_folds, n_repeats=n_repeats, seed=seed, verbose=verbose,
        save_timing=save_timing,
    )

    print(f"\n{sep}\n[JUSTIFIKASI RESAMPLER] {dataset_name} (backbone={backbone_winner})\n{sep}")
    resampler_winner, results_resampler = run_justification_axis(
        dataset_name, "justifikasi_resampler", RESAMPLER_NAMES, X, y, class_names,
        model_name=backbone_winner, resampler=None, loss_type="CE", axis="resampler",
        groups=groups, n_folds=n_folds, n_repeats=n_repeats, seed=seed, verbose=verbose,
        save_timing=save_timing,
    )

    print(f"\n{sep}\n[JUSTIFIKASI LOSS] {dataset_name} "
          f"(backbone={backbone_winner}, resampler={resampler_winner})\n{sep}")
    loss_winner, results_loss = run_justification_axis(
        dataset_name, "justifikasi_loss", LOSS_NAMES, X, y, class_names,
        model_name=backbone_winner, resampler=resampler_winner, loss_type=None, axis="loss_type",
        groups=groups, n_folds=n_folds, n_repeats=n_repeats, seed=seed, verbose=verbose,
        save_timing=save_timing,
    )

    print(f"\n{sep}\n[JUSTIFIKASI META] {dataset_name} "
          f"(backbone={backbone_winner}, resampler={resampler_winner}, loss={loss_winner})\n{sep}")
    # treatment_label diberi prefix "justifikasi_" dan sengaja TIDAK memakai
    # save_artifacts=True default ke namespace backbone HDMC, supaya tidak
    # bentrok nama folder kalau kebetulan resampler_winner+loss_winner sama
    # dengan salah satu dari 3 treatment HDMC terkunci (mis. "None"+"CE").
    # Tanpa prefix ini, artifact justifikasi (3-fold) bisa tertimpa/tercampur
    # dengan artifact HDMC final (10-fold) di folder yang sama.
    oof_proba, _ = generate_oof_proba_single(
        dataset_name, backbone_winner, resampler_winner, loss_winner, X, y, class_names,
        groups=groups, n_folds=n_folds, seed=seed, verbose=verbose,
        treatment_label=f"justifikasi_meta_{backbone_winner}_{resampler_winner}_{loss_winner}",
    )
    X_meta = build_meta_features(oof_proba)
    meta_winner, results_meta = run_justification_meta(
        dataset_name, X_meta, y, class_names, groups=groups,
        n_folds=n_folds, n_repeats=n_repeats, seed=seed, verbose=verbose,
    )

    summary = {
        "dataset": dataset_name,
        "backbone_winner": backbone_winner,
        "resampler_winner": resampler_winner,
        "loss_winner": loss_winner,
        "meta_winner": meta_winner,
        "catatan": ("Pemenang di sini adalah justifikasi desain (protokol ringkas "
                    "n_folds=3/n_repeats=1). Treatment dan meta-learner HDMC final "
                    "TETAP mengikuti keputusan terkunci (CNN-BiLSTM, 3 treatment tetap, "
                    "LogisticRegression), lihat hdmc_multiview.py."),
    }
    summary_path = os.path.join(TAB_DIR, f"{dataset_name}_justifikasi_winners.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{sep}\n[SELESAI] {dataset_name}\n{json.dumps(summary, indent=2)}\n"
          f"Ringkasan tersimpan: {summary_path}\n{sep}")
    return summary


# ══════════════════════════════════════════════════════════════════════════
# 4. SINTESIS PEMILIHAN 3 TREATMENT HDMC (DARI HASIL AXIS YANG SUDAH ADA)
# ══════════════════════════════════════════════════════════════════════════
# TIDAK ADA training baru di sini. Fungsi ini cuma MEMBACA checkpoint CSV
# yang sudah dihasilkan justifikasi_resampler/justifikasi_loss (Bagian 1)
# dan hasil justifikasi_treatment_pair (hdmc_multiview.py), lalu menyusun
# satu tabel+narasi yang menjelaskan KENAPA tepat 3 kombinasi ini
# (SMOTETomek+Focal, ADASYN+WCE, None+CE) dipilih untuk HDMC, bukan
# kombinasi resampler x loss lain.
#
# PRINSIP DESAIN yang disintesis di sini (bukan hasil pencarian baru):
# ketiga treatment sengaja direntangkan di SPEKTRUM INTENSITAS penanganan
# imbalance, dengan intensitas resampler dipasangkan KONSISTEN dengan
# intensitas loss-nya:
#   - SMOTETomek+Focal : AGRESIF  (oversampling+pembersihan batas kelas,
#                                   dipasangkan loss yang menekan keras
#                                   kontribusi sampel mudah)
#   - ADASYN+WCE        : MODERAT  (oversampling fokus area sulit,
#                                   dipasangkan reweighting proporsional biasa)
#   - None+CE           : BASELINE (tanpa intervensi resampling maupun loss)
#
# Justifikasi ini SENGAJA tidak menguji seluruh kombinasi resampler x loss
# (4x5=20 kombinasi, apalagi triplet dari 20 itu C(20,3)=1140, tidak
# tractable dan REDUNDAN dengan justifikasi_resampler/justifikasi_loss yang
# sudah menguji tiap axis secara terpisah). Prinsip AGRESIF/MODERAT/BASELINE
# di atas yang menjustifikasi PEMILIHAN pasangannya, angka macro-F1 dari
# justifikasi_resampler/justifikasi_loss yang menjustifikasi bahwa tiap
# KOMPONEN (SMOTETomek, ADASYN, Focal, WCE) memang kandidat kuat di axis-nya
# masing-masing, dan justifikasi_treatment_pair yang mengonfirmasi PASCA-FAKTA
# bahwa hasil pemasangan ini benar-benar saling melengkapi (disagreement
# tinggi), bukan kebetulan redundan.

def sintesis_pemilihan_treatment(dataset_name, verbose=True):
    """
    Baca ulang checkpoint justifikasi_resampler, justifikasi_loss (Bagian 1
    file ini), dan hasil justifikasi_treatment_pair (hdmc_multiview.py),
    susun satu tabel+narasi kenapa SMOTETomek+Focal, ADASYN+WCE, None+CE
    dipilih. WAJIB dijalankan SETELAH run_justification_pipeline() dan
    justifikasi_treatment_pair() selesai untuk dataset yang sama (kalau
    filenya belum ada, fungsi ini kasih pesan jelas, bukan crash cryptic).

    Return: dict ringkasan, juga disimpan ke
    results/tables/<dataset>_sintesis_treatment.json, siap dikutip di
    bagian metodologi naskah.
    """
    resampler_path = os.path.join(TAB_DIR, f"{dataset_name}_checkpoint_justifikasi_resampler.csv")
    loss_path = os.path.join(TAB_DIR, f"{dataset_name}_checkpoint_justifikasi_loss.csv")
    pair_path = os.path.join(TAB_DIR, f"{dataset_name}_justifikasi_treatment_pair.json")

    missing = [p for p in [resampler_path, loss_path, pair_path] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"[{dataset_name}] File berikut belum ada, jalankan run_justification_pipeline() "
            f"dan justifikasi_treatment_pair() dulu untuk dataset ini:\n  " + "\n  ".join(missing)
        )

    import pandas as pd
    df_resampler = read_checkpoint_csv(resampler_path)
    df_loss = read_checkpoint_csv(loss_path)
    with open(pair_path) as f:
        pair_stats = json.load(f)

    def _skor(df, nama_kolom, nama_kandidat):
        row = df[df[nama_kolom] == nama_kandidat]
        if row.empty:
            return None
        return float(row["macro_f1_mean"].iloc[0])

    skor_resampler = {
        "SMOTETomek": _skor(df_resampler, "resampler", "SMOTETomek"),
        "ADASYN": _skor(df_resampler, "resampler", "ADASYN"),
        "None": _skor(df_resampler, "resampler", "None"),
        "SMOTE": _skor(df_resampler, "resampler", "SMOTE"),
    }
    skor_loss = {
        "Focal": _skor(df_loss, "loss_type", "Focal"),
        "WCE": _skor(df_loss, "loss_type", "WCE"),
        "CE": _skor(df_loss, "loss_type", "CE"),
        "CB_Focal": _skor(df_loss, "loss_type", "CB_Focal"),
        "LDAM": _skor(df_loss, "loss_type", "LDAM"),
    }

    ringkasan = {
        "dataset": dataset_name,
        "prinsip_desain": "Spektrum intensitas (agresif/moderat/baseline), resampler dan loss dipasangkan konsisten intensitasnya",
        "treatment_terpilih": {
            "SMOTETomek+Focal": {
                "intensitas": "agresif",
                "skor_resampler_axis": skor_resampler["SMOTETomek"],
                "skor_loss_axis": skor_loss["Focal"],
            },
            "ADASYN+WCE": {
                "intensitas": "moderat",
                "skor_resampler_axis": skor_resampler["ADASYN"],
                "skor_loss_axis": skor_loss["WCE"],
            },
            "None+CE": {
                "intensitas": "baseline",
                "skor_resampler_axis": skor_resampler["None"],
                "skor_loss_axis": skor_loss["CE"],
            },
        },
        "skor_kandidat_lain_axis_resampler": skor_resampler,
        "skor_kandidat_lain_axis_loss": skor_loss,
        "disagreement_antar_treatment_pasca_pemilihan": pair_stats,
    }

    out_path = os.path.join(TAB_DIR, f"{dataset_name}_sintesis_treatment.json")
    with open(out_path, "w") as f:
        json.dump(ringkasan, f, indent=2)

    if verbose:
        sep = "=" * 70
        print(f"\n{sep}\n[SINTESIS PEMILIHAN TREATMENT] {dataset_name}\n{sep}")
        print("Skor axis resampler (justifikasi_resampler, backbone+loss dikunci):")
        for k, v in skor_resampler.items():
            tanda = " <- dipakai" if k in ("SMOTETomek", "ADASYN", "None") else ""
            print(f"  {k:12s} macro_f1={v if v is not None else float('nan'):.4f}{tanda}" if v is not None
                  else f"  {k:12s} (tidak ditemukan di checkpoint)")
        print("\nSkor axis loss (justifikasi_loss, backbone+resampler dikunci):")
        for k, v in skor_loss.items():
            tanda = " <- dipakai" if k in ("Focal", "WCE", "CE") else ""
            print(f"  {k:12s} macro_f1={v if v is not None else float('nan'):.4f}{tanda}" if v is not None
                  else f"  {k:12s} (tidak ditemukan di checkpoint)")
        print(f"\nDisagreement pasca-pemilihan (justifikasi_treatment_pair): {len(pair_stats)} pasangan")
        for name, stats in sorted(pair_stats.items(), key=lambda kv: kv[1]["mean_l1"], reverse=True):
            print(f"  {stats['mean_l1']:.4f} (L1)  {name}")
        print(f"\nTersimpan: {out_path}")
        print(sep)

    return ringkasan


# ══════════════════════════════════════════════════════════════════════════
# 5. SINTESIS BACKBONE LINTAS DATASET (JAWAB: "MENANG DI DATASET MANA?")
# ══════════════════════════════════════════════════════════════════════════
# TIDAK ADA training baru di sini. Fungsi ini cuma MEMBACA checkpoint
# justifikasi_backbone dari KEEMPAT dataset (WAJIB sudah selesai dijalankan
# untuk semua dataset dulu), lalu menjawab langsung: apakah CNN-BiLSTM
# konsisten menang di semua dataset, atau menang di sebagian saja?
#
# CNN-BiLSTM dikunci sebagai backbone HDMC untuk SEMUA dataset (satu
# arsitektur, bukan pilih arsitektur beda-beda per dataset), KONSISTEN
# dengan kriteria yang sama dipakai untuk pemilihan meta-learner
# (LogisticRegression, lihat instruksi proyek): kriterianya KONSISTENSI
# lintas dataset (performa baik di semua kondisi imbalance), BUKAN
# argmax per dataset (yang bisa menghasilkan 4 arsitektur berbeda untuk
# 4 dataset, sistem jadi tidak seragam dan sulit dijelaskan sebagai satu
# metode HDMC yang utuh).

def sintesis_backbone_lintas_dataset(dataset_order, verbose=True):
    """
    Baca checkpoint justifikasi_backbone dari SEMUA dataset di
    `dataset_order`, susun tabel skor+rank tiap kandidat backbone per
    dataset, terapkan kriteria konsistensi (rank rata-rata + rank
    terburuk/minimax) untuk menjustifikasi backbone yang dipakai SEMUA
    dataset. WAJIB dijalankan SETELAH run_justification_pipeline() selesai
    untuk SEMUA dataset di dataset_order (bukan cuma satu).

    Return: dict ringkasan, juga disimpan ke
    results/tables/sintesis_backbone_lintas_dataset.json. TIDAK mengubah
    HDMC_BACKBONE secara otomatis, cuma memberi bukti data untuk klaim di
    naskah (atau peringatan kalau ternyata tidak konsisten).
    """
    import pandas as pd
    from src.hdmc_multiview import HDMC_BACKBONE

    missing = []
    df_per_dataset = {}
    for ds in dataset_order:
        path = os.path.join(TAB_DIR, f"{ds}_checkpoint_justifikasi_backbone.csv")
        if not os.path.exists(path):
            missing.append(path)
            continue
        df_per_dataset[ds] = read_checkpoint_csv(path)
    if missing:
        raise FileNotFoundError(
            "Checkpoint justifikasi_backbone berikut belum ada, jalankan "
            "run_justification_pipeline() untuk SEMUA dataset dulu:\n  " + "\n  ".join(missing)
        )

    kandidat_all = sorted(set().union(*[set(df["candidate"]) for df in df_per_dataset.values()]))
    tabel_skor = pd.DataFrame(index=kandidat_all, columns=dataset_order, dtype=float)
    for ds, df in df_per_dataset.items():
        for _, row in df.iterrows():
            tabel_skor.loc[row["candidate"], ds] = row["macro_f1_mean"]

    tabel_rank = tabel_skor.rank(axis=0, ascending=False)  # rank 1 = terbaik, per kolom (per dataset)

    rank_rata2 = tabel_rank.mean(axis=1)
    rank_terburuk = tabel_rank.max(axis=1)  # minimax: rank PALING JELEK di dataset manapun

    pemenang_rank_rata2 = rank_rata2.idxmin()
    pemenang_minimax = rank_terburuk.idxmin()

    backbone_terkunci = HDMC_BACKBONE
    menang_di = [ds for ds in dataset_order if tabel_rank.loc[backbone_terkunci, ds] == 1]
    kalah_di = [ds for ds in dataset_order if tabel_rank.loc[backbone_terkunci, ds] != 1]

    ringkasan = {
        "backbone_terkunci": backbone_terkunci,
        "menang_rank1_di": menang_di,
        "TIDAK_menang_rank1_di": kalah_di,
        "rank_per_dataset": {ds: int(tabel_rank.loc[backbone_terkunci, ds]) for ds in dataset_order},
        "skor_per_dataset": {ds: float(tabel_skor.loc[backbone_terkunci, ds]) for ds in dataset_order},
        "rank_rata2_backbone_terkunci": float(rank_rata2[backbone_terkunci]),
        "rank_terburuk_backbone_terkunci": float(rank_terburuk[backbone_terkunci]),
        "pemenang_rank_rata2_kriteria": pemenang_rank_rata2,
        "pemenang_minimax_kriteria": pemenang_minimax,
        "cocok_dengan_kriteria_rank_rata2": pemenang_rank_rata2 == backbone_terkunci,
        "cocok_dengan_kriteria_minimax": pemenang_minimax == backbone_terkunci,
        "tabel_skor_lengkap": tabel_skor.to_dict(),
        "tabel_rank_lengkap": tabel_rank.to_dict(),
    }

    out_path = os.path.join(TAB_DIR, "sintesis_backbone_lintas_dataset.json")
    with open(out_path, "w") as f:
        json.dump(ringkasan, f, indent=2, default=str)

    if verbose:
        sep = "=" * 70
        print(f"\n{sep}\n[SINTESIS BACKBONE LINTAS DATASET]\n{sep}")
        print(f"\nTabel macro-F1 tiap kandidat backbone x dataset:")
        print(tabel_skor.round(4).to_string())
        print(f"\nTabel rank tiap kandidat backbone x dataset (1=terbaik):")
        print(tabel_rank.astype(int).to_string())
        print(f"\nBackbone terkunci: {backbone_terkunci}")
        if menang_di:
            print(f"  Menang (rank 1) di: {menang_di}")
        if kalah_di:
            print(f"  TIDAK menang (bukan rank 1) di: {kalah_di}")
            for ds in kalah_di:
                pemenang_ds = tabel_rank[ds].idxmin()
                print(f"    -> {ds}: yang menang adalah {pemenang_ds} "
                      f"(macro_f1={tabel_skor.loc[pemenang_ds, ds]:.4f}, "
                      f"{backbone_terkunci}={tabel_skor.loc[backbone_terkunci, ds]:.4f})")
        print(f"\nKriteria rank rata-rata: pemenang={pemenang_rank_rata2} "
              f"(cocok dengan terkunci: {ringkasan['cocok_dengan_kriteria_rank_rata2']})")
        print(f"Kriteria minimax (rank terburuk terbaik): pemenang={pemenang_minimax} "
              f"(cocok dengan terkunci: {ringkasan['cocok_dengan_kriteria_minimax']})")
        if not ringkasan["cocok_dengan_kriteria_rank_rata2"] and not ringkasan["cocok_dengan_kriteria_minimax"]:
            print(f"\n[PERINGATAN] {backbone_terkunci} TIDAK menang di kriteria rank rata-rata "
                  f"MAUPUN minimax. Ini perlu ditinjau manual, HDMC_BACKBONE TIDAK diubah otomatis.")
        else:
            print(f"\n[OK] {backbone_terkunci} konsisten sebagai kandidat terbaik atau mendekati "
                  f"terbaik lintas dataset (menang di minimal satu kriteria konsistensi).")
        print(f"\nTersimpan: {out_path}")
        print(sep)

    return ringkasan


# ══════════════════════════════════════════════════════════════════════════
# 6. DIAGNOSTIK OVERFITTING BACKBONE (BACA ULANG CHECKPOINT, TANPA TRAINING BARU)
# ══════════════════════════════════════════════════════════════════════════
# TIDAK ADA training baru di sini. Model tiap kandidat backbone SUDAH
# tersimpan dari justifikasi_backbone (Bagian A), fungsi ini cuma LOAD
# ulang model itu dan bandingkan performa di data TRAINING (yang sudah
# dilihat model) vs data VALIDASI (yang tidak pernah dilihat model).
#
# INTERPRETASI: gap = macro_f1_train - macro_f1_val. Gap besar (model jauh
# lebih bagus di training dibanding validasi) = indikasi overfitting.
# Kandidat dengan kapasitas lebih besar (BiLSTM/BiGRU, parameter lebih
# banyak) secara umum LEBIH RENTAN overfitting dibanding kandidat sederhana
# (CNN Only), terutama pada dataset dengan IR ekstrem seperti SOR di mana
# sinyal kelas minoritas sangat sedikit untuk dipelajari tanpa menghafal.
#
# Fungsi ini TIDAK mengambil keputusan otomatis (tidak memilih backbone).
# Hasilnya jadi BUKTI TAMBAHAN untuk keputusan manual, dilengkapi
# significance/rank dari sintesis_backbone_lintas_dataset di atas.

def analisis_overfitting_backbone(dataset_name, X, y, class_names, groups=None,
                                   n_folds=N_FOLDS_JUSTIFIKASI, n_repeats=N_REPEATS_JUSTIFIKASI,
                                   seed=SEED, verbose=True, stage_name="justifikasi_backbone"):
    """
    Load ulang model tiap kandidat backbone (dari checkpoint justifikasi_backbone
    yang sudah tersimpan), hitung macro-F1 di data training (resampled, None+CE,
    sama seperti axis backbone) vs data validasi, laporkan gap-nya.

    WAJIB dijalankan SETELAH justifikasi_backbone selesai untuk dataset ini
    (kalau checkpoint model belum ada, fold itu dilewati dengan pesan jelas,
    bukan crash cryptic).

    stage_name: default "justifikasi_backbone" (hasil 3-fold/15-epoch biasa).
    Bisa diarahkan ke stage_name lain (mis. "justifikasi_backbone_epoch_besar")
    untuk membaca hasil axis backbone yang dijalankan ulang dengan budget
    epoch berbeda, tanpa menimpa hasil asli.

    Return: dict ringkasan per kandidat, juga disimpan ke
    results/tables/<dataset>_<stage_name>_analisis_overfitting.json
    """
    import torch
    from src.backbone_architectures import BACKBONE_REGISTRY, get_model
    from src.resampling_and_loss import apply_resampling
    from src.dataset_loaders import get_repeated_folds, get_repeated_folds_grouped, scale_fold
    from src.training_engine import predict_proba, fold_seed
    from src.metrics import compute_metrics

    if groups is not None:
        folds = list(get_repeated_folds_grouped(X, y, groups, n_splits=n_folds, n_repeats=n_repeats, seed=seed))
    else:
        folds = list(get_repeated_folds(X, y, n_splits=n_folds, n_repeats=n_repeats, seed=seed))

    n_classes = len(class_names)
    hasil = {}
    dilewati = []

    for cand in BACKBONE_REGISTRY:
        gaps, val_scores, train_scores = [], [], []
        for split_idx, (train_idx, val_idx) in enumerate(folds):
            r_idx, f_idx = _fold_repeat_index(n_folds, split_idx)
            model_path = os.path.join(RAW_DIR, dataset_name, stage_name, cand,
                                       f"repeat{r_idx}_fold{f_idx}", "model_state.pt")
            if not os.path.exists(model_path):
                dilewati.append(f"{cand} repeat{r_idx}_fold{f_idx}")
                continue

            run_seed = fold_seed(r_idx, f_idx, base_seed=seed)
            X_tr_raw, X_va_raw = X[train_idx], X[val_idx]
            y_tr_raw, y_va_raw = y[train_idx], y[val_idx]
            X_tr_sc, X_va_sc, _ = scale_fold(X_tr_raw, X_va_raw)
            # resampler="None" konsisten dengan axis backbone (justifikasi_backbone
            # mengunci resampler=None, loss=CE selama pengujian kandidat backbone)
            X_tr_res, y_tr_res = apply_resampling(X_tr_sc, y_tr_raw, method="None", seed=run_seed)

            model = get_model(cand, in_features=X.shape[1], n_classes=n_classes)
            model.load_state_dict(torch.load(model_path, map_location="cpu"))

            proba_tr = predict_proba(model, X_tr_res)
            proba_va = predict_proba(model, X_va_sc)
            m_tr = compute_metrics(y_tr_res, proba_tr.argmax(axis=1), class_names)
            m_va = compute_metrics(y_va_raw, proba_va.argmax(axis=1), class_names)

            gaps.append(m_tr["macro_f1"] - m_va["macro_f1"])
            val_scores.append(m_va["macro_f1"])
            train_scores.append(m_tr["macro_f1"])

        if gaps:
            hasil[cand] = {
                "macro_f1_train_mean": float(np.mean(train_scores)),
                "macro_f1_val_mean": float(np.mean(val_scores)),
                "gap_train_minus_val_mean": float(np.mean(gaps)),
                "gap_train_minus_val_std": float(np.std(gaps)),
                "n_fold_terbaca": len(gaps),
            }

    out_path = os.path.join(TAB_DIR, f"{dataset_name}_{stage_name}_analisis_overfitting.json")
    with open(out_path, "w") as f:
        json.dump({"dataset": dataset_name, "hasil": hasil, "fold_dilewati": dilewati}, f, indent=2)

    if verbose:
        sep = "=" * 70
        print(f"\n{sep}\n[ANALISIS OVERFITTING BACKBONE] {dataset_name} (stage: {stage_name})\n{sep}")
        if dilewati:
            print(f"[PERINGATAN] {len(dilewati)} fold dilewati (checkpoint belum ada): {dilewati[:3]}"
                  f"{' ...' if len(dilewati) > 3 else ''}")
        urutan = sorted(hasil.items(), key=lambda kv: kv[1]["gap_train_minus_val_mean"])
        print(f"{'Kandidat':15s} {'F1 Train':>10s} {'F1 Val':>10s} {'Gap':>10s}  Interpretasi")
        for cand, r in urutan:
            gap = r["gap_train_minus_val_mean"]
            label = "rendah" if gap < 0.02 else ("sedang" if gap < 0.05 else "TINGGI (overfitting)")
            print(f"{cand:15s} {r['macro_f1_train_mean']:10.4f} {r['macro_f1_val_mean']:10.4f} "
                  f"{gap:10.4f}  {label}")
        print(f"\nTersimpan: {out_path}")
        print(sep)

    return hasil
