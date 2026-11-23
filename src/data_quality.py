# src/data_quality.py — Hibah Kemdiktisaintek Part 3
"""
Deteksi outlier OTOMATIS (selalu jalan, aman, read-only), TANPA penghapusan
otomatis. Prinsip inti: kelas minoritas SECARA DEFINISI adalah outlier
dibanding kelas Normal, jadi keputusan MEMBUANG baris data adalah keputusan
berisiko yang TIDAK BOLEH diambil otomatis tanpa tinjauan, betapapun
konservatif aturannya. Yang aman untuk dijalankan otomatis cuma ANALISIS
dan PENCATATAN, bukan tindakan pengubahan data.

Daftar isi:
  1. Deteksi per kelas, per fitur (IQR)       : detect_outliers_per_class
  2. Kandidat kasus ekstrem (TIDAK dihapus)   : flag_severe_outliers
  3. Pipeline analisis otomatis, TANPA hapus  : run_outlier_analysis

ATURAN KESELAMATAN:
- Deteksi IQR dihitung TERPISAH per kelas (Q1/Q3 kelas itu sendiri), bukan
  lintas seluruh dataset. Nilai yang "ekstrem" untuk kelas Normal bisa jadi
  nilai NORMAL untuk kelas Fiber Cut, dan sebaliknya.
- `run_outlier_analysis()` HANYA mendeteksi dan mencatat, TIDAK PERNAH
  menghapus atau mengubah data secara otomatis. Baris X/y/groups yang
  dikembalikan SAMA PERSIS dengan input, tidak ada baris yang hilang.
- Kandidat kasus paling ekstrem (>3x IQR dari kuartil kelasnya sendiri, di
  >=50% fitur sekaligus) tetap dihitung dan DICATAT ke CSV terpisah
  (`<dataset>_outlier_kandidat_dihapus.csv`), supaya kalau suatu saat
  memang diputuskan untuk ditindaklanjuti (setelah ditinjau manual), datanya
  sudah siap tanpa perlu deteksi ulang. TAPI tindakan itu WAJIB manual,
  tidak dieksekusi oleh kode ini.

Semua hasil deteksi dicatat ke results/tables/<dataset>_outlier_report.csv
dan results/tables/<dataset>_outlier_kandidat_dihapus.csv, OTOMATIS, tanpa
mengubah X/y yang dipakai eksperimen.
"""
import os
import numpy as np
import pandas as pd

from src.config import TAB_DIR


# ══════════════════════════════════════════════════════════════════════════
# 1. DETEKSI PER KELAS, PER FITUR (IQR)
# ══════════════════════════════════════════════════════════════════════════

def detect_outliers_per_class(X, y, class_names, k_iqr=1.5):
    """
    Untuk tiap kelas, tiap fitur: hitung Q1/Q3 kelas ITU SENDIRI, tandai
    baris yang nilainya di luar [Q1 - k_iqr*IQR, Q3 + k_iqr*IQR].

    Return: outlier_mask (n_samples, n_features) boolean, DAN df_report
    (ringkasan jumlah+persentase outlier per kelas per fitur).
    """
    n_samples, n_features = X.shape
    outlier_mask = np.zeros((n_samples, n_features), dtype=bool)
    rows = []

    for c in np.unique(y):
        idx_c = np.where(y == c)[0]
        if len(idx_c) < 10:
            # kelas terlalu kecil, kuartil tidak stabil, lewati deteksi
            # (bukan berarti kelas ini "bersih", cuma sampelnya belum cukup
            # untuk estimasi IQR yang bisa dipercaya)
            continue
        Xc = X[idx_c]
        q1 = np.percentile(Xc, 25, axis=0)
        q3 = np.percentile(Xc, 75, axis=0)
        iqr = q3 - q1
        iqr_safe = np.where(iqr == 0, 1e-9, iqr)  # hindari div-by-zero fitur konstan
        lower = q1 - k_iqr * iqr_safe
        upper = q3 + k_iqr * iqr_safe

        mask_c = (Xc < lower) | (Xc > upper)
        outlier_mask[idx_c] = mask_c

        n_outlier_rows = int((mask_c.any(axis=1)).sum())
        cname = class_names[c] if c < len(class_names) else str(c)
        rows.append({
            "class_id": int(c), "class_name": cname, "n_samples": len(idx_c),
            "n_baris_ada_outlier_1_fitur": n_outlier_rows,
            "pct_baris_ada_outlier_1_fitur": round(n_outlier_rows / len(idx_c) * 100, 2),
        })

    df_report = pd.DataFrame(rows)
    return outlier_mask, df_report


# ══════════════════════════════════════════════════════════════════════════
# 2. KANDIDAT KASUS EKSTREM (TIDAK DIHAPUS OTOMATIS)
# ══════════════════════════════════════════════════════════════════════════

def flag_severe_outliers(X, y, k_iqr_severe=3.0, min_features_frac=0.5):
    """
    Tandai baris sebagai KANDIDAT kasus ekstrem: ekstrem (>k_iqr_severe x IQR
    dari kuartil KELASNYA SENDIRI) di >=min_features_frac fraksi fitur
    SEKALIGUS. INI HANYA PENANDAAN, BUKAN PENGHAPUSAN. Baris ini kemungkinan
    besar sensor glitch/baris korup (bukan sinyal fault asli yang biasanya
    cuma ekstrem di fitur yang relevan dengan jenis fault itu), TAPI
    keputusan menghapusnya tetap harus manual, tidak diambil di sini.
    """
    n_samples, n_features = X.shape
    severe_mask = np.zeros((n_samples, n_features), dtype=bool)

    for c in np.unique(y):
        idx_c = np.where(y == c)[0]
        if len(idx_c) < 10:
            continue
        Xc = X[idx_c]
        q1 = np.percentile(Xc, 25, axis=0)
        q3 = np.percentile(Xc, 75, axis=0)
        iqr = q3 - q1
        iqr_safe = np.where(iqr == 0, 1e-9, iqr)
        lower = q1 - k_iqr_severe * iqr_safe
        upper = q3 + k_iqr_severe * iqr_safe
        severe_mask[idx_c] = (Xc < lower) | (Xc > upper)

    frac_extreme_per_row = severe_mask.mean(axis=1)
    rows_to_remove = frac_extreme_per_row >= min_features_frac
    return rows_to_remove, frac_extreme_per_row


# ══════════════════════════════════════════════════════════════════════════
# 3. PIPELINE ANALISIS OTOMATIS, TANPA HAPUS
# ══════════════════════════════════════════════════════════════════════════

def run_outlier_analysis(X, y, class_names, dataset_name, groups=None,
                          k_iqr_report=1.5, k_iqr_severe=3.0, min_features_frac=0.5,
                          verbose=True):
    """
    Jalankan deteksi (pelaporan, 1.5x IQR) + penandaan kandidat kasus
    ekstrem (3x IQR, >=50% fitur sekaligus), OTOMATIS, dicatat ke CSV.
    TIDAK ADA baris yang dihapus atau diubah di sini. X/y/groups yang
    dikembalikan SAMA PERSIS dengan input.

    Dipanggil dari src/hdmc_multiview.py::load_dataset(), jadi SEMUA
    notebook otomatis dapat laporan kualitas data tanpa langkah manual,
    tapi data yang dipakai eksperimen tetap data ASLI (belum ada baris
    yang dibuang atas nama "pembersihan otomatis").

    Return: X, y, groups (SAMA PERSIS dengan input, tidak diubah), summary dict.
    """
    _, df_report = detect_outliers_per_class(X, y, class_names, k_iqr=k_iqr_report)
    rows_kandidat, frac_extreme = flag_severe_outliers(
        X, y, k_iqr_severe=k_iqr_severe, min_features_frac=min_features_frac)

    n_kandidat = int(rows_kandidat.sum())
    n_total = len(y)

    kandidat_summary = []
    if n_kandidat > 0:
        for c in np.unique(y[rows_kandidat]):
            cname = class_names[c] if c < len(class_names) else str(c)
            n_c_kandidat = int((y[rows_kandidat] == c).sum())
            n_c_total = int((y == c).sum())
            kandidat_summary.append({
                "class_id": int(c), "class_name": cname,
                "n_kandidat_kasus_ekstrem": n_c_kandidat, "n_total_kelas": n_c_total,
                "pct_kandidat": round(n_c_kandidat / max(n_c_total, 1) * 100, 2),
            })

    report_path = os.path.join(TAB_DIR, f"{dataset_name}_outlier_report.csv")
    df_report.to_csv(report_path, index=False)

    # Baris index kandidat kasus ekstrem DISIMPAN LENGKAP (bukan cuma
    # ringkasan), supaya kalau nanti diputuskan untuk ditindaklanjuti
    # (manual, setelah tinjauan), tidak perlu deteksi ulang dari nol.
    kandidat_idx_path = os.path.join(TAB_DIR, f"{dataset_name}_outlier_kandidat_index.csv")
    pd.DataFrame({
        "row_index": np.where(rows_kandidat)[0],
        "class_id": y[rows_kandidat],
        "class_name": [class_names[c] if c < len(class_names) else str(c) for c in y[rows_kandidat]],
        "frac_fitur_ekstrem": frac_extreme[rows_kandidat],
    }).to_csv(kandidat_idx_path, index=False)

    summary_path = os.path.join(TAB_DIR, f"{dataset_name}_outlier_kandidat_dihapus.csv")
    pd.DataFrame(kandidat_summary).to_csv(summary_path, index=False)

    if verbose:
        print(f"[{dataset_name}] Deteksi outlier (pelaporan, {k_iqr_report}x IQR per kelas): "
              f"tersimpan {report_path}")
        print(f"[{dataset_name}] Kandidat kasus ekstrem ({k_iqr_severe}x IQR, "
              f">={min_features_frac*100:.0f}% fitur sekaligus): "
              f"{n_kandidat}/{n_total} baris ({n_kandidat/n_total*100:.3f}%) DITANDAI, "
              f"TIDAK DIHAPUS. Index tersimpan: {kandidat_idx_path}")
        if kandidat_summary:
            for r in kandidat_summary:
                print(f"    {r['class_name']:20s} {r['n_kandidat_kasus_ekstrem']}/{r['n_total_kelas']} "
                      f"({r['pct_kandidat']}%) ditandai sebagai kandidat")
        else:
            print(f"    Tidak ada baris yang memenuhi ambang kandidat kasus ekstrem.")
        print(f"[{dataset_name}] Data yang dipakai eksperimen TIDAK DIUBAH "
              f"({n_total} baris, sama dengan sebelum analisis ini).")

    summary = {
        "dataset": dataset_name, "n_total": n_total,
        "n_kandidat_kasus_ekstrem": n_kandidat,
        "pct_kandidat": round(n_kandidat / n_total * 100, 4),
    }
    return X, y, groups, summary
