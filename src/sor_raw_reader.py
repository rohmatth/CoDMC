# src/sor_raw_reader.py — Hibah Kemdiktisaintek Part 3
"""
Baca file mentah SOR (snapshot .xlsx per event + Anotasi.csv). Fungsi di
sini bersifat INSPEKSI/PARSING, melaporkan apa yang benar-benar ada di
file, tidak mengasumsikan struktur tertentu tanpa verifikasi.

Daftar isi:
  1. Daftar file snapshot         : list_snapshot_files
  2. Baca dan verifikasi anotasi  : load_annotation, annotation_structure_report,
                                     verify_prefix_map_against_annotation
  3. Cek integritas file          : check_duplicate_and_missing_files
  4. Inspeksi struktur trace      : inspect_sample_trace, inspect_sample_across_prefixes

Pemetaan prefix nama file ke kelas (SOR_PREFIX_MAP_TENTATIVE di config.py)
adalah DUGAAN AWAL. Kolom "Class" di Anotasi.csv adalah sumber label
OTORITATIF (lihat dataset_loaders.py::_build_file_records), prefix map di
sini cuma dipakai untuk verifikasi silang kasar, bukan sumber label utama.
"""
import os
import re
import numpy as np
import pandas as pd

from src.config import SOR_DIR, SOR_ANNOTATION_FILE, SOR_SNAPSHOT_DATES, SOR_PREFIX_MAP_TENTATIVE




# ══════════════════════════════════════════════════════════════════════════
# 1. DAFTAR FILE SNAPSHOT
# ══════════════════════════════════════════════════════════════════════════

def list_snapshot_files(sor_dir=SOR_DIR, snapshot_dates=SOR_SNAPSHOT_DATES):
    """
    Daftar seluruh file .xlsx SOR, dengan prefix dan index yang diparse
    dari nama file. Tidak melakukan validasi isi file.

    Mendukung DUA kemungkinan struktur folder (auto-detect):
    1. FLAT: seluruh file .xlsx langsung di dalam sor_dir (mis. folder
       "Dataset Gabungan" tunggal, seperti dipakai rekan tim untuk riset
       paralel mereka).
    2. SUBFOLDER PER TANGGAL: sor_dir berisi subfolder per snapshot_dates
       (asumsi awal Part 1/2, sesuai README lama, tapi TERNYATA belum
       tentu sesuai struktur disk yang sebenarnya).

    Auto-detect: kalau ada file .xlsx LANGSUNG di sor_dir, pakai struktur
    flat (skip subfolder). Kalau tidak, coba subfolder per snapshot_dates
    seperti sebelumnya. Kalau KEDUANYA kosong, cetak peringatan jelas
    supaya gampang didiagnosis, bukan diam-diam mengembalikan dataframe
    kosong yang bikin error membingungkan di langkah selanjutnya.
    """
    rows = []

    flat_files = [f for f in sorted(os.listdir(sor_dir)) if f.lower().endswith(".xlsx")] \
        if os.path.isdir(sor_dir) else []

    if flat_files:
        print(f"[list_snapshot_files] Struktur FLAT terdeteksi: {len(flat_files)} "
              f"file .xlsx langsung di {sor_dir}")
        for fname in flat_files:
            m = re.match(r"([A-Za-z]+)(\d+)\.xlsx$", fname, re.IGNORECASE)
            prefix = m.group(1) if m else None
            idx = int(m.group(2)) if m else None
            path = os.path.join(sor_dir, fname)
            rows.append({
                "snapshot": "flat", "filename": fname, "prefix": prefix, "index": idx,
                "path": path, "size_bytes": os.path.getsize(path),
            })
    else:
        print(f"[list_snapshot_files] Tidak ada .xlsx langsung di {sor_dir}, "
              f"coba struktur subfolder per tanggal: {snapshot_dates}")
        for snap in snapshot_dates:
            snap_dir = os.path.join(sor_dir, snap)
            if not os.path.isdir(snap_dir):
                print(f"[PERINGATAN] Folder tidak ditemukan: {snap_dir}")
                continue
            for fname in sorted(os.listdir(snap_dir)):
                if not fname.lower().endswith(".xlsx"):
                    continue
                m = re.match(r"([A-Za-z]+)(\d+)\.xlsx$", fname, re.IGNORECASE)
                prefix = m.group(1) if m else None
                idx = int(m.group(2)) if m else None
                path = os.path.join(snap_dir, fname)
                rows.append({
                    "snapshot": snap, "filename": fname, "prefix": prefix, "index": idx,
                    "path": path, "size_bytes": os.path.getsize(path),
                })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        print(f"[PERINGATAN] TIDAK ADA file .xlsx ditemukan sama sekali, baik "
              f"langsung di {sor_dir} maupun di subfolder {snapshot_dates}. "
              f"Cek ulang SOR_DIR di src/config.py, pastikan path itu benar "
              f"mengarah ke folder tempat file .xlsx SOR disimpan.")
    return df




# ══════════════════════════════════════════════════════════════════════════
# 2. BACA DAN VERIFIKASI ANOTASI
# ══════════════════════════════════════════════════════════════════════════

def load_annotation(path=SOR_ANNOTATION_FILE):
    """
    Load Anotasi.csv APA ADANYA. Mencoba beberapa separator umum kalau
    default gagal, karena format file belum dikonfirmasi.
    """
    for sep in [",", ";", "\t"]:
        try:
            df = pd.read_csv(path, sep=sep)
            if df.shape[1] > 1:
                print(f"[load_annotation] Terbaca dengan separator '{sep}', shape={df.shape}")
                return df
        except Exception:
            continue
    raise ValueError(f"Tidak bisa membaca {path} dengan separator umum (, ; tab). "
                      f"Cek manual formatnya.")


def annotation_structure_report(anno_df):
    """
    Laporan struktur Anotasi.csv: kolom, dtype, jumlah unique, contoh nilai,
    jumlah missing. Dipakai untuk memahami apa isi file ini sebelum dipakai
    sebagai sumber label.
    """
    rows = []
    for col in anno_df.columns:
        rows.append({
            "column": col,
            "dtype": str(anno_df[col].dtype),
            "n_unique": anno_df[col].nunique(),
            "n_missing": int(anno_df[col].isna().sum()),
            "example_values": anno_df[col].dropna().unique()[:5].tolist(),
        })
    return pd.DataFrame(rows)


def verify_prefix_map_against_annotation(anno_df, file_list_df,
                                          prefix_map=SOR_PREFIX_MAP_TENTATIVE):
    """
    Cek silang: apakah jumlah file per prefix di folder snapshot konsisten
    dengan jumlah entri per label di Anotasi.csv (kalau kolom label bisa
    ditemukan secara heuristik). Ini pengecekan KASAR, hasilnya harus dibaca
    manual, bukan diverifikasi otomatis penuh karena nama kolom belum pasti.
    """
    prefix_counts = file_list_df.groupby("prefix").size().rename("n_files_by_prefix")
    print("Jumlah file per prefix (dari nama file, seluruh snapshot):")
    print(prefix_counts)
    print("\nDugaan pemetaan prefix -> kelas (BELUM diverifikasi):")
    for p, cname in prefix_map.items():
        n = prefix_counts.get(p, 0)
        print(f"  {p:4s} -> {cname:<18s} ({n} file)")

    unmapped = set(file_list_df["prefix"].dropna().unique()) - set(prefix_map.keys())
    if unmapped:
        print(f"\n[PERINGATAN] Prefix ditemukan di nama file tapi TIDAK ada di "
              f"SOR_PREFIX_MAP_TENTATIVE: {unmapped}. Perlu dikonfirmasi manual.")

    print("\nKolom Anotasi.csv yang tersedia untuk verifikasi manual:", list(anno_df.columns))
    return prefix_counts




# ══════════════════════════════════════════════════════════════════════════
# 3. CEK INTEGRITAS FILE
# ══════════════════════════════════════════════════════════════════════════

def check_duplicate_and_missing_files(file_list_df, anno_df, filename_col_guess=None):
    """
    Cek dua arah:
    1. File di folder snapshot yang tidak punya baris di Anotasi.csv (kalau
       ada kolom nama file yang bisa dicocokkan).
    2. Baris di Anotasi.csv yang menyebut file yang tidak ada di folder.
    filename_col_guess: nama kolom di anno_df yang diduga berisi nama file.
    Kalau None, fungsi ini hanya melaporkan duplikat nama file di folder.
    """
    report = {}

    dup_files = file_list_df[file_list_df.duplicated(subset=["snapshot", "filename"], keep=False)]
    report["duplicate_filenames_within_snapshot"] = len(dup_files)
    if len(dup_files) > 0:
        print("[PERINGATAN] Ada nama file duplikat dalam satu snapshot:")
        print(dup_files)

    empty_files = file_list_df[file_list_df["size_bytes"] < 1024]  # < 1KB, kemungkinan korup/kosong
    report["suspicious_small_files"] = len(empty_files)
    if len(empty_files) > 0:
        print(f"[PERINGATAN] {len(empty_files)} file berukuran < 1KB, kemungkinan kosong/korup:")
        print(empty_files[["snapshot", "filename", "size_bytes"]])

    if filename_col_guess and filename_col_guess in anno_df.columns:
        anno_files = set(anno_df[filename_col_guess].astype(str))
        actual_files = set(file_list_df["filename"])
        missing_on_disk = anno_files - actual_files
        missing_in_annotation = actual_files - anno_files
        report["in_annotation_not_on_disk"] = len(missing_on_disk)
        report["on_disk_not_in_annotation"] = len(missing_in_annotation)
        if missing_on_disk:
            print(f"[PERINGATAN] {len(missing_on_disk)} entri Anotasi.csv menyebut file "
                  f"yang tidak ditemukan di folder: {list(missing_on_disk)[:10]}")
        if missing_in_annotation:
            print(f"[PERINGATAN] {len(missing_in_annotation)} file di folder tidak "
                  f"tercatat di Anotasi.csv: {list(missing_in_annotation)[:10]}")
    else:
        print("[INFO] Kolom nama file di Anotasi.csv belum ditentukan "
              "(filename_col_guess=None atau kolom tidak ditemukan), lewati cross-check dua arah.")

    return report




# ══════════════════════════════════════════════════════════════════════════
# 4. INSPEKSI STRUKTUR TRACE
# ══════════════════════════════════════════════════════════════════════════

def inspect_sample_trace(path):
    """
    Baca satu file xlsx trace OTDR, laporkan sheet name, shape, kolom,
    dtype, dan jumlah missing value. Dipakai untuk memastikan struktur
    file trace konsisten sebelum dipakai untuk windowing.
    """
    xls = pd.ExcelFile(path)
    report = {"path": path, "sheet_names": xls.sheet_names}
    sheet_reports = {}
    for sheet in xls.sheet_names:
        df = xls.parse(sheet)
        sheet_reports[sheet] = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": {c: str(df[c].dtype) for c in df.columns},
            "n_missing_total": int(df.isna().sum().sum()),
        }
    report["sheets"] = sheet_reports
    return report


def inspect_sample_across_prefixes(file_list_df, n_per_prefix=1):
    """
    Ambil beberapa sampel file per prefix, inspeksi strukturnya, laporkan
    apakah struktur (kolom, sheet) KONSISTEN antar prefix/kelas. Struktur
    yang tidak konsisten antar kelas berarti perlu penanganan khusus
    sebelum windowing/stride bisa diterapkan seragam.
    """
    reports = {}
    for prefix, group in file_list_df.groupby("prefix"):
        sample_paths = group["path"].head(n_per_prefix).tolist()
        for p in sample_paths:
            try:
                reports[os.path.basename(p)] = inspect_sample_trace(p)
            except Exception as e:
                reports[os.path.basename(p)] = {"error": str(e)}
    return reports
