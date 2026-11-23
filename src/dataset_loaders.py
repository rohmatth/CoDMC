# src/dataset_loaders.py — Hibah Kemdiktisaintek Part 3
"""
Loading dan preprocessing 4 dataset (OTDR, HardFailure, SoftFailure, SOR).
TIDAK ADA kolom atau langkah terkait kDN/complexity di sini (eksklusif tesis).

Daftar isi:
  1. OTDR (tabular)                         : load_otdr
  2. HardFailure / SoftFailure (tabular)    : load_failure_wide
  3. SOR (windowed, sequential)             : load_sor_windowed, _build_file_records
  4. Util pelaporan                         : _report_class_balance
  5. Skema CV                               : get_repeated_folds, get_repeated_folds_grouped, scale_fold

class_names di-pass eksplisit per fungsi (dari src/config.py DATASETS[...]
["class_names"]), bukan konstanta global tunggal, karena keempat dataset
punya skema kelas berbeda (OTDR 8 kelas, HardFailure/SoftFailure biner,
SOR 6 kelas) dimuat dalam satu pipeline yang sama.

TIDAK ADA pencarian window/stride SOR di sini. load_sor_windowed() memakai
SOR_WINDOW_SIZE/SOR_STRIDE TETAP dari config.py (TIDAK DIUBAH dari Part 2).
"""
import os
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.config import SEED, SOR_WINDOW_SIZE, SOR_STRIDE, SOR_POST_EVENT_POLICY


OTDR_DROP = ["Class", "Position", "Reflectance", "loss"]




# ══════════════════════════════════════════════════════════════════════════
# 1. OTDR (TABULAR)
# ══════════════════════════════════════════════════════════════════════════

def load_otdr(path, class_names):
    """
    Label   : kolom 'Class'
    Fitur   : SNR, P1-P30, Maximum_Amplitude
    Drop    : Class, Position, Reflectance, loss (leakage, sama seperti Part 1)
    """
    df = pd.read_excel(path)
    if "Class" not in df.columns:
        raise ValueError(f"Kolom 'Class' tidak ditemukan. Kolom: {list(df.columns)}")

    le = LabelEncoder()
    y  = le.fit_transform(df["Class"].astype(int)).astype(int)

    drop         = [c for c in OTDR_DROP if c in df.columns]
    feature_cols = [c for c in df.columns if c not in drop]

    X = (df[feature_cols]
         .apply(pd.to_numeric, errors="coerce")
         .fillna(0)
         .values.astype(np.float32))

    _report_class_balance("OTDR", y, class_names, extra=f"features={len(feature_cols)}")
    return X, y, feature_cols




# ══════════════════════════════════════════════════════════════════════════
# 2. HARDFAILURE / SOFTFAILURE (TABULAR)
# ══════════════════════════════════════════════════════════════════════════

def load_failure_wide(path, class_names, name="Failure"):
    """HardFailure / SoftFailure, format panjang -> lebar (pivot per Timestamp)."""
    df = pd.read_csv(path, sep=";")
    df["_fail"] = df["Failure"].notna().astype(int)

    for c in ["BER", "OSNR", "InputPower", "OutputPower"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    y_ts   = df.groupby("Timestamp")["_fail"].max()
    pivots = []
    for metric in ["BER", "OSNR", "InputPower", "OutputPower"]:
        if metric not in df.columns:
            continue
        sub = df[df[metric].notna()][["Timestamp", "ID", metric]].copy()
        if sub.empty:
            continue
        piv = sub.pivot_table(index="Timestamp", columns="ID", values=metric, aggfunc="mean")
        piv.columns = [f"{metric}_{str(c).replace('/', '_').replace(' ', '_')}" for c in piv.columns]
        pivots.append(piv)

    if not pivots:
        raise ValueError(f"Tidak ada fitur pivotable dari {path}")

    wide = pd.concat(pivots, axis=1).reindex(y_ts.index).sort_index()
    y_ts = y_ts.sort_index()
    wide = wide.ffill().bfill()
    for col in wide.columns:
        if wide[col].isna().any():
            med = wide[col].median()
            wide[col] = wide[col].fillna(0 if pd.isna(med) else med)
    wide = wide.fillna(0)

    X = wide.values.astype(np.float32)
    y = y_ts.values.astype(int)
    _report_class_balance(name, y, class_names)
    return X, y, list(wide.columns)




# ══════════════════════════════════════════════════════════════════════════
# 3. SOR (WINDOWED, SEQUENTIAL)
# ══════════════════════════════════════════════════════════════════════════

def load_sor_windowed(class_names, window_size=SOR_WINDOW_SIZE, stride=SOR_STRIDE,
                       post_event_policy=SOR_POST_EVENT_POLICY, seed=SEED):
    """
    Bangun dataset window SOR dari sor_snapshots/ dengan window_size/stride
    TETAP (lihat README.md "Provenance window/stride SOR", TIDAK ada
    pencarian di sini). Mengembalikan X, y (index kelas, bukan string),
    dan groups (id file asal tiap window), WAJIB dipakai bersama
    get_repeated_folds_grouped() supaya tidak ada window dari file yang
    sama nyebrang ke fold train dan val sekaligus.
    """
    from src.sor_raw_reader import list_snapshot_files, load_annotation
    from src.sor_windowing import build_windowed_dataset

    file_list_df = list_snapshot_files()
    anno_df      = load_annotation()

    file_records = _build_file_records(file_list_df, anno_df, class_names)

    X, y_str, source = build_windowed_dataset(
        file_records, window_size=window_size, stride=stride,
        post_event_policy=post_event_policy)

    name_to_idx = {c: i for i, c in enumerate(class_names)}
    unknown = sorted(set(y_str) - set(name_to_idx))
    if unknown:
        raise ValueError(f"Label SOR tidak dikenal di class_names: {unknown}")
    y = np.array([name_to_idx[c] for c in y_str], dtype=int)

    X = X.astype(np.float32)
    _report_class_balance("SOR", y, class_names, extra=f"n_files={len(file_records)}")
    return X, y, source  # source dipakai sebagai groups di CV


def _build_file_records(file_list_df, anno_df, class_names):
    """
    Gabungkan daftar file snapshot SOR dengan label dari Anotasi.csv.

    Struktur Anotasi.csv (dikonfirmasi dari file asli):
    kolom = File, Splice loss, Refl loss, Slope, Peak, Index Peak,
    Start of curr, End of curr, Panjang Event (m), Resolution, Class.

    - "File"       : nama file TANPA ekstensi (mis. "A1", "D11", "N1"),
                      dicocokkan ke file_list_df["filename"] (BERISI
                      ekstensi .xlsx) dengan strip ekstensi + lowercase.
    - "Class"      : label kelas LANGSUNG per file (mis. "Air Gap",
                      "Dirty Connector", "Normal"). Kolom ini OTORITATIF,
                      SOR_PREFIX_MAP_TENTATIVE di config.py TIDAK dipakai
                      lagi untuk menentukan kelas (cuma bekas dugaan awal
                      sebelum Anotasi.csv diperiksa).
    - "Index Peak" : posisi event di dalam trace (index integer). KOSONG
                      untuk file "Normal" murni (tidak ada event), yang
                      berarti event_idx=None, seluruh trace dilabeli
                      "Normal" oleh windows_from_trace().

    Nilai trace (kolom power/amplitude di dalam file .xlsx) dibaca dari
    kolom bernama SOR_VALUE_COLUMN (src/config.py). WAJIB diisi manual
    dulu setelah inspect_sample_trace() dijalankan di
    nb01_eda_all_datasets.ipynb, karena Anotasi.csv sendiri tidak memuat
    trace mentah, cuma metadata event per file.
    """
    from src.config import SOR_VALUE_COLUMN

    if SOR_VALUE_COLUMN is None:
        raise ValueError(
            "SOR_VALUE_COLUMN di src/config.py masih None. Jalankan "
            "inspect_sample_trace() di nb01_eda_all_datasets.ipynb dulu "
            "untuk melihat nama kolom nilai trace (power/amplitude) di "
            "file .xlsx SOR, lalu isi SOR_VALUE_COLUMN dengan nama kolom "
            "itu sebelum _build_file_records() bisa dipakai."
        )

    anno = anno_df.copy()
    anno["_file_key"] = anno["File"].astype(str).str.strip().str.lower()

    files = file_list_df.copy()
    files["_file_key"] = (
        files["filename"].astype(str).str.strip().str.lower()
        .str.replace(r"\.xlsx$", "", regex=True)
    )

    merged = files.merge(anno, on="_file_key", how="inner", suffixes=("", "_anno"))

    missing_anno = set(files["_file_key"]) - set(anno["_file_key"])
    missing_file = set(anno["_file_key"]) - set(files["_file_key"])
    if missing_anno:
        print(f"[PERINGATAN] {len(missing_anno)} file di folder snapshot TIDAK "
              f"punya baris di Anotasi.csv, dilewati: {sorted(missing_anno)[:10]}")
    if missing_file:
        print(f"[PERINGATAN] {len(missing_file)} baris Anotasi.csv menyebut "
              f"file yang TIDAK ditemukan di folder snapshot, dilewati: "
              f"{sorted(missing_file)[:10]}")
    if len(merged) == 0:
        raise ValueError(
            "Tidak ada file yang cocok antara folder snapshot dan Anotasi.csv. "
            "Cek SOR_DIR/SOR_SNAPSHOT_DATES di config.py dan format kolom "
            "'File' di Anotasi.csv."
        )

    unknown_classes = set(merged["Class"].dropna().unique()) - set(class_names)
    if unknown_classes:
        raise ValueError(
            f"Kelas di Anotasi.csv tidak ada di class_names: {unknown_classes}. "
            f"class_names saat ini: {class_names}. Perbaiki SOR_CLASS_NAMES "
            f"di config.py supaya cocok dengan label asli di Anotasi.csv."
        )

    file_records = []
    skipped_read_error = []
    for _, row in merged.iterrows():
        try:
            df_trace = pd.read_excel(row["path"])
        except Exception as e:
            skipped_read_error.append((row["filename"], str(e)))
            continue

        if SOR_VALUE_COLUMN not in df_trace.columns:
            raise ValueError(
                f"Kolom '{SOR_VALUE_COLUMN}' tidak ditemukan di {row['path']}. "
                f"Kolom tersedia: {list(df_trace.columns)}. Periksa ulang "
                f"SOR_VALUE_COLUMN di config.py."
            )
        # Urutkan eksplisit berdasarkan kolom "Index" sebelum ambil nilai
        # trace, JANGAN asumsikan baris di file .xlsx sudah terurut posisi.
        # Pola ini mengikuti load_series() pada notebook pengolahan SOR
        # rekan tim, yang sudah divalidasi bekerja terhadap 610 file asli.
        if "Index" in df_trace.columns:
            df_trace = df_trace.sort_values("Index")
        values = df_trace[SOR_VALUE_COLUMN].to_numpy(dtype=np.float32)

        cls = row["Class"]
        index_peak = row.get("Index Peak")
        is_normal = (cls == "Normal") or pd.isna(index_peak)
        event_class = None if is_normal else cls
        event_idx = None if pd.isna(index_peak) else int(index_peak)

        file_records.append({
            "filename": row["filename"],
            "values": values,
            "event_class": event_class,
            "event_idx": event_idx,
        })

    if skipped_read_error:
        print(f"[PERINGATAN] {len(skipped_read_error)} file gagal dibaca, "
              f"dilewati: {skipped_read_error[:5]}")

    print(f"[_build_file_records] {len(file_records)} file berhasil digabung "
          f"dengan anotasi (dari {len(files)} file di folder, {len(anno)} "
          f"baris Anotasi.csv).")
    return file_records




# ══════════════════════════════════════════════════════════════════════════
# 4. UTIL PELAPORAN
# ══════════════════════════════════════════════════════════════════════════

def _report_class_balance(name, y, class_names, extra=""):
    cnt     = Counter(y.tolist())
    major_n = max(cnt.values())
    print(f"[{name}] n={len(y)}  n_classes={len(class_names)}  {extra}")
    for i, cname in enumerate(class_names):
        n_i = cnt.get(i, 0)
        ir  = major_n / max(n_i, 1)
        print(f"  [{i}]  {cname:<20} n={n_i:>7}  IR={ir:>7.1f}")




# ══════════════════════════════════════════════════════════════════════════
# 5. SKEMA CV
# ══════════════════════════════════════════════════════════════════════════

def get_repeated_folds(X, y, n_splits, n_repeats, seed=SEED):
    """
    Fold biasa, level baris data. Dipakai untuk OTDR, HardFailure,
    SoftFailure, di mana tiap baris independen (bukan window dari trace
    kontinu). JANGAN dipakai untuk SOR, lihat get_repeated_folds_grouped().
    """
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    return list(rskf.split(X, y))


def get_repeated_folds_grouped(X, y, groups, n_splits, n_repeats, seed=SEED):
    """
    Fold anti-bocor level file, WAJIB untuk SOR. Window dari file (source)
    yang sama tidak pernah nyebrang ke fold train dan val dalam split yang
    sama. StratifiedGroupKFold TIDAK punya varian "repeated" bawaan
    sklearn, jadi repeat diimplementasikan manual: tiap repeat memakai
    random_state berbeda (seed + repeat_idx * 1000), split TETAP
    deterministik dan reproducible.

    GUARD PENTING: StratifiedGroupKFold TIDAK error kalau suatu kelas
    berasal dari file lebih sedikit daripada n_splits, tapi diam-diam
    menghasilkan fold validasi dengan NOL sampel kelas itu (bukan
    stratifikasi gagal, cuma kelas itu kosong di fold tersebut). Fold
    kosong begini mencemari rata-rata metrik tanpa error apapun, sangat
    mungkin terjadi di SOR karena kelas langka (mis. Fiber Cut) bisa cuma
    berasal dari segelintir file. Fungsi ini mengecek jumlah file (group)
    unik per kelas, dan kalau lebih kecil dari n_splits yang diminta,
    n_splits DITURUNKAN otomatis ke jumlah minimum itu (minimal 2), dengan
    peringatan eksplisit dicetak. Ini konsisten untuk semua kandidat dalam
    dataset SOR yang sama, jadi tidak merusak fairness antar kandidat,
    cuma membuat n_splits realistis mengikuti data yang benar-benar ada.

    Return: list of (train_idx, val_idx), format identik dengan
    get_repeated_folds().
    """
    import pandas as pd
    groups = np.asarray(groups)
    y = np.asarray(y)

    n_groups_per_class = pd.Series(groups).groupby(y).nunique()
    min_groups = int(n_groups_per_class.min())
    effective_n_splits = min(n_splits, max(min_groups, 2))
    if effective_n_splits < n_splits:
        print(f"[PERINGATAN] n_splits diminta={n_splits}, tapi kelas dengan "
              f"file paling sedikit cuma punya {min_groups} file unik. "
              f"n_splits DITURUNKAN ke {effective_n_splits} supaya tidak ada "
              f"fold validasi dengan nol sampel kelas tersebut. Distribusi "
              f"file per kelas:\n{n_groups_per_class.to_string()}")

    all_splits = []
    for r in range(n_repeats):
        sgkf = StratifiedGroupKFold(n_splits=effective_n_splits, shuffle=True,
                                     random_state=seed + r * 1000)
        all_splits.extend(list(sgkf.split(X, y, groups=groups)))
    return all_splits


def scale_fold(X_tr, X_va):
    sc = StandardScaler()
    return sc.fit_transform(X_tr).astype(np.float32), sc.transform(X_va).astype(np.float32), sc
