# src/sor_windowing.py — Hibah Kemdiktisaintek Part 3
"""
Windowing/stride untuk trace OTDR mentah SOR. window_size dan stride
TETAP (src/config.py SOR_WINDOW_SIZE/SOR_STRIDE = 80/40, TIDAK DIUBAH dari
Part 2), diwarisi dari protokol pengumpulan data lapangan riset paralel
rekan tim. TIDAK ADA pencarian atau optimasi window/stride di project ini.

Daftar isi:
  1. Split file, sebelum windowing  : split_files_by_class, assert_no_file_leakage
  2. Ekstraksi window                : extract_windows, label_windows_by_position,
                                        estimate_event_position_from_signal, windows_from_trace
  3. Bangun dataset window           : build_windowed_dataset, build_split_datasets

PRINSIP ANTI-BOCOR:
1. Split train/val/test dilakukan di level FILE, SEBELUM windowing, bukan
   di level window setelah windowing. Window dari file yang sama sangat
   berkorelasi (bertetangga/overlap), kalau split dilakukan setelah
   windowing, window dari file yang sama bisa tersebar ke train dan
   validation sekaligus, itu kebocoran data.
2. Untuk CV berulang (justifikasi dan final Part 3), prinsip yang sama
   berlaku lewat get_repeated_folds_grouped() di src/dataset_loaders.py
   (group=nama file), BUKAN split_files_by_class()/build_split_datasets()
   di modul ini yang dipakai untuk split tetap gaya train/val/test. Pilih
   salah satu sesuai kebutuhan notebook, jangan campur keduanya untuk
   output yang sama.
3. Window SETELAH posisi event, secara default DIBUANG (bukan dilabeli
   Normal). Untuk gangguan total seperti Fiber Cut, sinyal setelah titik
   putus bukan representasi kondisi serat normal. Perilaku lama (label
   "Normal") masih tersedia lewat post_event_policy='normal' kalau
   diperlukan pembanding.

ASUMSI STRUKTUR, SUDAH DIVERIFIKASI di nb01_eda_all_datasets.ipynb Part 2:
- Tiap file xlsx = satu trace OTDR utuh untuk satu event (atau satu trace
  Normal murni untuk file berprefix N).
- Trace punya satu kolom "posisi" (jarak di sepanjang serat, index sekuens)
  dan satu atau lebih kolom nilai (power/amplitude/dB).

DISTRIBUSI NORMAL SENGAJA TIDAK DIBATASI (TIDAK di-undersample). Window
dari bagian non-event di semua trace dihitung penuh sebagai Normal,
membuat IR SOR jauh lebih ekstrem dibanding OTDR/HardFailure/SoftFailure,
dipakai sebagai uji ketahanan pipeline terhadap imbalance ekstrem.
"""
import numpy as np


# ── Split di level FILE, sebelum windowing (dipakai kalau butuh split tetap) ──



# ══════════════════════════════════════════════════════════════════════════
# 1. SPLIT FILE, SEBELUM WINDOWING
# ══════════════════════════════════════════════════════════════════════════

def split_files_by_class(file_records, train_ratio=0.6, val_ratio=0.2,
                          test_ratio=0.2, seed=42):
    """
    Split file_records (bukan window) jadi train/val/test, stratifikasi per
    kelas (event_class, atau 'Normal' kalau None). Window dari satu file
    HANYA akan muncul di satu split, tidak pernah tersebar.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Total rasio harus 1.0"

    rng = np.random.RandomState(seed)
    by_class = {}
    for i, rec in enumerate(file_records):
        cls = rec["event_class"] if rec["event_class"] is not None else "Normal"
        by_class.setdefault(cls, []).append(i)

    train_idx, val_idx, test_idx = [], [], []
    for cls, idxs in by_class.items():
        idxs = idxs.copy()
        rng.shuffle(idxs)
        n = len(idxs)
        n_test = max(1, round(n * test_ratio))
        n_val = max(1, round(n * val_ratio))
        n_train = n - n_test - n_val
        if n_train < 1:  # kelas dengan file sangat sedikit
            n_train = max(1, n - 2)
            n_val = 1 if (n - n_train - 1) >= 1 else 0
            n_test = n - n_train - n_val
        train_idx.extend(idxs[:n_train])
        val_idx.extend(idxs[n_train:n_train + n_val])
        test_idx.extend(idxs[n_train + n_val:])

    train_records = [file_records[i] for i in train_idx]
    val_records = [file_records[i] for i in val_idx]
    test_records = [file_records[i] for i in test_idx]
    return train_records, val_records, test_records


def assert_no_file_leakage(*record_groups):
    """Sanity check wajib: pastikan tidak ada nama file yang muncul di lebih
    dari satu grup (train/val/test)."""
    seen = {}
    for gi, recs in enumerate(record_groups):
        for rec in recs:
            fname = rec["filename"]
            if fname in seen and seen[fname] != gi:
                raise AssertionError(
                    f"KEBOCORAN: file '{fname}' muncul di grup {seen[fname]} dan {gi}")
            seen[fname] = gi
    return True


# ── Ekstraksi window ─────────────────────────────────────────────────────────



# ══════════════════════════════════════════════════════════════════════════
# 2. EKSTRAKSI WINDOW
# ══════════════════════════════════════════════════════════════════════════

def extract_windows(values, window_size, stride):
    values = np.asarray(values, dtype=np.float32)
    n = len(values)
    if n < window_size:
        return np.empty((0, window_size), dtype=np.float32), np.array([], dtype=int)
    starts = np.arange(0, n - window_size + 1, stride)
    windows = np.stack([values[s:s + window_size] for s in starts])
    return windows, starts


def label_windows_by_position(starts, window_size, event_idx=None, event_class=None,
                               exclusion_margin=0, post_event_policy="discard"):
    """
    event_idx=None -> semua window "Normal" (trace Normal murni).

    Kalau event_idx ada, tiga kondisi:
      - window sebelum event -> "Normal"
      - window berisi event  -> event_class
      - window setelah event -> DIBUANG (None) kalau post_event_policy=
        "discard" (default), atau "Normal" kalau post_event_policy="normal"
        (skema lama, dipertahankan untuk pembanding kalau diperlukan).

    exclusion_margin>0 memperlebar zona "berisi event" di kedua sisi,
    dipakai kalau posisi event dari anotasi diperkirakan kurang presisi.
    """
    labels = []
    for s in starts:
        w_start, w_end = s, s + window_size - 1
        if event_idx is None:
            labels.append("Normal")
            continue

        event_lo, event_hi = event_idx - exclusion_margin, event_idx + exclusion_margin
        if w_start <= event_hi and w_end >= event_lo:
            labels.append(event_class)
        elif w_end < event_lo:
            labels.append("Normal")
        else:  # w_start > event_hi, window setelah event
            labels.append("Normal" if post_event_policy == "normal" else None)
    return np.array(labels, dtype=object)


def estimate_event_position_from_signal(values):
    """FALLBACK SAJA kalau Anotasi.csv tidak menyediakan posisi event
    eksplisit."""
    values = np.asarray(values, dtype=np.float32)
    diff = np.abs(np.diff(values))
    return int(np.argmax(diff))


def windows_from_trace(trace_values, window_size, stride, event_idx=None,
                        event_class=None, exclusion_margin=0, post_event_policy="discard"):
    windows, starts = extract_windows(trace_values, window_size, stride)
    if len(windows) == 0:
        return windows, np.array([])
    labels = label_windows_by_position(starts, window_size, event_idx, event_class,
                                        exclusion_margin, post_event_policy)
    keep = np.array([l is not None for l in labels])
    return windows[keep], labels[keep]




# ══════════════════════════════════════════════════════════════════════════
# 3. BANGUN DATASET WINDOW
# ══════════════════════════════════════════════════════════════════════════

def build_windowed_dataset(file_records, window_size, stride,
                            exclusion_margin=0, post_event_policy="discard"):
    """
    file_records: list of dict {'values', 'event_class', 'event_idx', 'filename'}.
    TIDAK ADA batasan jumlah window Normal (sesuai desain, lihat docstring modul).
    Return X, y, source (nama file asal tiap window, WAJIB dipakai sebagai
    group di get_repeated_folds_grouped() untuk ablasi CV, atau sebagai
    dasar split_files_by_class() untuk split tetap).
    """
    all_windows, all_labels, all_source = [], [], []
    for rec in file_records:
        w, l = windows_from_trace(
            rec["values"], window_size, stride,
            event_idx=rec.get("event_idx"), event_class=rec.get("event_class"),
            exclusion_margin=exclusion_margin, post_event_policy=post_event_policy,
        )
        if len(w) == 0:
            continue
        all_windows.append(w)
        all_labels.append(l)
        all_source.extend([rec.get("filename")] * len(w))

    if not all_windows:
        raise ValueError("Tidak ada window dihasilkan, cek window_size vs panjang trace.")

    X = np.concatenate(all_windows, axis=0)
    y = np.concatenate(all_labels, axis=0)
    source = np.array(all_source)
    return X, y, source


def build_split_datasets(file_records, window_size, stride,
                          train_ratio=0.6, val_ratio=0.2, test_ratio=0.2,
                          exclusion_margin=0, post_event_policy="discard", seed=42):
    """
    Split file DULU, baru windowing per split. Dipakai kalau butuh split
    tetap train/val/test (bukan CV berulang). Untuk ablasi justifikasi/final Part 3,
    pakai build_windowed_dataset() atas SELURUH file_records lalu
    get_repeated_folds_grouped() di src/data_utils.py.
    """
    train_recs, val_recs, test_recs = split_files_by_class(
        file_records, train_ratio, val_ratio, test_ratio, seed)
    assert_no_file_leakage(train_recs, val_recs, test_recs)

    result = {}
    for name, recs in [("train", train_recs), ("val", val_recs), ("test", test_recs)]:
        X, y, source = build_windowed_dataset(
            recs, window_size, stride, exclusion_margin, post_event_policy)
        result[name] = {"X": X, "y": y, "source": source, "n_files": len(recs)}
    return result
