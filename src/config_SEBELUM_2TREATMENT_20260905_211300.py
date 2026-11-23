# src/config.py — Hibah Kemdiktisaintek Part 3
"""
Satu sumber kebenaran untuk path, hyperparameter, dan skema CV Part 3.

Daftar isi:
  1. Path dataset dan hasil       (baris ~30)
  2. Konfigurasi tetap SOR        (baris ~55)
  3. Registry dataset             (baris ~80)
  4. Skema CV: final vs justifikasi (baris ~105)
  5. Hyperparameter training      (baris ~130)

BEDA DARI PART 2:
Part 2 punya skema "explore" (3-fold/1-repeat, sementara) dan "final"
(5-fold/3-repeat). Part 3 TIDAK memakai pola itu lagi. Part 3 punya DUA
skema paralel dengan tujuan berbeda, bukan tahapan progresif:
  - CV_JUSTIFIKASI (3-fold/1-repeat): dipakai KHUSUS notebook nb02 Bagian A
    (alasan pemilihan backbone/resampler/loss/meta/treatment pair), bukan
    klaim utama, sengaja ringkas.
  - CV_FINAL (10-fold/1-repeat): dipakai proposed (HDMC multiview) DAN
    seluruh pembanding (baseline+eksternal) di nb02 Bagian B dan nb03,
    WAJIB identik antara proposed dan pembanding supaya uji Wilcoxon
    berpasangan valid. Keputusan pindah dari 5-fold/3-repeat (protokol
    yang sudah dipakai di laporan kemajuan) ke 10-fold/1-repeat diambil
    karena artifact (model/OOF/prediksi) belum tersimpan lengkap di
    proses sebelumnya sehingga rerun tidak terhindarkan, dan 10-fold/1-
    repeat tetap 10 observasi (di atas batas minimum 6 untuk Wilcoxon)
    dengan biaya waktu lebih rendah. Angka di laporan kemajuan (15 kali
    pengujian) TETAP SAH sebagai laporan progress, tidak direvisi. Angka
    final ada di naskah jurnal, memakai protokol Part 3 ini.

TIDAK ADA kDN, complexity profiling, atau Complexity-Aware Treatment
Assignment di project ini (eksklusif tesis CAMP-IFD).

TIDAK ADA pencarian window/stride SOR di project ini. window=80, stride=40
adalah KONFIGURASI TETAP, TIDAK BERUBAH dari Part 2, supaya IR SOR (558,51)
tetap konsisten dengan yang sudah dilaporkan ke laporan kemajuan dan draft
jurnal. Lihat README.md.
"""
import os

# ══════════════════════════════════════════════════════════════════════════
# 1. PATH DATASET DAN HASIL
# ══════════════════════════════════════════════════════════════════════════
# Struktur folder tidak berubah dari Part 2 (dataset dipakai bersama):
#   Code/
#   ├── dataset/
#   │   ├── OTDR_Imbalance_Final.xlsx
#   │   ├── HardFailure_dataset.csv
#   │   ├── SoftFailure_dataset.csv
#   │   └── Hibah/
#   │       ├── Anotasi.csv
#   │       └── sor_snapshots/
#   ├── hibah_part1/
#   ├── hibah_part2/
#   ├── hibah_part3/    <- project ini
#   └── ...
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.abspath(os.path.join(PROJ_ROOT, "..", "dataset"))

OTDR_FILE = os.path.join(DATA_DIR, "OTDR_Imbalance_Final.xlsx")
HARD_FILE = os.path.join(DATA_DIR, "HardFailure_dataset.csv")
SOFT_FILE = os.path.join(DATA_DIR, "SoftFailure_dataset.csv")

RESULT_DIR = os.path.join(PROJ_ROOT, "results")
RAW_DIR    = os.path.join(RESULT_DIR, "raw")          # artifact lengkap per fold
TAB_DIR    = os.path.join(RESULT_DIR, "tables")        # checkpoint CSV/JSON
FIG_DIR    = os.path.join(RESULT_DIR, "figures")
MDL_DIR    = os.path.join(RESULT_DIR, "models")        # model_state_dict penuh
PRED_DIR   = os.path.join(RESULT_DIR, "predictions")   # OOF proba + y_true per fold

# ══════════════════════════════════════════════════════════════════════════
# 2. KONFIGURASI TETAP SOR (TIDAK DIUBAH DARI PART 2)
# ══════════════════════════════════════════════════════════════════════════
SOR_DIR             = os.path.join(DATA_DIR, "Hibah", "sor_snapshots")
SOR_ANNOTATION_FILE = os.path.join(DATA_DIR, "Hibah", "Anotasi.csv")
SOR_SNAPSHOT_DATES  = ["snapshot_2026-07-18", "snapshot_2026-07-26"]

# Window/stride TETAP, warisan protokol tim pengumpul data lapangan.
# JANGAN DIUBAH, IR SOR (558,51) sudah dilaporkan ke BIMA dan draft jurnal
# berdasarkan angka ini.
SOR_WINDOW_SIZE = 80
SOR_STRIDE      = 40
SOR_POST_EVENT_POLICY = "discard"

SOR_PREFIX_MAP_TENTATIVE = {
    "A": "Air Gap", "BD": "Bending", "BS": "Bad Splice",
    "D": "Dirty Connector", "FC": "Fiber Cut", "N": "Normal",
}
SOR_CLASS_NAMES = ["Normal", "Air Gap", "Bad Splice", "Bending", "Dirty Connector", "Fiber Cut"]

# Kolom nilai trace (amplitudo) di file .xlsx snapshot SOR. Dikonfirmasi
# dari inspect_sample_trace() (file A1.xlsx): Distance, Backscatter, Slope,
# Index, Wavelength_nm, PulseWidth_ns, AveragingTime_s tersedia sebagai
# kolom. "Backscatter" dipakai sebagai fitur, "Distance" sumbu posisi.
SOR_VALUE_COLUMN = "Backscatter"

# ══════════════════════════════════════════════════════════════════════════
# 3. REGISTRY DATASET
# ══════════════════════════════════════════════════════════════════════════
# "grouped": True -> CV WAJIB pakai get_repeated_folds_grouped() (per file,
# anti-bocor), bukan get_repeated_folds() biasa. Lihat dataset_loaders.py.
DATASETS = {
    "sor": {
        "file": None,  # dibangun dari SOR_DIR via sor_windowing.py
        "class_names": SOR_CLASS_NAMES,
        "grouped": True,
    },
    "otdr": {
        "file": OTDR_FILE,
        "class_names": ["Normal", "Fiber Tapping", "Bad Splice", "Bending Event",
                         "Dirty Connector", "Fiber Cut", "PC Connector", "Reflector"],
        "grouped": False,
    },
    "hardfailure": {
        "file": HARD_FILE,
        "class_names": ["Normal", "Failure"],
        "grouped": False,
    },
    "softfailure": {
        "file": SOFT_FILE,
        "class_names": ["Normal", "Failure"],
        "grouped": False,
    },
}
# Urutan tersulit -> termudah (berdasar Imbalance Ratio), dipakai konsisten
# di seluruh notebook (EDA, eksperimen utama, baseline/eksternal):
# SOR (IR=558,51) -> OTDR (IR=33,36) -> HardFailure (IR=18,62) ->
# SoftFailure (IR=9,41).
DATASET_ORDER = ["sor", "otdr", "hardfailure", "softfailure"]

IMBALANCE_RATIO = {  # dicatat manual untuk referensi cepat di notebook/plot,
    "sor": 558.51,   # nilai SEBENARNYA tetap dihitung ulang dari label saat
    "otdr": 33.36,   # EDA (nb01), ini cuma acuan cepat.
    "hardfailure": 18.62,
    "softfailure": 9.41,
}

for _d in [RAW_DIR, TAB_DIR, FIG_DIR, PRED_DIR,
           os.path.join(MDL_DIR, "backbones"), os.path.join(MDL_DIR, "comparators"),
           os.path.join(MDL_DIR, "meta")]:
    os.makedirs(_d, exist_ok=True)
for _ds in DATASETS:
    os.makedirs(os.path.join(RAW_DIR, _ds), exist_ok=True)
    os.makedirs(os.path.join(PRED_DIR, _ds), exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════
# 3b. NAMA TAMPILAN UNTUK VISUALISASI (naskah IEEE Access, Bahasa Inggris)
# ══════════════════════════════════════════════════════════════════════════
# Nama internal (key dict, nama folder model, nama file checkpoint) TETAP
# memakai nama penulis paper (mis. "Abdelli-BiGRU") untuk ketertelusuran ke
# sumber arsitektur. Nama TAMPILAN di plot/figure yang akan masuk naskah
# TIDAK memakai nama penulis, cuma nama arsitektur teknis, konsisten dengan
# gaya penamaan IEEE Access. Dipakai di nb01/nb04/nb05 lewat display_name().
DISPLAY_NAMES = {
    # Proposed
    "HDMC": "HDMC (Proposed)",
    # Eksternal — TANPA nama penulis, cuma nama arsitektur teknis (gaya IEEE Access)
    "Abdelli-BiGRU": "BiGRU-Attention",
    "Soothar-CNNLSTM": "CNN-LSTM",
    "Yang-IMCBiLSTM": "IMC-BiLSTM",
    "Healy-GRUClassifier": "GRU-Autoencoder",
    "Titouni-CNNExtractor": "CNN-Ensemble",
    # Baseline
    "Baseline-CNNOnly": "CNN-Only",
    "Baseline-BiGRUOnly": "BiGRU-Only",
    "Baseline-BiLSTMOnly": "BiLSTM-Only",
    "Baseline-LogReg": "Logistic Regression",
    "Baseline-RandomForest": "Random Forest",
    "Baseline-SVM": "SVM (Linear)",
}


def display_name(internal_name):
    """Nama tampilan bersih untuk plot (bukan nama penulis paper), dipakai
    di seluruh judul/label/legend figure yang berpotensi masuk naskah.
    Kalau nama tidak ada di mapping, dikembalikan apa adanya (aman untuk
    nama yang belum terdaftar, tidak crash)."""
    return DISPLAY_NAMES.get(internal_name, internal_name)

# ══════════════════════════════════════════════════════════════════════════
# 4. SKEMA CV: JUSTIFIKASI vs FINAL
# ══════════════════════════════════════════════════════════════════════════
# Justifikasi (nb02 Bagian A): alasan desain (backbone/resampler/loss/
# meta/treatment pair), BUKAN klaim utama, sengaja ringkas.
N_FOLDS_JUSTIFIKASI   = 3
N_REPEATS_JUSTIFIKASI = 1

# Final (nb02 Bagian B + nb03): proposed (HDMC multiview) dan SELURUH
# pembanding (baseline+eksternal), WAJIB identik untuk validitas uji
# Wilcoxon berpasangan. 10 observasi, di atas batas minimum 6.
N_FOLDS_FINAL   = 10
N_REPEATS_FINAL = 1

SEED = 42

# ══════════════════════════════════════════════════════════════════════════
# 5. HYPERPARAMETER TRAINING (SATU untuk semua kandidat, semua dataset)
# ══════════════════════════════════════════════════════════════════════════
BATCH_SIZE   = 256
MAX_EPOCHS   = 50   # protokol FINAL (run_hdmc_pipeline, comparator_runner), rigor penuh
LR           = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE     = 7    # protokol FINAL

# Budget LEBIH RINGAN, KHUSUS tahap justifikasi (justifikasi_backbone/
# resampler/loss, justifikasi_pemasangan_treatment). Tujuan justifikasi
# cuma PERBANDINGAN RELATIF antar kandidat (siapa lebih baik dari siapa),
# bukan angka absolut final, jadi epoch budget lebih kecil sudah cukup
# valid untuk menentukan ranking, dengan biaya waktu jauh lebih rendah.
# TIDAK dipakai di run_hdmc_pipeline (protokol final tetap MAX_EPOCHS/
# PATIENCE penuh di atas).
MAX_EPOCHS_JUSTIFIKASI = 15
PATIENCE_JUSTIFIKASI   = 5

# Batch size yang diukur saat inference timing sweep, dicatat untuk SEMUA
# kandidat (proposed, baseline, eksternal), sesuai kebutuhan pelaporan
# waktu training+inference lengkap per model (poin metrik wajib Part 3).
INFER_BATCH_SIZES = [1, 32, 256]
TIMING_N_REPEATS  = 5

# Meta-learner search budget (SETARA untuk kelima kandidat justifikasi meta
# dan meta-learner final HDMC).
META_GRID_BUDGET = 5

# SOR: repeat run untuk stabilitas (nondeterminism cuDNN, lihat
# training_engine.py). Berlaku untuk justifikasi dan final.
SOR_STABILITY_CHECK = True

# ── Plot ──────────────────────────────────────────────────────────────────
PLOT_DPI    = 300
FIGSIZE_STD = (10, 6)
FIGSIZE_WIDE = (14, 6)
CLASS_COLORS_LIST = [
    "#2196F3", "#FF5722", "#4CAF50", "#FFC107",
    "#9C27B0", "#F44336", "#00BCD4", "#795548",
]
