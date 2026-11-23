# src/metrics.py — Hibah Kemdiktisaintek Part 3
"""
Perhitungan metrik lengkap: Accuracy, Balanced Accuracy, macro-F1, MCC,
precision/recall/F1 KESELURUHAN dan PER KELAS, confusion matrix, data
PR curve. Semua dihitung dari y_true/y_pred/y_proba, bisa dipanggil ulang
kapan saja dari prediksi yang tersimpan di results/predictions/ tanpa
retrain (poin metrik lengkap Part 3).

Daftar isi:
  1. Metrik satu fold         : compute_metrics
  2. Agregasi lintas fold     : aggregate_fold_metrics (mean+-std, plus waktu
                                 training/inference kalau timing_list diisi)
  3. Data kurva PR per kelas  : pr_curve_data

Accuracy TETAP bukan metrik utama untuk keputusan (macro-F1 yang utama,
lihat instruksi proyek), tapi selalu dilaporkan untuk kelengkapan.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    balanced_accuracy_score, matthews_corrcoef, confusion_matrix,
)




# ══════════════════════════════════════════════════════════════════════════
# 1. METRIK SATU FOLD
# ══════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred, class_names):
    """
    Mengembalikan dict lengkap: metrik agregat + metrik per kelas
    (precision, recall, F1) + confusion matrix.
    """
    n_classes = len(class_names)
    labels = list(range(n_classes))

    acc      = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)
    bacc     = balanced_accuracy_score(y_true, y_pred)
    mcc      = matthews_corrcoef(y_true, y_pred)

    precision_pc = precision_score(y_true, y_pred, average=None, zero_division=0, labels=labels)
    recall_pc    = recall_score(y_true, y_pred, average=None, zero_division=0, labels=labels)
    f1_pc        = f1_score(y_true, y_pred, average=None, zero_division=0, labels=labels)
    cm           = confusion_matrix(y_true, y_pred, labels=labels)

    per_class = {
        class_names[i]: {
            "precision": float(precision_pc[i]),
            "recall": float(recall_pc[i]),
            "f1": float(f1_pc[i]),
        }
        for i in range(n_classes)
    }

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "bacc": float(bacc),
        "mcc": float(mcc),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }




# ══════════════════════════════════════════════════════════════════════════
# 2. AGREGASI LINTAS FOLD
# ══════════════════════════════════════════════════════════════════════════

def aggregate_fold_metrics(fold_metrics_list, timing_list=None):
    """
    Agregasi metrik dari beberapa observasi fold-repeat jadi mean +/- std.
    fold_metrics_list : list of dict dari compute_metrics().
    timing_list       : list of dict {"t_train_s": ..., "t_infer_bs*_s": ...,
                         "t_infer_bs*_per_sample_ms": ...} per fold-repeat
                         yang sama urutannya dengan fold_metrics_list. Kalau
                         diisi, mean+std waktu training/inference ikut
                         diagregasi supaya bisa dibaca langsung dari
                         checkpoint CSV tanpa buka results/raw/.
    """
    accs      = np.array([m["accuracy"] for m in fold_metrics_list])
    macro_f1s = np.array([m["macro_f1"] for m in fold_metrics_list])
    baccs     = np.array([m["bacc"] for m in fold_metrics_list])
    mccs      = np.array([m["mcc"] for m in fold_metrics_list])

    class_names = list(fold_metrics_list[0]["per_class"].keys())
    per_class_agg = {}
    for cname in class_names:
        f1s = np.array([m["per_class"][cname]["f1"] for m in fold_metrics_list])
        pre = np.array([m["per_class"][cname]["precision"] for m in fold_metrics_list])
        rec = np.array([m["per_class"][cname]["recall"] for m in fold_metrics_list])
        per_class_agg[cname] = {
            "precision_mean": float(pre.mean()), "precision_std": float(pre.std()),
            "recall_mean": float(rec.mean()), "recall_std": float(rec.std()),
            "f1_mean": float(f1s.mean()), "f1_std": float(f1s.std()),
        }

    agg = {
        "accuracy_mean": float(accs.mean()), "accuracy_std": float(accs.std()),
        "macro_f1_mean": float(macro_f1s.mean()), "macro_f1_std": float(macro_f1s.std()),
        "bacc_mean": float(baccs.mean()), "bacc_std": float(baccs.std()),
        "mcc_mean": float(mccs.mean()), "mcc_std": float(mccs.std()),
        "per_class": per_class_agg,
        "n_observations": len(fold_metrics_list),
        # Per-fold arrays — dibutuhkan oleh uji signifikansi Wilcoxon berpasangan
        "accuracy_per_fold":  accs.tolist(),
        "macro_f1_per_fold":  macro_f1s.tolist(),
        "bacc_per_fold":      baccs.tolist(),
        "mcc_per_fold":       mccs.tolist(),
    }

    if timing_list:
        timing_keys = sorted({k for t in timing_list for k in t.keys()})
        timing_agg = {}
        for k in timing_keys:
            vals = np.array([t[k] for t in timing_list if k in t])
            timing_agg[f"{k}_mean"] = float(vals.mean())
            timing_agg[f"{k}_std"] = float(vals.std())
        agg["timing"] = timing_agg

    return agg




# ══════════════════════════════════════════════════════════════════════════
# 3. DATA KURVA PR PER KELAS
# ══════════════════════════════════════════════════════════════════════════

def pr_curve_data(y_true, y_proba, class_names):
    """
    Data precision-recall curve per kelas (one-vs-rest), disimpan mentah
    supaya plot bisa dibuat ulang kapan saja dari y_proba yang tersimpan.
    """
    from sklearn.metrics import precision_recall_curve
    n_classes = len(class_names)
    curves = {}
    for i in range(n_classes):
        y_bin = (np.asarray(y_true) == i).astype(int)
        precision, recall, thresholds = precision_recall_curve(y_bin, y_proba[:, i])
        curves[class_names[i]] = {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
        }
    return curves
