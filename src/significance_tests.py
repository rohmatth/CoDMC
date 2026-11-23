# src/significance_tests.py — Hibah Kemdiktisaintek Part 3
"""
Uji statistik dipakai KONSISTEN di seluruh justifikasi (backbone/resampler/
loss/meta) dan uji signifikansi final (proposed HDMC vs seluruh 10
pembanding, nb02/nb03).

Daftar isi:
  1. Uji beda >=3 kandidat        : friedman_test
  2. Effect size non-parametrik   : cliffs_delta
  3. Uji post-hoc berpasangan     : wilcoxon_pairwise
  4. Pemilihan pemenang otomatis  : pick_winner, auto_select_winner

friedman_test    : uji beda >=3 kandidat pada observasi berpasangan (fold sama)
wilcoxon_pairwise: uji post-hoc tiap pasangan kandidat, atau tiap kandidat
                   vs satu baseline (dipakai untuk proposed vs 10 pembanding,
                   baseline=nama_proposed)
cliffs_delta     : effect size non-parametrik, dilaporkan berdampingan
                   dengan p-value, bukan pengganti p-value

CATATAN VALIDITAS: Wilcoxon berpasangan mensyaratkan jumlah observasi dan
struktur fold yang SAMA PERSIS antara dua kandidat yang dibandingkan. Untuk
Part 3, ini berarti proposed dan seluruh pembanding WAJIB dievaluasi di
protokol final yang identik (n_folds=10, n_repeats=1, lihat config.py),
bukan hasil dari protokol berbeda (mis. proposed 10-fold dibanding
eksternal yang masih di protokol justifikasi 3-fold).
"""
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import friedmanchisquare, wilcoxon




# ══════════════════════════════════════════════════════════════════════════
# 1. UJI BEDA >=3 KANDIDAT
# ══════════════════════════════════════════════════════════════════════════

def friedman_test(scores_by_candidate):
    """
    scores_by_candidate: dict {nama_kandidat: array skor per fold (harus
    urutan fold yang SAMA persis antar kandidat, karena Friedman menguji
    observasi berpasangan).
    """
    names  = list(scores_by_candidate.keys())
    arrays = [np.asarray(scores_by_candidate[n]) for n in names]
    lengths = {len(a) for a in arrays}
    if len(lengths) != 1:
        raise ValueError(f"Jumlah observasi tidak konsisten antar kandidat: {lengths}. "
                          f"Pastikan semua kandidat dievaluasi pada fold index yang sama.")
    stat, p = friedmanchisquare(*arrays)
    return {"statistic": float(stat), "p_value": float(p), "n_candidates": len(names),
            "n_observations": len(arrays[0])}




# ══════════════════════════════════════════════════════════════════════════
# 2. EFFECT SIZE NON-PARAMETRIK
# ══════════════════════════════════════════════════════════════════════════

def cliffs_delta(x, y):
    """
    Cliff's delta non-parametrik. Interpretasi umum:
    |d| < 0.147 negligible, < 0.33 small, < 0.474 medium, else large
    (Romano et al. 2006).
    """
    x, y = np.asarray(x), np.asarray(y)
    n_x, n_y = len(x), len(y)
    more = sum(xi > yi for xi in x for yi in y)
    less = sum(xi < yi for xi in x for yi in y)
    delta = (more - less) / (n_x * n_y)

    ad = abs(delta)
    if ad < 0.147:
        effect = "negligible"
    elif ad < 0.33:
        effect = "small"
    elif ad < 0.474:
        effect = "medium"
    else:
        effect = "large"
    return float(delta), effect




# ══════════════════════════════════════════════════════════════════════════
# 3. UJI POST-HOC BERPASANGAN
# ══════════════════════════════════════════════════════════════════════════

def wilcoxon_pairwise(scores_by_candidate, baseline=None):
    """
    Uji post-hoc tiap pasangan kandidat (atau tiap kandidat vs baseline
    kalau baseline ditentukan). Mengembalikan DataFrame p_value + Cliff's delta.
    """
    names = list(scores_by_candidate.keys())
    pairs = ([(baseline, n) for n in names if n != baseline] if baseline
             else list(combinations(names, 2)))

    rows = []
    for a, b in pairs:
        xa = np.asarray(scores_by_candidate[a])
        xb = np.asarray(scores_by_candidate[b])
        try:
            stat, p = wilcoxon(xa, xb)
        except ValueError:
            # semua selisih nol, atau observasi terlalu sedikit
            stat, p = float("nan"), float("nan")
        delta, effect = cliffs_delta(xa, xb)
        rows.append({
            "candidate_a": a, "candidate_b": b,
            "p_value": p, "cliffs_delta": delta, "effect": effect,
        })
    return pd.DataFrame(rows)




# ══════════════════════════════════════════════════════════════════════════
# 4. PEMILIHAN PEMENANG OTOMATIS
# ══════════════════════════════════════════════════════════════════════════

def pick_winner(mean_scores_by_candidate):
    """Pemenang murni berdasarkan mean skor tertinggi. Dipakai secara
    internal oleh auto_select_winner(), lihat catatan di sana soal
    kenapa mean-based cukup untuk seleksi OTOMATIS."""
    return max(mean_scores_by_candidate, key=mean_scores_by_candidate.get)


def auto_select_winner(results_by_candidate, metric_key="macro_f1_per_fold", verbose=True):
    """
    Pilih pemenang OTOMATIS (mean skor tertinggi), TANPA perlu tinjauan
    atau input manual nama kandidat di notebook. Dipakai konsisten di
    Tabel A, B1, B2, C, di semua dataset, supaya seluruh alur ablasi bisa
    dijalankan ujung ke ujung tanpa berhenti menunggu keputusan manusia.

    Uji statistik (Friedman + Wilcoxon post-hoc + Cliff's delta) TETAP
    dihitung dan dikembalikan/dicetak, bukan dihilangkan, karena bukti
    statistik itu tetap wajib ada untuk penulisan manuskrip (lihat aturan
    keras klaim hasil). Bedanya, di sini uji statistik itu TIDAK
    memblokir keputusan mana yang dipakai lanjut ke tabel berikutnya,
    cuma dicatat untuk pelaporan dan tinjauan pasca-hoc kalau diperlukan.

    results_by_candidate: dict {nama_kandidat: agg}, agg dari
    run_candidate()/run_meta_candidate(), WAJIB punya key metric_key
    (list skor per fold).

    Return: (winner, mean_scores, friedman_result, wilcoxon_df).
    friedman_result/wilcoxon_df bisa None kalau cuma ada 1 kandidat atau
    uji gagal dihitung (mis. observasi terlalu sedikit), winner tetap
    valid dari mean tertinggi.
    """
    scores = {name: agg[metric_key] for name, agg in results_by_candidate.items()}
    mean_scores = {name: float(np.mean(v)) for name, v in scores.items()}
    winner = pick_winner(mean_scores)

    friedman_result, wilcoxon_df = None, None
    if len(scores) >= 3:
        try:
            friedman_result = friedman_test(scores)
        except Exception as e:
            if verbose:
                print(f"[PERINGATAN] Friedman test gagal dihitung ({e}), dilewati.")
    if len(scores) >= 2:
        try:
            wilcoxon_df = wilcoxon_pairwise(scores, baseline=winner)
        except Exception as e:
            if verbose:
                print(f"[PERINGATAN] Wilcoxon pairwise gagal dihitung ({e}), dilewati.")

    if verbose:
        print(f"Kandidat terpilih OTOMATIS: {winner!r} (mean {metric_key.replace('_per_fold','')}={mean_scores[winner]:.4f})")
        for name, m in sorted(mean_scores.items(), key=lambda kv: -kv[1]):
            marker = "  <-- WINNER" if name == winner else ""
            print(f"  {name:<25} mean={m:.4f}{marker}")
        if friedman_result:
            print(f"Friedman: statistic={friedman_result['statistic']:.4f} "
                  f"p_value={friedman_result['p_value']:.4f} "
                  f"(n_observations={friedman_result['n_observations']})")

    return winner, mean_scores, friedman_result, wilcoxon_df
