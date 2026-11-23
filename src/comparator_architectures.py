# src/comparator_architectures.py — Hibah Kemdiktisaintek Part 3
"""
Reproduksi 5 arsitektur dari paper eksternal + 2 baseline deep sederhana,
SEMUA memakai fitur yang SUDAH DISERAGAMKAN (P1-P30 + SNR + Max Amplitude
untuk OTDR, fitur setara untuk 3 dataset lain), TIDAK memakai fitur asli
tiap paper (beberapa di antaranya bocor, lihat catatan Titouni di bawah).
Semua dilatih lewat training_engine.py yang SAMA dengan backbone HDMC
(AdamW, hyperparameter dari config.py) supaya perbandingan murni soal
arsitektur, bukan soal siapa dapat training recipe lebih baik.

Daftar isi:
  1. Abdelli-BiGRU (attention)         : AbdelliBiGRU
  2. Soothar-CNNLSTM (reuse backbone)  : SoothaCNNLSTM (alias)
  3. Yang-IMCBiLSTM (4 BiLSTM independen) : YangIMCBiLSTM
  4. Healy-GRUClassifier (autoencoder) : HealyGRUClassifier
  5. Titouni-CNNExtractor (feature extractor untuk ensemble) : TitouniCNNExtractor
  6. Simple-CNNBaseline (reuse backbone) : SimpleBaseline (alias)
  7. Baseline-BiLSTMOnly (baseline deep tambahan) : BiLSTMOnly
  8. Registry gabungan                 : EXTERNAL_REGISTRY, BASELINE_DEEP_REGISTRY,
                                          COMPARATOR_REGISTRY

Sumber tiap arsitektur (dibaca langsung dari PDF paper, bukan tebakan):

1. AbdelliBiGRU — Abdelli dkk. (2022), "Machine-learning-based anomaly
   detection in optical fiber monitoring". Arsitektur diagnosis (bagian
   kedua paper, A-BiGRU): 2 lapis BiGRU (64, 32 cells) + attention layer
   + task-specific FC. Tahap deteksi anomali (autoencoder tahap 1)
   SENGAJA tidak direproduksi karena tugas kita klasifikasi multi-kelas
   langsung, bukan deteksi-lalu-diagnosis dua tahap seperti paper asli.

2. SoothaCNNLSTM — Soothar dkk. (2024), "Optimizing Optical Fiber Faults
   Detection". Conv1D 64 filter kernel=3 -> LSTM -> FC. Ini SAMA PERSIS
   dengan kandidat "CNN-LSTM" di backbone_architectures.py, jadi tidak
   perlu implementasi baru, tinggal reuse dengan nama berbeda untuk
   kejelasan pelaporan.

3. YangIMCBiLSTM — Yang dkk. (2025). IMC-BiLSTM (BUKAN SMC-BiLSTM):
   4 cabang CNN (kernel 1,3,5,7, 64 filter) -> EMPAT BiLSTM INDEPENDEN
   (bukan berbagi bobot) -> konkatenasi 4x128=512 -> FC1(->64)+ReLU ->
   FC2(->kelas). Dipilih IMC (bukan SMC) karena itu varian yang paper
   asli klaim sebagai performa terbaik mereka (F1=97.37%, Tabel 8 paper).

4. HealyGRUClassifier — Healy dkk. (2025). Autoencoder GRU (encoder 30
   cells + 15 cells, decoder simetris) + FC 16 unit -> kelas. Sesuai
   Fig. 3 paper (arsitektur classifier, BUKAN generator cGAN/cVAE-nya).

5. TitouniCNNExtractor — Titouni dkk. (2025). CNN 3-lapis FC (128,64,32
   per Section IV-D, versi yang benar-benar divalidasi paper, BUKAN
   512,256,128 di Section IV-A yang kontradiktif) sebagai feature
   extractor -> XGBoost+SVM+RF -> meta-classifier. PENTING: paper asli
   memakai posisi+reflektansi+loss sebagai fitur (Algorithm 1 Step 1),
   BOCOR terhadap label di dataset ini. Reproduksi di sini SENGAJA
   memakai fitur yang sudah diseragamkan (tanpa posisi/reflektansi/loss),
   supaya perbandingan adil dan tidak mewariskan kebocoran itu.

6. SimpleBaseline — reuse CNNOnly dari backbone_architectures.py, dipakai
   ulang di sini sebagai baseline "model biasa" pembanding.

7. BiLSTMOnly — baseline rekuren murni (tanpa CNN sama sekali), setara
   level kesederhanaan dengan SimpleBaseline dari sisi keluarga arsitektur
   yang berbeda (konvolusi vs rekuren).
"""
import torch
import torch.nn as nn


# ── 1. Abdelli — Attention-BiGRU classifier ─────────────────────────────────
# CATATAN: dropout 0,2 sebelum FC TIDAK disebutkan di deskripsi A-BiGRU
# paper Abdelli dkk. Paper ini genuinely tidak menyebutkan regularisasi
# apapun di bagian ini. Dropout 0,2 ditambahkan sebagai praktik umum yang
# konsisten dengan kandidat lain (Yang, Titouni, dst juga pakai 0,2-0,3),
# BUKAN dari paper Abdelli sendiri. Ini gap tak terhindarkan (paper tidak
# jelaskan), bukan kesalahan membaca paper.



# ══════════════════════════════════════════════════════════════════════════
# 1. ABDELLI-BIGRU (ATTENTION)
# ══════════════════════════════════════════════════════════════════════════

class AbdelliBiGRU(nn.Module):
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.bigru1 = nn.GRU(1, 64, batch_first=True, bidirectional=True)
        self.bigru2 = nn.GRU(128, 32, batch_first=True, bidirectional=True)
        self.attn_w = nn.Linear(64, 64)
        self.attn_u = nn.Linear(64, 1, bias=False)
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        x = x.unsqueeze(-1)                      # (B, L, 1)
        h1, _ = self.bigru1(x)                    # (B, L, 128)
        h2, _ = self.bigru2(h1)                    # (B, L, 64)
        e = torch.tanh(self.attn_w(h2))            # (B, L, 64)
        scores = self.attn_u(e).squeeze(-1)         # (B, L)
        alpha = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, L, 1)
        c = (alpha * h2).sum(dim=1)                 # (B, 64) attention context
        return self.fc(c)


# ── 2. Soothar — CNN-LSTM (reuse langsung dari models.py) ───────────────────
# CATATAN: paper Soothar TIDAK PERNAH menyebutkan angka jumlah hidden
# unit LSTM-nya (cuma bilang "the LSTM is specially designed for
# processing data sequences", tanpa angka). Kernel=3, filter=64 itu
# eksplisit disebutkan paper dan sudah match. Hidden unit LSTM dipakai
# 64 di sini (reuse arsitektur CNN-LSTM internal Anda), gap tak
# terhindarkan, bukan kesalahan baca paper.


# ══════════════════════════════════════════════════════════════════════════
# 2. SOOTHAR-CNNLSTM (REUSE BACKBONE)
# ══════════════════════════════════════════════════════════════════════════

from src.backbone_architectures import CNN_LSTM as SoothaCNNLSTM  # noqa: E402  (alias, arsitektur identik)




# ══════════════════════════════════════════════════════════════════════════
# 3. YANG-IMCBILSTM (4 BILSTM INDEPENDEN)
# ══════════════════════════════════════════════════════════════════════════

class YangIMCBiLSTM(nn.Module):
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=k, padding=k // 2),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3)
            )
            for k in [1, 3, 5, 7]
        ])
        # EMPAT BiLSTM independen, satu per cabang (beda dari SMC-BiLSTM Anda
        # yang berbagi satu modul). Ini yang membuat IMC lebih mahal (4x
        # parameter rekuren) tapi diklaim paper aslinya paling akurat.
        self.bilstms = nn.ModuleList([
            nn.LSTM(64, 64, batch_first=True, bidirectional=True) for _ in range(4)
        ])
        self.drop = nn.Dropout(0.2)
        self.fc1 = nn.Linear(512, 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        x2 = x.unsqueeze(1)
        hs = []
        for branch, bilstm in zip(self.branches, self.bilstms):
            xb = branch(x2).permute(0, 2, 1)
            _, (h, _) = bilstm(xb)
            hs.append(torch.cat([h[0], h[1]], dim=1))
        h_cat = torch.cat(hs, dim=1)
        h_fc = torch.relu(self.fc1(self.drop(h_cat)))
        return self.fc2(h_fc)




# ══════════════════════════════════════════════════════════════════════════
# 4. HEALY-GRUCLASSIFIER (AUTOENCODER)
# ══════════════════════════════════════════════════════════════════════════

class HealyGRUClassifier(nn.Module):
    """
    KOREKSI: implementasi sebelumnya cuma pakai encoder (GRU 30->15) lalu
    langsung ke FC, MEMBUANG decoder. Fig. 3 paper Healy dkk. EKSPLISIT
    menunjukkan decoder (GRU 15->30) ikut di jalur klasifikasi, SEBELUM
    FC 16 unit -> output 8 kelas: Encoder(GRU30->GRU15) -> Decoder(GRU15
    ->GRU30) -> FC(16) -> Output. Sudah diperbaiki mengikuti Fig. 3 ini.
    """
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.enc_gru1 = nn.GRU(1, 30, batch_first=True)
        self.enc_gru2 = nn.GRU(30, 15, batch_first=True)
        self.dec_gru1 = nn.GRU(15, 15, batch_first=True)
        self.dec_gru2 = nn.GRU(15, 30, batch_first=True)
        self.fc1 = nn.Linear(30, 16)
        self.fc2 = nn.Linear(16, n_classes)

    def forward(self, x):
        x = x.unsqueeze(-1)
        h1, _ = self.enc_gru1(x)
        h2, _ = self.enc_gru2(h1)       # encoder output, (B, L, 15)
        d1, _ = self.dec_gru1(h2)
        d2, _ = self.dec_gru2(d1)        # decoder output, (B, L, 30)
        last = d2[:, -1, :]
        return self.fc2(torch.relu(self.fc1(last)))




# ══════════════════════════════════════════════════════════════════════════
# 5. TITOUNI-CNNEXTRACTOR (FEATURE EXTRACTOR)
# ══════════════════════════════════════════════════════════════════════════

class TitouniCNNExtractor(nn.Module):
    """
    Feature extractor saja (3 FC layer), OUTPUT embedding, BUKAN langsung
    kelas. Ensemble XGBoost/SVM/RF + meta-classifier di atas embedding ini
    dilakukan lewat jalur run_meta_candidate() yang sama seperti Tabel C
    Anda sendiri (disiplin OOF yang sama), BUKAN meniru cara Titouni
    membangun ensemble-nya sendiri (200 tree/box-constraint/dst tidak
    direplikasi, disengaja untuk fairness protokol meta yang setara
    dengan kandidat lain -- lihat compare_external.py).

    KOREKSI: paper Titouni PUNYA DUA deskripsi arsitektur CNN yang SALING
    KONTRADIKSI:
    - Section IV-A ("CNN-Based Feature Extraction"): 512->256->128,
      dropout 0,3.
    - Section IV-D ("Model Configuration and Hyperparameter
      Justification"): 128->64->32, dropout 0,5.
    Dipakai di sini: versi IV-D, karena judul bagian itu eksplisit
    membahas konfigurasi yang benar-benar divalidasi untuk hasil yang
    dilaporkan paper (bukan deskripsi umum di IV-A). Versi sebelumnya
    (IV-A) SALAH dipilih, sudah diperbaiki.
    """
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc_out = nn.Linear(32, n_classes)  # dipakai saat training CNN-nya sendiri

    def forward(self, x):
        h = torch.relu(self.bn1(self.fc1(x)))
        h = self.drop1(h)
        h = torch.relu(self.bn2(self.fc2(h)))
        h = self.drop2(h)
        h = torch.relu(self.bn3(self.fc3(h)))
        return self.fc_out(h)

    def forward_features(self, x):
        h = torch.relu(self.bn1(self.fc1(x)))
        h = torch.relu(self.bn2(self.fc2(h)))
        h = torch.relu(self.bn3(self.fc3(h)))
        return h  # 32-dim embedding untuk ensemble



# ══════════════════════════════════════════════════════════════════════════
# 6. SIMPLE-CNNBASELINE (REUSE BACKBONE)
# ══════════════════════════════════════════════════════════════════════════

from src.backbone_architectures import CNNOnly as SimpleBaseline  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════
# 7. BASELINE BILSTM POLOS (dulu di part2_baseline_comparison.py)
# ══════════════════════════════════════════════════════════════════════════

class BiLSTMOnly(nn.Module):
    """
    BiLSTM cabang tunggal -> Dense. Tanpa lapisan CNN sama sekali.
    Setara level kesederhanaan dengan CNNOnly (backbone_architectures.py),
    bedanya cuma tipe lapisan (rekuren murni vs konvolusi murni), supaya
    perbandingan "CNN sederhana vs RNN sederhana" adil (dua-duanya
    baseline paling dasar dari masing-masing keluarga arsitektur, tanpa
    trik tambahan). Input diperlakukan sebagai sekuens univariat (B, L, 1),
    sama seperti perlakuan AbdelliBiGRU/HealyGRUClassifier di atas.
    """
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.bilstm = nn.LSTM(1, 64, batch_first=True, bidirectional=True)
        self.drop1 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.drop2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        x = x.unsqueeze(-1)               # (B, L, 1)
        _, (h, _) = self.bilstm(x)         # h: (2, B, 64), maju+mundur
        h_cat = torch.cat([h[0], h[1]], dim=1)  # (B, 128)
        h_cat = self.drop1(h_cat)
        h_cat = self.relu(self.fc1(h_cat))
        return self.fc2(self.drop2(h_cat))


class BiGRUOnly(nn.Module):
    """
    BiGRU cabang tunggal -> Dense. Tanpa lapisan CNN sama sekali, struktur
    identik dengan BiLSTMOnly (dropout, dimensi, jumlah unit sama persis),
    beda cuma tipe sel rekuren (GRU, bukan LSTM). Ditambahkan supaya
    baseline "RNN murni" sejalan dengan backbone HDMC (CNN-BiGRU, sejak
    keputusan ganti dari CNN-BiLSTM), perbandingan "apa kontribusi CNN"
    jadi konsisten dengan tipe sel rekuren yang sama seperti proposed.

    BiLSTMOnly TETAP ada di file ini (tidak dihapus), sudah dipakai untuk
    dataset SOR (baseline lama). BiGRUOnly dipakai untuk dataset yang
    belum diuji (OTDR, HardFailure, SoftFailure), dan sebagai tambahan
    kandidat baru khusus SOR.
    """
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.bigru = nn.GRU(1, 64, batch_first=True, bidirectional=True)
        self.drop1 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.drop2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        x = x.unsqueeze(-1)               # (B, L, 1)
        _, h = self.bigru(x)               # h: (2, B, 64), maju+mundur
        h_cat = torch.cat([h[0], h[1]], dim=1)  # (B, 128)
        h_cat = self.drop1(h_cat)
        h_cat = self.relu(self.fc1(h_cat))
        return self.fc2(self.drop2(h_cat))


# ══════════════════════════════════════════════════════════════════════════
# 8. REGISTRY GABUNGAN
# ══════════════════════════════════════════════════════════════════════════

# 5 arsitektur eksternal (dari paper), dipakai di nb03 sebagai pembanding utama.
EXTERNAL_REGISTRY = {
    "Abdelli-BiGRU": AbdelliBiGRU,
    "Soothar-CNNLSTM": SoothaCNNLSTM,
    "Yang-IMCBiLSTM": YangIMCBiLSTM,
    "Healy-GRUClassifier": HealyGRUClassifier,
    "Titouni-CNNEnsemble": TitouniCNNExtractor,  # dipakai khusus lewat jalur meta, lihat experiment_runner.py
}

# 5 baseline konvensional (2 deep + 3 classical). Classical (LogReg/RF/SVM)
# TIDAK dari sklearn di sini, didaftar dan dilatih terpisah di
# comparator_runner.py (bukan nn.Module), jadi tidak masuk registry ini.
BASELINE_DEEP_REGISTRY = {
    "Baseline-CNNOnly": SimpleBaseline,
    "Baseline-BiLSTMOnly": BiLSTMOnly,
    "Baseline-BiGRUOnly": BiGRUOnly,
}

# Registry gabungan (eksternal + baseline deep), dipakai run_deep_model()
# di comparator_runner.py supaya satu fungsi bisa melayani keduanya.
COMPARATOR_REGISTRY = {**EXTERNAL_REGISTRY, **BASELINE_DEEP_REGISTRY}
