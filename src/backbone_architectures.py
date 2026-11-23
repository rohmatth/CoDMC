# src/backbone_architectures.py — Hibah Kemdiktisaintek Part 3
"""
Lima kandidat backbone untuk justifikasi pemilihan arsitektur (nb02
Bagian A, dulu disebut "Tabel A" di Part 2).

Daftar isi:
  1. Blok CNN dasar          : _vanilla_cnn_block
  2. 5 kandidat CNN-cabang-tunggal : CNNOnly, CNN_LSTM, CNN_GRU, CNN_BiLSTM, CNN_BiGRU
  3. Registry + factory      : BACKBONE_REGISTRY, get_model

FAIRNESS DESIGN:
Kelima kandidat (CNN Only, CNN-LSTM, CNN-GRU, CNN-BiLSTM, CNN-BiGRU)
semuanya memakai CNN cabang tunggal, kernel size sama (3), filter count sama
(64), padding sama. Satu-satunya yang berbeda antar kelima kandidat ini
adalah lapisan rekuren di atasnya (tidak ada / LSTM / GRU / BiLSTM / BiGRU).
Ini memastikan perbandingan lima kandidat itu murni mengisolasi pengaruh
pilihan lapisan rekuren, bukan tercampur dengan perbedaan CNN.

CATATAN: kandidat multi-skala (SMC-BiLSTM, adaptasi Yang dkk.) SENGAJA
dikeluarkan dari justifikasi_backbone. Alasannya bukan performa, tapi
independensi argumen penulisan: kalau nanti paper Yang dkk. tidak dipakai
lagi sebagai pembanding eksternal di naskah (lihat instruksi proyek soal
kemungkinan pembanding dengan hasil serupa dihapus dari badan paper),
justifikasi backbone tidak boleh bergantung pada rujukan yang keberadaannya
sendiri belum pasti di draft final. Kelima kandidat CNN-cabang-tunggal
sudah cukup untuk menjustifikasi pilihan lapisan rekuren tanpa
menggantungkan argumen ke paper eksternal tertentu.

Semua kandidat dilatih lewat satu training harness yang sama
(lihat training_engine.py), dengan hyperparameter yang identik dari
config.py. Tidak ada percabangan khusus per model di luar perbedaan
arsitektur ini sendiri.

TIDAK ADA kDN, complexity profiling, atau Treatment Assignment di file ini
(eksklusif tesis).
"""
import torch
import torch.nn as nn

_CNN_FILTERS = 64
_CNN_KERNEL  = 3
_REC_HIDDEN  = 64




# ══════════════════════════════════════════════════════════════════════════
# 1. BLOK CNN DASAR
# ══════════════════════════════════════════════════════════════════════════

def _vanilla_cnn_block():
    """CNN cabang tunggal, dipakai identik oleh 5 kandidat default."""
    return nn.Sequential(
        nn.Conv1d(1, _CNN_FILTERS, kernel_size=_CNN_KERNEL, padding=_CNN_KERNEL // 2),
        nn.BatchNorm1d(_CNN_FILTERS),
        nn.ReLU(),
    )




# ══════════════════════════════════════════════════════════════════════════
# 2. LIMA KANDIDAT CNN-CABANG-TUNGGAL
# ══════════════════════════════════════════════════════════════════════════

class CNNOnly(nn.Module):
    """CNN cabang tunggal -> Flatten -> Dense. Tanpa lapisan rekuren.

    Memakai Flatten, BUKAN GlobalAveragePooling. Dimensi fitur OTDR tetap
    (32), bukan sekuens panjang variabel yang butuh invariansi panjang lewat
    GAP. GlobalAveragePooling meratakan seluruh posisi jadi satu nilai per
    channel dan menghilangkan informasi POSISI di sepanjang serat, padahal
    posisi itu justru bagian penting yang membedakan satu jenis gangguan
    OTDR dari yang lain. Versi awal file ini sempat memakai GAP dan
    menyebabkan CNN Only kolaps ke macro-F1 jauh di bawah kandidat lain
    (~0.29 vs ~0.94), bukan karena tanpa lapisan rekuren, tapi karena
    kehilangan informasi posisi akibat GAP. Flatten memperbaiki ini.
    """
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.conv    = _vanilla_cnn_block()
        self.flatten = nn.Flatten()
        self.drop1   = nn.Dropout(0.3)
        self.fc1     = nn.Linear(_CNN_FILTERS * in_features, _REC_HIDDEN)
        self.relu    = nn.ReLU()
        self.drop2   = nn.Dropout(0.2)
        self.fc2     = nn.Linear(_REC_HIDDEN, n_classes)

    def forward(self, x):
        x = self.conv(x.unsqueeze(1))
        x = self.flatten(x)
        x = self.drop1(x)
        x = self.relu(self.fc1(x))
        return self.fc2(self.drop2(x))

    def forward_features(self, x):
        """Representasi sebelum lapisan klasifikasi terakhir, dipakai sebagai
        fitur kaya untuk ensemble base learner (XGBoost/SVM/RF), mengikuti
        struktur asli Titouni dkk. (CNN feature extractor -> base learner),
        bukan cuma probabilitas akhir yang lebih ringkas."""
        x2 = self.conv(x.unsqueeze(1))
        x2 = self.flatten(x2)
        x2 = self.drop1(x2)
        return self.relu(self.fc1(x2))


class CNN_LSTM(nn.Module):
    """CNN cabang tunggal -> LSTM (unidirectional) -> Dense."""
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.conv = _vanilla_cnn_block()
        self.rec  = nn.LSTM(_CNN_FILTERS, _REC_HIDDEN, batch_first=True, bidirectional=False)
        self.drop = nn.Dropout(0.2)
        self.fc   = nn.Linear(_REC_HIDDEN, n_classes)

    def forward(self, x):
        x = self.conv(x.unsqueeze(1)).permute(0, 2, 1)
        _, (h, _) = self.rec(x)
        return self.fc(self.drop(h[-1]))

    def forward_features(self, x):
        x2 = self.conv(x.unsqueeze(1)).permute(0, 2, 1)
        _, (h, _) = self.rec(x2)
        return h[-1]


class CNN_GRU(nn.Module):
    """CNN cabang tunggal -> GRU (unidirectional) -> Dense."""
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.conv = _vanilla_cnn_block()
        self.rec  = nn.GRU(_CNN_FILTERS, _REC_HIDDEN, batch_first=True, bidirectional=False)
        self.drop = nn.Dropout(0.2)
        self.fc   = nn.Linear(_REC_HIDDEN, n_classes)

    def forward(self, x):
        x = self.conv(x.unsqueeze(1)).permute(0, 2, 1)
        _, h = self.rec(x)
        return self.fc(self.drop(h[-1]))

    def forward_features(self, x):
        x2 = self.conv(x.unsqueeze(1)).permute(0, 2, 1)
        _, h = self.rec(x2)
        return h[-1]


class CNN_BiLSTM(nn.Module):
    """CNN cabang tunggal -> BiLSTM -> Dense."""
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.conv = _vanilla_cnn_block()
        self.rec  = nn.LSTM(_CNN_FILTERS, _REC_HIDDEN, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.2)
        self.fc   = nn.Linear(_REC_HIDDEN * 2, n_classes)

    def forward(self, x):
        x = self.conv(x.unsqueeze(1)).permute(0, 2, 1)
        _, (h, _) = self.rec(x)
        h = torch.cat([h[0], h[1]], dim=1)
        return self.fc(self.drop(h))

    def forward_features(self, x):
        x2 = self.conv(x.unsqueeze(1)).permute(0, 2, 1)
        _, (h, _) = self.rec(x2)
        return torch.cat([h[0], h[1]], dim=1)


class CNN_BiGRU(nn.Module):
    """CNN cabang tunggal -> BiGRU -> Dense.

    Nama arsitektur ini WAJIB "CNN-BiGRU", bukan "CNN-GRU", karena
    konfigurasinya bidirectional. Konsisten dengan aturan penamaan
    arsitektur yang berlaku di seluruh dokumen dan kode.
    """
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.conv = _vanilla_cnn_block()
        self.rec  = nn.GRU(_CNN_FILTERS, _REC_HIDDEN, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.2)
        self.fc   = nn.Linear(_REC_HIDDEN * 2, n_classes)

    def forward(self, x):
        x = self.conv(x.unsqueeze(1)).permute(0, 2, 1)
        _, h = self.rec(x)
        h = torch.cat([h[0], h[1]], dim=1)
        return self.fc(self.drop(h))

    def forward_features(self, x):
        x2 = self.conv(x.unsqueeze(1)).permute(0, 2, 1)
        _, h = self.rec(x2)
        return torch.cat([h[0], h[1]], dim=1)




# ══════════════════════════════════════════════════════════════════════════
# 3. REGISTRY + FACTORY
# ══════════════════════════════════════════════════════════════════════════

BACKBONE_REGISTRY = {
    "CNN Only"    : CNNOnly,
    "CNN-LSTM"    : CNN_LSTM,
    "CNN-GRU"     : CNN_GRU,
    "CNN-BiLSTM"  : CNN_BiLSTM,
    "CNN-BiGRU"   : CNN_BiGRU,
}


def get_model(name, in_features, n_classes):
    if name in BACKBONE_REGISTRY:
        return BACKBONE_REGISTRY[name](in_features, n_classes)
    # Lazy import untuk hindari circular import (external_models.py meng-import
    # dari models.py juga, untuk reuse CNN_LSTM/CNNOnly)
    from src.comparator_architectures import EXTERNAL_REGISTRY
    if name in EXTERNAL_REGISTRY:
        return EXTERNAL_REGISTRY[name](in_features, n_classes)
    raise ValueError(
        f"Model '{name}' tidak dikenal. Pilihan internal: {list(BACKBONE_REGISTRY.keys())}. "
        f"Pilihan eksternal: {list(EXTERNAL_REGISTRY.keys())}."
    )
