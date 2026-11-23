# src/resampling_and_loss.py — Hibah Kemdiktisaintek Part 3
"""
Teknik penanganan imbalance, SEMUA diterapkan UNIFORM ke seluruh kelas.
Tidak ada keputusan per kelas di sini (beda mendasar dengan Treatment
Assignment tesis yang menentukan resampler/loss per kelas berdasarkan
profil kDN, sengaja tidak dipakai di hibah ini).

Daftar isi:
  1. Resampler (justifikasi_resampler) : RESAMPLER_NAMES, apply_resampling
  2. Util frekuensi kelas              : get_samples_per_class, get_class_weights
  3. Loss function (justifikasi_loss)  : FocalLoss, CBFocalLoss, LDAMLoss, build_loss

Kandidat resampler (justifikasi_resampler, loss ditahan CE): None, SMOTE,
SMOTETomek, ADASYN.
Kandidat loss (justifikasi_loss, resampler ditahan ke pemenang): CE, WCE,
Focal, CB-Focal, LDAM.

File ini TIDAK meng-copy atau meng-import kode kDN milik tesis.
"""
import numpy as np
import torch
import torch.nn as nn

try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.combine import SMOTETomek
except ImportError as e:
    raise ImportError(
        "Paket imbalanced-learn dibutuhkan. Install: "
        "pip install imbalanced-learn --break-system-packages"
    ) from e


# ── Tabel B1: Resampler (uniform, seluruh kelas) ───────────────────────────────



# ══════════════════════════════════════════════════════════════════════════
# 1. RESAMPLER (justifikasi_resampler)
# ══════════════════════════════════════════════════════════════════════════

RESAMPLER_NAMES = ["None", "SMOTE", "SMOTETomek", "ADASYN"]


def apply_resampling(X_tr, y_tr, method="None", seed=42, n_classes=None):
    """
    Terapkan resampler UNIFORM ke seluruh kelas minoritas sekaligus.
    Tidak ada logika per kelas. method salah satu dari RESAMPLER_NAMES.
    """
    if method == "None":
        return X_tr, y_tr

    counts   = np.bincount(y_tr)
    min_count = counts[counts > 0].min()
    # k_neighbors tidak boleh >= jumlah sampel kelas terkecil
    k = max(1, min(5, min_count - 1))

    if method == "SMOTE":
        sampler = SMOTE(random_state=seed, k_neighbors=k)
    elif method == "SMOTETomek":
        sampler = SMOTETomek(random_state=seed,
                              smote=SMOTE(random_state=seed, k_neighbors=k))
    elif method == "ADASYN":
        sampler = ADASYN(random_state=seed, n_neighbors=k)
    elif method == "cVAE":
        from src.training_engine import get_device
        sampler = CVAEResampler(num_classes=n_classes if n_classes is not None else len(np.unique(y_tr)),
                                 max_epochs=300, patience=20, batch_size=128, device=str(get_device()))
    else:
        raise ValueError(f"Resampler '{method}' tidak dikenal. Pilihan: {RESAMPLER_NAMES}")

    if method == "cVAE":
        X_res, y_res = sampler.fit_resample(X_tr, y_tr)
    else:
        X_res, y_res = sampler.fit_resample(X_tr, y_tr)
    return X_res.astype(np.float32), y_res.astype(np.int64)


# ── Tabel B2: Loss function (uniform, seluruh kelas) ────────────────────────────



# ══════════════════════════════════════════════════════════════════════════
# 2. UTIL FREKUENSI KELAS
# ══════════════════════════════════════════════════════════════════════════

LOSS_NAMES = ["CE", "WCE", "Focal", "CB_Focal", "LDAM"]


def get_samples_per_class(y, n_classes):
    counts = np.bincount(y, minlength=n_classes)
    return counts.astype(np.float64)


def get_class_weights(y, n_classes, device="cpu"):
    """Inverse-frequency class weight, dinormalisasi rata-rata 1."""
    spc = get_samples_per_class(y, n_classes)
    spc = np.maximum(spc, 1)
    w = spc.sum() / (n_classes * spc)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32, device=device)




# ══════════════════════════════════════════════════════════════════════════
# 3. LOSS FUNCTION (justifikasi_loss)
# ══════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """Focal Loss (Lin et al. 2017), diterapkan uniform seluruh kelas."""
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, target):
        logp = torch.log_softmax(logits, dim=1)
        pt   = logp.gather(1, target.unsqueeze(1)).squeeze(1).exp()
        ce   = nn.functional.nll_loss(logp, target, weight=self.weight, reduction="none")
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


class CBFocalLoss(nn.Module):
    """Class-Balanced Focal Loss (Cui et al. 2019), diterapkan uniform seluruh kelas."""
    def __init__(self, samples_per_class, n_classes, beta=0.9999, gamma=2.0):
        super().__init__()
        eff_num = 1.0 - np.power(beta, samples_per_class)
        eff_num = np.maximum(eff_num, 1e-8)
        weights = (1.0 - beta) / eff_num
        weights = weights / weights.sum() * n_classes
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))
        self.gamma = gamma

    def forward(self, logits, target):
        logp = torch.log_softmax(logits, dim=1)
        pt   = logp.gather(1, target.unsqueeze(1)).squeeze(1).exp()
        w    = self.weights[target]
        ce   = nn.functional.nll_loss(logp, target, reduction="none")
        loss = w * ((1 - pt) ** self.gamma) * ce
        return loss.mean()


class LDAMLoss(nn.Module):
    """
    Label-Distribution-Aware Margin Loss (Cao et al. 2019, NeurIPS),
    tanpa Deferred Re-Weighting (DRW). Margin per kelas dihitung dari
    frekuensi sampel training: kelas langka dapat margin lebih besar,
    kelas mayoritas margin lebih kecil. Margin dikurangkan dari logit
    kelas benar sebelum softmax, mendorong kelas langka punya wilayah
    keputusan lebih lebar.

    CATATAN: paper asli melaporkan LDAM paling kuat saat digabung DRW
    (loss uniform di awal training, baru re-weighting kelas diaktifkan
    di 2/3 akhir epoch). Versi ini TIDAK memakai DRW karena train_model()
    di trainer.py belum mendukung jadwal loss per-epoch. Kalau LDAM
    murni ini menang di ablasi, DRW adalah kandidat perbaikan lanjutan,
    bukan prasyarat untuk mencoba LDAM sekarang.
    """
    def __init__(self, samples_per_class, n_classes, max_m=0.5, weight=None, s=30.0):
        super().__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(np.maximum(samples_per_class, 1)))
        m_list = m_list * (max_m / np.max(m_list))
        self.register_buffer("m_list", torch.tensor(m_list, dtype=torch.float32))
        self.s = s
        self.weight = weight

    def forward(self, logits, target):
        index = torch.zeros_like(logits, dtype=torch.bool)
        index.scatter_(1, target.unsqueeze(1), True)
        batch_m = self.m_list[target].unsqueeze(1)
        logits_m = logits.clone()
        logits_m[index] = logits_m[index] - batch_m.squeeze(1)
        return nn.functional.cross_entropy(self.s * logits_m, target, weight=self.weight)


def build_loss(loss_type, y_train, n_classes, device):
    if loss_type == "CE":
        return nn.CrossEntropyLoss()
    elif loss_type == "WCE":
        w = get_class_weights(y_train, n_classes, device=device)
        return nn.CrossEntropyLoss(weight=w)
    elif loss_type == "Focal":
        w = get_class_weights(y_train, n_classes, device=device)
        return FocalLoss(gamma=2.0, weight=w)
    elif loss_type == "CB_Focal":
        spc = get_samples_per_class(y_train, n_classes)
        return CBFocalLoss(spc, n_classes, beta=0.9999, gamma=2.0).to(device)
    elif loss_type == "LDAM":
        spc = get_samples_per_class(y_train, n_classes)
        return LDAMLoss(spc, n_classes, max_m=0.5, s=30.0).to(device)
    else:
        raise ValueError(f"Loss '{loss_type}' tidak dikenal. Pilihan: {LOSS_NAMES}")

import torch
import torch.nn as nn
import torch.optim as optim

# ══════════════════════════════════════════════════════════════════════════
# CONDITIONAL VAE — Healy dkk. (2025), augmentasi generatif berkondisi kelas
# ══════════════════════════════════════════════════════════════════════════
# KOREKSI dari versi sebelumnya:
# 1. Conditioning HANYA label kelas (embedding), BUKAN lagi "2 kolom pertama
#    fitur mentah". Versi sebelumnya salah asumsi 2 kolom pertama = SNR +
#    Maximum Amplitude, padahal urutan fitur OTDR sebenarnya adalah
#    [SNR, P1..P30, Maximum_Amplitude] (lihat dataset_loaders.py::load_otdr),
#    jadi 2 kolom pertama sebenarnya [SNR, P1] BUKAN [SNR, MaxAmplitude].
#    Conditioning kelas-saja dipilih SENGAJA (bukan kompromi terpaksa)
#    supaya perlakuan SERAGAM di 4 dataset, fitur classifier Healy TIDAK
#    berkurang sama sekali (tetap fitur penuh sama seperti kandidat lain),
#    cuma variabel bantu di dalam cVAE yang disederhanakan jadi label kelas,
#    karena itu satu-satunya conditioning yang pasti ada di semua dataset.
# 2. Rekonstruksi sekarang mencakup SELURUH vektor fitur (bukan cuma "trace"
#    dari kolom ke-3 dst), supaya bekerja identik untuk dataset apapun,
#    tidak ada asumsi struktur SNR+trace+MaxAmplitude yang OTDR-spesifik.
# 3. Epoch: EARLY STOPPING berbasis konvergensi loss rekonstruksi+KLD
#    (bukan angka epoch tetap yang dipotong sewenang-wenang), max_epochs
#    sebagai plafon aman, patience untuk berhenti kalau loss sudah stabil.
#    Ini metodologis lebih valid dibanding sekadar mengecilkan angka epoch.

class ConditionalVAE(nn.Module):
    """
    Encoder-decoder GRU, conditioning HANYA embedding label kelas.
    n_features: dimensi vektor fitur penuh dataset (beda per dataset,
    dihitung otomatis dari X.shape[1] saat fit_resample dipanggil).
    """
    def __init__(self, n_features, num_classes, latent_dim=64, class_emb_dim=8):
        super().__init__()
        self.class_emb = nn.Embedding(num_classes, class_emb_dim)
        cond_dim = class_emb_dim

        self.enc_gru1 = nn.GRU(n_features + cond_dim, 128, batch_first=True, bidirectional=True)
        self.enc_gru2 = nn.GRU(256, 128, batch_first=True, bidirectional=True)
        self.enc_fc1 = nn.Linear(256, 128)
        self.enc_fc2_mu = nn.Linear(128, latent_dim)
        self.enc_fc2_logvar = nn.Linear(128, latent_dim)
        self.relu = nn.LeakyReLU()

        self.dec_gru1 = nn.GRU(latent_dim + cond_dim, 256, batch_first=True)
        self.dec_gru2 = nn.GRU(256, 128, batch_first=True)
        self.dec_fc1 = nn.Linear(128, 64)
        self.dec_fc2 = nn.Linear(64, n_features)

    def encode(self, x, cond):
        inp = torch.cat([x, cond], dim=2)
        h1, _ = self.enc_gru1(inp)
        h2, _ = self.enc_gru2(h1)
        h2 = h2[:, -1, :]
        hidden = self.relu(self.enc_fc1(h2))
        return self.enc_fc2_mu(hidden), self.enc_fc2_logvar(hidden)

    def decode(self, z, cond):
        inp = torch.cat([z, cond], dim=2)
        h1, _ = self.dec_gru1(inp)
        h2, _ = self.dec_gru2(h1)
        h2 = h2[:, -1, :]
        h = self.relu(self.dec_fc1(h2))
        return self.dec_fc2(h).unsqueeze(1)

    def forward(self, x, c_int):
        c_emb = self.class_emb(c_int).unsqueeze(1)
        x_seq = x.unsqueeze(1)

        mu, logvar = self.encode(x_seq, c_emb)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = (mu + eps * std).unsqueeze(1)

        recon_x = self.decode(z, c_emb).squeeze(1)
        return recon_x, mu, logvar


class CVAEResampler:
    """
    max_epochs: plafon aman (bukan target tetap). patience: berhenti kalau
    loss (rekonstruksi+KLD) tidak membaik selama `patience` epoch berturut,
    dicek terhadap 10% data training yang disisihkan sebagai validasi
    internal cVAE (terpisah dari validasi fold CV utama).
    """
    def __init__(self, num_classes=8, max_epochs=300, patience=20, batch_size=128,
                 lr=1e-3, latent_dim=64, device='cpu', verbose=False):
        self.num_classes = num_classes
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.lr = lr
        self.latent_dim = latent_dim
        self.device = device
        self.verbose = verbose

    def fit_resample(self, X, y):
        counts = np.bincount(y)
        max_count = counts.max()
        minority_classes = [c for c in range(len(counts)) if counts[c] < max_count and counts[c] > 0]
        if len(minority_classes) == 0:
            return X, y

        n_features = X.shape[1]
        model = ConditionalVAE(n_features=n_features, num_classes=self.num_classes,
                                latent_dim=self.latent_dim).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        # split internal train/val 90/10 KHUSUS untuk early stopping cVAE
        # (independen dari validasi fold CV utama, tidak bocor label test)
        rng = np.random.RandomState(0)
        idx_perm = rng.permutation(len(y))
        n_val = max(1, int(0.1 * len(y)))
        val_idx, tr_idx = idx_perm[:n_val], idx_perm[n_val:]

        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y, dtype=torch.long).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_t[tr_idx], y_t[tr_idx])
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        def _loss_batch(bx, by):
            recon_x, mu, logvar = model(bx, by)
            recon_loss = nn.functional.mse_loss(recon_x, bx, reduction='sum')
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            return recon_loss + kld_loss

        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_state = None

        for epoch in range(self.max_epochs):
            model.train()
            for bx, by in loader:
                optimizer.zero_grad()
                loss = _loss_batch(bx, by)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_loss = _loss_batch(X_t[val_idx], y_t[val_idx]).item() / max(len(val_idx), 1)

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                epochs_no_improve += 1

            if self.verbose and epoch % 20 == 0:
                print(f"    [cVAE] epoch {epoch} val_loss={val_loss:.2f} "
                      f"(no_improve={epochs_no_improve}/{self.patience})")

            if epochs_no_improve >= self.patience:
                if self.verbose:
                    print(f"    [cVAE] early stop di epoch {epoch} (val_loss terbaik={best_val_loss:.2f})")
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        X_syn, y_syn = [], []
        with torch.no_grad():
            for c in minority_classes:
                num_to_gen = max_count - counts[c]
                if num_to_gen <= 0:
                    continue
                c_emb = model.class_emb(torch.full((num_to_gen,), c, dtype=torch.long, device=self.device))
                cond = c_emb.unsqueeze(1)
                z = torch.randn(num_to_gen, self.latent_dim).to(self.device).unsqueeze(1)
                recon_x = model.decode(z, cond).squeeze(1)
                X_syn.append(recon_x.cpu().numpy())
                y_syn.append(np.full(num_to_gen, c))

        if len(X_syn) > 0:
            X_syn = np.vstack(X_syn)
            y_syn = np.concatenate(y_syn)
            X_res = np.vstack([X, X_syn])
            y_res = np.concatenate([y, y_syn])
            idx = np.random.permutation(len(y_res))
            return X_res[idx].astype(np.float32), y_res[idx].astype(np.int64)
        return X, y
