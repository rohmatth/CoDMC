# src/training_engine.py — Hibah Kemdiktisaintek Part 3
"""
Satu training harness dipakai IDENTIK oleh semua kandidat (backbone
justifikasi, HDMC multiview, baseline, eksternal), di keempat dataset.
Tidak ada percabangan khusus per model atau per dataset. Perbedaan antar
kandidat murni dari model, resampler, dan loss yang di-pass sebagai
argumen, bukan dari logika training itu sendiri.

Daftar isi:
  1. Seed dan device               : fold_seed, set_seed, get_device
  2. DataLoader                    : make_loader
  3. Satu epoch train/eval         : train_epoch, eval_model
  4. Training loop lengkap         : train_model (return model + history lengkap,
                                      dipakai untuk learning curve nb04)
  5. Inference-only                : predict_proba, extract_features, inference_timing_sweep
  6. Partial fine-tune (scaffolding, belum dipakai di Part 3) : partial_finetune

FAIRNESS: seed training terikat ke (repeat_idx, fold_idx), bukan ke nama
model, sehingga tidak ada model yang kebetulan dapat inisialisasi bobot
yang lebih menguntungkan dibanding model lain pada fold yang sama.

train_model() mengembalikan `history` (train_loss, val_f1 per epoch)
LENGKAP, dipakai langsung sebagai sumber learning curve di nb04 tanpa
perlu retrain, ASALKAN pemanggil (hdmc_multiview.py, comparator_runner.py)
menyimpan history itu ke disk per fold. Modul ini sendiri tidak menyimpan
apapun ke disk, murni komputasi.
"""
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import BATCH_SIZE, MAX_EPOCHS, LR, WEIGHT_DECAY, PATIENCE, SEED, INFER_BATCH_SIZES
from src.resampling_and_loss import build_loss




# ══════════════════════════════════════════════════════════════════════════
# 1. SEED DAN DEVICE
# ══════════════════════════════════════════════════════════════════════════

def fold_seed(repeat_idx, fold_idx, base_seed=SEED):
    """Seed run ditentukan dari posisi fold-repeat, SAMA untuk semua kandidat
    yang dievaluasi pada fold-repeat yang sama. Ini kunci fairness inisialisasi
    bobot antar kandidat."""
    return base_seed + repeat_idx * 100 + fold_idx


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Operasi cuDNN (Conv1d, LSTM/GRU) punya beberapa algoritma internal yang
    # dipilih otomatis dan bisa sedikit berbeda urutan komputasi antar run
    # walau seed sama persis. Untuk dataset dengan IR ekstrem (mis. SOR),
    # perbedaan kecil ini bisa menggeser hasil akhir cukup jauh karena
    # keseimbangan kelas yang rapuh. Baris di bawah memaksa cuDNN pakai
    # algoritma deterministik, mengurangi (bukan menghilangkan total) variasi
    # run-to-run. Trade-off: sedikit lebih lambat dari mode non-deterministik.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")




# ══════════════════════════════════════════════════════════════════════════
# 2. DATALOADER
# ══════════════════════════════════════════════════════════════════════════

def make_loader(X, y, batch_size=BATCH_SIZE, shuffle=True):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    ds  = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                       num_workers=0, pin_memory=False)




# ══════════════════════════════════════════════════════════════════════════
# 3. SATU EPOCH TRAIN/EVAL
# ══════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        logit = model(X_b)
        loss  = criterion(logit, y_b)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(X_b)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_model(model, loader, device):
    from sklearn.metrics import f1_score
    model.eval()
    preds, trues, probas = [], [], []
    for X_b, y_b in loader:
        X_b = X_b.to(device)
        logit = model(X_b)
        proba = torch.softmax(logit, dim=1)
        preds.extend(logit.argmax(1).cpu().numpy())
        trues.extend(y_b.numpy())
        probas.append(proba.cpu().numpy())
    preds  = np.array(preds)
    trues  = np.array(trues)
    probas = np.concatenate(probas, axis=0)
    macro_f1 = f1_score(trues, preds, average="macro", zero_division=0)
    return macro_f1, preds, trues, probas




# ══════════════════════════════════════════════════════════════════════════
# 4. TRAINING LOOP LENGKAP
# ══════════════════════════════════════════════════════════════════════════

def train_model(model, X_tr, y_tr, X_va, y_va,
                 loss_type="CE", n_classes=8,
                 max_epochs=MAX_EPOCHS, patience=PATIENCE,
                 batch_size=BATCH_SIZE, lr=LR,
                 weight_decay=WEIGHT_DECAY, seed=SEED,
                 verbose=False):
    """
    Training loop identik untuk semua kandidat. Mengembalikan model terbaik,
    metrik validasi, dan learning history lengkap untuk regenerasi plot
    (training curve) belakangan tanpa perlu retrain.
    """
    set_seed(seed)
    device = get_device()

    model     = model.to(device)
    crit      = build_loss(loss_type, y_tr, n_classes, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=1e-5)

    tr_loader = make_loader(X_tr, y_tr, batch_size, shuffle=True)
    va_loader = make_loader(X_va, y_va, batch_size, shuffle=False)

    best_f1, best_state, no_improve = -1.0, None, 0
    history = {"train_loss": [], "val_f1": [], "best_epoch": 0}

    t_start = time.perf_counter()
    for epoch in range(max_epochs):
        tr_loss = train_epoch(model, tr_loader, optimizer, crit, device)
        va_f1, _, _, _ = eval_model(model, va_loader, device)
        scheduler.step()

        history["train_loss"].append(float(tr_loss))
        history["val_f1"].append(float(va_f1))

        if va_f1 > best_f1:
            best_f1 = va_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            history["best_epoch"] = epoch
        else:
            no_improve += 1

        if verbose and (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:>3}/{max_epochs} | "
                  f"loss={tr_loss:.4f} | val_F1={va_f1:.4f} | best={best_f1:.4f}")

        if no_improve >= patience:
            break
    t_train_s = time.perf_counter() - t_start

    if best_state:
        model.load_state_dict(best_state)

    va_f1, va_preds, va_trues, va_probas = eval_model(model, va_loader, device)
    return model, va_f1, va_preds, va_trues, va_probas, history, t_train_s




# ══════════════════════════════════════════════════════════════════════════
# 5. INFERENCE-ONLY
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_proba(model, X, device=None, batch_size=BATCH_SIZE):
    """Inference-only, dipakai untuk Full Model (transfer ke dataset lain),
    bukan bagian dari training loop."""
    device = device or get_device()
    model = model.to(device)
    model.eval()
    loader = make_loader(X, np.zeros(len(X), dtype=int), batch_size, shuffle=False)
    probas = []
    for X_b, _ in loader:
        X_b = X_b.to(device)
        proba = torch.softmax(model(X_b), dim=1)
        probas.append(proba.cpu().numpy())
    return np.concatenate(probas, axis=0)


@torch.no_grad()
def extract_features(model, X, device=None, batch_size=BATCH_SIZE):
    """
    Ekstrak representasi kaya sebelum lapisan klasifikasi terakhir
    (forward_features()), dipakai sebagai fitur untuk ensemble base learner
    (XGBoost/SVM/RF), mengikuti struktur asli Titouni dkk. yang memberi
    ensemble representasi CNN penuh, bukan cuma probabilitas akhir.
    """
    device = device or get_device()
    model = model.to(device)
    model.eval()
    loader = make_loader(X, np.zeros(len(X), dtype=int), batch_size, shuffle=False)
    feats = []
    for X_b, _ in loader:
        X_b = X_b.to(device)
        f = model.forward_features(X_b)
        feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0)


@torch.no_grad()
def inference_timing_sweep(model, X, batch_sizes=INFER_BATCH_SIZES, n_repeats=5):
    """
    Ukur inference time di beberapa batch size. n_repeats>1 untuk stabilkan
    estimasi, terutama penting di batch=1 yang variansnya besar.
    Mengikuti pelajaran dari tesis: jangan sampai overhead sekali-pakai
    (mis. index rebuild) ikut terukur sebagai bagian dari inference murni.
    """
    device = get_device()
    model  = model.to(device)
    model.eval()

    results = {}
    for bs in batch_sizes:
        loader = make_loader(X, np.zeros(len(X), dtype=int), bs, shuffle=False)
        times  = []
        for _ in range(n_repeats):
            start = time.perf_counter()
            for X_b, _ in loader:
                X_b = X_b.to(device)
                _ = model(X_b)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        results[f"t_infer_bs{bs}_s"] = float(np.mean(times))
        results[f"t_infer_bs{bs}_per_sample_ms"] = float(np.mean(times) / len(X) * 1000)
    return results


# ══════════════════════════════════════════════════════════════════════════
# 6. PARTIAL FINE-TUNE (SCAFFOLDING, BELUM DIPAKAI DI PART 3)
# ══════════════════════════════════════════════════════════════════════════

def partial_finetune(pretrained_model, n_classes_new, X_tr, y_tr, X_va, y_va,
                      loss_type="CE", max_epochs=MAX_EPOCHS, patience=PATIENCE,
                      lr=1e-3, weight_decay=1e-4, seed=SEED, verbose=True):
    """
    Backbone (conv + lapisan rekuren) DIBEKUKAN (requires_grad=False), bawa
    representasi yang sudah dipelajari dari OTDR. Lapisan `fc` (klasifikasi
    akhir) DIGANTI baru sesuai n_classes_new, dan HANYA lapisan itu yang
    dilatih. Ini standar transfer learning "linear probing" / head fine-tune,
    jauh lebih masuk akal daripada frozen transfer mentah (yang gagal total
    kalau input di luar distribusi training asli), tapi tetap jauh lebih
    murah dari Full Retrain.

    Bekerja untuk SEMUA arsitektur di src/models.py karena semuanya
    konsisten menamai lapisan akhir `fc`. Panjang sekuens input BOLEH beda
    dari saat backbone dilatih (Conv1d padding='same' dan lapisan rekuren
    tidak bergantung pada panjang sekuens), jadi TIDAK PERLU zero-pad atau
    downsampling paksa ke dimensi asli OTDR, fitur asli domain target bisa
    dipakai apa adanya.
    """
    import copy
    set_seed(seed)
    device = get_device()

    model = copy.deepcopy(pretrained_model)
    for p in model.parameters():
        p.requires_grad = False

    # Ganti lapisan fc, cari dimensi input fc lama secara otomatis
    old_fc = model.fc
    in_dim = old_fc.in_features
    model.fc = nn.Linear(in_dim, n_classes_new)
    for p in model.fc.parameters():
        p.requires_grad = True

    model = model.to(device)
    crit = build_loss(loss_type, y_tr, n_classes_new, device)
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=lr, weight_decay=weight_decay)

    tr_loader = make_loader(X_tr, y_tr, BATCH_SIZE, shuffle=True)
    va_loader = make_loader(X_va, y_va, BATCH_SIZE, shuffle=False)

    best_f1, best_state, no_improve = -1.0, None, 0
    history = {"train_loss": [], "val_f1": [], "best_epoch": 0}

    for epoch in range(max_epochs):
        tr_loss = train_epoch(model, tr_loader, optimizer, crit, device)
        va_f1, _, _, _ = eval_model(model, va_loader, device)
        history["train_loss"].append(float(tr_loss))
        history["val_f1"].append(float(va_f1))

        if va_f1 > best_f1:
            best_f1 = va_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            history["best_epoch"] = epoch
        else:
            no_improve += 1

        if verbose and (epoch + 1) % 5 == 0:
            print(f"    [partial_finetune] epoch {epoch+1}/{max_epochs} "
                  f"loss={tr_loss:.4f} val_F1={va_f1:.4f} best={best_f1:.4f}")

        if no_improve >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)

    va_f1, va_preds, va_trues, va_probas = eval_model(model, va_loader, device)
    return model, va_f1, va_preds, va_trues, va_probas, history
