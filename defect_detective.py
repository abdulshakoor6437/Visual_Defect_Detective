"""
Visual Defect Detective System  -  ANN (MLP) + dashboard in ONE file   (version 2)
===================================================================================
Dataset : Severstal Steel Defect Detection (train.csv + train_images, or the competition
          .zip, anywhere inside the same folder as this file)
Task    : classify a steel image as DEFECT or CLEAN with a plain ANN (no CNN)

What is new in v2 (raw pixels overfit quickly, so the ANN now gets better inputs):
  * local block statistics (mean, spread, edge energy) on an 8 x 32 grid of the image
  * background removal (median polish) so scratches and patches stand out
  * global statistics (percentiles, skewness, kurtosis, edge histogram) and row/column profiles
  * flip augmentation (4 versions of every training image) + the same flips at prediction time
  * an ensemble of several small ANNs whose probabilities are averaged
  * a steel check (version 3): rejects images that do not look like the training steel images

Install : pip install streamlit torch numpy pandas opencv-python-headless scikit-learn matplotlib
Run     : python -m streamlit run defect_detective_v2.py     (dashboard: train + results + detective)
Or      : python defect_detective_v2.py train                (training only, in the terminal)
"""
import os, sys, json, time, zipfile, copy
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent            # folder of this file = project folder
OUT_DIR = BASE / "defect_detective_output"
OUT_DIR.mkdir(exist_ok=True)
MODEL_PATH = OUT_DIR / "model_v2.pt"              # v2 files never clash with old v1 files
HIST_PATH = OUT_DIR / "history_v2.json"
METRICS_PATH = OUT_DIR / "metrics_v2.json"
CACHE_PATH = OUT_DIR / "features_v2.npz"
GATE_PATH = OUT_DIR / "steel_check_v2.npz"      # the "is it steel?" check
EXTRACT_DIR = BASE / "severstal_data"

RH, RW = 128, 512     # every image is shrunk to 128 x 512 pixels before features are computed
BR, BC = 8, 32        # block grid: 8 rows x 32 columns (blocks of 16 x 16 pixels)
TARGET = 0.87
SEED = 42
SKIP_DIRS = {".git", "venv", ".venv", "__pycache__", "node_modules",
             "defect_detective_output", ".streamlit", "site-packages"}


# ----------------------------------------------------------------------------
# 1. Finding / unpacking the data
# ----------------------------------------------------------------------------
def _walk(root):
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in SKIP_DIRS and not d.startswith(".")]
        yield dp, fn


def find_csv():
    for dp, fn in _walk(BASE):
        if "train.csv" in fn:
            return Path(dp) / "train.csv"
    return None


def find_img_dir(probe):
    for dp, fn in _walk(BASE):
        if probe in set(fn):
            return Path(dp)
    return None


def find_zips():
    return [Path(dp) / f for dp, fn in _walk(BASE) for f in fn if f.lower().endswith(".zip")]


def extract_zip_tolerant(zip_path, dest=EXTRACT_DIR):
    """Extract file by file; damaged files are skipped. test_images are skipped (not needed)."""
    dest.mkdir(exist_ok=True)
    ok, bad = 0, 0
    with zipfile.ZipFile(zip_path) as zf:
        for n in zf.namelist():
            if n.replace("\\", "/").startswith("test_images/") or n.endswith("/"):
                continue
            try:
                zf.extract(n, dest)
                ok += 1
            except Exception:
                bad += 1
    return ok, bad


def load_index():
    """Read train.csv and list the images. Returns csv_path, img_dir, files, labels, df."""
    csv = find_csv()
    if csv is None:
        raise FileNotFoundError("train.csv was not found inside the project folder: " + str(BASE))
    df = pd.read_csv(csv)
    if "ImageId" not in df.columns and "ImageId_ClassId" in df.columns:   # older format
        df["ImageId"] = df["ImageId_ClassId"].str.split("_").str[0]
        df["ClassId"] = df["ImageId_ClassId"].str.split("_").str[1].astype(int)
    if "ImageId" not in df.columns:
        raise ValueError("train.csv has unexpected columns: " + str(list(df.columns)))
    df = df.dropna(subset=["EncodedPixels"])
    defective = set(df["ImageId"].unique())
    img_dir = find_img_dir(sorted(defective)[0])
    if img_dir is None:
        raise FileNotFoundError("Image folder (train_images) was not found next to train.csv.")
    files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    labels = np.array([1 if f in defective else 0 for f in files], dtype=np.int64)
    if labels.sum() == 0 or labels.sum() == len(labels):
        raise ValueError("All images got the same label - this is not the real train.csv.")
    return csv, img_dir, files, labels, df


# ----------------------------------------------------------------------------
# 2. Features (what the ANN sees)   <<FEATURES-START>>
# ----------------------------------------------------------------------------
_FLIPS = ((slice(None), slice(None)),                         # original
          (slice(None), slice(None, None, -1)),               # left-right flip
          (slice(None, None, -1), slice(None)),               # up-down flip
          (slice(None, None, -1), slice(None, None, -1)))     # both flips
_Q = [1, 5, 25, 50, 75, 95, 99]


def features_from_gray(img):
    """Gray image -> array (4, F): the feature vector of the image and of its 3 flipped versions."""
    g = cv2.resize(img, (RW, RH), interpolation=cv2.INTER_AREA).astype(np.float32)
    g = (g - g.mean()) / (g.std() + 1e-6)                                   # per-image normalisation
    r = g - np.median(g, axis=0, keepdims=True)                             # background removal
    r = r - np.median(r, axis=1, keepdims=True)                             # (median polish)
    gm = (np.abs(np.diff(g, axis=1, prepend=g[:, :1])) +
          np.abs(np.diff(g, axis=0, prepend=g[:1, :])))                     # edge strength
    rs = r / (r.std() + 1e-6)
    glob = np.concatenate([
        np.percentile(g, _Q), np.percentile(r, _Q), np.percentile(gm, _Q),
        np.array([(g ** 3).mean(), (g ** 4).mean() - 3, (rs ** 3).mean(), (rs ** 4).mean() - 3,
                  r.std(), np.abs(r).max()]),
        np.histogram(np.clip(gm, 0, 2), bins=16, range=(0, 2))[0] / gm.size]).astype(np.float32)
    out = []
    for fl in _FLIPS:
        gv, rv, mv = (np.ascontiguousarray(a[fl]) for a in (g, r, gm))
        gb = gv.reshape(BR, RH // BR, BC, RW // BC)
        rb = rv.reshape(BR, RH // BR, BC, RW // BC)
        mb = mv.reshape(BR, RH // BR, BC, RW // BC)
        parts = [gb.mean(axis=(1, 3)).ravel(), gb.std(axis=(1, 3)).ravel(),
                 mb.mean(axis=(1, 3)).ravel(), rb.std(axis=(1, 3)).ravel(),
                 np.abs(rb).max(axis=(1, 3)).ravel(),
                 gv.mean(0).reshape(64, 8).mean(1), rv.std(0).reshape(64, 8).mean(1),
                 gv.mean(1).reshape(16, 8).mean(1), rv.std(1).reshape(16, 8).mean(1), glob]
        out.append(np.concatenate(parts))
    return np.stack(out).astype(np.float32)
# <<FEATURES-END>>


def features_from_path(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return None if img is None else features_from_gray(img)          # None = damaged file


def build_features(img_dir, files, labels, progress=None):
    """Features for all images (cached on disk). Damaged images are dropped."""
    if CACHE_PATH.exists():
        z = np.load(CACHE_PATH, allow_pickle=False)
        if list(z["files"]) == list(files):
            return z["X"].astype(np.float32), z["y"]
    paths = [img_dir / f for f in files]
    feats = []
    with ThreadPoolExecutor(8) as ex:
        for i, f in enumerate(ex.map(features_from_path, paths)):
            feats.append(f)
            if progress and i % 250 == 0:
                progress(i / len(paths), f"Reading images {i}/{len(paths)}")
    keep = [i for i, f in enumerate(feats) if f is not None]
    X = np.stack([feats[i] for i in keep]).astype(np.float32)
    y = labels[keep]
    if len(keep) == len(files):          # cache only when no image was dropped
        np.savez(CACHE_PATH, X=X.astype(np.float16), y=y, files=np.array(files))
    return X, y


# ----------------------------------------------------------------------------
# 3. The ANN (an ensemble of small MLPs)
# ----------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


@torch.no_grad()
def ens_probs(models, X3, device, bs=4096):
    """Average probability over all models AND over the 4 flipped versions of each image."""
    n, V, F = X3.shape
    flat = X3.reshape(n * V, F)
    total = 0
    for m in models:
        m.eval()
        outs = [torch.sigmoid(m(flat[i:i + bs].to(device))).cpu() for i in range(0, len(flat), bs)]
        total = total + torch.cat(outs).reshape(n, V).mean(1)
    return (total / len(models)).numpy()


def bce(p, y):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def train_one(seed, Xtr, ytr, Xva, yva, device, epochs, batch, lr, on_epoch=None):
    torch.manual_seed(seed)
    n, V, F = Xtr.shape
    Xf = Xtr.reshape(n * V, F)
    yf = ytr.repeat_interleave(V)
    model = MLP(F).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3, min_lr=1e-5)
    lossf = nn.BCEWithLogitsLoss()
    yva_np = yva.numpy()
    hist = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
    best_loss, best_state, best_ep, wait, patience = 1e9, None, 0, 0, 8

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xf))
        run_loss, run_ok, seen = 0.0, 0, 0
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            if len(b) < 2:                               # BatchNorm needs >= 2 samples
                continue
            xb, yb = Xf[b].to(device), yf[b].to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = lossf(logits, yb)
            loss.backward(); opt.step()
            run_loss += loss.item() * len(b)
            run_ok += ((logits > 0).float() == yb).sum().item()
            seen += len(b)
        v_prob = ens_probs([model], Xva, device)
        v_loss = bce(v_prob, yva_np)
        v_acc = float(((v_prob > 0.5) == yva_np).mean())
        hist["loss"].append(run_loss / seen); hist["accuracy"].append(run_ok / seen)
        hist["val_loss"].append(v_loss); hist["val_accuracy"].append(v_acc)
        sched.step(v_loss)
        if v_loss < best_loss - 1e-4:
            best_loss, best_ep, wait = v_loss, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
        if on_epoch:
            on_epoch(ep + 1, epochs, hist)
        if wait >= patience:
            break
    model.load_state_dict(best_state)
    return model, hist, best_ep + 1


def train_model(epochs=60, batch=256, lr=1e-3, n_models=3, progress=None, on_epoch=None):
    t0 = time.time()
    csv, img_dir, files, labels, _ = load_index()
    X, y = build_features(img_dir, files, labels, progress)          # X: (N, 4, F)

    idx = np.arange(len(y))
    tr, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=SEED)

    F = X.shape[2]
    mu = X[tr].reshape(-1, F).mean(0); sd = X[tr].reshape(-1, F).std(0) + 1e-6   # training set only
    X -= mu; X /= sd
    np.clip(X, -10, 10, out=X)
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y.astype(np.float32))
    Xtr, ytr, Xva, yva, Xte, yte = Xt[tr], yt[tr], Xt[va], yt[va], Xt[te], yt[te]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models, hist0, best_eps = [], None, []
    for k in range(n_models):
        cb = (lambda ep, total, h, k=k: on_epoch(k + 1, n_models, ep, total, h)) if on_epoch else None
        m, h, be = train_one(SEED + k, Xtr, ytr, Xva, yva, device, epochs, batch, lr, cb)
        models.append(m); best_eps.append(be)
        if hist0 is None:
            hist0 = h

    yva_np, yte_np, ytr_np = yva.numpy(), yte.numpy(), ytr.numpy()
    v_prob = ens_probs(models, Xva, device)
    thr = float(max(np.arange(0.30, 0.71, 0.02), key=lambda t: ((v_prob > t) == yva_np).mean()))
    tr_prob = ens_probs(models, Xtr, device)
    te_prob = ens_probs(models, Xte, device)
    pred = (te_prob > thr).astype(int)

    metrics = {
        "train_acc": float(((tr_prob > thr) == ytr_np).mean()),
        "val_acc": float(((v_prob > thr) == yva_np).mean()),
        "test_acc": float((pred == yte_np).mean()),
        "test_loss": bce(te_prob, yte_np),
        "threshold": thr, "best_epoch": int(best_eps[0]), "epochs_run": len(hist0["loss"]),
        "n_models": n_models, "n_features": int(F),
        "n_train": len(tr), "n_val": len(va), "n_test": len(te),
        "report": classification_report(yte_np, pred, target_names=["clean", "defect"],
                                        output_dict=True, zero_division=0),
        "confusion": confusion_matrix(yte_np, pred).tolist(),
        "seconds": round(time.time() - t0, 1), "device": device,
    }
    torch.save({"states": [{k: v.cpu() for k, v in m.state_dict().items()} for m in models],
                "mu": torch.from_numpy(mu), "sd": torch.from_numpy(sd),
                "n_in": int(F), "threshold": thr}, MODEL_PATH)
    HIST_PATH.write_text(json.dumps(hist0))
    METRICS_PATH.write_text(json.dumps(metrics))
    GATE_PATH.unlink(missing_ok=True)      # steel check is refitted on the next use
    return hist0, metrics


def load_bundle():
    ck = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    models = []
    for s in ck["states"]:
        m = MLP(ck["n_in"]); m.load_state_dict(s); m.eval(); models.append(m)
    return models, ck["mu"].numpy(), ck["sd"].numpy(), float(ck["threshold"])


def predict_gray(img, bundle):
    models, mu, sd, thr = bundle
    x = np.clip((features_from_gray(img) - mu) / sd, -10, 10).astype(np.float32)
    p = float(ens_probs(models, torch.from_numpy(x[None]), "cpu")[0])
    return p, thr



# ----------------------------------------------------------------------------
# 3b. Steel check: "does this image look like the steel images the ANN was trained on?"
#     The ANN only knows steel, so it cannot tell by itself. This check learns the normal
#     range of the training images (no retraining of the ANN, no extra data needed) and
#     rejects images that are the wrong shape, colored, or have an unusual texture pattern.
# ----------------------------------------------------------------------------
# <<GATE-START>>
ASPECT_MIN, ASPECT_MAX = 3.5, 9.5      # steel images are 1600 x 256 (about 6.3 : 1)
GATE_PCA = 40                          # number of principal components kept
GATE_PCT = 99.5                        # about 1% of real steel images may be flagged as unusual


def _colorfulness(bgr):
    """0 for a gray image; larger for a colored one."""
    if bgr is None or bgr.ndim < 3:
        return 0.0
    b = bgr.astype(np.int16)
    return float((b.max(axis=2) - b.min(axis=2)).mean())


def _fit_gate_arrays(Xtr0, Xva0, color_thr):
    """Xtr0 / Xva0: standardized feature vectors (one per image). Returns the steel-check parameters."""
    m = Xtr0.mean(0)
    _, S, Vt = np.linalg.svd(Xtr0 - m, full_matrices=False)
    V = Vt[:GATE_PCA]
    lam = (S[:GATE_PCA] ** 2) / (len(Xtr0) - 1)

    def scores(A):
        Ac = A - m
        z = Ac @ V.T
        return (z ** 2 / lam).sum(1), ((Ac - z @ V) ** 2).sum(1)
    mv, sv = scores(Xva0)
    return {"m": m, "V": V, "lam": lam,
            "thr_m": np.array(np.percentile(mv, GATE_PCT)),
            "thr_s": np.array(np.percentile(sv, GATE_PCT)),
            "color_thr": np.array(color_thr)}


def steel_check(bgr, bundle, gate):
    """Returns (is_steel, reasons). bgr = image as read by OpenCV in color."""
    reasons = []
    h, w = bgr.shape[:2]
    aspect = w / h
    if not (ASPECT_MIN <= aspect <= ASPECT_MAX):
        reasons.append(f"The shape is {w} x {h} pixels (width / height = {aspect:.1f}); "
                       "the steel images used for training are about 6.3 times wider than tall.")
    col = _colorfulness(bgr)
    if col > float(gate["color_thr"]):
        reasons.append(f"The image is colored (color level {col:.0f}); the steel images are gray.")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    _, mu, sd, _ = bundle
    x = np.clip((features_from_gray(gray)[0] - mu) / sd, -10, 10)
    xc = x - gate["m"]
    z = xc @ gate["V"].T
    maha = float((z ** 2 / gate["lam"]).sum())
    spe = float(((xc - z @ gate["V"]) ** 2).sum())
    if maha > float(gate["thr_m"]) or spe > float(gate["thr_s"]):
        reasons.append("The texture and brightness pattern is unlike the steel images the model "
                       "was trained on.")
    return len(reasons) == 0, reasons
# <<GATE-END>>


def fit_steel_check(progress=None):
    """Learn the steel check from the cached training features (about one minute)."""
    csv, img_dir, files, labels, _ = load_index()
    X, y = build_features(img_dir, files, labels, progress)                 # (N, 4, F)
    ck = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    mu, sd = ck["mu"].numpy(), ck["sd"].numpy()
    idx = np.arange(len(y))
    tr, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=SEED)

    def prep(a):
        return np.clip((a - mu) / sd, -10, 10).astype(np.float32)
    color_thr = 15.0
    if len(y) == len(files):                  # measure how gray the real steel images are
        vals = [_colorfulness(cv2.imread(str(img_dir / files[i]), cv2.IMREAD_COLOR)) for i in va[:600]]
        color_thr = float(max(10.0, 3.0 * np.percentile(vals, 99)))
    g = _fit_gate_arrays(prep(X[tr, 0, :]), prep(X[va, 0, :]), color_thr)
    np.savez(GATE_PATH, **g)
    return g


def load_steel_check():
    z = np.load(GATE_PATH)
    return {k: z[k] for k in z.files}


# ----------------------------------------------------------------------------
# 4. Dashboard
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cached_index(_stamp):
    csv, img_dir, files, labels, df = load_index()
    return str(csv), str(img_dir), files, labels, df


@st.cache_resource(show_spinner=False)
def cached_bundle(_mtime):
    return load_bundle()


@st.cache_resource(show_spinner=False)
def cached_gate(_mtime):
    return load_steel_check()


def plot_curves(h):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].plot(h["accuracy"], label="train"); ax[0].plot(h["val_accuracy"], label="validation")
    ax[0].axhline(TARGET, color="gray", ls="--", lw=1, label=f"target {TARGET}")
    ax[0].set_title("Accuracy (first model of the ensemble)"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(h["loss"], label="train"); ax[1].plot(h["val_loss"], label="validation")
    ax[1].set_title("Loss (first model of the ensemble)"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout()
    return fig


def run_dashboard():
    st.set_page_config(page_title="Visual Defect Detective System", layout="wide")
    st.title("Visual Defect Detective System")
    st.caption("Severstal steel surface images  |  plain ANN (fully connected network)  |  DEFECT vs CLEAN")

    if "stamp" not in st.session_state:
        st.session_state.stamp = 0

    with st.sidebar:
        st.header("Training settings")
        epochs = st.slider("Max epochs", 10, 150, 60, 5)
        batch = st.select_slider("Batch size", [128, 256, 512], value=256)
        lr = st.select_slider("Learning rate", [3e-4, 5e-4, 1e-3, 2e-3], value=1e-3)
        n_models = st.slider("Ensemble size (number of ANNs)", 1, 5, 3)
        st.caption("Early stopping keeps the best epoch of every ANN automatically.")
        if st.button("Rescan project folder"):
            st.session_state.stamp += 1
            cached_index.clear()

    # ----- make sure the data exists -----
    try:
        csv, img_dir, files, labels, df = cached_index(st.session_state.stamp)
    except Exception as e:
        st.error(str(e))
        zips = find_zips()
        if zips:
            st.info("A zip file was found in the project folder. You can unpack it here "
                    "(damaged files are skipped, test_images are not needed).")
            z = st.selectbox("Zip file", [str(p) for p in zips])
            if st.button("Extract zip"):
                with st.spinner("Extracting... this can take a few minutes"):
                    try:
                        ok, bad = extract_zip_tolerant(Path(z))
                        st.success(f"Extracted {ok} files, skipped {bad} damaged files.")
                        st.session_state.stamp += 1
                        cached_index.clear()
                        st.rerun()
                    except Exception as ex:
                        st.error(f"The zip is damaged and cannot be read: {ex}")
        else:
            st.warning("Put train.csv + the train_images folder (or the competition zip) in the same "
                       "folder as this file, then press 'Rescan project folder'.")
        return

    tab_data, tab_train, tab_res, tab_det = st.tabs(["Data", "Train", "Results", "Detective"])

    # ----- DATA -----
    with tab_data:
        c1, c2, c3 = st.columns(3)
        c1.metric("Images", f"{len(files):,}")
        c2.metric("Defective", f"{int(labels.sum()):,}")
        c3.metric("Clean", f"{int((labels == 0).sum()):,}")
        st.caption(f"labels: {csv}\n\nimages: {img_dir}")
        left, right = st.columns(2)
        with left:
            st.subheader("Class balance")
            st.bar_chart(pd.Series({"clean": int((labels == 0).sum()),
                                    "defect": int(labels.sum())}))
        with right:
            if "ClassId" in df.columns:
                st.subheader("Defect types (ClassId)")
                st.bar_chart(df.drop_duplicates(["ImageId", "ClassId"])["ClassId"]
                             .value_counts().sort_index())
        st.subheader("Samples")
        rng = np.random.default_rng(1)
        d_idx = rng.choice(np.where(labels == 1)[0], 2, replace=False)
        c_idx = rng.choice(np.where(labels == 0)[0], 2, replace=False)
        for i in list(d_idx) + list(c_idx):
            im = cv2.imread(os.path.join(img_dir, files[i]), cv2.IMREAD_GRAYSCALE)
            if im is not None:
                st.image(im, caption=("DEFECT" if labels[i] else "CLEAN") + "  -  " + files[i],
                         use_container_width=True)

    # ----- TRAIN -----
    with tab_train:
        st.write("Training reads every image once (about 5 to 10 minutes the first time, then the features "
                 "are cached), splits the data 70 / 15 / 15, and trains an ensemble of ANNs with early stopping.")
        if st.button("Start training", type="primary"):
            bar = st.progress(0.0, text="Starting...")
            chart = st.empty()

            def prog(frac, msg):
                bar.progress(min(frac, 1.0), text=msg)

            def on_epoch(k, K, ep, total, h):
                bar.progress(ep / total,
                             text=f"ANN {k}/{K} | epoch {ep}/{total} | val acc {h['val_accuracy'][-1]:.3f}")
                chart.line_chart(pd.DataFrame({"train acc": h["accuracy"],
                                               "val acc": h["val_accuracy"]}))
            with st.spinner("Training..."):
                hist, met = train_model(epochs, batch, lr, n_models, prog, on_epoch)
            cached_bundle.clear()
            bar.progress(1.0, text="Done")
            st.success(f"Finished in {met['seconds']} s  |  test accuracy {met['test_acc']:.4f}. "
                       "Open the Results tab.")

    # ----- RESULTS -----
    with tab_res:
        if not (HIST_PATH.exists() and METRICS_PATH.exists()):
            st.info("Train the model first (Train tab).")
        else:
            h = json.loads(HIST_PATH.read_text()); m = json.loads(METRICS_PATH.read_text())
            a, b, c, d = st.columns(4)
            a.metric("Train accuracy", f"{m['train_acc']:.3f}")
            b.metric("Validation accuracy", f"{m['val_acc']:.3f}")
            c.metric("Test accuracy", f"{m['test_acc']:.3f}",
                     delta=f"{m['test_acc'] - TARGET:+.3f} vs target {TARGET}")
            d.metric("ANNs in ensemble", f"{m.get('n_models', 1)}")
            if m["test_acc"] >= TARGET:
                st.success(f"Target reached: test accuracy {m['test_acc']:.3f} >= {TARGET}")
            else:
                st.warning(f"Target not reached yet: {m['test_acc']:.3f} < {TARGET}. "
                           "Try a larger ensemble, more epochs or a lower learning rate.")
            gap = m["train_acc"] - m["val_acc"]
            if gap > 0.08:
                st.error(f"Train accuracy is {gap:.3f} above validation accuracy: the model is overfitting.")
            elif m["val_acc"] < 0.75:
                st.error("Both accuracies are low: the model is underfitting.")
            else:
                st.info(f"Train/validation gap is {gap:.3f}: no strong overfitting.")
            st.pyplot(plot_curves(h))
            cm = np.array(m["confusion"])
            l, r = st.columns(2)
            with l:
                st.subheader("Confusion matrix (test set)")
                fig, ax = plt.subplots(figsize=(3.6, 3.2))
                ax.imshow(cm, cmap="Blues")
                ax.set_xticks([0, 1], ["clean", "defect"]); ax.set_yticks([0, 1], ["clean", "defect"])
                ax.set_xlabel("predicted"); ax.set_ylabel("actual")
                for i in range(2):
                    for j in range(2):
                        ax.text(j, i, cm[i, j], ha="center", va="center",
                                color="white" if cm[i, j] > cm.max() / 2 else "black")
                fig.tight_layout(); st.pyplot(fig)
            with r:
                st.subheader("Classification report")
                st.dataframe(pd.DataFrame(m["report"]).T.round(3))

    # ----- DETECTIVE -----
    with tab_det:
        if not MODEL_PATH.exists():
            st.info("Train the model first (Train tab).")
        else:
            bundle = cached_bundle(MODEL_PATH.stat().st_mtime)
            if not GATE_PATH.exists():
                with st.spinner("Setting up the steel check (one time only, about a minute)..."):
                    fit_steel_check()
            gate = cached_gate(GATE_PATH.stat().st_mtime)
            src = st.radio("Image source", ["Upload an image", "Pick from dataset"], horizontal=True)
            bgr, truth, name = None, None, ""
            if src == "Upload an image":
                up = st.file_uploader("Image to inspect", type=["jpg", "jpeg", "png"])
                if up is not None:
                    bgr = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
                    name = up.name
                    if bgr is None:
                        st.error("This file could not be read as an image.")
            else:
                name = st.selectbox("Image", files[:500])
                bgr = cv2.imread(os.path.join(img_dir, name), cv2.IMREAD_COLOR)
                truth = "DEFECT" if labels[files.index(name)] else "CLEAN"
            if bgr is not None:
                st.image(bgr, caption=name, channels="BGR", use_container_width=True)
                is_steel, reasons = steel_check(bgr, bundle, gate)
                if is_steel:
                    st.success("Steel check: this looks like a steel surface image.")
                else:
                    st.error("Steel check: this does NOT look like a steel surface image.")
                    for r in reasons:
                        st.write("- " + r)
                    st.caption("The check compares the image with the steel images the model was trained on. "
                               "It cannot recognize every kind of steel, and it may rarely flag a real one.")
                show = is_steel or st.checkbox("Show the defect prediction anyway (not reliable)")
                if show:
                    p, thr = predict_gray(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), bundle)
                    label = "DEFECT" if p > thr else "CLEAN"
                    (st.error if label == "DEFECT" else st.success)(
                        f"Prediction: {label}   |   defect probability {p:.1%}")
                    st.progress(min(max(p, 0.0), 1.0), text="defect probability")
                    if truth:
                        st.write(f"Actual label: **{truth}**  -  " + ("correct" if truth == label else "wrong"))


# ----------------------------------------------------------------------------
def _in_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


def cli():
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        def on_epoch(k, K, ep, total, h):
            print(f"ANN {k}/{K} epoch {ep:3d}/{total} | loss {h['loss'][-1]:.4f} acc {h['accuracy'][-1]:.4f} | "
                  f"val_loss {h['val_loss'][-1]:.4f} val_acc {h['val_accuracy'][-1]:.4f}")
        hist, met = train_model(on_epoch=on_epoch, progress=lambda f, msg: print(msg))
        print(f"\nTRAIN {met['train_acc']:.4f} | VAL {met['val_acc']:.4f} | TEST {met['test_acc']:.4f} "
              f"(target {TARGET}) | confusion {met['confusion']}")
    else:
        print("Start the dashboard with:   python -m streamlit run defect_detective_v2.py\n"
              "Train only (terminal):      python defect_detective_v2.py train")


if _in_streamlit():
    run_dashboard()
elif __name__ == "__main__":
    cli()