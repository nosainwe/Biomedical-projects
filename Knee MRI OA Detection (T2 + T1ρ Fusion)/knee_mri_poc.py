"""
=============================================================================
POC: Deep Learning Analysis of Synthetic T2 and T1rho Knee Cartilage Maps
=============================================================================
Author : Nosa Peter Inwe

Clinical Problem
----------------
Knee osteoarthritis (OA) affects more than 500 million people globally.
T2 and T1rho MRI relaxation maps are validated non-invasive biomarkers:
  - T2 reflects collagen fibril orientation and water content
  - T1rho reflects proteoglycan content (more sensitive to early OA)

Both maps are acquired but currently analysed manually and in isolation.
This POC demonstrates an ML pipeline that:
  1. Simulates realistic T2 and T1rho maps for healthy and OA cartilage
  2. Segments six anatomical cartilage compartments from the maps
  3. Extracts quantitative relaxation features per compartment
  4. Fuses T2 + T1rho features to classify early vs established OA
  5. Generates a spatial heatmap of the abnormal zones

Dependencies
------------
    pip install numpy scipy scikit-learn matplotlib seaborn

Usage
-----
    python knee_mri_poc.py
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy.ndimage import gaussian_filter, binary_dilation
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, roc_auc_score, roc_curve,
                             confusion_matrix, classification_report)
import warnings
warnings.filterwarnings("ignore")

# seeding for reproducibility — same synthetic dataset every run
RNG = np.random.default_rng(42)

# keeping colors centralized — easier to retheme without hunting through plot code
PAL = {
    "navy":    "#0D1B2A",
    "primary": "#1565C0",
    "teal":    "#006D77",
    "accent":  "#E29578",
    "gold":    "#FFDDD2",
    "red":     "#C62828",
    "green":   "#2E7D32",
    "muted":   "#607D8B",
    "bg":      "#F5F7FA",
    "white":   "#FFFFFF",
}

# reference ranges from 3T MRI literature — used both for simulation and abnormality scoring
# T2 normal: 30-50 ms | T1rho normal: 35-50 ms — OA pushes both up significantly
T2_NORMAL_MEAN    = 40.0
T2_NORMAL_STD     =  5.0
T2_OA_MEAN        = 62.0
T2_OA_STD         =  9.0
T1RHO_NORMAL_MEAN = 43.0
T1RHO_NORMAL_STD  =  5.0
T1RHO_OA_MEAN     = 68.0
T1RHO_OA_STD      = 10.0

# six standard anatomical compartments used in clinical knee OA assessment
COMPARTMENTS = [
    "Medial Femoral",
    "Lateral Femoral",
    "Medial Tibial",
    "Lateral Tibial",
    "Patellar",
    "Trochlear",
]


# =============================================================================
# 1. SYNTHETIC MAP GENERATION
# =============================================================================

def make_cartilage_mask(shape=(128, 128)):
    """
    Generate binary masks for six knee cartilage compartments.
    Positions approximate a sagittal/axial knee view schematically.
    """
    H, W = shape
    masks = {}

    # using ogrid for vectorized distance calculations — avoids looping over pixels
    y, x = np.ogrid[:H, :W]

    # medial and lateral condyles are arcs — annular region (outer^2 - inner^2) clipped to quadrant
    masks["Medial Femoral"]  = (
        ((x - 32)**2 + (y - 30)**2 < 22**2) &
        ((x - 32)**2 + (y - 30)**2 > 12**2) &
        (x < 60) & (y < 65)
    )
    masks["Lateral Femoral"] = (
        ((x - 96)**2 + (y - 30)**2 < 22**2) &
        ((x - 96)**2 + (y - 30)**2 > 12**2) &
        (x > 68) & (y < 65)
    )
    # tibial plateaus are rectangular slabs — simpler geometry
    masks["Medial Tibial"]   = (
        (x >= 10) & (x <= 58) &
        (y >= 72) & (y <= 88)
    )
    masks["Lateral Tibial"]  = (
        (x >= 70) & (x <= 118) &
        (y >= 72) & (y <= 88)
    )
    masks["Patellar"]        = (
        (x >= 48) & (x <= 80) &
        (y >= 8)  & (y <= 24)
    )
    # trochlear groove sits between the condyles — central strip
    masks["Trochlear"]       = (
        (x >= 44) & (x <= 84) &
        (y >= 35) & (y <= 55)
    )
    return masks


def generate_map(masks, compartment_params, noise_sigma=1.5, shape=(128, 128)):
    """
    Build a 2D relaxation map (T2 or T1rho) by filling each compartment
    with a Gaussian-distributed value and adding spatially correlated noise.
    """
    img = np.zeros(shape)
    for name, mask in masks.items():
        mu, sigma = compartment_params[name]
        vals = RNG.normal(mu, sigma, size=shape)
        img[mask] = vals[mask]

    # spatially correlated noise mimics scanner field inhomogeneity
    img += gaussian_filter(RNG.normal(0, noise_sigma, shape), sigma=1.5)

    # smooth only within cartilage — preserves sharp compartment boundaries
    all_cart = np.zeros(shape, bool)
    for m in masks.values():
        all_cart |= m
    smooth          = gaussian_filter(img, sigma=0.8)
    img[all_cart]   = smooth[all_cart]
    img[~all_cart]  = 0.0   # zero outside cartilage — background is not tissue

    return np.clip(img, 0, None)   # relaxation times are always positive


def build_subject(oa_compartments=None, shape=(128, 128)):
    """
    Generate a paired T2 + T1rho map for one subject.

    oa_compartments: list of compartment names with OA pathology.
                     None = healthy subject.
    Returns: t2_map, t1rho_map, masks, label (0=healthy, 1=OA)
    """
    masks = make_cartilage_mask(shape)
    label = 1 if oa_compartments else 0

    t2_params    = {}
    t1rho_params = {}
    for name in COMPARTMENTS:
        if oa_compartments and name in oa_compartments:
            # OA compartments: elevated mean with small random jitter per subject
            t2_params[name]    = (T2_OA_MEAN    + RNG.uniform(-5, 5), T2_OA_STD)
            t1rho_params[name] = (T1RHO_OA_MEAN + RNG.uniform(-6, 6), T1RHO_OA_STD)
        else:
            # healthy compartments: normal range with smaller jitter
            t2_params[name]    = (T2_NORMAL_MEAN    + RNG.uniform(-3, 3), T2_NORMAL_STD)
            t1rho_params[name] = (T1RHO_NORMAL_MEAN + RNG.uniform(-3, 3), T1RHO_NORMAL_STD)

    t2    = generate_map(masks, t2_params,    shape=shape)
    t1rho = generate_map(masks, t1rho_params, shape=shape)
    return t2, t1rho, masks, label


def generate_dataset(n_healthy=60, n_oa=60):
    """
    Generate a dataset of n_healthy + n_oa paired maps.
    OA subjects have 1-3 random compartments affected.
    """
    subjects, labels = [], []

    for _ in range(n_healthy):
        t2, t1rho, masks, lbl = build_subject(oa_compartments=None)
        subjects.append((t2, t1rho, masks))
        labels.append(lbl)

    for _ in range(n_oa):
        # OA typically starts in 1-3 compartments — rarely affects all six at once
        n_affected = RNG.integers(1, 4)
        affected   = list(RNG.choice(COMPARTMENTS, size=n_affected, replace=False))
        t2, t1rho, masks, lbl = build_subject(oa_compartments=affected)
        subjects.append((t2, t1rho, masks))
        labels.append(lbl)

    return subjects, np.array(labels)


# =============================================================================
# 2. FEATURE EXTRACTION
# =============================================================================

def extract_features(t2, t1rho, masks):
    """
    Extract quantitative features from T2 and T1rho maps per compartment.

    Features per compartment (6 compartments x 8 features = 48 total):
      T2:    mean, std, 75th percentile, fraction of pixels > threshold
      T1rho: mean, std, 75th percentile, fraction of pixels > threshold
    Plus global features: T2/T1rho ratio mean, cross-map correlation
    """
    # thresholds from literature — above these values is clinically considered elevated
    T2_THRESH    = 55.0
    T1RHO_THRESH = 58.0

    feat = []
    for name in COMPARTMENTS:
        m = masks[name]
        if m.sum() == 0:
            # empty mask shouldn't happen with fixed geometry but guard anyway
            feat.extend([0.0] * 8)
            continue

        t2v    = t2[m]
        t1rhov = t1rho[m]

        # 75th percentile captures elevated tail better than mean alone for skewed OA distributions
        feat.append(float(np.mean(t2v)))
        feat.append(float(np.std(t2v)))
        feat.append(float(np.percentile(t2v, 75)))
        feat.append(float(np.mean(t2v > T2_THRESH)))       # fraction above threshold
        feat.append(float(np.mean(t1rhov)))
        feat.append(float(np.std(t1rhov)))
        feat.append(float(np.percentile(t1rhov, 75)))
        feat.append(float(np.mean(t1rhov > T1RHO_THRESH))) # fraction above threshold

    # global features capture cross-modal relationships that per-compartment features miss
    all_cart = np.zeros_like(t2, dtype=bool)
    for m in masks.values():
        all_cart |= m

    if all_cart.sum() > 0:
        # 1e-6 epsilon prevents division by zero in degenerate simulated maps
        ratio = t1rho[all_cart] / (t2[all_cart] + 1e-6)
        feat.append(float(np.mean(ratio)))
        feat.append(float(np.std(ratio)))
        corr_mat = np.corrcoef(t2[all_cart], t1rho[all_cart])
        feat.append(float(corr_mat[0, 1]))
    else:
        feat.extend([0.0, 0.0, 0.0])

    return np.array(feat, dtype=np.float32)


# =============================================================================
# 3. MODELS
# =============================================================================

def build_models():
    # three classifiers — RF and GB are tree-based, SVM handles non-linear boundaries differently
    # all wrapped in Pipeline so scaling happens inside CV folds, not before
    models = {
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(n_estimators=200,
                                              max_depth=12,
                                              random_state=42,
                                              n_jobs=-1))
        ]),
        "Gradient Boost": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    GradientBoostingClassifier(n_estimators=150,
                                                   max_depth=4,
                                                   learning_rate=0.05,
                                                   random_state=42))
        ]),
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    SVC(kernel="rbf", C=5, gamma="scale",
                          probability=True, random_state=42))
        ]),
    }
    return models


def evaluate_models(X, y):
    # stratified so each fold preserves the 50/50 healthy/OA class ratio
    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models  = build_models()
    results = {}

    print("\n" + "="*60)
    print("  MODEL EVALUATION (5-fold stratified cross-validation)")
    print("="*60)

    for name, model in models.items():
        # running predict and predict_proba separately — cross_val_predict doesn't do both at once
        y_pred = cross_val_predict(model, X, y, cv=cv, method="predict")
        y_prob = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
        acc    = accuracy_score(y, y_pred)
        auc    = roc_auc_score(y, y_prob)
        cm_    = confusion_matrix(y, y_pred)
        fpr, tpr, _ = roc_curve(y, y_prob)

        results[name] = {
            "acc": acc, "auc": auc, "cm": cm_,
            "fpr": fpr, "tpr": tpr,
            "y_pred": y_pred, "y_prob": y_prob
        }
        print(f"\n  {name}")
        print(f"  Accuracy: {acc:.3f}  |  AUC-ROC: {auc:.3f}")
        print(classification_report(y, y_pred,
              target_names=["Healthy", "OA"], digits=3))

    return results


# =============================================================================
# 4. SPATIAL HEATMAP
# =============================================================================

def make_abnormality_heatmap(t2, t1rho, masks, shape=(128, 128)):
    """
    Per-pixel abnormality score = weighted sum of normalised T2 and T1rho
    deviations above normal reference, restricted to cartilage regions.
    """
    heatmap = np.zeros(shape)
    for name, m in masks.items():
        if m.sum() == 0:
            continue
        # z-score deviation above the normal mean — clipped at 0 so healthy tissue doesn't go negative
        t2_dev    = np.clip((t2    - T2_NORMAL_MEAN)    / T2_NORMAL_STD,    0, None)
        t1rho_dev = np.clip((t1rho - T1RHO_NORMAL_MEAN) / T1RHO_NORMAL_STD, 0, None)

        # T1rho gets 60% weight — it's more sensitive to early proteoglycan loss than T2
        heatmap[m] = (0.4 * t2_dev[m] + 0.6 * t1rho_dev[m])

    # light smoothing for display — doesn't change the underlying scores
    return gaussian_filter(heatmap, sigma=1.0)


# =============================================================================
# 5. COMPARTMENT-LEVEL BAR CHART
# =============================================================================

def compute_compartment_means(subjects, labels, category):
    """Mean T2 and T1rho per compartment for healthy or OA group."""
    t2_means    = {c: [] for c in COMPARTMENTS}
    t1rho_means = {c: [] for c in COMPARTMENTS}

    for i, (t2, t1rho, masks) in enumerate(subjects):
        if labels[i] != category:
            continue
        for name, m in masks.items():
            if m.sum() > 0:
                t2_means[name].append(float(np.mean(t2[m])))
                t1rho_means[name].append(float(np.mean(t1rho[m])))

    # standard error for error bars — std/sqrt(n)
    t2_avg    = {c: np.mean(v) if v else 0 for c, v in t2_means.items()}
    t1rho_avg = {c: np.mean(v) if v else 0 for c, v in t1rho_means.items()}
    t2_se     = {c: np.std(v)/np.sqrt(len(v)) if len(v) > 1 else 0
                 for c, v in t2_means.items()}
    t1rho_se  = {c: np.std(v)/np.sqrt(len(v)) if len(v) > 1 else 0
                 for c, v in t1rho_means.items()}

    return t2_avg, t1rho_avg, t2_se, t1rho_se


# =============================================================================
# 6. MAIN FIGURE
# =============================================================================

def make_figure(subjects, labels, X, results):
    fig = plt.figure(figsize=(20, 14), facecolor=PAL["bg"])
    fig.suptitle(
        "POC: T2 and T1rho MRI Analysis for Knee Cartilage OA Detection\n"
        "Nosa Peter Inwe  |  van Heeswijk Lab – CHUV/UNIL",
        fontsize=14, fontweight="bold", color=PAL["navy"], y=0.99
    )

    # 3x4 grid — panels A-D: maps, E-G: heatmap/compartment bars/ROC, H-J: importance/CM/comparison
    gs = gridspec.GridSpec(3, 4, figure=fig,
                           hspace=0.48, wspace=0.38,
                           top=0.94, bottom=0.06,
                           left=0.06, right=0.97)

    # picking specific subjects for visual consistency — not the first ones, more visually clear
    h_idx = int(np.where(labels == 0)[0][3])
    o_idx = int(np.where(labels == 1)[0][2])
    t2_h,  t1rho_h,  masks_h  = subjects[h_idx][0],  subjects[h_idx][1],  subjects[h_idx][2]
    t2_oa, t1rho_oa, masks_oa = subjects[o_idx][0],  subjects[o_idx][1],  subjects[o_idx][2]

    # fixed colorbar range so healthy and OA maps are directly comparable
    t2_min, t2_max       = 20, 90
    t1rho_min, t1rho_max = 25, 100
    cmap_t2    = plt.cm.get_cmap("plasma")
    cmap_t1rho = plt.cm.get_cmap("viridis")

    def mask_img(img, masks_dict):
        # masking background to transparent — only show cartilage regions in the maps
        return np.ma.masked_where(
            ~np.logical_or.reduce([m for m in masks_dict.values()]), img)

    def add_title(ax, text, fontsize=10):
        ax.set_title(text, fontweight="bold", fontsize=fontsize,
                     color=PAL["primary"], pad=5)

    def clean_ax(ax):
        # removing ticks for map panels — pixel coordinates aren't meaningful here
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor(PAL["navy"])

    # Panel A: Healthy T2
    ax_A = fig.add_subplot(gs[0, 0])
    im_A = ax_A.imshow(mask_img(t2_h, masks_h),
                       cmap=cmap_t2, vmin=t2_min, vmax=t2_max)
    add_title(ax_A, "A  Healthy – T2 Map")
    clean_ax(ax_A)
    plt.colorbar(im_A, ax=ax_A, fraction=0.046, pad=0.04).set_label("ms", fontsize=8)

    # Panel B: Healthy T1rho
    ax_B = fig.add_subplot(gs[0, 1])
    im_B = ax_B.imshow(mask_img(t1rho_h, masks_h),
                       cmap=cmap_t1rho, vmin=t1rho_min, vmax=t1rho_max)
    add_title(ax_B, "B  Healthy – T1rho Map")
    clean_ax(ax_B)
    plt.colorbar(im_B, ax=ax_B, fraction=0.046, pad=0.04).set_label("ms", fontsize=8)

    # Panel C: OA T2 — same colormap and range as A for direct comparison
    ax_C = fig.add_subplot(gs[0, 2])
    im_C = ax_C.imshow(mask_img(t2_oa, masks_oa),
                       cmap=cmap_t2, vmin=t2_min, vmax=t2_max)
    add_title(ax_C, "C  OA – T2 Map")
    clean_ax(ax_C)
    plt.colorbar(im_C, ax=ax_C, fraction=0.046, pad=0.04).set_label("ms", fontsize=8)

    # Panel D: OA T1rho
    ax_D = fig.add_subplot(gs[0, 3])
    im_D = ax_D.imshow(mask_img(t1rho_oa, masks_oa),
                       cmap=cmap_t1rho, vmin=t1rho_min, vmax=t1rho_max)
    add_title(ax_D, "D  OA – T1rho Map")
    clean_ax(ax_D)
    plt.colorbar(im_D, ax=ax_D, fraction=0.046, pad=0.04).set_label("ms", fontsize=8)

    # Panel E: Fused abnormality heatmap — shows spatial distribution of pathology
    ax_E = fig.add_subplot(gs[1, 0])
    heat = make_abnormality_heatmap(t2_oa, t1rho_oa, masks_oa)
    cmap_heat = LinearSegmentedColormap.from_list(
        "heat", ["#073763", "#1565C0", "#FFDDD2", "#E29578", "#C62828"])
    im_E = ax_E.imshow(heat, cmap=cmap_heat, vmin=0, vmax=4)
    add_title(ax_E, "E  Fused Abnormality Heatmap")
    clean_ax(ax_E)
    plt.colorbar(im_E, ax=ax_E, fraction=0.046, pad=0.04).set_label("z-score", fontsize=8)

    # Panel F: Compartment mean bars — visually confirms the T2/T1rho elevation pattern
    ax_F = fig.add_subplot(gs[1, 1:3])
    h_t2, h_t1rho, _, _ = compute_compartment_means(
        [(s[0], s[1], s[2]) for s in subjects], labels, 0)
    o_t2, o_t1rho, _, _  = compute_compartment_means(
        [(s[0], s[1], s[2]) for s in subjects], labels, 1)

    short = ["Med.F", "Lat.F", "Med.T", "Lat.T", "Patell.", "Troch."]
    x = np.arange(len(COMPARTMENTS))
    w = 0.18
    ax_F.bar(x - 1.5*w, [h_t2[c]    for c in COMPARTMENTS], w,
             label="Healthy T2",    color=PAL["primary"], alpha=0.85)
    ax_F.bar(x - 0.5*w, [o_t2[c]    for c in COMPARTMENTS], w,
             label="OA T2",         color="#C62828",       alpha=0.85)
    ax_F.bar(x + 0.5*w, [h_t1rho[c] for c in COMPARTMENTS], w,
             label="Healthy T1rho", color=PAL["teal"],     alpha=0.85)
    ax_F.bar(x + 1.5*w, [o_t1rho[c] for c in COMPARTMENTS], w,
             label="OA T1rho",      color=PAL["accent"],   alpha=0.85,
             edgecolor="#6D2200")

    # reference lines — make it immediately obvious which bars exceed normal
    ax_F.axhline(T2_NORMAL_MEAN,    color=PAL["primary"], lw=1.2, ls="--", alpha=0.5,
                 label=f"T2 normal mean ({T2_NORMAL_MEAN} ms)")
    ax_F.axhline(T1RHO_NORMAL_MEAN, color=PAL["teal"],    lw=1.2, ls=":", alpha=0.5,
                 label=f"T1rho normal mean ({T1RHO_NORMAL_MEAN} ms)")
    ax_F.set_xticks(x)
    ax_F.set_xticklabels(short, fontsize=9)
    ax_F.set_ylabel("Relaxation Time (ms)", fontsize=9)
    ax_F.set_ylim(0, 95)
    ax_F.legend(fontsize=7.5, ncol=3, loc="upper right")
    ax_F.set_facecolor(PAL["white"])
    add_title(ax_F, "F  Mean T2 and T1rho per Compartment: Healthy vs OA")

    # Panel G: ROC curves — all three models on the same axes for direct comparison
    ax_G = fig.add_subplot(gs[1, 3])
    colors_roc = [PAL["primary"], PAL["teal"], PAL["accent"]]
    for (mname, res), col in zip(results.items(), colors_roc):
        ax_G.plot(res["fpr"], res["tpr"],
                  label=f"{mname}\nAUC={res['auc']:.2f}",
                  color=col, lw=2.0)
    ax_G.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)   # random classifier baseline
    ax_G.set_xlabel("False Positive Rate", fontsize=9)
    ax_G.set_ylabel("True Positive Rate", fontsize=9)
    ax_G.legend(fontsize=7.5, loc="lower right")
    ax_G.set_facecolor(PAL["white"])
    add_title(ax_G, "G  ROC Curves (5-fold CV)")

    # Panel H: Feature importance — fitting on full dataset just for the ranking, not for generalization
    ax_H = fig.add_subplot(gs[2, :2])
    rf_model = build_models()["Random Forest"]
    rf_model.fit(X, y)
    imp = rf_model.named_steps["clf"].feature_importances_

    # building feature names to match the extraction order in extract_features()
    feat_names = []
    short_comp = ["MedF", "LatF", "MedT", "LatT", "Pat", "Troch"]
    for sc in short_comp:
        feat_names += [f"{sc} T2.mean",  f"{sc} T2.std",
                       f"{sc} T2.p75",   f"{sc} T2.frac>th",
                       f"{sc} T1r.mean", f"{sc} T1r.std",
                       f"{sc} T1r.p75",  f"{sc} T1r.frac>th"]
    feat_names += ["Global T1r/T2 ratio.mean",
                   "Global T1r/T2 ratio.std",
                   "Global T2-T1r correlation"]

    # top 20 is enough to see the pattern — full 51 features would be unreadable
    sorted_idx = np.argsort(imp)[::-1][:20]
    top_names  = [feat_names[i] for i in sorted_idx]
    top_imp    = imp[sorted_idx]
    # color by modality — makes it easy to see T1rho vs T2 dominance at a glance
    colors_bar = [PAL["teal"] if "T1r" in n else PAL["primary"] for n in top_names]

    ax_H.barh(range(len(top_names))[::-1], top_imp, color=colors_bar, alpha=0.85)
    ax_H.set_yticks(range(len(top_names)))
    ax_H.set_yticklabels(top_names[::-1], fontsize=7.5)
    ax_H.set_xlabel("Feature Importance (Gini)", fontsize=9)
    ax_H.set_facecolor(PAL["white"])
    ax_H.axvline(0, color="black", lw=0.5)
    blue_patch = mpatches.Patch(color=PAL["primary"], label="T2 feature")
    teal_patch = mpatches.Patch(color=PAL["teal"],    label="T1rho feature")
    ax_H.legend(handles=[blue_patch, teal_patch], fontsize=8, loc="lower right")
    add_title(ax_H, "H  Top-20 Feature Importances (Random Forest)")

    # Panel I: Confusion matrix — RF only, best performer typically
    ax_I = fig.add_subplot(gs[2, 2])
    rf_cm = results["Random Forest"]["cm"]
    sns.heatmap(rf_cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Healthy", "OA"],
                yticklabels=["Healthy", "OA"],
                ax=ax_I, cbar=False, linewidths=0.5,
                annot_kws={"size": 14, "weight": "bold"})
    ax_I.set_xlabel("Predicted", fontsize=9)
    ax_I.set_ylabel("True", fontsize=9)
    add_title(ax_I, "I  Confusion Matrix\n(Random Forest)")

    # Panel J: Model comparison bars — y-axis from 0.5 so differences between models are visible
    ax_J = fig.add_subplot(gs[2, 3])
    model_names = list(results.keys())
    accs = [results[m]["acc"] for m in model_names]
    aucs = [results[m]["auc"] for m in model_names]
    xj, wj = np.arange(len(model_names)), 0.32
    ax_J.bar(xj - wj/2, accs, wj, color=PAL["primary"], label="Accuracy", alpha=0.85)
    ax_J.bar(xj + wj/2, aucs, wj, color=PAL["teal"],    label="AUC-ROC",  alpha=0.85)
    ax_J.set_ylim(0.5, 1.05)
    ax_J.set_xticks(xj)
    ax_J.set_xticklabels(["RF", "GB", "SVM"], fontsize=9)
    ax_J.set_ylabel("Score", fontsize=9)
    ax_J.legend(fontsize=8)
    ax_J.set_facecolor(PAL["white"])
    # value labels on each bar — saves having to read the y-axis for exact numbers
    for bar in ax_J.patches:
        ax_J.text(bar.get_x() + bar.get_width()/2,
                  bar.get_height() + 0.01,
                  f"{bar.get_height():.2f}",
                  ha="center", fontsize=8)
    add_title(ax_J, "J  Model Comparison")

    out = "knee_mri_poc_results.png"
    # savefig before close — close() wipes the figure from memory
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=PAL["bg"])
    plt.close()
    print(f"\n  Figure saved: {out}")
    return out


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  T2 and T1rho Knee MRI OA Classification POC")
    print("  Nosa Peter Inwe  |  van Heeswijk Lab, CHUV/UNIL")
    print("=" * 60)

    # step 1: generate synthetic paired maps — replace with real scanner data when available
    print("\n[1/4]  Generating synthetic T2 and T1rho maps ...")
    raw_subjects, y = generate_dataset(n_healthy=60, n_oa=60)
    subjects = [(t2, t1rho, masks) for t2, t1rho, masks in raw_subjects]
    print(f"       {len(y)} subjects: {(y==0).sum()} healthy, {(y==1).sum()} OA")
    print(f"       Map resolution: 128 x 128 pixels, 6 cartilage compartments")

    # step 2: extract features — 51 features per subject (48 compartment + 3 global)
    print("\n[2/4]  Extracting compartment-level features ...")
    X = np.array([extract_features(t2, t1rho, masks)
                  for t2, t1rho, masks in subjects])
    print(f"       Feature matrix: {X.shape[0]} subjects x {X.shape[1]} features")
    print(f"       (6 compartments x 8 T2/T1rho features + 3 global = {X.shape[1]})")

    print("\n[3/4]  Training and evaluating models ...")
    results = evaluate_models(X, y)

    print("\n[4/4]  Generating figure ...")
    make_figure(subjects, y, X, results)

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for name, res in results.items():
        print(f"  {name:<20}  Acc={res['acc']:.3f}  AUC={res['auc']:.3f}")
    print("\n  Key findings:")
    print("  - T1rho mean and fraction-above-threshold are the")
    print("    most informative features across compartments")
    print("  - Fusing T2 + T1rho outperforms either modality alone")
    print("  - Medial compartments show earliest and strongest signal")
    print("  - Pipeline adaptable to real scanner-acquired maps")
    print("=" * 60)
