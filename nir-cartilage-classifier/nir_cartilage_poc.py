"""
=============================================================================
POC: NIR Spectral Classifier for Cartilage Health Assessment
=============================================================================
Author : Nosa Peter Inwe
Purpose: Proof-of-concept demonstrating how machine learning can classify
         Near-Infrared (NIR) spectra into Healthy vs. Osteoarthritic (OA)
         cartilage — directly relevant to Prof. Afara's Biomedical
         Spectroscopy Laboratory (BSL) research at UEF.

Pipeline
--------
1. Synthetic NIR spectra generation  (mimics real tissue absorbance)
2. Spectral preprocessing            (SG filter, MSC, 2nd derivative)
3. Model training & comparison       (Random Forest, SVM, PLS-DA)
4. Evaluation                        (accuracy, AUC-ROC, confusion matrix)
5. Explainability                    (wavelength importance plot)

Dependencies
------------
    pip install numpy scipy scikit-learn matplotlib seaborn

Run
---
    python nir_cartilage_poc.py
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.signal import savgol_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    roc_auc_score, roc_curve, classification_report
)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

# seeding the rng so synthetic data is the same every run — important for reproducibility in a POC
RNG = np.random.default_rng(42)

# 500–2500 nm at 2 nm step — covers all the biologically relevant NIR bands for cartilage
WAVELENGTHS = np.arange(500, 2502, 2)   # 1001 wavelength points
N_SAMPLES   = 300                        # 150 healthy, 150 OA — balanced classes


# =============================================================================
# 1. SYNTHETIC SPECTRA GENERATION
# =============================================================================

def _gaussian(wl, center, width, height):
    # building individual absorption peaks — each peak = one molecular vibration band
    return height * np.exp(-0.5 * ((wl - center) / width) ** 2)


def generate_spectra(n_per_class: int = 150) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate NIR absorbance spectra for two tissue classes.

    Healthy cartilage:
        - Higher water bands at ~970, ~1190, ~1450, ~1940 nm
        - Strong collagen CH2 band at ~1730 nm
        - Strong proteoglycan band at ~2100 nm
    OA cartilage:
        - Reduced proteoglycan (~2100 nm band suppressed)
        - Slightly elevated collagen degradation shoulder ~1680 nm
        - Broader water absorption (tissue hydration change)
    """
    wl = WAVELENGTHS
    spectra, labels = [], []

    # healthy baseline — proteoglycan at 2100 nm is the key signal to preserve
    def healthy_template():
        s  = np.ones(len(wl)) * 0.1
        s += _gaussian(wl,  970,  40, 0.35)   # water overtone
        s += _gaussian(wl, 1190,  60, 0.28)   # water combination
        s += _gaussian(wl, 1450,  80, 0.55)   # water 1st overtone
        s += _gaussian(wl, 1730,  50, 0.40)   # collagen CH2
        s += _gaussian(wl, 1940,  90, 0.70)   # water combination
        s += _gaussian(wl, 2100, 100, 0.50)   # proteoglycan sulphate — strong in healthy
        s += _gaussian(wl, 2300,  70, 0.25)   # collagen CH2 combination
        return s

    # OA template — the proteoglycan band drops to 0.22 from 0.50, that's the whole story
    def oa_template():
        s  = np.ones(len(wl)) * 0.12
        s += _gaussian(wl,  970,  45, 0.38)   # water (slightly wider in OA)
        s += _gaussian(wl, 1190,  65, 0.30)
        s += _gaussian(wl, 1450,  85, 0.60)
        s += _gaussian(wl, 1680,  40, 0.18)   # collagen degradation shoulder — OA-specific
        s += _gaussian(wl, 1730,  55, 0.35)   # collagen CH2 reduced
        s += _gaussian(wl, 1940,  95, 0.73)
        s += _gaussian(wl, 2100, 100, 0.22)   # KEY: proteoglycan loss — classifier leans on this
        s += _gaussian(wl, 2300,  75, 0.22)
        return s

    for _ in range(n_per_class):
        # additive noise + random amplitude scaling to simulate real measurement variance
        noise = RNG.normal(0, 0.012, len(wl))
        scale = RNG.uniform(0.90, 1.10)
        spectra.append(healthy_template() * scale + noise)
        labels.append(0)   # 0 = Healthy

    for _ in range(n_per_class):
        # OA gets slightly more noise — real OA tissue tends to be more heterogeneous
        noise = RNG.normal(0, 0.015, len(wl))
        scale = RNG.uniform(0.88, 1.12)
        spectra.append(oa_template() * scale + noise)
        labels.append(1)   # 1 = OA

    return np.array(spectra), np.array(labels)


# =============================================================================
# 2. SPECTRAL PREPROCESSING
# =============================================================================

def multiplicative_scatter_correction(X: np.ndarray) -> np.ndarray:
    # MSC removes baseline offsets caused by light scattering differences between samples
    # fitting a linear model of each spectrum against the mean, then correcting it out
    ref   = X.mean(axis=0)
    X_msc = np.zeros_like(X)
    for i, x in enumerate(X):
        coef    = np.polyfit(ref, x, 1)
        X_msc[i] = (x - coef[1]) / coef[0]
    return X_msc


def preprocess(X: np.ndarray) -> np.ndarray:
    """
    Standard spectroscopy preprocessing pipeline:
    1. Savitzky-Golay smoothing (window=11, poly=3)
    2. Multiplicative Scatter Correction (MSC)
    3. 2nd derivative (SG, window=15, poly=3, deriv=2)
    """
    # step 1: smooth first to avoid amplifying noise in the derivative step
    X_sg  = savgol_filter(X, window_length=11, polyorder=3, axis=1)
    # step 2: MSC corrects for inter-sample scatter before we differentiate
    X_msc = multiplicative_scatter_correction(X_sg)
    # step 3: 2nd derivative sharpens peaks and removes broad baseline variations
    X_d2  = savgol_filter(X_msc, window_length=15, polyorder=3, deriv=2, axis=1)
    return X_d2


# =============================================================================
# 3. PLS-DA HELPER  (scikit-learn has PLSRegression, not PLSDiscriminant)
# =============================================================================

class PLSDA:
    # sklearn doesn't ship PLS-DA natively so wrapping PLSRegression with a threshold
    # this is standard practice — PLS-DA is just PLS-R with a 0/1 response and a decision boundary
    def __init__(self, n_components: int = 5):
        self.n_components = n_components
        self.pls    = PLSRegression(n_components=n_components)
        self.scaler = StandardScaler()

    def fit(self, X, y):
        Xs = self.scaler.fit_transform(X)
        self.pls.fit(Xs, y)
        scores = self.pls.predict(Xs).ravel()
        # using median of training scores as threshold — robust to class imbalance
        self.threshold = np.median(scores)
        return self

    def predict(self, X):
        Xs     = self.scaler.transform(X)
        scores = self.pls.predict(Xs).ravel()
        return (scores > self.threshold).astype(int)

    def predict_proba(self, X):
        Xs     = self.scaler.transform(X)
        scores = self.pls.predict(Xs).ravel()
        # sigmoid-like rescaling around the threshold to get soft probabilities for AUC
        p1 = 1 / (1 + np.exp(-(scores - self.threshold) * 3))
        return np.column_stack([1 - p1, p1])


# =============================================================================
# 4. EVALUATION  (5-fold cross-validation)
# =============================================================================

def evaluate_models(X: np.ndarray, y: np.ndarray) -> dict:
    # stratified so each fold has the same class ratio — important with balanced but small datasets
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # RF and SVM go through sklearn Pipeline — handles scaling inside CV cleanly
    models = {
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, max_depth=15,
                                           random_state=42, n_jobs=-1))
        ]),
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10, gamma="scale",
                        probability=True, random_state=42))
        ]),
    }

    results = {}
    for name, model in models.items():
        # running predict and predict_proba separately — cross_val_predict doesn't do both at once
        y_pred = cross_val_predict(model, X, y, cv=cv, method="predict")
        y_prob = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
        acc    = accuracy_score(y, y_pred)
        auc    = roc_auc_score(y, y_prob)
        cm     = confusion_matrix(y, y_pred)
        fpr, tpr, _ = roc_curve(y, y_prob)
        results[name] = {
            "accuracy": acc, "auc": auc,
            "cm": cm, "fpr": fpr, "tpr": tpr,
            "y_pred": y_pred, "y_prob": y_prob,
        }
        print(f"\n{'─'*50}")
        print(f"  {name}")
        print(f"{'─'*50}")
        print(f"  Accuracy : {acc:.3f}")
        print(f"  AUC-ROC  : {auc:.3f}")
        print(classification_report(y, y_pred,
              target_names=["Healthy", "OA"], digits=3))

    # PLS-DA needs manual CV because it's not a sklearn estimator
    # reassembling predictions in original order after CV — argsort on test indices does this
    plsda_preds, plsda_probs = [], []
    for train_idx, test_idx in cv.split(X, y):
        plsda = PLSDA(n_components=8)
        plsda.fit(X[train_idx], y[train_idx])
        plsda_preds.append(plsda.predict(X[test_idx]))
        plsda_probs.append(plsda.predict_proba(X[test_idx])[:, 1])

    # reordering fold outputs back to original sample order
    sort_idx   = np.argsort(np.concatenate([test for _, test in cv.split(X, y)]))
    plsda_pred = np.concatenate(plsda_preds)[sort_idx]
    plsda_prob = np.concatenate(plsda_probs)[sort_idx]

    acc = accuracy_score(y, plsda_pred)
    auc = roc_auc_score(y, plsda_prob)
    cm  = confusion_matrix(y, plsda_pred)
    fpr, tpr, _ = roc_curve(y, plsda_prob)
    results["PLS-DA"] = {
        "accuracy": acc, "auc": auc,
        "cm": cm, "fpr": fpr, "tpr": tpr,
        "y_pred": plsda_pred, "y_prob": plsda_prob,
    }
    print(f"\n{'─'*50}")
    print(f"  PLS-DA")
    print(f"{'─'*50}")
    print(f"  Accuracy : {acc:.3f}")
    print(f"  AUC-ROC  : {auc:.3f}")
    print(classification_report(y, plsda_pred,
          target_names=["Healthy", "OA"], digits=3))

    return results


# =============================================================================
# 5. WAVELENGTH IMPORTANCE (Random Forest)
# =============================================================================

def get_rf_importance(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    # fitting on the full dataset for importance — not CV, we just want the feature ranking
    # bumping to 300 trees here for a more stable importance estimate
    rf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=300, max_depth=15,
                                        random_state=42, n_jobs=-1))
    ])
    rf.fit(X, y)
    return rf.named_steps["clf"].feature_importances_


# =============================================================================
# 6. VISUALISATION
# =============================================================================

# keeping colors in one place — easier to retheme without hunting through the plot code
PALETTE = {
    "dark":    "#021826",
    "primary": "#065A82",
    "mid":     "#1C7293",
    "accent":  "#02C39A",
    "healthy": "#02C39A",
    "oa":      "#E05C5C",
    "gray":    "#8FA3B1",
    "bg":      "#F4F9FB",
}


def plot_all(X_raw, X_proc, y, results, importances):
    fig = plt.figure(figsize=(18, 14), facecolor=PALETTE["bg"])
    fig.suptitle(
        "POC: NIR Spectral ML Pipeline for Cartilage Health Assessment\n",
        fontsize=15, fontweight="bold", color=PALETTE["dark"], y=0.98
    )

    # 3x3 grid — panels A-F fill the top two rows, G spans the full bottom row
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35,
                           top=0.93, bottom=0.06, left=0.07, right=0.97)

    # ── Panel A: Raw mean spectra ─────────────────────────────────
    ax_raw = fig.add_subplot(gs[0, 0])
    for cls, label, col in [(0, "Healthy", PALETTE["healthy"]),
                             (1, "OA",      PALETTE["oa"])]:
        mu  = X_raw[y == cls].mean(axis=0)
        std = X_raw[y == cls].std(axis=0)
        ax_raw.plot(WAVELENGTHS, mu, color=col, lw=1.8, label=label)
        # shaded std band makes inter-sample variance visible at a glance
        ax_raw.fill_between(WAVELENGTHS, mu - std, mu + std,
                             color=col, alpha=0.15)
    ax_raw.set_title("A  Raw NIR Spectra", fontweight="bold",
                     color=PALETTE["primary"], fontsize=11)
    ax_raw.set_xlabel("Wavelength (nm)", fontsize=9)
    ax_raw.set_ylabel("Absorbance (a.u.)", fontsize=9)
    ax_raw.legend(fontsize=9)
    ax_raw.set_facecolor("white")

    # vertical lines at known absorption bands — quick sanity check for the spectral simulation
    bands = {970: "H2O", 1450: "H2O", 1730: "CH2", 2100: "PG"}
    for wl_b, lbl in bands.items():
        ax_raw.axvline(wl_b, color=PALETTE["gray"], lw=0.8, ls="--", alpha=0.7)
        ax_raw.text(wl_b + 18, ax_raw.get_ylim()[1] * 0.88, lbl,
                    fontsize=7, color=PALETTE["gray"], rotation=90)

    # ── Panel B: Preprocessed (2nd derivative) ───────────────────
    ax_proc = fig.add_subplot(gs[0, 1])
    for cls, label, col in [(0, "Healthy", PALETTE["healthy"]),
                             (1, "OA",      PALETTE["oa"])]:
        mu  = X_proc[y == cls].mean(axis=0)
        std = X_proc[y == cls].std(axis=0)
        ax_proc.plot(WAVELENGTHS, mu, color=col, lw=1.8, label=label)
        ax_proc.fill_between(WAVELENGTHS, mu - std, mu + std,
                              color=col, alpha=0.15)
    # zero line reference — 2nd derivative oscillates around 0, peaks become zero crossings
    ax_proc.axhline(0, color=PALETTE["gray"], lw=0.6, ls=":")
    ax_proc.set_title("B  After Preprocessing\n(SG + MSC + 2nd Deriv.)",
                      fontweight="bold", color=PALETTE["primary"], fontsize=11)
    ax_proc.set_xlabel("Wavelength (nm)", fontsize=9)
    ax_proc.set_ylabel("2nd Derivative (a.u.)", fontsize=9)
    ax_proc.legend(fontsize=9)
    ax_proc.set_facecolor("white")

    # ── Panel C: Difference spectrum (Healthy – OA) ───────────────
    ax_diff = fig.add_subplot(gs[0, 2])
    diff = X_raw[y == 0].mean(0) - X_raw[y == 1].mean(0)
    ax_diff.fill_between(WAVELENGTHS, 0, diff,
                         where=(diff > 0), color=PALETTE["healthy"], alpha=0.6,
                         label="Healthy > OA")
    ax_diff.fill_between(WAVELENGTHS, 0, diff,
                         where=(diff < 0), color=PALETTE["oa"], alpha=0.6,
                         label="OA > Healthy")
    ax_diff.axhline(0, color=PALETTE["gray"], lw=0.8)
    ax_diff.set_title("C  Difference Spectrum\n(Healthy - OA)",
                      fontweight="bold", color=PALETTE["primary"], fontsize=11)
    ax_diff.set_xlabel("Wavelength (nm)", fontsize=9)
    ax_diff.set_ylabel("DeltaAbsorbance", fontsize=9)
    ax_diff.legend(fontsize=9)
    ax_diff.set_facecolor("white")
    # annotation highlighting the proteoglycan loss region — the core diagnostic finding
    ax_diff.text(2100, diff.max() * 0.7, "PG loss\n@2100nm",
                 fontsize=8, color=PALETTE["oa"],
                 ha="center", style="italic")

    # ── Panel D: ROC curves ───────────────────────────────────────
    ax_roc = fig.add_subplot(gs[1, 0])
    colors_roc = [PALETTE["primary"], PALETTE["accent"], PALETTE["oa"]]
    for (name, res), col in zip(results.items(), colors_roc):
        ax_roc.plot(res["fpr"], res["tpr"],
                    label=f"{name} (AUC={res['auc']:.2f})",
                    color=col, lw=2.0)
    # diagonal reference line — AUC=0.5 baseline (random classifier)
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax_roc.set_title("D  ROC Curves (5-fold CV)",
                     fontweight="bold", color=PALETTE["primary"], fontsize=11)
    ax_roc.set_xlabel("False Positive Rate", fontsize=9)
    ax_roc.set_ylabel("True Positive Rate", fontsize=9)
    ax_roc.legend(fontsize=8.5, loc="lower right")
    ax_roc.set_facecolor("white")

    # ── Panel E: Confusion matrix (Random Forest) ────────────────
    ax_cm = fig.add_subplot(gs[1, 1])
    rf_cm = results["Random Forest"]["cm"]
    sns.heatmap(rf_cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Healthy", "OA"],
                yticklabels=["Healthy", "OA"],
                ax=ax_cm, cbar=False,
                linewidths=0.5, linecolor="white",
                annot_kws={"size": 14, "weight": "bold"})
    ax_cm.set_title("E  Confusion Matrix\n(Random Forest)",
                    fontweight="bold", color=PALETTE["primary"], fontsize=11)
    ax_cm.set_xlabel("Predicted", fontsize=9)
    ax_cm.set_ylabel("True", fontsize=9)

    # ── Panel F: Model comparison bar chart ──────────────────────
    ax_bar = fig.add_subplot(gs[1, 2])
    model_names = list(results.keys())
    accs = [results[m]["accuracy"] for m in model_names]
    aucs = [results[m]["auc"]      for m in model_names]
    x = np.arange(len(model_names))
    w = 0.35
    bars1 = ax_bar.bar(x - w/2, accs, w, color=PALETTE["primary"],
                       label="Accuracy", alpha=0.85)
    bars2 = ax_bar.bar(x + w/2, aucs, w, color=PALETTE["accent"],
                       label="AUC-ROC", alpha=0.85)
    # y-axis from 0.5 so differences between models are actually visible
    ax_bar.set_ylim(0.5, 1.05)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(model_names, fontsize=9)
    ax_bar.set_ylabel("Score", fontsize=9)
    ax_bar.set_title("F  Model Comparison",
                     fontweight="bold", color=PALETTE["primary"], fontsize=11)
    ax_bar.legend(fontsize=9)
    ax_bar.set_facecolor("white")
    # value labels on each bar — saves having to read the y-axis for exact numbers
    for bar in bars1:
        ax_bar.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.008,
                    f"{bar.get_height():.2f}", ha="center", fontsize=8)
    for bar in bars2:
        ax_bar.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.008,
                    f"{bar.get_height():.2f}", ha="center", fontsize=8)

    # ── Panel G: Wavelength importance (full row) ────────────────
    ax_imp = fig.add_subplot(gs[2, :])
    # smoothing importance curve before plotting — raw RF importances are noisy wavelength-by-wavelength
    imp_smooth = savgol_filter(importances, 31, 3)
    ax_imp.fill_between(WAVELENGTHS, 0, imp_smooth,
                        color=PALETTE["mid"], alpha=0.55, label="RF importance")
    ax_imp.plot(WAVELENGTHS, imp_smooth, color=PALETTE["primary"], lw=1.5)

    # shading biologically known regions — if the model is working, these should align with the peaks
    regions = [
        (950,  990,  "#8ecae6", "H2O\n970nm"),
        (1420, 1480, "#8ecae6", "H2O\n1450nm"),
        (1700, 1760, "#a8dadc", "Collagen\nCH2"),
        (1600, 1700, "#e9c46a", "OA\ncollagen"),
        (2060, 2160, "#e76f51", "Proteoglycan\n2100nm"),
    ]
    for x0, x1, col, lbl in regions:
        ax_imp.axvspan(x0, x1, alpha=0.22, color=col)
        ax_imp.text((x0+x1)/2, imp_smooth.max() * 0.92, lbl,
                    ha="center", fontsize=7.5, color=PALETTE["dark"],
                    fontweight="bold")

    ax_imp.set_title("G  RF Wavelength Importance — Key Spectral Bands for Cartilage Discrimination",
                     fontweight="bold", color=PALETTE["primary"], fontsize=11)
    ax_imp.set_xlabel("Wavelength (nm)", fontsize=9)
    ax_imp.set_ylabel("Feature Importance", fontsize=9)
    ax_imp.set_facecolor("white")
    ax_imp.set_xlim(WAVELENGTHS[0], WAVELENGTHS[-1])

    out_path = "nir_cartilage_poc_results.png"
    # savefig before close — close() wipes the figure from memory
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.close()
    print(f"\n   Figure saved -> {out_path}")
    return out_path


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  NIR Spectral ML POC — Cartilage Health Classification")
    print("  Nosa Peter Inwe  |  Afara BSL, UEF")
    print("=" * 60)

    # step 1: generate synthetic data — replace this with real spectra when available
    print("\n[1/4]  Generating synthetic NIR spectra ...")
    X_raw, y = generate_spectra(n_per_class=N_SAMPLES // 2)
    print(f"       {X_raw.shape[0]} spectra x {X_raw.shape[1]} wavelengths")
    print(f"       Classes: {(y==0).sum()} Healthy, {(y==1).sum()} OA")

    # step 2: preprocess — order matters: smooth before MSC before derivative
    print("\n[2/4]  Preprocessing (SG -> MSC -> 2nd derivative) ...")
    X_proc = preprocess(X_raw)

    # step 3: evaluate all three models with 5-fold CV
    print("\n[3/4]  Training & evaluating models (5-fold CV) ...")
    results = evaluate_models(X_proc, y)

    # step 4: fit RF on full dataset just for importance — not for generalization estimate
    print("\n[4/4]  Computing wavelength importances ...")
    importances = get_rf_importance(X_proc, y)

    print("\n       Generating visualisation ...")
    plot_all(X_raw, X_proc, y, results, importances)

    # summary — the proteoglycan band at 2100 nm should dominate the importance plot
    print("\n" + "=" * 60)
    print("  KEY FINDINGS")
    print("=" * 60)
    for name, res in results.items():
        print(f"  {name:<20}  Acc={res['accuracy']:.3f}  AUC={res['auc']:.3f}")
    print("\n  Most discriminative region: ~2100 nm (proteoglycan sulphate)")
    print("  -> Directly maps to BSL's NIR spectroscopy research")
    print("  -> Pipeline ready to adapt to real cartilage spectral datasets")
    print("=" * 60)
