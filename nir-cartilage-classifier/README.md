# NIR Spectral Classifier for Cartilage Health Assessment

> Proof-of-concept ML pipeline for classifying Near-Infrared (NIR) spectra into **Healthy vs. Osteoarthritic (OA)** cartilage tissue - built in the context of Prof. Isaac Afara's [Biomedical Spectroscopy Laboratory (BSL)](https://uefconnect.uef.fi/en/group/biomedical-spectroscopy-laboratory/) research at the University of Eastern Finland (UEF).

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](../LICENSE)

---

## Background

Osteoarthritis (OA) is a degenerative joint disease characterised by the breakdown of articular cartilage. A key early marker is the loss of **proteoglycans (PG)** - the molecules that give cartilage its compressive strength. This loss produces a measurable change in NIR absorbance around **~2100 nm**, well before structural damage is visible on imaging.

NIR spectroscopy can detect these biochemical changes non-destructively, making it a promising tool for early OA diagnosis. This project demonstrates the full ML pipeline that would sit on top of such a spectroscopy system.

---

## What This Project Does

```
Synthetic NIR spectra (500–2500 nm)
        ↓
Spectral preprocessing  (SG smoothing → MSC → 2nd derivative)
        ↓
Model training & comparison  (Random Forest | SVM-RBF | PLS-DA)
        ↓
5-fold cross-validated evaluation  (Accuracy, AUC-ROC, Confusion Matrix)
        ↓
Wavelength importance plot  (which bands drive classification)
```

The most discriminative spectral region identified is **~2100 nm (proteoglycan sulphate)**, which aligns directly with the known biochemistry of OA - a good sign that the pipeline is learning the right signal.

---

## Output

Running the script produces a single figure (`nir_cartilage_poc_results.png`) with 7 panels:

| Panel | Content |
|---|---|
| A | Raw mean NIR spectra (Healthy vs OA) with ±1 SD band |
| B | Preprocessed spectra after SG + MSC + 2nd derivative |
| C | Difference spectrum (Healthy − OA), highlighting PG loss at 2100 nm |
| D | ROC curves for all three models (5-fold CV) |
| E | Confusion matrix (Random Forest) |
| F | Model comparison bar chart (Accuracy + AUC-ROC) |
| G | Random Forest wavelength importance across the full NIR range |

---

## Models

Three classifiers are benchmarked under identical 5-fold stratified cross-validation:

**Random Forest** - ensemble of decision trees; handles high-dimensional spectral data well and provides feature (wavelength) importance scores natively.

**SVM with RBF kernel** - finds the maximum-margin hyperplane in a high-dimensional feature space; well-suited to spectral data with many correlated features.

**PLS-DA** - Partial Least Squares Discriminant Analysis; the standard chemometrics baseline for spectral classification. scikit-learn doesn't ship PLS-DA natively, so it is implemented here as a wrapper around `PLSRegression` with a learned threshold.

---

## Preprocessing Pipeline

Order matters here - each step builds on the previous one:

1. **Savitzky-Golay smoothing** (window=11, poly=3) - reduces high-frequency noise without distorting peak shapes
2. **Multiplicative Scatter Correction (MSC)** - removes inter-sample baseline offsets caused by light scattering differences in tissue
3. **2nd derivative** (SG, window=15, poly=3) - sharpens absorption peaks and removes any remaining broad baseline drift

---

## Synthetic Data

Real cartilage NIR spectra are not publicly available. The spectra here are **physically motivated synthetic data** - each absorption peak corresponds to a real molecular vibration band in tissue:

| Wavelength (nm) | Assignment | Healthy → OA change |
|---|---|---|
| ~970 | Water overtone | Slight broadening |
| ~1450 | Water 1st overtone | Slight increase |
| ~1680 | Collagen degradation | New shoulder in OA |
| ~1730 | Collagen CH₂ | Reduced in OA |
| ~2100 | Proteoglycan sulphate | **Strong reduction** - key diagnostic marker |
| ~2300 | Collagen CH₂ combination | Slight reduction |

The synthetic approach makes the pipeline reproducible and shareable without requiring proprietary datasets. The code is structured so that `generate_spectra()` can be replaced with a real data loader with no changes elsewhere.

---

## Installation & Usage

```bash
# Clone the repo
git clone https://github.com/<your-username>/biomedical-spectroscopy-projects.git
cd biomedical-spectroscopy-projects/nir-cartilage-classifier

# Install dependencies (no GPU needed - CPU only)
pip install -r requirements.txt

# Run
python nir_cartilage_poc.py
```

Expected output:

```
============================================================
  NIR Spectral ML POC - Cartilage Health Classification
  Nosa Peter Inwe  |  Afara BSL, UEF
============================================================

[1/4]  Generating synthetic NIR spectra ...
       300 spectra x 1001 wavelengths
       Classes: 150 Healthy, 150 OA

[2/4]  Preprocessing (SG -> MSC -> 2nd derivative) ...

[3/4]  Training & evaluating models (5-fold CV) ...
  ...

[4/4]  Computing wavelength importances ...
   Figure saved -> nir_cartilage_poc_results.png

============================================================
  KEY FINDINGS
============================================================
  Random Forest         Acc=0.xxx  AUC=0.xxx
  SVM (RBF)             Acc=0.xxx  AUC=0.xxx
  PLS-DA                Acc=0.xxx  AUC=0.xxx

  Most discriminative region: ~2100 nm (proteoglycan sulphate)
  -> Directly maps to BSL's NIR spectroscopy research
  -> Pipeline ready to adapt to real cartilage spectral datasets
============================================================
```

---

## Adapting to Real Data

To use this pipeline with real NIR spectra, replace the `generate_spectra()` call in `__main__` with your own loader:

```python
# Example: load from CSV (rows = samples, columns = wavelengths)
import pandas as pd

df = pd.read_csv("your_spectra.csv")
X_raw = df.drop(columns=["label"]).values   # (N, 1001)
y     = df["label"].values                  # 0 = Healthy, 1 = OA
```

Everything downstream (preprocessing, model training, evaluation, plots) works unchanged.

---

## Key Learnings

**Preprocessing order matters.** Applying the 2nd derivative before MSC amplifies scatter noise into the derivative signal. The correct order - smooth, then scatter-correct, then differentiate - produces much cleaner features.

**PLS-DA is not in scikit-learn.** The standard workaround is to wrap `PLSRegression` with a threshold on the continuous prediction. Using the median of training scores as the threshold is robust to mild class imbalance and avoids hardcoding a `0.5` cutoff that may not be appropriate for non-symmetric class distributions.

**The 2100 nm band tells most of the story.** All three models assign their highest importance to this region - which is exactly what the biology predicts. When your model's feature importance aligns with domain knowledge, that's a good sign it's learning something real.

---

## Acknowledgements

| Resource | Role |
|---|---|
| [Prof. Isaac Afara - BSL, UEF](https://uefconnect.uef.fi/en/group/biomedical-spectroscopy-laboratory/) | Research context and domain framing |
| [Afara et al. (2020)](https://doi.org/10.1038/s41598-020-73040-0) *Scientific Reports* | NIR spectroscopy for articular cartilage assessment |
| [Sarin et al. (2021)](https://doi.org/10.1016/j.joca.2021.02.004) | Compositional mapping of cartilage via NIR |
| [Rinnan et al. (2009)](https://doi.org/10.1016/j.trac.2009.07.007) | Pre-processing of NIR spectra - review of MSC and derivatives |
| [scikit-learn](https://scikit-learn.org) | RF, SVM, PLSRegression, cross-validation framework |
| [scipy.signal](https://docs.scipy.org/doc/scipy/reference/signal.html) | Savitzky-Golay filter implementation |

> This is an independent learning project. It is not affiliated with UEF, the BSL, or any of the cited authors.

---

## License

MIT - see [LICENSE](../LICENSE).
