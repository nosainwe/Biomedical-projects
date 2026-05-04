# NIR Spectral Classifier for Cartilage Health Assessment


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

---

## Mathematical Background

This project is not just a machine learning demo. The pipeline is built around the mathematics of spectral signal processing, chemometrics, and supervised classification.

Each NIR spectrum is treated as a numerical vector:

\[
x_i = [A(\lambda_1), A(\lambda_2), ..., A(\lambda_p)]
\]

where:

- \(x_i\) is the spectrum for sample \(i\)
- \(A(\lambda)\) is the absorbance at wavelength \(\lambda\)
- \(p\) is the number of wavelength points, here 1001
- the target label is \(y_i = 0\) for healthy cartilage and \(y_i = 1\) for OA cartilage

So the dataset can be written as:

\[
X \in \mathbb{R}^{n \times p}, \quad y \in \{0,1\}^{n}
\]

where \(n\) is the number of spectra and \(p\) is the number of wavelength variables.

---

### Absorbance and Spectral Features

NIR spectroscopy is based on how tissue absorbs light at different wavelengths. In simple form, absorbance is defined as:

\[
A = \log_{10}\left(\frac{I_0}{I}\right)
\]

where:

- \(I_0\) is the incident light intensity
- \(I\) is the transmitted or reflected intensity measured by the detector

Higher absorbance at a wavelength means the tissue is interacting more strongly with light at that wavelength. In cartilage, different molecules produce absorption features at different regions of the NIR spectrum.

The important idea is that the machine learning model does not see cartilage directly. It sees a high-dimensional absorbance vector and learns which wavelength patterns separate healthy cartilage from OA-like cartilage.

---

### Savitzky-Golay Smoothing

The Savitzky-Golay filter reduces high-frequency noise while preserving the shape of spectral peaks. Instead of simply averaging neighbouring points, it fits a local polynomial to a small moving window.

For a local wavelength window, the signal is approximated by:

\[
A(\lambda) \approx a_0 + a_1\lambda + a_2\lambda^2 + ... + a_k\lambda^k
\]

where \(k\) is the polynomial order.

In this project:

- smoothing window = 11
- polynomial order = 3

This means each point is replaced using a cubic polynomial fitted to nearby wavelength values. This is useful because NIR peaks are broad and smooth, so preserving peak shape matters more than preserving random noise.

---

### Multiplicative Scatter Correction

Cartilage spectra can vary because of tissue scattering, surface roughness, sample thickness, and measurement geometry. These effects can shift or scale the whole spectrum even when the underlying chemistry is similar.

MSC assumes that each measured spectrum \(x_i\) can be approximated as a scaled and shifted version of a reference spectrum \(x_{ref}\):

\[
x_i = a_i + b_i x_{ref} + e_i
\]

where:

- \(a_i\) is the additive offset
- \(b_i\) is the multiplicative scaling factor
- \(e_i\) is the residual error
- \(x_{ref}\) is usually the mean spectrum of the dataset

The corrected spectrum is then:

\[
x_{i,MSC} = \frac{x_i - a_i}{b_i}
\]

This makes the spectra more comparable by reducing non-chemical variation. In practical terms, MSC tries to make the model focus on biochemical differences instead of measurement artefacts.

---

### Second Derivative Spectroscopy

The second derivative is used to sharpen overlapping absorption bands and remove broad baseline drift.

For a spectrum \(A(\lambda)\), the second derivative is:

\[
\frac{d^2A}{d\lambda^2}
\]

In discrete spectral data, this is estimated numerically using the Savitzky-Golay derivative filter.

The second derivative helps because broad baseline effects change slowly with wavelength, while real absorption peaks change more rapidly. After differentiation, subtle peaks around chemically meaningful wavelengths become easier for the model to detect.

This is especially useful near the **~2100 nm** region, where proteoglycan-related absorption changes may be small but diagnostically important.

---

### Difference Spectrum

The difference spectrum compares the average healthy spectrum with the average OA spectrum:

\[
\Delta A(\lambda) = \bar{A}_{healthy}(\lambda) - \bar{A}_{OA}(\lambda)
\]

where:

\[
\bar{A}_{healthy}(\lambda) = \frac{1}{n_h}\sum_{i=1}^{n_h} A_i(\lambda)
\]

and:

\[
\bar{A}_{OA}(\lambda) = \frac{1}{n_o}\sum_{i=1}^{n_o} A_i(\lambda)
\]

A large value of \(\Delta A(\lambda)\) means that wavelength carries useful class-separating information.

In this project, the largest biologically meaningful difference appears around **~2100 nm**, which is associated with proteoglycan sulphate. This supports the idea that the classifier is not only separating two artificial classes, but is doing so using a chemically meaningful spectral region.

---

## Classification Mathematics

The task is binary classification:

\[
f(x_i) \rightarrow y_i
\]

where the model learns a function \(f\) that maps each preprocessed spectrum to either:

- \(0\): Healthy
- \(1\): OA

Three different classifiers are used because they represent three different mathematical approaches to classification.

---

### Random Forest

A Random Forest is an ensemble of decision trees. Each tree splits the spectral feature space into regions based on wavelength values.

A single decision tree makes a prediction:

\[
h_t(x)
\]

The full Random Forest prediction is based on the average or majority vote of many trees:

\[
\hat{y} = \text{majority vote}\{h_1(x), h_2(x), ..., h_T(x)\}
\]

where \(T\) is the number of trees.

Random Forests are useful for spectral data because they can handle:

- many wavelength variables
- non-linear relationships
- interactions between spectral bands
- noisy features

The wavelength importance is estimated by measuring how much each wavelength reduces impurity across the trees. A wavelength is more important if splits using that wavelength strongly improve class separation.

---

### Support Vector Machine with RBF Kernel

The SVM tries to find a decision boundary that separates healthy and OA spectra with the largest possible margin.

For a linear SVM, the decision function is:

\[
f(x) = w^Tx + b
\]

The model predicts one class or the other depending on the sign of \(f(x)\).

However, spectral data is often not linearly separable. The RBF kernel solves this by comparing samples in a non-linear feature space:

\[
K(x_i, x_j) = \exp(-\gamma ||x_i - x_j||^2)
\]

where:

- \(K(x_i, x_j)\) measures similarity between two spectra
- \(\gamma\) controls how local or broad the similarity function is
- \(||x_i - x_j||^2\) is the squared distance between spectra

The RBF kernel is useful when the difference between healthy and OA spectra is subtle and spread across several wavelength regions.

---

### PLS-DA

PLS-DA is widely used in chemometrics because spectra usually have many highly correlated variables. Nearby wavelengths often carry similar information, so ordinary regression or classification methods can struggle.

PLS first projects the spectral matrix \(X\) into a smaller set of latent variables:

\[
X = TP^T + E
\]

where:

- \(T\) contains the latent scores
- \(P\) contains the loadings
- \(E\) is the residual matrix

The response variable is also modelled:

\[
y = Tq + f
\]

PLS-DA uses these latent variables for classification. In this project, `PLSRegression` produces a continuous prediction score:

\[
\hat{y}_{score} \in \mathbb{R}
\]

A threshold is then applied:

\[
\hat{y} =
\begin{cases}
1, & \hat{y}_{score} \geq \tau \\
0, & \hat{y}_{score} < \tau
\end{cases}
\]

where \(\tau\) is the learned threshold.

This is why PLS-DA is implemented as a wrapper around `PLSRegression`. The regression output is converted into a class label.

---

## Evaluation Metrics

The models are evaluated with stratified 5-fold cross-validation. This means the dataset is split into five parts while keeping the healthy/OA class balance similar in each fold.

For each fold:

1. the model trains on 80% of the data
2. the model tests on the remaining 20%
3. the process repeats five times
4. the final score is averaged across folds

This gives a more reliable estimate than a single train-test split.

---

### Accuracy

Accuracy measures the proportion of correct predictions:

\[
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
\]

where:

- \(TP\): OA correctly classified as OA
- \(TN\): healthy correctly classified as healthy
- \(FP\): healthy incorrectly classified as OA
- \(FN\): OA incorrectly classified as healthy

Accuracy is easy to understand, but it can be misleading if the dataset is imbalanced. That is why AUC-ROC is also used.

---

### Confusion Matrix

The confusion matrix shows the types of correct and incorrect predictions:

\[
\begin{bmatrix}
TN & FP \\
FN & TP
\end{bmatrix}
\]

For a medical screening-style problem, false negatives are especially important. A false negative means an OA-like sample is classified as healthy, which would be more serious than a false positive in many diagnostic contexts.

---

### ROC Curve and AUC

The ROC curve compares:

\[
True Positive Rate = \frac{TP}{TP + FN}
\]

against:

\[
False Positive Rate = \frac{FP}{FP + TN}
\]

at different classification thresholds.

The AUC-ROC measures the area under this curve. A perfect classifier has:

\[
AUC = 1.0
\]

A random classifier has:

\[
AUC \approx 0.5
\]

A high AUC means the model ranks OA samples higher than healthy samples consistently, even before choosing a final threshold.

---

## Wavelength Importance

For the Random Forest model, each wavelength receives an importance score. Mathematically, this is based on how much a wavelength reduces node impurity across all trees.

For classification, impurity is often measured using the Gini impurity:

\[
Gini = 1 - \sum_{c=1}^{C} p_c^2
\]

where:

- \(C\) is the number of classes
- \(p_c\) is the proportion of samples belonging to class \(c\) at a node

A useful wavelength produces splits that reduce impurity. If the **~2100 nm** wavelength region repeatedly helps separate healthy from OA spectra, it receives a high importance score.

This is important because spectral machine learning should not be treated as a black box. The model should ideally highlight wavelength regions that make chemical and biological sense. In this project, the strong importance around **~2100 nm** agrees with the expected proteoglycan-related changes in cartilage.
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
