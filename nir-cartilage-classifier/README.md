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

---

## Mathematical Background

The pipeline treats each NIR spectrum as a high-dimensional signal. Each sample is represented as an absorbance vector measured across the wavelength range:

$$
\mathbf{x}_i =
\left[
A(\lambda_1), A(\lambda_2), \ldots, A(\lambda_p)
\right]
$$

where:

| Symbol | Meaning |
|---|---|
| $\mathbf{x}_i$ | Spectrum for sample $i$ |
| $A(\lambda_j)$ | Absorbance at wavelength $\lambda_j$ |
| $p$ | Number of wavelength points |
| $y_i$ | Class label: $0 =$ Healthy, $1 =$ OA |

The full dataset is therefore:

$$
\mathbf{X} \in \mathbb{R}^{n \times p},
\qquad
\mathbf{y} \in \{0,1\}^{n}
$$

where $n$ is the number of spectra and $p$ is the number of wavelength variables.

---

### 1. Absorbance Model

NIR spectroscopy measures how strongly tissue interacts with light at different wavelengths. Absorbance is commonly expressed as:

$$
A(\lambda) =
\log_{10}
\left(
\frac{I_0(\lambda)}{I(\lambda)}
\right)
$$

where:

| Symbol | Meaning |
|---|---|
| $I_0(\lambda)$ | Incident light intensity |
| $I(\lambda)$ | Measured transmitted or reflected intensity |
| $A(\lambda)$ | Absorbance at wavelength $\lambda$ |

The classifier does not see cartilage directly. It sees numerical absorbance patterns. The goal is to learn a mapping:

$$
f: \mathbf{x}_i \mapsto y_i
$$

so that biochemical changes in cartilage are detected from spectral changes.

In this project, the most important region is around **2100 nm**, where proteoglycan-related absorption changes are expected.

---

### 2. Savitzky-Golay Smoothing

Raw spectra contain noise. Savitzky-Golay smoothing reduces this noise while preserving peak shape. For each local wavelength window, the spectrum is approximated by a polynomial:

$$
A(\lambda)
\approx
a_0 + a_1\lambda + a_2\lambda^2 + \cdots + a_k\lambda^k
$$

where $k$ is the polynomial order.

In this project:

| Parameter | Value |
|---|---|
| Window length | 11 |
| Polynomial order | 3 |

So each local region is fitted with a cubic polynomial. This is better than a simple moving average because it smooths the signal without flattening chemically meaningful absorption peaks.

---

### 3. Multiplicative Scatter Correction

Biological tissue spectra are affected by scattering, sample thickness, surface roughness, and measurement geometry. These effects can shift or scale the whole spectrum.

MSC models each measured spectrum as:

$$
\mathbf{x}_i
=
a_i
+
b_i\mathbf{x}_{ref}
+
\mathbf{e}_i
$$

where:

| Symbol | Meaning |
|---|---|
| $\mathbf{x}_i$ | Measured spectrum |
| $\mathbf{x}_{ref}$ | Reference spectrum, usually the mean spectrum |
| $a_i$ | Additive offset |
| $b_i$ | Multiplicative scaling coefficient |
| $\mathbf{e}_i$ | Residual error |

The corrected spectrum is:

$$
\mathbf{x}_{i,MSC}
=
\frac{\mathbf{x}_i - a_i}{b_i}
$$

MSC reduces variation that comes from scattering rather than chemistry. This helps the classifier focus on molecular differences between healthy and OA-like cartilage.

---

### 4. Second Derivative Spectroscopy

The second derivative enhances subtle spectral features and suppresses broad baseline drift:

$$
A''(\lambda)
=
\frac{d^2 A(\lambda)}{d\lambda^2}
$$

For discrete spectra, this derivative is estimated numerically using a Savitzky-Golay derivative filter.

In this project:

| Parameter | Value |
|---|---|
| Derivative order | 2 |
| Window length | 15 |
| Polynomial order | 3 |

The second derivative is useful because broad baseline effects change slowly with wavelength, while absorption bands change more sharply. This makes hidden or overlapping peaks easier to detect.

---

### 5. Difference Spectrum

The difference spectrum shows which wavelengths separate the two classes most strongly:

$$
\Delta A(\lambda)
=
\overline{A}_{Healthy}(\lambda)
-
\overline{A}_{OA}(\lambda)
$$

where:

$$
\overline{A}_{Healthy}(\lambda)
=
\frac{1}{n_H}
\sum_{i=1}^{n_H}
A_i(\lambda)
$$

and:

$$
\overline{A}_{OA}(\lambda)
=
\frac{1}{n_{OA}}
\sum_{i=1}^{n_{OA}}
A_i(\lambda)
$$

A large value of $\Delta A(\lambda)$ indicates that the wavelength carries useful class-separating information.

In this project, the strongest meaningful difference appears near **2100 nm**, matching the expected reduction in proteoglycan-related absorption for OA-like cartilage.

---

## Classification Mathematics

Three classifiers are compared because they represent different ways of learning the boundary between healthy and OA-like spectra.

---

### Random Forest

A Random Forest combines many decision trees. Each tree produces a class prediction:

$$
h_t(\mathbf{x})
$$

The final prediction is obtained by majority vote:

$$
\hat{y}
=
\operatorname{mode}
\left(
h_1(\mathbf{x}),
h_2(\mathbf{x}),
\ldots,
h_T(\mathbf{x})
\right)
$$

where $T$ is the number of trees.

Random Forests are useful here because they can model non-linear relationships between wavelength regions and class labels.

Feature importance is estimated from the impurity reduction caused by each wavelength. For classification, impurity is often measured using Gini impurity:

$$
Gini
=
1
-
\sum_{c=1}^{C}
p_c^2
$$

where $p_c$ is the proportion of samples from class $c$ at a node.

A wavelength receives high importance if it repeatedly helps split healthy and OA-like spectra across many trees.

---

### Support Vector Machine with RBF Kernel

An SVM tries to find a decision boundary with the largest margin between the two classes.

For a linear SVM, the decision function is:

$$
f(\mathbf{x})
=
\mathbf{w}^{T}\mathbf{x}
+
b
$$

The predicted class depends on the sign of $f(\mathbf{x})$.

Because spectral data is often not linearly separable, this project uses the radial basis function kernel:

$$
K(\mathbf{x}_i, \mathbf{x}_j)
=
\exp
\left(
-\gamma
\lVert
\mathbf{x}_i - \mathbf{x}_j
\rVert^2
\right)
$$

where:

| Symbol | Meaning |
|---|---|
| $K(\mathbf{x}_i, \mathbf{x}_j)$ | Similarity between two spectra |
| $\gamma$ | Controls the width of the RBF kernel |
| $\lVert \mathbf{x}_i - \mathbf{x}_j \rVert^2$ | Squared Euclidean distance between spectra |

The RBF kernel allows the SVM to learn non-linear class boundaries from subtle spectral differences.

---

### PLS-DA

Partial Least Squares Discriminant Analysis is a standard chemometric method for spectral classification.

Spectral data usually has many highly correlated wavelength variables. PLS handles this by projecting the original spectra into a smaller latent-variable space:

$$
\mathbf{X}
=
\mathbf{T}\mathbf{P}^{T}
+
\mathbf{E}
$$

where:

| Symbol | Meaning |
|---|---|
| $\mathbf{X}$ | Spectral data matrix |
| $\mathbf{T}$ | Latent score matrix |
| $\mathbf{P}$ | Loading matrix |
| $\mathbf{E}$ | Residual matrix |

The target vector is modelled as:

$$
\mathbf{y}
=
\mathbf{T}\mathbf{q}
+
\mathbf{f}
$$

where $\mathbf{q}$ contains regression weights and $\mathbf{f}$ is the residual error.

Since scikit-learn provides `PLSRegression` rather than a native PLS-DA classifier, the model first produces a continuous score:

$$
\hat{y}_{score} \in \mathbb{R}
$$

The score is converted into a class label using a learned threshold $\tau$:

$$
\hat{y}
=
\begin{cases}
1, & \hat{y}_{score} \geq \tau \\
0, & \hat{y}_{score} < \tau
\end{cases}
$$

This gives a practical PLS-DA implementation using the tools available in scikit-learn.

---

## Evaluation Mathematics

The models are evaluated using stratified 5-fold cross-validation. The dataset is split into five folds while preserving the healthy/OA class ratio.

For each fold:

1. train on four folds
2. test on the remaining fold
3. repeat until each fold has been used once for testing
4. average the final scores

This gives a more stable estimate than a single train-test split.

---

### Accuracy

Accuracy measures the proportion of correct predictions:

$$
Accuracy
=
\frac{TP + TN}
{TP + TN + FP + FN}
$$

where:

| Term | Meaning |
|---|---|
| $TP$ | OA correctly classified as OA |
| $TN$ | Healthy correctly classified as healthy |
| $FP$ | Healthy incorrectly classified as OA |
| $FN$ | OA incorrectly classified as healthy |

Accuracy is useful, but it can hide important errors. In medical-style classification, false negatives are especially important because they represent OA-like samples being classified as healthy.

---

### Confusion Matrix

The confusion matrix is:

$$
\begin{bmatrix}
TN & FP \\
FN & TP
\end{bmatrix}
$$

It shows not only how many predictions were correct, but also what type of mistakes the model made.

---

### ROC Curve and AUC

The ROC curve plots the true positive rate against the false positive rate across different thresholds.

The true positive rate is:

$$
TPR
=
\frac{TP}{TP + FN}
$$

The false positive rate is:

$$
FPR
=
\frac{FP}{FP + TN}
$$

The AUC-ROC measures the area under the ROC curve:

| AUC value | Interpretation |
|---|---|
| $1.0$ | Perfect class separation |
| $0.5$ | Random guessing |
| $< 0.5$ | Worse than random ranking |

A high AUC means the model consistently ranks OA-like spectra above healthy spectra, even before choosing a final classification threshold.

---

## Why the Mathematics Matters

The mathematical pipeline is designed to make the final classification chemically meaningful:

```text
Raw absorbance vector
        ↓
Noise reduction
        ↓
Scatter correction
        ↓
Derivative-based feature enhancement
        ↓
Classification
        ↓
Wavelength importance analysis

'''



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
