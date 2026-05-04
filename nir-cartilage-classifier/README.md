
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

## Mathematical Background

The pipeline treats each NIR spectrum as a high-dimensional signal. Each sample is represented as an absorbance vector measured across the wavelength range:

$$
\mathbf{x}_i = \left[ A(\lambda_1), A(\lambda_2), \ldots, A(\lambda_p) \right]
$$

where:

| Symbol | Meaning |
|---|---|
| $\mathbf{x}_i$ | Spectrum for sample $i$ |
| $A(\lambda_j)$ | Absorbance at wavelength $\lambda_j$ |
| $p$ | Number of wavelength points |
| $y_i$ | Class label: $0$ = Healthy, $1$ = OA |

The full dataset is therefore:

$$
\mathbf{X} \in \mathbb{R}^{n \times p},\qquad \mathbf{y} \in \{0,1\}^{n}
$$

where $n$ is the number of spectra and $p$ is the number of wavelength variables.

---

### 1. Absorbance Model

NIR spectroscopy measures how strongly tissue interacts with light at different wavelengths. Absorbance is commonly expressed as:

$$
A(\lambda) = \log_{10} \left( \frac{I_0(\lambda)}{I(\lambda)} \right)
$$

where:

| Symbol | Meaning |
|---|---|
| $I_0(\lambda)$ | Incident light intensity |
| $I(\lambda)$ | Measured transmitted or reflected intensity |
| $A(\lambda)$ | Absorbance at wavelength $\lambda$ |

The classifier does not see cartilage directly. It sees numerical absorbance patterns. The goal is to learn a mapping:

$f: \mathbf{x}_i \mapsto y_i$

so that biochemical changes in cartilage are detected from spectral changes.

In this project, the most important region is around **2100 nm**, where proteoglycan-related absorption changes are expected.

---

### 2. Savitzky-Golay Smoothing

Raw spectra contain noise. Savitzky-Golay smoothing reduces this noise while preserving peak shape. For each local wavelength window, the spectrum is approximated by a polynomial:

$$
A(\lambda) \approx a_0 + a_1\lambda + a_2\lambda^2 + \cdots + a_k\lambda^k
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
\mathbf{x}_i = a_i + b_i\mathbf{x}_{ref} + \mathbf{e}_i
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
\mathbf{x}_{i,MSC} = \frac{\mathbf{x}_i - a_i}{b_i}
$$

MSC reduces variation that comes from scattering rather than chemistry. This helps the classifier focus on molecular differences between healthy and OA-like cartilage.

---

### 4. Second Derivative Spectroscopy

The second derivative enhances subtle spectral features and suppresses broad baseline drift:

$$
A''(\lambda) = \frac{d^2 A(\lambda)}{d\lambda^2}
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
\Delta A(\lambda) = \overline{A}_{Healthy}(\lambda) - \overline{A}_{OA}(\lambda)
$$

where:

$$
\overline{A}_{Healthy}(\lambda) = \frac{1}{n_H} \sum_{i=1}^{n_H} A_i(\lambda)
$$

and:

$$
\overline{A}_{OA}(\lambda) = \frac{1}{n_{OA}} \sum_{i=1}^{n_{OA}} A_i(\lambda)
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
\hat{y} = \operatorname{mode} \left( h_1(\mathbf{x}), h_2(\mathbf{x}), \ldots, h_T(\mathbf{x}) \right)
$$

where $T$ is the number of trees.

Random Forests are useful here because they can model non-linear relationships between wavelength regions and class labels.

Feature importance is estimated from the impurity reduction caused by each wavelength. For classification, impurity is often measured using Gini impurity:

$$
Gini = 1 - \sum_{c=1}^{C} p_c^2
$$

where $p_c$ is the proportion of samples from class $c$ at a node.

A wavelength receives high importance if it repeatedly helps split healthy and OA-like spectra across many trees.

---

### Support Vector Machine with RBF Kernel

An SVM tries to find a decision boundary with the largest margin between the two classes.

For a linear SVM, the decision function is:

$$
f(\mathbf{x}) = \mathbf{w}^{T}\mathbf{x} + b
$$

The predicted class depends on the sign of $f(\mathbf{x})$.

Because spectral data is often not linearly separable, this project uses the radial basis function kernel:

$$
K(\mathbf{x}_i, \mathbf{x}_j) = \exp\left( -\gamma \lVert \mathbf{x}_i - \mathbf{x}_j \rVert^2 \right)
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
\mathbf{X} = \mathbf{T}\mathbf{P}^{T} + \mathbf{E}
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
\mathbf{y} = \mathbf{T}\mathbf{q} + \mathbf{f}
$$

where $\mathbf{q}$ contains regression weights and $\mathbf{f}$ is the residual error.

Since scikit-learn provides `PLSRegression` rather than a native PLS-DA classifier, the model first produces a continuous score:

$$
\hat{y}_{score} \in \mathbb{R}
$$

The score is converted into a class label using a learned threshold $\tau$:

$$
\hat{y} = \begin{cases}
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
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
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
TPR = \frac{TP}{TP + FN}
$$

The false positive rate is:

$$
FPR = \frac{FP}{FP + TN}
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
