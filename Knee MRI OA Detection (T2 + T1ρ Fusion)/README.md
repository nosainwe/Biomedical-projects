# 🧠 Knee MRI OA Detection (T2 + T1ρ Fusion)

**Deep learning-inspired ML pipeline for early osteoarthritis detection using quantitative MRI biomarkers**

---

## 📌 Overview

This project is a **proof-of-concept (POC)** for automated analysis of knee cartilage using **T2 and T1ρ relaxation maps**.

Knee osteoarthritis affects over **500 million people globally**, and current workflows still rely on:
- manual segmentation  
- isolated analysis of T2 and T1ρ  
- coarse compartment-level averages  

This pipeline shows how **joint analysis + machine learning** can extract earlier and richer signals of cartilage degeneration.

---

## 🧪 What This Project Does

The pipeline simulates a full clinical workflow:

### 1. Synthetic MRI Generation
- Generates realistic **T2 and T1ρ maps**
- Models **healthy vs OA cartilage**
- Includes **spatial noise and variability**
- 6 anatomical compartments:
  - Medial Femoral  
  - Lateral Femoral  
  - Medial Tibial  
  - Lateral Tibial  
  - Patellar  
  - Trochlear  

---

### 2. Feature Extraction

For each compartment:

**T2 Features**
- Mean  
- Standard deviation  
- 75th percentile  
- Fraction above threshold  

**T1ρ Features**
- Mean  
- Standard deviation  
- 75th percentile  
- Fraction above threshold  

**Global Features**
- T1ρ / T2 ratio (mean + std)  
- Cross-map correlation  

➡️ Total: **51 features per subject**

---

### 3. Machine Learning Models

Evaluated using **5-fold stratified cross-validation**:

- Random Forest  
- Gradient Boosting  
- SVM (RBF kernel)  

**Metrics**
- Accuracy  
- ROC-AUC  
- Confusion matrix  

---

### 4. Spatial Abnormality Heatmap

Pixel-level abnormality detection:

- Combines:
  - T2 deviation (40%)
  - T1ρ deviation (60%)
- Produces a **z-score heatmap**
- Highlights *where degeneration starts inside cartilage*

---

### 5. Visual Analytics Dashboard

The pipeline generates a full figure including:

- T2 / T1ρ maps (healthy vs OA)  
- Fused abnormality heatmap  
- Compartment-level comparison  
- ROC curves  
- Feature importance  
- Confusion matrix  
- Model comparison  

---

## 🚀 How to Run

### Install dependencies

```bash
pip install numpy scipy scikit-learn matplotlib seaborn
```

### Run the pipeline

```bash
python knee_mri_poc.py
```

**Output**
```
knee_mri_poc_results.png
```

---

## 📊 Key Insights

- **T1ρ features dominate** → strongest signal for early OA  
- **T1ρ / T2 ratio emerges as a powerful biomarker**  
- **Feature fusion outperforms single-modality analysis**  
- **Medial compartments show earliest degeneration**  
- **Heatmaps reveal sub-compartment damage** (missed by averages)

---

## 🧬 Why This Matters

Current clinical workflows:
- treat T2 and T1ρ separately  
- rely on manual segmentation  
- lose spatial information  

This pipeline shows a path toward:
- automated analysis  
- earlier detection  
- richer biomarkers  
- scalable clinical adoption  

---

## ⚠️ Limitations

- Uses **synthetic data** (not real MRI scans)  
- Performance is optimistic (AUC ≈ 1.0 expected here)  
- Real-world validation required  

---

## 🔮 Next Steps

- Apply to **real MRI datasets**  
- Add **automated cartilage segmentation** (e.g. nnU-Net)  
- Extend to **deep learning (CNN on T2 + T1ρ maps)**  
- Validate against clinical grading (MOAKS / OARSI)  
- Explore **longitudinal progression modelling**

---

## 👤 Author

**Nosa Peter Inwe**  
MSc Intelligent Photonics (Erasmus Mundus)  
Biometrics & Intelligent Vision  

---

## 📎 Context

Built as part of research work combining:

- Quantitative MRI  
- Machine learning  
- Cartilage degeneration modelling  
