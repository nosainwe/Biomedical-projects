# ❤️ Accelerated 3D Cardiac T2 Mapping with KWIC

**3D whole-heart T2 mapping using radial k-space sampling and KWIC filtering**

---

## 📌 Overview

This project is a **proof-of-concept (POC)** for accelerating cardiac T2 mapping using:

- Golden-angle radial sampling  
- KWIC (K-space Weighted Image Contrast) filtering  
- Extension from **2D → 3D whole-heart imaging**

It builds on the SKRATCH framework and demonstrates how to reduce acquisition time while preserving quantitative accuracy.

---

## 🧪 What This Project Does

### 1. 3D Cardiac Phantom
- Multi-slice heart model (base → apex)
- Includes:
  - Healthy myocardium  
  - Blood pool  
  - Edema region  
- Realistic T2 values:
  - Myocardium: ~42 ms  
  - Edema: ~62 ms  
  - Blood: ~250 ms  

---

### 2. T2-weighted Simulation
- Simulates **T2-prepared MRI acquisitions**
- Uses exponential decay model:

S(TE) = exp(-TE / T2)

- Four T2prep times:
  - 0, 25, 45, 60 ms  

---

### 3. Radial k-space Sampling
- Golden-angle radial acquisition  
- Undersampling for acceleration  
- Comparison:
  - Full sampling  
  - Undersampled  

---

### 4. KWIC Filtering

**Core idea:**
- k-space centre → contrast (keep original)
- k-space periphery → structure (share across images)

#### 2D KWIC
- Shares periphery across T2prep images  

#### 3D KWIC (Proposed)
- Shares periphery:
  - across T2prep images  
  - across neighbouring slices (kz)

→ improves SNR and reconstruction quality  

---

### 5. T2 Mapping
- Pixel-wise estimation using log-linear regression  
- Vectorised least-squares fitting  
- Produces full 3D T2 maps  

---

### 6. Evaluation

Four conditions compared:
- Full (reference)
- Undersampled
- 2D KWIC
- 3D KWIC (proposed)

Metrics:
- Bias  
- Standard deviation  
- RMSE  

---

### 7. Phantom Validation
- Known T2 values tested (20–200 ms)  
- Confirms reconstruction accuracy  

---

## 🚀 How to Run

Open MATLAB and run:

cardiac_t2_3d_kwic

No external toolboxes required.

---

## 📊 Outputs

The script generates:

- T2 map comparisons (all methods)
- Multi-slice 3D visualisation
- RMSE comparison chart
- T2 decay curves
- Phantom validation plot
- AHA segment analysis

Saved file:

cardiac_t2_3d_poc_results.png

---

## 📊 Key Insights

- 3D KWIC reduces RMSE compared to 2D KWIC  
- k-space sharing across slices improves SNR  
- Edema regions (>50 ms) clearly detected  
- Undersampling alone introduces strong artefacts  
- KWIC restores image quality without increasing scan time  

---

## 🧬 Why This Matters

Cardiac T2 mapping is limited by:
- long acquisition times  
- motion sensitivity  
- low SNR in accelerated scans  

This approach shows how to:
- accelerate acquisition  
- maintain quantitative accuracy  
- extend to full 3D coverage  

---

## ⚠️ Limitations

- Simulated phantom (not in-vivo data)  
- Simplified noise model  
- No motion modelling (cardiac/respiratory)  

---

## 🔮 Next Steps

- Apply to real cardiac MRI datasets  
- Integrate motion correction  
- Extend to deep reconstruction models  
- Combine with T1 mapping (multi-parametric MRI)  

---

## 📎 Context

Developed for research direction in:

- Cardiac MRI  
- Quantitative imaging  
- k-space reconstruction methods  
- Accelerated MRI acquisition  
