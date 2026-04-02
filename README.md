# Biomedical-projects
Machine learning pipelines for biomedical imaging/spectroscopy - NIR, Raman, terahertz, and beyond. Tissue characterisation, disease detection, physiological signal classification, and spectral analysis across the electromagnetic spectrum.

Each project is self-contained with its own setup and run instructions.

---

## Projects

| Project | Description | Technologies |
|---------|-------------|--------------|
| [🔬 NIR Cartilage Classifier](./nir-cartilage-classifier/) | Proof-of-concept ML pipeline classifying **Near-Infrared spectra** into Healthy vs. Osteoarthritic cartilage. Full preprocessing chain (SG + MSC + 2nd derivative) benchmarked across Random Forest, SVM, and PLS-DA with wavelength importance analysis. Built in the context of Prof. Afara's BSL research at UEF. | scikit-learn, scipy, NumPy, Matplotlib |
| [🦴 Knee OA Severity Classifier](./knee-oa-classifier/) | DenseNet201 fine-tuned to grade knee osteoarthritis severity from X-ray images using the **Kellgren-Lawrence (KL) scale** (grades 0–4). Includes custom Random Erasing augmentation, heavy spatial augmentation, and early stopping on validation accuracy. | TensorFlow, Keras, OpenCV, scikit-learn |
| [🧠 Knee MRI T2 + T1ρ OA Detection]([./Knee-MRI-OA-Detection-(T2-+-T1ρ-Fusion)/](https://github.com/nosainwe/Biomedical-projects/tree/main/Knee%20MRI%20OA%20Detection%20(T2%20%2B%20T1%CF%81%20Fusion))) | Joint analysis pipeline for **T2 and T1ρ MRI cartilage maps** to detect early osteoarthritis. Includes synthetic data generation, compartment-based feature extraction (51 features), multi-model classification (RF, GB, SVM), and a fused spatial abnormality heatmap highlighting early degeneration zones. Built as a research POC for quantitative MRI analysis. | scikit-learn, NumPy, SciPy, Matplotlib, Seaborn |

---

## What's inside each project folder

Each folder typically contains:

- `README.md` - project overview, background, setup, and notes
- `requirements.txt` - Python dependencies
- `*.py` - the main script(s)
- `assets/` - optional output figures (gitignored unless small and useful)

Click any project above to see the details.

---

## Themes covered

This repo focuses on the intersection of **spectroscopy, biomedical signal analysis, and machine learning**:

- Near-Infrared (NIR) spectroscopy and tissue characterisation
- Spectral preprocessing: Savitzky-Golay filtering, Multiplicative Scatter Correction (MSC), derivative transforms
- Chemometrics methods: PLS-DA, feature importance, ROC/AUC evaluation
- Deep learning for medical image classification: transfer learning, augmentation strategies, KL grading
- Biomedical applications: cartilage health, osteoarthritis detection, tissue composition
- Quantitative MRI analysis: T2 and T1ρ mapping, multi-parametric feature fusion, spatial abnormality detection

---

## License

MIT - see [LICENSE](./LICENSE).
