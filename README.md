# Biomedical-projects
Machine learning pipelines for biomedical Imaging/spectroscopy - NIR, Raman, terahertz, and beyond. Tissue characterisation, disease detection, physiological signal classification, and spectral analysis across the electromagnetic spectrum
Each project is self-contained with its own setup and run instructions.

---

## Projects

| Project | Description | Technologies |
|---------|-------------|--------------|
| [🔬 NIR Cartilage Classifier](./nir-cartilage-classifier/) | Proof-of-concept ML pipeline classifying **Near-Infrared spectra** into Healthy vs. Osteoarthritic cartilage. Full preprocessing chain (SG + MSC + 2nd derivative) benchmarked across Random Forest, SVM, and PLS-DA with wavelength importance analysis. Built in the context of Prof. Afara's BSL research at UEF. | scikit-learn, scipy, NumPy, Matplotlib |

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
- Biomedical applications: cartilage health, osteoarthritis detection, tissue composition

---

## License

MIT - see [LICENSE](./LICENSE).
