# Biomedical-projects

Machine learning pipelines for biomedical imaging and spectroscopy, from NIR tissue analysis to histology, CT, MRI, and temporal modelling. The repo focuses on tissue characterisation, disease detection, longitudinal monitoring, and signal-aware analysis across imaging and spectral data.

Each project is self-contained and includes its own setup and run instructions.

---

## Projects

| Project | Description | Technologies |
|---------|-------------|--------------|
| [🔬 NIR Cartilage Classifier](./nir-cartilage-classifier/) | Proof-of-concept ML pipeline classifying **Near-Infrared spectra** into Healthy vs. Osteoarthritic cartilage. Includes SG filtering, MSC, second-derivative preprocessing, and model benchmarking across Random Forest, SVM, and PLS-DA, with wavelength-importance analysis. Built in the context of Prof. Afara's BSL research at UEF. | scikit-learn, SciPy, NumPy, Matplotlib |
| [🦴 Knee OA Severity Classifier](./knee%20osteoarthritis%20Classifier/) | DenseNet201 fine-tuned to grade knee osteoarthritis severity from X-ray images using the **Kellgren-Lawrence (KL) scale** from grade 0 to grade 4. Includes strong spatial augmentation, custom Random Erasing, and early stopping on validation accuracy. | TensorFlow, Keras, OpenCV, scikit-learn |
| [🧠 Knee MRI OA Detection (T2 + T1ρ Fusion)](./Knee%20MRI%20OA%20Detection%20(T2%20%2B%20T1%CF%81%20Fusion)/) | Joint analysis pipeline for **T2 and T1ρ MRI cartilage maps** to detect early osteoarthritis. Includes synthetic data generation, compartment-based feature extraction with 51 features, multi-model classification, and a fused spatial abnormality heatmap to highlight early degeneration zones. | scikit-learn, NumPy, SciPy, Matplotlib, Seaborn |
| [🫀 Accelerated 3D Cardiac T2 Mapping with KWIC Filter](./Accelerated%203D%20Cardiac%20T2%20Mapping%20with%20KWIC%20Filter/) | Research-oriented reconstruction project for accelerated cardiac MRI. Focuses on radial k-space handling, KWIC filtering, multi-echo reconstruction logic, and quantitative T2 mapping from undersampled acquisitions. Built as a learning and research preparation project around quantitative cardiac MRI workflows. | Python, NumPy, SciPy, Matplotlib |
| [🧬 Histology First POC](./histology_first_poc/) | Histology analysis proof of concept built around tissue-image classification and feature learning. Includes dataset inspection, classical handcrafted baselines, CNN modelling, and embedding export for later downstream or cross-modal analysis. Designed as a clean biomedical imaging starting point rather than an overblown demo. | PyTorch, scikit-learn, NumPy, pandas, Matplotlib |
| [🩻 CT First POC](./CT_first_poc_repo/) | Separate CT analysis proof of concept focused on grayscale medical image understanding. Includes image inspection, handcrafted texture baselines, CNN classification, confusion-matrix analysis, and embedding export. Built as a stand-alone CT branch rather than pretending CT is already paired with other modalities. | PyTorch, scikit-learn, scikit-image, NumPy, pandas, Matplotlib |
| [⏱️ Temporal ConvLSTM POC for Spectral Monitoring](./Temporal%20ConvLSTM%20POC%20for%20Spectral%20Monitoring/) | First-principles temporal modelling project built to understand longitudinal prediction before applying it to tissue-engineering spectra. Compares dense, LSTM, and ConvLSTM baselines on multivariate sequences, with the later goal of modelling spectral maturity trajectories rather than only endpoint separation. | TensorFlow, Keras, NumPy, pandas, scikit-learn, Matplotlib |

---

## What's inside each project folder

Each folder typically contains:

- `README.md` - project overview, background, setup, and notes
- `requirements.txt` - Python dependencies
- `*.py` or `*.ipynb` - the main script or notebook
- `assets/` or `outputs/` - optional output figures or saved results

Click any project above to inspect the code and project notes directly.

---

## Themes covered

This repo sits at the intersection of **biomedical imaging, spectroscopy, temporal modelling, and machine learning**:

- Near-Infrared spectroscopy and tissue characterisation
- Spectral preprocessing: Savitzky-Golay filtering, Multiplicative Scatter Correction, derivative transforms
- Chemometrics methods: PLS-DA, feature importance, ROC/AUC evaluation
- Deep learning for medical image classification: transfer learning, augmentation strategies, KL grading
- Quantitative MRI analysis: T2 and T1ρ mapping, feature fusion, spatial abnormality detection
- Histology and CT image analysis: texture features, CNN baselines, embeddings
- Temporal modelling for repeated measurements: sliding windows, sequence learning, ConvLSTM
- Biomedical applications: cartilage health, osteoarthritis detection, tissue maturity monitoring, image-derived biomarkers

---

## Why this repo exists

I use this repo to keep my biomedical work in one place, but also to make the progression visible.

Some projects are straightforward classifiers. Some are research-style proof-of-concept studies. A few are there because I needed to understand the method properly before using it in a real biomedical setting. That part matters. I would rather keep an honest learning project in public than fake maturity with a polished title and shaky logic.

---

## License

MIT - see [LICENSE](./LICENSE).
