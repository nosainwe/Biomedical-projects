# 🩻 CT First POC

We built this proof of concept because scaling complexity too early kills projects. I don't want to guess whether a bare-bones pipeline extracts meaningful features from medical imaging — I want to see it work. You'll often find yourself stuck tweaking hyperparameters instead of understanding the data. I reject that approach entirely. Start small. Learn the signals.

---

## Motivation

I hate over-engineered starting points. This repository answers one simple question: does a small convolutional neural network beat classical grayscale texture features?

Real medical data is messy. Algorithms fail silently on edge cases. You'll doubt the generalisability of tiny datasets, and rightly so. [Kaggle's CT-Scan Images dataset by Orvile](https://www.kaggle.com/datasets/orvile/ct-scan-images) gives us clear Cancer vs. Non-Cancer separation for quick testing. We need this friction to test pipeline design choices locally. Training cycles finish in minutes. We're running this right here in Joensuu on a standard laptop.

---

## What This POC Actually Does

1. Load the CT images and inspect the class distribution
2. Train a classical baseline (grayscale texture features)
3. Train a small CNN baseline
4. Evaluate both using confusion matrices and standard metrics
5. Export learned embeddings for downstream use

---

## Why Embeddings Matter

The network converts each scan into a compact learned representation. These vectors are immediately useful for similarity search, clustering, visualisation, and future alignment with clinical metadata. The embedding space is what carries this beyond a one-off classifier.

---

## Dataset Setup

Structure the dataset exactly like this before running anything:

```
data/
└── ct-scan-images/
    ├── Cancer/
    └── Non-Cancer/
```

Download from Kaggle and extract into `data/`. The notebook expects this layout exactly.

---

## Repository Layout

```
ct_first_poc_repo/
├── README.md
├── requirements.txt
├── .gitignore
└── notebooks/
    └── ct_first_poc.ipynb
```

---

## Setup and Execution

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run the notebook:

```bash
jupyter notebook notebooks/ct_first_poc.ipynb
```

---

## Key Questions This POC Answers

- Can we extract useful structure from CT images fast, with minimal infrastructure?
- Does the small CNN obliterate handcrafted texture features, or does classical win here?
- Do the learned embeddings actually help downstream tasks?
- Can we trust the raw learning signals from a dataset this small?

---

## License

MIT
