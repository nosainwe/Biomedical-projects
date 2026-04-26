🧠 CT First POC

A lightweight proof of concept for CT image analysis, focused on
building a clean, reproducible pipeline rather than chasing heavy
compute.

------------------------------------------------------------------------

🎯 Motivation

This repo is a small, focused experiment to answer a simple question:

Can I build a solid end-to-end CT analysis pipeline that produces
meaningful features and reusable representations?

Instead of scaling complexity early, the emphasis here is on clarity,
structure, and learning signals.

------------------------------------------------------------------------

📦 Dataset

CT-Scan Images dataset from Kaggle:
https://www.kaggle.com/datasets/orvile/ct-scan-images

Why this dataset?

-   Small enough to run locally without long training cycles
-   Clear class separation (Cancer vs Non-Cancer)
-   Ideal for testing pipeline design choices
-   Covers key steps: feature extraction, baseline models, CNNs,
    embeddings

------------------------------------------------------------------------

🛠️ What This POC Does

1.  Load and inspect CT images
2.  Analyse class distribution
3.  Build a classical baseline (grayscale texture features)
4.  Train a small CNN baseline
5.  Evaluate using confusion matrices and classification reports
6.  Export embeddings and metadata

------------------------------------------------------------------------

💡 Why Embeddings Matter

Each CT scan is converted into a compact learned representation,
enabling:

-   similarity search
-   clustering and grouping
-   visualisation
-   future alignment with multimodal or clinical data

------------------------------------------------------------------------

📁 Expected Dataset Structure

data/ └── ct-scan-images/ ├── Cancer/ └── Non-Cancer/

------------------------------------------------------------------------

🗂️ Repo Structure

ct_first_poc_repo/ ├── README.md ├── requirements.txt ├── .gitignore └──
notebooks/ └── ct_first_poc.ipynb

------------------------------------------------------------------------

⚙️ Setup

python -m venv .venv source .venv/bin/activate pip install -r
requirements.txt

------------------------------------------------------------------------

▶️ Run

jupyter notebook notebooks/ct_first_poc.ipynb

------------------------------------------------------------------------

🔍 Key Questions

-   Can a clean pipeline extract useful structure?
-   Does a small CNN outperform handcrafted features?
-   Are embeddings useful for downstream tasks?

------------------------------------------------------------------------

📜 License

MIT License
