````markdown
# 🧠 CT First POC

A lightweight proof of concept for CT image analysis, focused on building a clean, reproducible pipeline rather than chasing heavy compute.

---

## 🎯 Motivation

This repo is a small, focused experiment to answer a simple question:

> *Can I build a solid end-to-end CT analysis pipeline that produces meaningful features and reusable representations?*

Instead of scaling complexity early, the emphasis here is on clarity, structure, and learning signals.

---

## 📦 Dataset

I use the **CT-Scan Images** dataset from Kaggle:

🔗 https://www.kaggle.com/datasets/orvile/ct-scan-images

### Why this dataset?

- ⚡ Small enough to run locally without long training cycles  
- 🧩 Clear class separation (Cancer vs Non-Cancer)  
- 🧪 Ideal for testing pipeline design choices  
- 🧠 Covers key steps: feature extraction, baseline models, CNNs, embeddings  

---

## 🛠️ What This POC Does

The notebook walks through a full mini-pipeline:

1. 📂 Load and inspect CT images  
2. ⚖️ Analyse class distribution  
3. 🧱 Build a classical baseline (grayscale texture features)  
4. 🤖 Train a small CNN baseline  
5. 📊 Evaluate using confusion matrices + classification reports  
6. 🧬 Export embeddings + metadata for each image  

---

## 💡 Why Embeddings Matter

This project goes beyond classification.

Each CT scan is converted into a compact learned representation (embedding), enabling:

- 🔍 similarity search  
- 🧭 clustering and grouping  
- 📉 visualisation (e.g. PCA, t-SNE)  
- 🔗 future alignment with multimodal or clinical data  

Think of it as turning images into structured, reusable signals.

---

## 📁 Expected Dataset Structure

After downloading and extracting:

```text
data/
└── ct-scan-images/
    ├── Cancer/
    └── Non-Cancer/
````

> If the folder name differs, update `DATA_DIR` in the notebook.

---

## 🗂️ Repo Structure

```text
ct_first_poc_repo/
├── README.md
├── requirements.txt
├── .gitignore
└── notebooks/
    └── ct_first_poc.ipynb
```

---

## ⚙️ Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## ▶️ Run

```bash
jupyter notebook notebooks/ct_first_poc.ipynb
```

---

## 🔍 Key Questions

This POC is designed to answer:

* ❓ Can a clean baseline pipeline extract useful structure from CT images?
* ❓ Does a small CNN outperform handcrafted texture features?
* ❓ Are the learned embeddings stable and useful for downstream tasks?

If the answer is **yes**, this repo has done exactly what it was meant to do.

---

## 🚧 Future Directions

* 📈 Stronger architectures (ResNet, EfficientNet)
* 🧪 Data augmentation and regularisation
* 🧬 Embedding evaluation (clustering metrics, retrieval tasks)
* 🔗 Integration with clinical or multimodal datasets

---

## 📜 License

MIT License

```
MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🤝 Contributions

This is a personal POC, but ideas, suggestions, and improvements are always welcome.

---

## ✨ Final Note

Simple, clear pipelines beat complex setups that are hard to reason about.

This repo stays intentionally small so every step is easy to follow and justify.

```
```
