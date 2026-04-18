# Histology POC

I built this repo as a small, clean proof of concept.

I kept the first version narrow on purpose.

Instead of pretending I already have a public dataset that cleanly joins synchrotron CT, histology, and spatial transcriptomics, I started with one modality that is public, small enough to run locally, and easy to explain in a meeting: histology image tiles.

## Why I chose this dataset

I chose **Colorectal Histology MNIST** from Kaggle.

Why I picked it:

- it is small enough to run without turning the notebook into a week-long training job
- it has **5,000 RGB histology tiles**
- each tile is **150 x 150 px**
- it contains **8 tissue classes**
- it is easy to explain as a first step toward cross-modal biomedical analysis

Kaggle page:
https://www.kaggle.com/datasets/kmader/colorectal-histology-mnist

## What this POC does

In this notebook, I do four things:

1. load and inspect histology tiles
2. build a classical baseline using texture and colour features
3. train a small CNN baseline
4. export per-tile embeddings and metadata so the workflow is ready for later matching with CT regions or transcriptomic spots

That last part matters.

My point is not only to classify tissue tiles. My point is to show a clean path from image data to embeddings that can later be linked across modalities.

## Expected dataset structure

After downloading the Kaggle dataset, I place it like this:

```text
data/
└── colorectal-histology-mnist/
    └── Kather_texture_2016_image_tiles_5000/
        ├── 01_TUMOR/
        ├── 02_STROMA/
        ├── 03_COMPLEX/
        ├── 04_LYMPHO/
        ├── 05_DEBRIS/
        ├── 06_MUCOSA/
        ├── 07_ADIPOSE/
        └── 08_EMPTY/

If the extracted folder name differs slightly, I just update the DATA_DIR path in the notebook.

## Files

```text
marina_histology_poc/
├── README.md
├── requirements.txt
├── .gitignore
└── notebooks/
    └── histology_first_poc.ipynb
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
jupyter notebook notebooks/histology_first_poc.ipynb
```


