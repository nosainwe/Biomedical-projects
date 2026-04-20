# CT First POC

I built this repo as a small, separate proof of concept for CT image analysis.

I kept it narrow on purpose.

I did not want to pretend I already had a public dataset that pairs cleanly with histology or transcriptomics. So I treated CT as its own branch and built a workflow I can actually explain, defend, and extend later.

## Why I chose this dataset

I use the **CT-Scan images** dataset from Kaggle.

Kaggle page:
https://www.kaggle.com/datasets/orvile/ct-scan-images

Why I picked it:

- it is small enough to run locally without turning this into a long training job
- the class story is easy to explain
- it lets me focus on pipeline quality instead of compute
- it is a good fit for learning the basics of CT image analysis, feature extraction, baseline modelling, CNN training, and embedding export

## What this POC does

In this notebook, I do six things:

1. load and inspect the CT images
2. check the class balance
3. build a classical baseline using grayscale texture features
4. train a small CNN baseline
5. evaluate both models with confusion matrices and classification reports
6. export per-image embeddings and metadata for later downstream analysis

That last part matters.

The point is not only to classify scans. The point is to turn each scan into a compact learned fingerprint that I can later compare, group, visualise, or align with other sources of information once I have real paired data.

## Expected dataset structure

After downloading the Kaggle dataset, I place it like this:

```text
data/
└── ct-scan-images/
    ├── Cancer/
    └── Non-Cancer/
```

If the extracted folder name differs slightly, I just update the `DATA_DIR` path in the notebook.

## Repo structure

```text
ct_first_poc_repo/
├── README.md
├── requirements.txt
├── .gitignore
└── notebooks/
    └── ct_first_poc.ipynb
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
jupyter notebook notebooks/ct_first_poc.ipynb
```

## What I am trying to get from this POC

I am trying to answer three simple questions:

- can I extract meaningful structure from these CT images using a clean baseline pipeline?
- does a small CNN learn something stronger than handcrafted texture features?
- can I export stable embeddings that could later support more serious downstream analysis?

If the answer to those is yes, then this repo has done its job.

## Notes

This repo is intentionally simple.

I would rather show a smaller pipeline that is real, readable, and easy to explain than build something bloated that I cannot defend line by line.
