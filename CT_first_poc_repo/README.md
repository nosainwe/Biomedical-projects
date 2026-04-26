🩻 CT First POC

We've built this proof of concept because scaling complexity too early kills projects. I don't want to guess if a bare-bones pipeline extracts meaningful features from medical imaging; I want to see it work. You'll often find yourselves stuck tweaking hyperparameters instead of understanding the data. I reject that approach entirely. Start small. Learn the signals.

Motivation
I hate over-engineered starting points. This repository answers one simple question: does a small convolutional neural network beat classical grayscale texture features? Real medical data is messy; algorithms fail silently on edge cases. You'll doubt the generalisability of tiny datasets, and rightly so. Kaggle's CT-Scan Images dataset by Orvile gives us clear Cancer versus Non-Cancer separation for quick testing. We need this friction to test pipeline design choices locally. Training cycles finish in minutes. We're running this right here in Joensuu on a standard laptop.

What This POC Actually Does
You'll load the CT images. You'll analyse the class distribution. You'll train a classical baseline. You'll train a small CNN baseline. You'll evaluate the outputs using confusion matrices. You'll export the embeddings.

Why Embeddings Matter
The network converts each scan into a compact learned representation. You'll use these vectors for similarity search, clustering, visualisation, and future alignment with clinical metadata.

Directory Setup
You've got to structure the dataset exactly like this:
data/
└── ct-scan-images/
    ├── Cancer/
    └── Non-Cancer/

The repository follows a tight layout:
ct_first_poc_repo/
├── README.md
├── requirements.txt
├── .gitignore
└── notebooks/
    └── ct_first_poc.ipynb

Execution

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Run the notebook:
jupyter notebook notebooks/ct_first_poc.ipynb

Key Questions
Can we extract useful structure fast?
Does the small CNN obliterate handcrafted features?
Do these embeddings actually help downstream tasks?
Can I trust the raw learning signals?

License
MIT License.
