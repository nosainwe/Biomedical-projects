# Temporal ConvLSTM POC for Spectral Monitoring

I built this repo as a first-principles learning project for temporal modelling.

I did not want to jump straight into a tissue-engineering notebook full of shortcuts and unexplained code. So I start with a real public multivariate time-series dataset and learn the modelling logic first.

## Why this repo exists

Temporal monitoring of tissue constructs rather than one-shot endpoint prediction.

That matters. A model trained on only two far-apart time points can look strong because the biological states are already very different. The harder and more useful problem is different:

- how does the signal evolve over time?
- can I learn the trajectory rather than only classify the endpoints?
- can I predict when the system is approaching a target state?

## Why the temporal angle is justified

The 2025 tissue-engineered cartilage maturity paper used constructs cultured for only **7 or 28 days**, then predicted GAG and DNA and used the predicted GAG/DNA ratio to distinguish immature from more mature constructs. The paper itself says this setup used only **two culture durations with distinct biochemical differences**, and that **closer time points are needed** to test robustness in real-world tissue-engineering problems. It also says continuous monitoring could reveal **plateaus, fluctuations, and post-peak declines** that endpoint measurements would miss.

The 2025 culture-medium paper is closer to a true temporal setup. It collected conditioned medium every **3 days over 28 days**, then linked NIR spectra to biomarker release patterns such as hyaluronan, lactate, and collagen.

So the gap is clear:

- endpoint models answer “are these two states different?”
- temporal models answer “how is this construct changing, and where is it heading?”

## Dataset choice

I use the Jena Climate dataset.

The original task is weather forecasting, not tissue engineering. I am using it because the modelling logic is the part I want to learn first:

- each timestamp has a vector of measurements
- those measurements evolve over time
- future values depend on temporal context, not only the current reading

In a tissue-engineering setting, I would later replace “sensor vector at time t” with “spectral vector or biomarker vector at time t”.

## What the notebook does

1. downloads and inspects the dataset
2. builds sliding windows
3. trains a dense baseline
4. trains an LSTM baseline
5. trains a ConvLSTM baseline by reshaping windows into pseudo-images
6. compares the models and connects the result back to longitudinal spectroscopy

## Repo structure

```text
Temporal ConvLSTM POC for Spectral Monitoring/
├── README.md
├── requirements.txt
├── .gitignore
└── notebooks/
    └── temporal_convlstm_first_principles.ipynb
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
jupyter notebook notebooks/temporal_convlstm_first_principles.ipynb
```

## What I want to demonstarte from this repo

- what makes a problem temporal rather than static
- how sliding windows turn sequences into supervised learning samples
- what an LSTM remembers
- what a ConvLSTM adds on top of that
- when temporal modelling is actually useful for longitudinal spectroscopy

## How I would translate this later

Once I understand this notebook properly, the next step is straightforward:

- replace the public time-series dataset with repeated spectral measurements of the same construct over time
- let each time step contain a spectrum or a compact feature vector
- predict a future biomarker, maturity score, or time-to-threshold
- compare whether dense, LSTM, or ConvLSTM handles the trajectory best
