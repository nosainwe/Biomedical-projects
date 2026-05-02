# 🌱 Temporal ConvLSTM POC for Spectral Monitoring

I built this repo as a **first-principles learning project** for **temporal modelling**. 

I did not want to jump straight into a complex tissue-engineering notebook full of shortcuts and unexplained code. Instead, I wanted to start with a real, publicly available **multivariate time-series dataset** and focus on understanding the core modelling logic first. 🚀

---

## 🎯 Why this repo exists

This repo is built to explore **temporal monitoring of tissue constructs**, rather than just predicting endpoints. 🧬

That matters because:
- A model trained on only **two far-apart time points** can look strong due to large differences in biological states. But that’s not a real-world solution.
- The **real problem** is learning how the signal evolves over time:  
  - **How does the signal evolve over time?**
  - **Can I learn the trajectory** rather than just classify the endpoints?
  - **Can I predict** when the system is approaching a target state? 🎯

---

## ⏳ Why the Temporal Angle is Justified

The 2025 tissue-engineered cartilage maturity paper used constructs cultured for only **7 or 28 days**, predicting **GAG** and **DNA** levels. This setup was based on just two culture durations with distinct biochemical differences. The authors state that **closer time points are needed** to test the model’s robustness for real-world tissue-engineering problems.

The paper highlights that **continuous monitoring** can reveal **plateaus, fluctuations, and post-peak declines** that endpoint measurements alone cannot capture. 📉

The gap is clear:
- **Endpoint models** answer “**Are these two states different?**”
- **Temporal models** answer “**How is this construct changing, and where is it heading?**”

---

## 🌍 Dataset Choice

For this project, I’m using the **Jena Climate Dataset**. 🌦️

The original task in this dataset is **weather forecasting**, but I’m using it because:
- Each timestamp contains a vector of measurements (e.g., temperature, humidity, pressure).
- These measurements evolve over time, and future values depend on **temporal context**, not just the current reading.

In a **tissue-engineering setting**, I would later replace the "sensor vector at time t" with "**spectral vector**" or "**biomarker vector**" at time t.

---

## 🛠️ What the notebook does

1. **Download and inspect** the Jena Climate dataset 🌦️
2. **Build sliding windows** to prepare the time-series data for supervised learning ⏳
3. Train a **dense baseline model** to predict future values 💻
4. Train an **LSTM baseline model** for temporal learning 📊
5. Train a **ConvLSTM baseline model** by reshaping the windows into **pseudo-images** 🌌
6. Compare the models and connect the results back to **longitudinal spectroscopy** 💡

---

## 📁 Repo Structure

```text
Temporal ConvLSTM POC for Spectral Monitoring/
├── README.md
├── requirements.txt
├── .gitignore
└── notebooks/
    └── temporal_convlstm_first_principles.ipynb
```

---

## ⚙️ Install

First, create a virtual environment and install the required dependencies:

```bash id="mk1zk3"
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts ctivate
pip install -r requirements.txt
```

---

## ▶️ Run

To run the Jupyter notebook:

```bash id="1o7oal"
jupyter notebook notebooks/temporal_convlstm_first_principles.ipynb
```

---

## 💡 What I Want to Demonstrate from This Repo

- **What makes a problem temporal** rather than static ⏳
- How **sliding windows** convert sequences into supervised learning samples 📊
- What **LSTM** remembers when predicting over time 🔄
- What **ConvLSTM** adds on top of that 🧠
- **When temporal modelling** is actually useful for **longitudinal spectroscopy** 🔬

---

## 🚀 How I Would Translate This Later

Once I fully understand the notebook, the next step would be straightforward:

- Replace the public time-series dataset with **repeated spectral measurements** of the same tissue construct over time 🌱
- Let each time step contain a **spectrum** or a **compact feature vector**
- **Predict a future biomarker**, maturity score, or **time-to-threshold** ⏱️
- Compare whether **dense, LSTM, or ConvLSTM** models handle the trajectory best 🤖

