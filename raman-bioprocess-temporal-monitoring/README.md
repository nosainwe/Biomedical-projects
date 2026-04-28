🧫 Raman Bioprocess Temporal Monitoring
A GitHub-ready machine learning project for real-time temporal monitoring of a growing biological production process using Raman spectroscopy and process time-series data.
I built this project because I wanted something closer to the monitoring idea in the tissue-engineering spectroscopy papers I have been studying, but with a public dataset that actually has repeated measurements over time.
The lab papers motivated the question:
> Can spectroscopy and machine learning move us away from endpoint-only testing and toward real-time monitoring of biological growth?
This project answers that in a practical way using the IndPenSim biopharmaceutical manufacturing dataset, a large public dataset with online process variables, offline measurements, and simulated Raman spectra from industrial penicillin fermentation batches.
I am not claiming this is cartilage tissue engineering. It is not.  
I am using a bioprocess dataset because it gives me what the static cartilage datasets do not: spectral measurements across time.
---
🎯 Project idea
The core idea is simple:
> Treat each Raman spectrum as a snapshot of the biological process at a moment in time, then train models to estimate where the batch is in its growth/production trajectory.
In tissue engineering language, this is like asking:
How old is this construct in culture?
Is the system early, middle, or close to harvest?
How is the spectral signal changing over time?
Can I forecast the future state from recent spectral history?
In this project, the biological process is penicillin fermentation, not cartilage growth.  
The monitoring logic is the same.
---
🔬 Why this connects to tissue-engineering spectroscopy
The cartilage maturity paper by Elkadi et al. used visible and NIR spectroscopy with machine learning to predict GAG and DNA in tissue-engineered cartilage constructs. The constructs were cultured for 7 or 28 days, giving two relative maturity levels. The model predicted GAG and DNA, then used the predicted GAG/DNA ratio to classify maturity.
That paper is strong, but the limitation is also clear: two endpoints are not the same as continuous monitoring.
The culture-medium paper by Sadeesh et al. gets closer to the monitoring idea. It measured conditioned medium during a 28-day culture period and connected NIR spectral changes to biomarkers released into the medium, including hyaluronan, lactate, and collagen.
That is exactly the direction I wanted to model here:
repeated measurements
temporal behaviour
spectral signals
biological process state
prediction before the endpoint
So this repo is my practical bridge from “spectroscopy predicts biomarkers” to “spectroscopy tracks process trajectory”.
---
📦 Dataset
This project uses the Kaggle dataset:
Big Data - Biopharmaceutical Manufacturing  
Dataset page: `https://www.kaggle.com/datasets/stephengoldie/big-databiopharmaceutical-manufacturing`
The dataset is based on IndPenSim, an industrial-scale penicillin fermentation simulator. Public descriptions of the dataset state that it includes 100 batches with online, offline, and Raman data.
This is a useful dataset for this project because it contains:
batch time
batch ID
process variables
penicillin concentration
fault flags
Raman spectral columns across many wavelengths
The file usually used is:
```text
100_Batches_IndPenSim_V3.csv
```
The full file is large, so I do not store it in this GitHub repo.
---
🧠 What the notebook does
The notebook is heavily commented and built as a teaching notebook.
It does the following:
Finds the IndPenSim CSV file automatically
Detects key columns such as:
batch ID
time in hours
penicillin concentration
fault flag
Raman wavelength columns
Loads a manageable subset of batches and Raman wavelengths
Converts time from hours to days
Creates temporal targets:
current batch age in days
days remaining until an 80% production threshold
Visualises the Raman trajectory over time
Builds baseline machine learning models
Builds a small LSTM model for temporal sequence learning
Saves plots to `outputs/plots/`
Explains how this maps back to tissue-engineering monitoring
---
🧪 Machine learning tasks
Task 1: Real-time day estimator
Input:
```text
current Raman spectrum + current process variables
```
Output:
```text
estimated batch age in days
```
This is the closest practical version of:
> Given a spectrum from a growing biological system, can I estimate where it is in the culture timeline?
Task 2: Days-to-threshold predictor
Input:
```text
current Raman spectrum + current process state
```
Output:
```text
estimated days remaining until the batch reaches 80% of its maximum penicillin concentration
```
This is the bioprocess version of:
> How long until the construct reaches a target maturity threshold?
Task 3: Temporal LSTM model
Input:
```text
sequence of recent measurements
```
Output:
```text
current or near-future process state
```
This matters because biological systems are not static. The recent trajectory can contain information that a single spectrum misses.
---
📊 Expected plots
The notebook saves plots such as:
Raman spectral trajectory heatmap
Penicillin concentration over time
PCA trajectory of Raman spectra over batch time
Actual vs predicted batch age in days
Days-to-threshold prediction plot
LSTM training curve
These go into:
```text
outputs/plots/
```
---
🧱 Repo structure
```text
raman-bioprocess-temporal-monitoring/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── notebooks/
│   └── raman_bioprocess_temporal_monitoring.ipynb
├── outputs/
│   └── plots/
├── models/
└── src/
    └── bioprocess_utils.py
```
---
⚙️ Setup
Option A: Run on Kaggle
This is the easiest route because the dataset is already on Kaggle.
Create a new Kaggle notebook.
Add this dataset as input:
```text
stephengoldie/big-databiopharmaceutical-manufacturing
```
Upload or copy the notebook from this repo.
Run the notebook from top to bottom.
Option B: Run locally
Download the Kaggle dataset manually and place the CSV here:
```text
data/100_Batches_IndPenSim_V3.csv
```
Then install dependencies:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
Run:
```bash
jupyter notebook notebooks/raman_bioprocess_temporal_monitoring.ipynb
```
---
📘 What I need to understand from this project
1. Spectra are high-dimensional fingerprints
A Raman spectrum is not a single number. It is a vector of intensity values across wavelengths or Raman shifts.
In this project:
```text
one row = one time point in one batch
many Raman columns = spectral fingerprint at that time point
```
2. A temporal dataset is not just a table
The order matters.
If I shuffle the time points blindly, I destroy the biological story.
For each batch:
```text
early time → mid process → late process → harvest
```
That is why the notebook groups data by batch and builds sequence windows.
3. Sliding windows turn time into supervised learning
A model cannot learn “recent history” unless I give it history.
So I build windows like:
```text
[t-9, t-8, ..., t] → target at t
```
or:
```text
[t-9, t-8, ..., t] → target at t+future
```
That is the basic trick behind many temporal prediction models.
4. LSTM learns sequence memory
An LSTM receives measurements step by step and learns which parts of the past matter.
A dense model sees only a flattened snapshot.  
An LSTM sees a short trajectory.
That distinction is the whole point of this project.
5. This is a soft-sensor problem
A soft sensor predicts a hard-to-measure variable from easier-to-measure signals.
In the cartilage papers, spectroscopy is used to estimate destructive or slow reference measurements such as GAG, DNA, hyaluronan, lactate, or collagen.
In this project, Raman and process variables estimate batch age, product concentration, and days-to-threshold.
---
🧬 Translation back to tissue engineering
If I later had real tissue-engineering spectral data, I would replace:
```text
Raman spectra from penicillin fermentation
```
with:
```text
NIR/Raman spectra from tissue constructs or culture medium
```
and replace:
```text
penicillin concentration / batch age
```
with:
```text
GAG/DNA ratio / biomarker concentration / culture day / maturity threshold
```
The modelling logic stays the same:
```text
spectral trajectory → temporal model → maturity estimate
```
---
⚠️ Honest limitations
This project is useful, but I should not oversell it.
The dataset is penicillin fermentation, not cartilage.
IndPenSim is simulated, although it is designed as a realistic industrial bioprocess benchmark.
The target “days-to-threshold” is engineered from penicillin concentration, not measured tissue maturity.
The biological translation still needs real longitudinal tissue-engineering spectra.
A strong result here proves that I understand the temporal modelling workflow, not that I solved cartilage monitoring.
That honesty matters.
---
🚀 Next improvements
If I continue this project, I would add:
a Transformer time-series model
online streaming inference, one time point at a time
uncertainty estimates for days-to-threshold
fault-aware training, comparing normal and abnormal batches
a dashboard that updates the predicted day and maturity state as new spectra arrive
a proper tissue-engineering dataset when one becomes available
---
📚 References
Tissue-engineering motivation
Elkadi, O. A. et al. Non-destructive assessment of tissue engineered cartilage maturity using visible and near infrared spectroscopy combined with machine learning. Biosensors and Bioelectronics 286, 117587, 2025.
Sadeesh, N. et al. Non-destructive monitoring of cartilage tissue engineering via near-infrared (NIR) spectroscopic assessment of culture medium. Biosensors and Bioelectronics 288, 117809, 2025.
Dataset and bioprocess context
Kaggle dataset: Big Data - Biopharmaceutical Manufacturing
IndPenSim project: industrial-scale penicillin fermentation simulator with online, offline, and Raman-style process data.
