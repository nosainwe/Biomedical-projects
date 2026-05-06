# 🌱 Temporal ConvLSTM POC for Spectral Monitoring

I built this repo as a first-principles learning project for temporal modelling.

I am starting with a real public multivariate time-series dataset and use it to learn the modelling logic first.


The point is to learn the shape of the problem:

```text
repeated measurement over time -> temporal window -> model -> future state prediction
```

Later, I can replace the weather sensor vector with a spectral vector, biomarker vector, or tissue-culture measurement.

---

## 🎯 Why this repo exists

This repo explores temporal monitoring of tissue constructs, not only one-shot endpoint prediction.

That distinction matters.

A model trained on two far-apart time points can look good because the biological states are already easy to separate. For example, day 7 and day 28 constructs may differ strongly in matrix content, water content, scattering behaviour, and biochemical maturity. A classifier can exploit that separation without truly learning how the construct develops.

The harder question is this:

```text
Can I model the path between those endpoints?
```

That is the useful problem.

I want this repo to help me understand:

- how a signal evolves over time
- how sliding windows turn a sequence into supervised learning data
- how a model can use recent history, not just the current reading
- how temporal prediction could support non-destructive monitoring in tissue engineering
- how to move from "what class is this?" to "where is this system heading?"

---

## 🧬 Tissue-engineering motivation

The tissue-engineering use case I care about is longitudinal monitoring.

In a real cartilage or tissue-engineering experiment, researchers may care about variables such as:

```text
culture day
GAG content
DNA content
GAG/DNA ratio
collagen-related signals
lactate release
hyaluronan release
spectral intensity across wavelengths
```

Endpoint testing can destroy the sample or only tell me what happened at one time point. That is not enough if the goal is process control, early warning, or deciding when the construct is ready.

A temporal model should answer questions like:

```text
Is the construct still maturing?
Has the signal plateaued?
Is it approaching a target state?
How many days remain before a maturity threshold?
```

That is the bigger idea behind this repo.

---

## ⏳ Why the temporal angle is justified

The 2025 tissue-engineered cartilage maturity paper used constructs cultured for 7 or 28 days. It predicted GAG and DNA, then used the predicted GAG/DNA ratio to distinguish immature from more mature constructs.

That is useful work, but the time design is still endpoint-heavy. Two time points do not fully test whether a model can track gradual biological change.

The paper itself points towards the need for closer time points and continuous monitoring. That is the opening for this project.

A second 2025 culture-medium study moves closer to temporal monitoring by collecting conditioned medium every 3 days over 28 days, then linking NIR spectral behaviour to biomarker release patterns such as hyaluronan, lactate, and collagen.

So the gap is clear:

```text
Endpoint model:
day 7 versus day 28

Temporal model:
day 1, day 4, day 7, day 10, ..., day 28
```

One asks whether two states differ.
The other asks how the system moves.

That second question is harder. It is also more useful.

---

## 🌍 Dataset choice

For this first repo, I use the Jena Climate dataset.

The original task is weather forecasting. I use it because it has the exact machine-learning structure I need to practise:

```text
timestamp -> vector of measurements -> future target
```

Each time point contains a vector of sensor readings, such as temperature, pressure, humidity, wind speed, and other weather-related variables. The values change over time, and the future depends partly on recent history.

That is close to the mathematical structure of longitudinal spectroscopy:

```text
culture time -> spectral vector -> future biomarker or maturity score
```

In a tissue-engineering version, I would replace:

```text
weather sensor vector at time t
```

with:

```text
NIR/Raman spectrum at time t
```

or:

```text
culture-medium biomarker vector at time t
```

The model does not care whether the vector comes from weather sensors or a spectrometer. The domain matters for interpretation, but the temporal learning workflow stays similar.

Still, I should be honest: Jena Climate does not prove tissue maturity monitoring. It only helps me learn the sequence-modelling machinery without inventing fake tissue data.

---

## 🧠 The core machine-learning problem

At each time step, I have a measurement vector:

```text
x_t = [x_t1, x_t2, x_t3, ..., x_td]
```

where:

```text
t = time index
d = number of measured variables
```

For weather data, this vector may contain pressure, temperature, humidity, and wind variables.

For spectroscopy, this vector could be:

```text
x_t = [I_lambda1, I_lambda2, I_lambda3, ..., I_lambdad]
```

where each value is spectral intensity at one wavelength or Raman shift.

The modelling goal is:

```text
Given past measurements, predict a future value.
```

So instead of using only the current reading:

```text
x_t -> y_t
```

I use a window of recent readings:

```text
[x_(t-L+1), x_(t-L+2), ..., x_t] -> y_(t+h)
```

where:

```text
L = lookback window length
h = forecast horizon
```

Example:

```text
last 24 time steps -> temperature 6 hours ahead
```

Tissue-engineering version:

```text
last 5 spectral measurements -> GAG/DNA ratio at the next culture day
```

or:

```text
last 7 culture-medium spectra -> days remaining until target maturity
```

---

## 🧮 Sliding windows, the main trick

A raw time series is not automatically supervised learning data.

I have to convert it.

Suppose I have this sequence:

```text
x_1, x_2, x_3, x_4, x_5, x_6, x_7
```

If I choose a lookback window of 3 and predict the next value, I create samples like:

```text
[x_1, x_2, x_3] -> x_4
[x_2, x_3, x_4] -> x_5
[x_3, x_4, x_5] -> x_6
[x_4, x_5, x_6] -> x_7
```

That is how I teach the model to use recent history.

In matrix form, one training input becomes:

```text
X_i in R^(L x d)
```

where:

```text
L = number of past time steps
d = number of variables per time step
```

The target can be:

```text
y_i in R
```

for one future value, or:

```text
y_i in R^k
```

for several future values.

For tissue monitoring, this target could be:

```text
future GAG
future DNA
future GAG/DNA
future maturity score
days to threshold
```

The sliding-window step is boring but important. If I get this wrong, the model learns nonsense.

---

## 📐 Baseline model: Dense network

The dense baseline does not understand time by itself.

To feed it temporal data, I flatten the window:

```text
X_i in R^(L x d)
```

into:

```text
z_i in R^(L*d)
```

Then the dense model applies layers like:

```text
h = ReLU(Wz + b)
y_hat = Wh + c
```

This gives me a simple benchmark.

If the dense model performs well, the dataset may contain strong signals that do not need specialised memory. If it performs poorly, sequence models may help.

I need this baseline because deep learning projects without baselines are easy to fool myself with.

---

## 🔁 LSTM model: what it remembers

An LSTM processes the sequence one time step at a time.

At each time step, it receives:

```text
x_t
```

and updates an internal hidden state:

```text
h_t
```

The key idea is that the LSTM does not treat the window as one flat block. It reads the measurements in order.

A simplified LSTM uses gates:

```text
f_t = sigmoid(W_f [h_(t-1), x_t] + b_f)
i_t = sigmoid(W_i [h_(t-1), x_t] + b_i)
o_t = sigmoid(W_o [h_(t-1), x_t] + b_o)
```

where:

```text
f_t = forget gate
i_t = input gate
o_t = output gate
```

It also builds a candidate memory:

```text
c_t_candidate = tanh(W_c [h_(t-1), x_t] + b_c)
```

Then it updates memory:

```text
c_t = f_t * c_(t-1) + i_t * c_t_candidate
```

and hidden state:

```text
h_t = o_t * tanh(c_t)
```

Plain English:

```text
forget some old information
write some new information
keep a hidden summary of the recent trajectory
```

That matters for spectral monitoring because maturity may not sit in one spectrum alone. The direction of change can matter.

A construct that is moving steadily towards maturity and a construct that has plateaued may show similar values at one time point but different recent histories.

---

## 🧊 ConvLSTM model: why I use it here

A ConvLSTM extends the LSTM idea by replacing matrix multiplications with convolutions.

A standard LSTM works with vectors. A ConvLSTM works better when each time step has local structure, such as an image, map, or grid.

The ConvLSTM gate equations look like this:

```text
f_t = sigmoid(W_xf * X_t + W_hf * H_(t-1) + b_f)
i_t = sigmoid(W_xi * X_t + W_hi * H_(t-1) + b_i)
o_t = sigmoid(W_xo * X_t + W_ho * H_(t-1) + b_o)
```

The `*` symbol means convolution in this section.

The cell update follows the same memory logic:

```text
C_t = f_t * C_(t-1) + i_t * C_t_candidate
H_t = o_t * tanh(C_t)
```

Why does this matter for spectroscopy?

Spectra have neighbouring structure.

Adjacent wavelengths are not random independent columns. Peaks, slopes, shoulders, and broad absorption bands spread across neighbouring wavelengths. So if I reshape a spectral vector into a pseudo-image or spectral grid, ConvLSTM can learn local spectral patterns over time.

This is not perfect.

A spectrum is not truly a 2D image unless I design the representation carefully. But the experiment helps me understand how a model can learn:

```text
local feature structure + temporal structure
```

That is the reason ConvLSTM belongs in this first-principles repo.

---

## 📉 Loss function and evaluation

For regression, I use mean squared error or mean absolute error.

Mean squared error:

```text
MSE = (1/n) sum((y_i - y_hat_i)^2)
```

Mean absolute error:

```text
MAE = (1/n) sum(abs(y_i - y_hat_i))
```

MSE punishes large mistakes more strongly because the error is squared. MAE is easier to read because it stays in the original unit.

If the target is days, then:

```text
MAE = 0.8
```

means the model misses by roughly 0.8 days on average.

That kind of error would be easy to explain in a tissue-monitoring setting. A model that predicts maturity timing within one day may be useful. A model that misses by ten days is not.

I also compare predicted and actual curves. Metrics alone hide failure modes.

---

## 🧪 What the notebook does

The notebook follows this order:

1. Download and inspect the Jena Climate dataset
2. Clean and normalise the multivariate time-series data
3. Build sliding windows for supervised temporal learning
4. Train a dense baseline model
5. Train an LSTM baseline model
6. Train a ConvLSTM baseline model by reshaping windows into pseudo-images
7. Compare losses and prediction curves
8. Save plots for the README or project report
9. Connect the result back to longitudinal spectroscopy and tissue monitoring

The notebook is written as a learning notebook, not just a training script. I comment the code heavily because I want to know what each tensor shape means.

---

## 🧱 Tensor shapes I expect to see

This part matters because many temporal projects fail through shape confusion.

Raw dataframe:

```text
rows = time points
columns = variables
```

After selecting features:

```text
data shape = (T, d)
```

where:

```text
T = number of time points
d = number of variables
```

After sliding windows:

```text
X shape = (N, L, d)
y shape = (N,)
```

where:

```text
N = number of training windows
L = lookback length
d = number of variables
```

Dense model input after flattening:

```text
(N, L*d)
```

LSTM input:

```text
(N, L, d)
```

ConvLSTM input after pseudo-image reshaping:

```text
(N, L, H, W, C)
```

where:

```text
H * W * C = d
```

If I later use spectra, one possible representation is:

```text
(N, L, wavelength_bins, 1, 1)
```

or a 2D spectral map if I engineer one.

---

## 📁 Repo structure

```text
Temporal ConvLSTM POC for Spectral Monitoring/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── temporal_convlstm_first_principles.ipynb
├── outputs/
│   └── plots/
└── models/
```

The original minimal structure still works, but I prefer this expanded layout because it keeps outputs and model files out of the notebook folder.

---

## ⚙️ Install

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install packages:

```bash
pip install -r requirements.txt
```

If TensorFlow gives issues on Windows, check the Python version first. TensorFlow can be picky. That is not a modelling problem; it is an environment problem.

---

## ▶️ Run

Start Jupyter:

```bash
jupyter notebook notebooks/temporal_convlstm_first_principles.ipynb
```

Then run the notebook from top to bottom.

Do not jump to the model cells first. The dataset, windowing, scaling, and shape checks must run before training.

---

## 📊 Expected outputs

The notebook should produce plots such as:

```text
raw time-series plot
target variable over time
training and validation loss curves
actual vs predicted future value
dense vs LSTM vs ConvLSTM comparison
```

For the tissue-engineering translation, the most useful plot would be:

```text
predicted maturity trajectory against actual maturity trajectory
```

That is the plot I would want in a thesis discussion.

Not just accuracy.

A curve.

---

## 💡 What I want to demonstrate from this repo

This repo should prove that I understand:

- what makes a problem temporal rather than static
- how sliding windows convert sequences into supervised learning samples
- why a train/test split must respect time order
- what an LSTM stores in its hidden state
- what ConvLSTM adds when features have local structure
- how temporal modelling could support longitudinal spectroscopy
- why endpoint classification is easier than trajectory modelling
- how to explain tensor shapes without hiding behind model names

That is the real value of the project.

---

## 🚀 How I would translate this later

Once I understand the notebook properly, the next step is straightforward.

I would replace:

```text
weather variables
```

with:

```text
repeated NIR or Raman spectra from tissue constructs
```

Then I would replace the target:

```text
future temperature
```

with one of these:

```text
future GAG
future DNA
future GAG/DNA ratio
future collagen marker
future lactate level
culture day
days to maturity threshold
```

The modelling pipeline would look like this:

```text
spectra over past k days
-> temporal model
-> future biomarker or maturity estimate
```

A better future version would include repeated measurements from the same construct over time. That matters because different constructs can start from different baselines.

The clean experimental design would look like:

```text
Construct A: day 1, day 4, day 7, day 10, ...
Construct B: day 1, day 4, day 7, day 10, ...
Construct C: day 1, day 4, day 7, day 10, ...
```

Then I would split by construct, not by random row.

That prevents leakage.

---

## ⚠️ Limitations

This repo is a proof of concept.

It does not prove tissue-engineering maturity prediction because the dataset is weather data. It teaches the temporal modelling workflow.

Main limitations:

- the dataset is not biological
- there are no spectra in the current version
- ConvLSTM uses a pseudo-image representation, which may not match real spectral physics
- model performance on Jena Climate does not transfer automatically to tissue constructs
- real tissue data would need repeated measurements from the same sample or matched culture batches

I should not oversell it.

The correct claim is:

```text
I built a first-principles temporal modelling workflow that can later be adapted to longitudinal spectroscopy.
```

The wrong claim is:

```text
I solved tissue maturity monitoring.
```

That would fall apart in one serious question.

---

## 🔬 Future improvements

The next sensible upgrades are:

- replace Jena Climate with a real longitudinal spectroscopy or bioprocess dataset
- add a days-to-threshold target rather than only future-value prediction
- compare LSTM, GRU, Temporal CNN, ConvLSTM, and Transformer baselines
- add uncertainty estimates so the model says when it is unsure
- split by sample or batch, not by random rows
- build a small streaming demo where new spectra update the predicted maturity day

The streaming version is the one that excites me most.

It would look like this:

```text
new spectrum arrives
-> preprocess
-> update rolling window
-> predict maturity state
-> update plot
```

That is close to real temporal monitoring.

---

## 📌 Thesis-style framing

If this work later becomes part of a thesis, I would frame it as:

```text
Machine Learning-Based Temporal Monitoring of Tissue-Engineered Construct Maturation Using Spectroscopic Data
```

A shorter version:

```text
Temporal Spectroscopic Monitoring of Tissue-Engineered Constructs Using Machine Learning
```

The first title is stronger because it includes the method, the time component, the tissue-engineering application, and the target process.

---

## 🧾 One-sentence summary

I built this repo to learn how temporal models handle repeated multivariate signals, so I can later apply the same logic to non-destructive spectroscopic monitoring of tissue-engineered construct maturation.
