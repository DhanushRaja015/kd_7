# Machine Learning-Based Non-Intrusive Load Monitoring (NILM)

> **Academic Prototype** — Appliance-level outputs are ML estimates, not direct
> electrical measurements.  Do not use as a billing system or safety device.

---

## Abstract

Non-Intrusive Load Monitoring (NILM) disaggregates a household's total electrical
load into per-appliance contributions using a single measurement point (e.g., the
main meter).  This project implements a complete NILM pipeline in Python, including
data ingestion, time-series preprocessing, feature engineering, Random Forest
modelling, energy estimation, and an interactive Streamlit dashboard.

---

## Problem Statement

Installing a separate sensor on every appliance is costly and intrusive.  NILM
provides appliance-level consumption insights using only the aggregate signal
already available at the smart meter — enabling energy disaggregation without
additional hardware.

---

## Objectives

- Load and validate aggregate and appliance-level electricity datasets.
- Clean, align, and resample time-series data without introducing bias.
- Split chronologically into training, validation, and test sets.
- Train a Random Forest regressor (power estimation) and classifier (ON/OFF) per appliance.
- Compare ML predictions against a median-power baseline on the held-out test set.
- Estimate appliance energy (kWh) and cost with a user-configurable tariff.
- Display results on a professional interactive dashboard.
- Export predictions, energy summaries, and evaluation metrics.

---

## System Architecture

```
┌───────────────────────────────────────────────────────────┐
│                     Streamlit Dashboard                    │
│  Overview │ Dataset │ Preprocess │ Train │ Predict │ ...   │
└────────────────────────┬──────────────────────────────────┘
                         │ calls
┌────────────────────────▼──────────────────────────────────┐
│                     src/  (backend)                        │
│  data_loader → preprocessing → features → train           │
│               predict → evaluate → energy                  │
└───────────────────────────────────────────────────────────┘
```

---

## Technology Stack

| Component | Library |
|-----------|---------|
| Language | Python 3.10+ |
| Dashboard | Streamlit 1.32 |
| Data processing | pandas 2.2, NumPy 1.26 |
| Machine learning | scikit-learn 1.4 |
| Visualisation | Plotly 5.20 |
| Model persistence | joblib 1.3 |
| Testing | pytest 8.1 |

---

## Dataset Requirements

**Minimum (aggregate-only):**

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | datetime | Reading time (any parseable format) |
| `aggregate_w` | float | Total household power in **watts** |

**With appliance labels (required for supervised training):**

| Column | Appliance |
|--------|-----------|
| `refrigerator` | Refrigerator (W) |
| `ceiling_fan` | Ceiling fan (W) |
| `air_conditioner` | Air conditioner (W) |
| `washing_machine` | Washing machine (W) |
| `electric_kettle` | Electric kettle (W) |

Column names are **configurable** — map any CSV column to the canonical keys via the
Dataset Management page.

**Supported public datasets:** UK-DALE, REFIT (see `data/README.md`).

---

## Project Structure

```
nilm-ml-project/
├── app.py                      ← Streamlit entry-point
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── README.md               ← Dataset acquisition instructions
│   └── sample/
│       └── sample_nilm_data.csv
├── models/                     ← Saved model artifacts (gitignored except .gitkeep)
│   └── .gitkeep
├── reports/                    ← Exported CSV/JSON reports
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── data_loader.py          ← CSV ingestion, validation, synthetic data
│   ├── preprocessing.py        ← Cleaning, chronological splitting
│   ├── features.py             ← Aggregate-only feature engineering
│   ├── train.py                ← RF models, baseline, save/load
│   ├── predict.py              ← Prediction pipeline
│   ├── evaluate.py             ← Metrics: MAE, RMSE, F1, energy error
│   └── energy.py               ← kWh / cost calculations, daily breakdown
└── tests/
    ├── __init__.py
    ├── test_data_loader.py
    ├── test_preprocessing.py
    ├── test_features.py
    ├── test_energy.py
    └── test_predictions.py
```

---

## Installation

### Prerequisites
- Python 3.10 or later
- pip

### Steps

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd nilm-ml-project

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Running the Application

```bash
streamlit run app.py
```

The dashboard opens at **http://localhost:8501** in your browser.

---

## Workflow

### Step 1 — Load Data
Go to **📂 Dataset Management**:
- Click **🧪 Use Sample Dataset** to load the built-in synthetic data, OR
- Upload your own CSV and configure column mapping.
- Download the sample CSV template if you need a format reference.

### Step 2 — Preprocess
Go to **⚙️ Data Preprocessing**:
- Adjust train / val / test fractions.
- Click **▶️ Run Preprocessing**.
- Review the split summary and feature preview.

### Step 3 — Train
Go to **🤖 Model Training**:
- Select appliances and configure Random Forest parameters.
- Click **🚀 Train Models**.
- Models are saved to `models/` automatically.

### Step 4 — Predict
Go to **🔮 Prediction Interface**:
- Select a date range and appliances.
- Click **▶️ Run Predictions**.

### Step 5 — Analyse Energy
Go to **⚡ Energy Analysis**:
- Set your electricity tariff.
- View per-appliance kWh, cost, daily trends, and discrepancy.

### Step 6 — Evaluate
Go to **📊 Model Evaluation**:
- Click **🔄 Evaluate on Test Set** to compute held-out metrics.
- Download results from **📥 Reports & Export**.

---

## Model Training (CLI alternative)

You can train from a Python script if preferred:

```python
import sys
sys.path.insert(0, ".")
from src.data_loader import generate_sample_dataset
from src.preprocessing import NILMPreprocessor, chronological_split
from src.train import NILMTrainer
from pathlib import Path

df = generate_sample_dataset(n_samples=1440, seed=42)
train, val, test = chronological_split(df)
prep = NILMPreprocessor()
train_c = prep.fit_transform(train)
val_c   = prep.transform(val)

trainer = NILMTrainer(
    appliance_keys=["refrigerator","ceiling_fan","air_conditioner",
                    "washing_machine","electric_kettle"],
    n_estimators=100,
)
trainer.fit(train_c, val_c)
trainer.save(Path("models"))
print("Trained:", trainer.trained_appliances())
```

---

## Running Tests

```bash
pytest tests/ -v
```

For coverage:

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

### What the tests verify
- CSV validation and schema checks
- Timestamp parsing and duplicate removal
- Chronological split sizes, order, and no overlap
- NILMPreprocessor: clipping, NaN fill, fit-before-transform guard
- Feature columns present, no appliance label leakage, bounded cyclical features
- Energy formula correctness (1 kWh = 1000 W × 3600 s)
- Cost calculation and zero/negative edge cases
- Model training on small fixture, prediction shape, no negative power
- ON/OFF state values are strictly {0, 1}
- Model save / load round-trip (predictions match)
- Baseline median estimation

---

## Model Evaluation Methodology

| Task | Metrics |
|------|---------|
| Power regression | MAE (W), RMSE (W), R² |
| ON/OFF classification | Precision, Recall, F1-score, Confusion matrix |
| Energy estimation | Absolute error (kWh), Relative error (%) |

- Metrics are computed on the **held-out test set only**.
- The ML model is compared against a **median-power baseline**.
- ML is reported as better only when test metrics support that claim.

---

## Energy & Cost Formula

```
Energy (kWh) = Σ [ Power_W × Δt_seconds ] / 3,600,000
Estimated cost = Energy_kWh × tariff_per_kWh
```

- For irregular timestamps, actual elapsed time between readings is used.
- Tariff is user-configurable (default ₹0.12/kWh).
- Aggregate disaggregation discrepancy is displayed; predictions are **not** forced to sum to the aggregate.

---

## Expected Outputs

| Output | Location |
|--------|----------|
| Interactive dashboard | http://localhost:8501 |
| Trained model files | `models/*.joblib` |
| Training summary | `models/training_summary.json` |
| Timestamped predictions CSV | Download from Reports page |
| Energy summary CSV | Download from Reports page |
| Evaluation metrics CSV/JSON | Download from Reports page |

---

## Limitations

- Different appliances can have similar power signatures.
- Low-frequency datasets miss short appliance events.
- Simultaneous appliance use makes disaggregation harder.
- Model trained on one household may not generalise to another.
- Synthetic data does not establish real-world accuracy.
- Cost estimates depend on tariff accuracy and data completeness.

---

## Future Enhancements

- 1D CNN or LSTM sequence models for better temporal modelling.
- Transfer learning for new households.
- Real-time aggregate feed from smart meter API.
- Experiment tracking and model comparison UI.
- Uncertainty / confidence indicators on predictions.

---

## Privacy

Electricity patterns reveal household routines and occupancy.
- Data is processed **locally** — nothing is sent to external servers.
- Do not upload identifiable household data to shared/public deployments.
- Predictions must not be used as the sole basis for billing or safety decisions.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: src` | Run `streamlit run app.py` from the project root directory |
| `No module named streamlit` | Run `pip install -r requirements.txt` |
| CSV upload fails | Check that timestamp and aggregate columns exist and are named correctly |
| Training fails — "No appliance labels" | Your CSV must contain labeled appliance columns for supervised training |
| Model not found after restart | Click **📂 Load Saved Models** on the Training page |
| Predictions look flat | Increase training data size or reduce `max_depth` to prevent overfitting |

---

## Academic Deliverables Checklist

- [x] Source code (`src/`)
- [x] Requirements file (`requirements.txt`)
- [x] Dataset schema documentation (`data/README.md`)
- [x] Data preprocessing and feature engineering
- [x] Trained model and baseline comparison
- [x] Test-set evaluation metrics
- [x] Interactive dashboard with Plotly charts
- [x] Automated test suite (`tests/`)
- [x] Export: predictions, energy summary, metrics

---

*Project: Machine Learning-Based Non-Intrusive Load Monitoring for Appliance-Level Energy Disaggregation*
