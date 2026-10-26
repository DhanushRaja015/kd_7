# Machine Learning-Based Non-Intrusive Load Monitoring (NILM)

## 1. Project Overview

**Project title:** Machine Learning-Based Non-Intrusive Load Monitoring
for Appliance-Level Energy Disaggregation

Non-Intrusive Load Monitoring (NILM) is a technique that estimates the
energy consumption or operating state of individual electrical
appliances using measurements collected at a single point, such as a
home's main electricity meter. Instead of installing a separate sensor
on every appliance, NILM analyzes the combined electrical load and uses
data-driven models to estimate appliance-level consumption.

This project develops a Python-based NILM application with a
machine-learning pipeline and an interactive dashboard. It is intended
as an academic prototype for EEE, machine learning, and
energy-monitoring applications.

> **Important:** Appliance-level outputs are estimates, not direct
> measurements. Results depend on the dataset, sampling rate,
> preprocessing, and model performance. Do not present simulated or
> predicted values as verified real-world measurements.

## 2. Objectives

-   Load and inspect aggregate and appliance-level electricity datasets.
-   Clean missing, invalid, and duplicate readings.
-   Align timestamps and resample data to a common interval.
-   Train and evaluate a machine-learning model for appliance-level
    energy disaggregation.
-   Estimate appliance power or on/off states from aggregate readings.
-   Compare predicted appliance consumption with ground-truth labels
    when available.
-   Display total and appliance-level consumption, trends, and estimated
    energy cost.
-   Export prediction results and evaluation metrics.
-   Document model limitations, assumptions, and reproducible setup
    steps.

## 3. Proposed Features

### Data and preprocessing

-   Upload a supported CSV dataset.
-   Support timestamped aggregate power readings and appliance submeter
    readings where available.
-   Validate required columns and show clear errors for malformed files.
-   Handle missing values, duplicates, irregular timestamps, and
    resampling.
-   Split data chronologically into training, validation, and test sets
    to reduce time-series leakage.

### Machine learning

-   Include a simple baseline, such as a threshold-based estimate or
    aggregate-proportional baseline.
-   Train a suitable initial model, such as Random Forest, for appliance
    state classification or power estimation.
-   Keep model code modular so that sequence models such as a 1D CNN or
    LSTM can be added later.
-   Save and load trained models and preprocessing configuration.
-   Report metrics on a held-out test set only.

### Dashboard

-   Show aggregate power over time.
-   Show estimated appliance-level power and energy.
-   Show appliance activity states where the model supports them.
-   Show estimated energy cost using a user-configurable electricity
    tariff.
-   Display model evaluation metrics and prediction-versus-ground-truth
    plots when labels exist.
-   Allow users to download predictions and summary reports as CSV.

## 4. Suggested Technology Stack

-   **Language:** Python 3.10 or later
-   **Data processing:** pandas, NumPy
-   **Machine learning:** scikit-learn
-   **Visualization:** Plotly and/or Matplotlib
-   **Dashboard:** Streamlit
-   **Model persistence:** joblib
-   **Testing:** pytest
-   **Version control:** Git and GitHub

Use compatible, pinned dependency versions in `requirements.txt`. Avoid
adding unnecessary dependencies.

## 5. Dataset Strategy

Start with a public dataset such as **UK-DALE** or **REFIT**, after
checking the dataset's access conditions, license, available channels,
sampling frequency, and file format. These datasets contain household
electricity measurements and appliance-level readings suitable for
research.

A practical alternative for development is a clearly labeled sample or
synthetic dataset generated for testing the pipeline. Synthetic data is
useful for verifying the application but must not be used to claim
real-world model accuracy.

Keep raw datasets out of version control unless their license explicitly
permits redistribution. Document the official source, access
instructions, and expected file schema.

### Suggested input schema

The exact schema should be configurable. A basic aligned CSV may
contain:

  -----------------------------------------------------------------------
  Column                              Meaning
  ----------------------------------- -----------------------------------
  `timestamp`                         Date and time of the reading

  `aggregate_power`                   Total measured household power, in
                                      watts

  `fridge_power`                      Ground-truth refrigerator power, in
                                      watts, if available

  `washing_machine_power`             Ground-truth washing machine power,
                                      in watts, if available

  `kettle_power`                      Ground-truth kettle power, in
                                      watts, if available

  `ac_power`                          Ground-truth air-conditioner power,
                                      in watts, if available
  -----------------------------------------------------------------------

Appliance columns are labels for training and evaluation; they must not
be used as model inputs when predicting those same appliances. For
deployment-style inference, the model input should use aggregate
measurements and features available at prediction time.

## 6. Modeling Approach

1.  Inspect the dataset and document its sampling interval and units.
2.  Align aggregate and appliance readings by timestamp.
3.  Resample to a justified interval appropriate for the dataset.
4.  Create time-series windows and aggregate-only features, such as
    recent aggregate power, rolling mean, rolling maximum, and
    time-of-day features where justified.
5.  Establish a baseline.
6.  Train one or more models:
    -   **Classification:** estimate whether an appliance is on or off.
    -   **Regression:** estimate appliance power in watts.
7.  Split chronologically into training, validation, and test periods.
8.  Tune using training/validation data only.
9.  Evaluate once on the held-out test period.
10. Save the model, feature configuration, and evaluation report.

Avoid random row-level splitting for time-series data when it would
allow near-identical adjacent readings to leak across sets. Make sure
the training pipeline does not use appliance ground-truth values as
features during inference.

## 7. Evaluation Metrics

Choose metrics based on the task and report them per appliance.

-   **MAE (Mean Absolute Error):** average absolute power-estimation
    error in watts.
-   **RMSE (Root Mean Squared Error):** penalizes larger errors more
    heavily.
-   **Precision, recall, and F1-score:** evaluate appliance on/off
    classification.
-   **Energy error:** compare estimated and measured energy over a
    defined period, in kWh.
-   **Inference time:** measure prediction latency if real-time behavior
    is claimed.

Always compare the machine-learning model against the baseline. Explain
limitations caused by sampling frequency, overlapping appliance
signatures, missing data, and limited training examples.

## 8. Energy and Cost Calculations

For power readings in watts sampled at interval `Δt` seconds:

`Energy (kWh) = sum(power_W × Δt_seconds) / 3,600,000`

`Estimated cost = Energy (kWh) × tariff_per_kWh`

For irregular sampling, calculate each interval using its actual elapsed
time. Clearly label cost as an estimate and let the user configure the
tariff. Do not hard-code a tariff as universally correct.

## 9. Proposed Project Structure

``` text
nilm-ml-project/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── README.md
│   └── sample/
├── models/
│   └── .gitkeep
├── reports/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   └── energy.py
└── tests/
    ├── test_preprocessing.py
    ├── test_energy.py
    └── test_predictions.py
```

The implementation may refine this structure, but keep data loading,
preprocessing, training, inference, evaluation, and UI code separated.

## 10. Installation and Setup

1.  Install Python 3.10 or later.
2.  Clone or download the repository.
3.  Create and activate a virtual environment.
4.  Install dependencies.
5.  Place a legally obtained dataset in the documented data directory or
    upload it through the application.
6.  Run the dashboard.

Example commands:

``` bash
python -m venv .venv
```

Windows:

``` bash
.venv\Scripts\activate
```

macOS/Linux:

``` bash
source .venv/bin/activate
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

Run the application:

``` bash
streamlit run app.py
```

Train the model using the documented training command or the training
interface implemented by the project. The README should be updated to
match the actual commands and files generated by the finished
implementation.

## 11. Expected Results

The completed application should:

-   Validate and visualize aggregate electricity data.
-   Train a reproducible model using labeled appliance readings.
-   Estimate appliance power or state from aggregate-only inputs.
-   Compare predictions with ground truth where available.
-   Report baseline and model metrics on held-out data.
-   Estimate appliance energy and cost with clearly stated assumptions.
-   Export predictions and evaluation summaries.
-   Handle invalid files and missing data gracefully.

Actual accuracy must be measured from the chosen dataset. Do not claim a
specific accuracy before running and documenting a valid evaluation.

## 12. Limitations

-   Different appliances can have similar electrical signatures.
-   Low-frequency datasets may miss short appliance events.
-   Simultaneous appliance use makes disaggregation more difficult.
-   Model performance can change between homes and electrical
    environments.
-   Synthetic data does not establish real-world performance.
-   Energy cost estimates depend on the tariff and measurement coverage.
-   This academic prototype is not a certified meter, billing system, or
    electrical safety device.

## 13. Future Enhancements

-   Add 1D CNN or LSTM sequence models.
-   Add transfer learning or fine-tuning for a new household.
-   Support real-time aggregate readings from a smart meter where lawful
    and technically available.
-   Add model comparison and experiment tracking.
-   Add uncertainty indicators and low-confidence warnings.
-   Support additional appliances and household datasets.
-   Add authenticated multi-user access only if there is a clear
    deployment requirement.

## 14. Responsible Use and Privacy

Electricity patterns can reveal household routines. Use datasets with
appropriate permissions, minimize collected information, and avoid
uploading identifiable household data to public services. Clearly
disclose that predictions may be wrong and should not be used as the
sole basis for billing or safety-critical decisions.

## 15. Academic Deliverables

-   Source code and requirements file
-   Dataset source and schema documentation
-   Data preprocessing and exploratory analysis
-   Trained model and baseline comparison
-   Test-set metrics and plots
-   Dashboard screenshots
-   Test cases and results
-   Project report with methodology, results, limitations, and future
    scope

## 16. Conclusion

This project demonstrates how machine learning can estimate
appliance-level electricity consumption from a combined household load.
By combining careful time-series preprocessing, a leakage-aware
evaluation process, and an interactive dashboard, the project connects
electrical energy monitoring with practical machine-learning software
development.
