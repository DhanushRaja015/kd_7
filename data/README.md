# Data Directory

## Purpose

This directory holds electricity datasets used by the NILM application.

## Structure

```
data/
├── README.md          ← this file
└── sample/
    └── sample_nilm_data.csv   ← synthetic demo dataset (NOT real-world data)
```

## Dataset Requirements

Your CSV must contain at minimum:

| Column | Description |
|--------|-------------|
| `timestamp` | Date and time of each reading (parseable by pandas) |
| `aggregate_w` | Total household power in watts |

Optional appliance columns (required for supervised training):

| Column | Appliance |
|--------|-----------|
| `refrigerator` | Refrigerator power (W) |
| `ceiling_fan` | Ceiling fan power (W) |
| `air_conditioner` | Air conditioner power (W) |
| `washing_machine` | Washing machine power (W) |
| `electric_kettle` | Electric kettle power (W) |

Column names are **configurable** through the Dataset Management page.

## Recommended Public Datasets

### UK-DALE
- **URL:** https://jack-kelly.com/data/
- **Format:** HDF5 files with 1-second and 6-second aggregate readings
- **License:** Creative Commons Attribution 4.0
- **Appliances:** fridge, kettle, washing machine, dishwasher, microwave, etc.
- **Import note:** Use the dataset's `convert_to_csv` utilities or write a custom
  adapter.  Column names and units must be mapped to this project's schema via
  the Dataset Management page.

### REFIT
- **URL:** https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements
- **Format:** CSV files, 8-second intervals
- **License:** Creative Commons Attribution 4.0
- **Import note:** Align house-level aggregate and appliance channels by timestamp.

## Preparing Your Dataset

1. Export your dataset to CSV format.
2. Ensure the file has a parseable timestamp column and a numeric aggregate power column.
3. Include appliance columns if you want supervised training and ground-truth evaluation.
4. Upload via the **Dataset Management** page and configure column mapping.

## Sampling Rate Notes

- 1-minute or finer resolution is recommended for good disaggregation.
- The application supports resampling; set the interval in Dataset Management.
- Very low-frequency data (e.g., 30-minute intervals) will miss short appliance events.

## Privacy

Electricity profiles can reveal household routines and occupancy.
Keep raw datasets local and do not upload identifiable data to public services.
