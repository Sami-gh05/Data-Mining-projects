# Data Mining — Course Projects

Two end-to-end data mining projects completed for the **Data Mining** course at
Amirkabir University of Technology (Tehran Polytechnic).

The first project is an exploratory analysis and preprocessing pipeline on a large
apartment-rental dataset. The second is a full supervised-learning pipeline on a loan
approval dataset, comparing six classifiers with cross-validated hyperparameter tuning.

Notebooks are written in Persian (RTL markdown) with English code and figure labels.

---

## Projects

### 1. Apartment Rent — Exploratory Data Analysis & Preprocessing

`notebooks/01_apartment_rent_eda.ipynb`

A ~99k-record US rental listings dataset taken from raw to analysis-ready, then explored
for the geographic and structural drivers of rent.

**Preprocessing**
- Duplicate detection and removal; dropping non-informative columns
- Missing-value strategy chosen per column: sentinel fill for `amenities` and
  `pets_allowed`, median imputation for `bathrooms` / `bedrooms` (chosen over mean
  because both are discrete and outlier-prone), row removal where `cityname`,
  `latitude` or `price` is missing
- Unix timestamp converted to datetime, with year / month / day extracted into
  separate columns
- Outlier removal on `price` using the IQR rule (Q1 − 1.5·IQR, Q3 + 1.5·IQR)

**Analysis**
- Distribution shape: skewness and kurtosis, histogram with KDE overlay, boxplot
- Three sampling strategies compared — simple random, systematic (every 10th record)
  and stratified — evaluated by how faithfully each preserves the original city
  proportions
- Discretization of `price` into cheap / medium / expensive, with the binning method
  chosen to suit a right-skewed distribution
- Correlation matrix heatmap over the numeric features
- Price per square foot (`pps`) derived, then used to rank the ten most expensive states
- Geospatial scatter of Dallas listings coloured by `pps`, overlaid on a map image, to
  test the effect of distance from the city centre
- Within-state comparison of `pps` across cities in California

**Stack:** pandas · NumPy · Matplotlib · seaborn

---

### 2. Loan Approval — Classification Pipeline

`notebooks/02_loan_approval_classification.ipynb`

A binary classification problem (approve / reject) over ~4,270 loan applications with
13 features, built as a reusable train-and-evaluate pipeline rather than six one-off
model fits.

**Preprocessing & feature work**
- Boxplot and IQR-based outlier handling (`remove_outliers_iqr`)
- Encoding of the categorical columns `education`, `self_employed` and `loan_status`
- Feature relevance measured two ways — Pearson correlation and mutual information —
  with the results compared, since MI also captures non-linear dependence
- Z-score standardization (`StandardScaler`), chosen over min-max because of the skewed
  distributions and the very different feature scales
- Class imbalance addressed with SMOTE

**Modelling**
- Six classifiers compared: Logistic Regression, Decision Tree, Random Forest, KNN, SVM
  and XGBoost
- K-Fold cross-validation with `GridSearchCV` hyperparameter search
  (`perform_grid_search`), tuning depth, estimator count, neighbour count, kernel and
  learning rate depending on the model
- Shared evaluation helpers: `evaluate_model` and `train_and_evaluate_model`

**Evaluation**
- Accuracy, precision, recall and F1, with cross-validated and test scores compared to
  check for overfitting
- Confusion matrices, ROC curves and AUC
- t-SNE projection of the test set (`plot_tsne_results`) to visualise how each model
  separates the classes in 2D
- Feature importance analysis on the best-performing model

**Stack:** scikit-learn · XGBoost · imbalanced-learn · pandas · NumPy · Matplotlib · seaborn

---

## Repository structure

```
data-mining/
├── notebooks/
│   ├── 01_apartment_rent_eda.ipynb
│   └── 02_loan_approval_classification.ipynb
├── data/
│   ├── apartment_rent.csv
│   └── loan_approval.csv
├── images/                 # map overlays used by notebook 01
├── reports/                # exported HTML/PDF versions of the notebooks
├── requirements.txt
└── README.md
```

## Getting started

```bash
git clone https://github.com/Sami-gh05/data-mining.git
cd data-mining
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Notebooks expect to be run from the `notebooks/` directory and read data via
`../data/`. If you run them from the repository root instead, adjust the
`pd.read_csv` paths accordingly.

## Datasets

| File | Rows | Target / focus |
|---|---|---|
| `apartment_rent.csv` | ~99,800 raw → 99,399 after cleaning | `price` (regression / discretization) |
| `loan_approval.csv` | ~4,270 | `loan_status` (binary classification) |

Both are public datasets used for coursework.

