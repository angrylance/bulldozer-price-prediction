# Bulldozer Price Prediction

Machine learning regression project predicting bulldozer auction sale prices using historical sales data, machine specifications, and auction information.

The project demonstrates an end-to-end machine learning workflow, including:

* Exploratory Data Analysis (EDA)
* Feature engineering
* Missing value handling
* Categorical feature encoding
* Time-based train/validation splitting
* Baseline model training
* Hyperparameter tuning
* Model evaluation
* Test prediction generation
* Feature importance analysis
* Model export with Joblib

The primary goal was to build a regression model that predicts future bulldozer sale prices while maintaining a reproducible and engineering-focused machine learning workflow.

---

## Project Overview

Used equipment prices can vary significantly depending on machine age, product type, configuration, usage history, and auction timing.

This project uses historical bulldozer auction data to train a machine learning model capable of estimating future sale prices.

Problem type:

* Supervised Machine Learning
* Regression

Target variable:

* `SalePrice`

Primary evaluation metric:

* Root Mean Squared Log Error (RMSLE)

---

## Dataset

Dataset source:

* Blue Book for Bulldozers Kaggle Competition

Dataset characteristics:

* ~412,698 training examples
* 53 original features
* 103 features after preprocessing
* Historical auction records through 2011
* Validation data from January 1, 2012 to April 30, 2012
* Test data from May 1, 2012 to November 2012

Example feature groups:

| Feature Group | Description |
| --- | --- |
| Machine specifications | Model, product class, size, configuration |
| Auction metadata | Sale ID, auctioneer ID, sale date |
| Product information | Product group, product class, enclosure type |
| Usage information | Machine hours and usage-related fields |
| Engineered date features | Sale year, month, day, day of week, day of year |

The raw dataset is not included in this repository due to size and licensing. It can be downloaded from Kaggle:

```text
https://www.kaggle.com/competitions/bluebook-for-bulldozers
```

---

## Exploratory Data Analysis

The EDA phase focused on:

* Dataset structure and quality checks
* Target variable distribution
* Missing value patterns
* Data type inspection
* Sale date analysis
* Preparation for time-based validation

The target variable `SalePrice` is right-skewed:

```text
count    412698
mean      31215
std       23141
min        4750
25%       14500
50%       24000
75%       40000
max      142000
```

Key observations:

* Most bulldozers are sold in the lower-to-mid price range.
* A smaller number of expensive machines create a long right tail.
* The mean sale price is higher than the median sale price.
* Percentage-based error is more meaningful than raw absolute error, which makes RMSLE suitable for this task.

---

## Feature Engineering

The original `saledate` column was parsed as a datetime feature and expanded into several time-based features:

* `saleYear`
* `saleMonth`
* `saleDay`
* `saleDayOfWeek`
* `saleDayOfYear`

These features help the model capture long-term market trends and seasonal auction patterns.

After extracting these features, the original `saledate` column was removed to avoid redundant input.

---

## Data Preprocessing

The dataset required preprocessing before model training because it contained missing values and categorical variables stored as strings.

### Numerical Features

Missing numerical values were filled using median imputation.

For numerical columns with missing values, additional binary indicator columns were created.

Example:

```text
MachineHoursCurrentMeter_is_missing
```

This preserves information about whether a value was originally missing, which can sometimes carry predictive signal.

### Categorical Features

String-based features were converted into pandas categorical values and then encoded numerically.

This was necessary because Scikit-Learn models require numerical input.

### Processed Dataset Persistence

The processed dataset was saved in csv format.

Persisting the processed dataset improves reproducibility and separates preprocessing from modeling, which is closer to how production machine learning workflows are usually organized.

---

## Train / Validation Split

A time-based split was used instead of a random split.

| Split | Time Period |
| --- | --- |
| Training data | Data through the end of 2011 |
| Validation data | January 1, 2012 to April 30, 2012 |

This prevents future data from leaking into training and provides a more realistic estimate of model performance on unseen future auctions.

---

## Model

The main model used in this project was:

* Random Forest Regressor

Random Forest was selected because it performs well on tabular data and can model non-linear relationships without requiring extensive feature scaling.

---

## Baseline Model

The baseline model was trained on a subset of the dataset to speed up experimentation:

```python
RandomForestRegressor(
    n_jobs=-1,
    random_state=42,
    max_samples=10000
)
```

Baseline validation performance:

| Metric | Score |
| --- | ---: |
| MAE | 7,144 |
| RMSLE | 0.293 |
| R² | 0.833 |

The baseline model provided a strong starting point and established a reference for later improvements.

---

## Hyperparameter Tuning

Model performance was improved using `RandomizedSearchCV`.

The search space included:

```python
rf_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": ["sqrt", 0.3, 0.5, 0.7],
    "max_samples": [10000]
}
```

Randomized search configuration:

```python
n_iter=50
cv=5
```

Runtime:

```text
Wall time: ~28 minutes
CPU time: ~3 hours
```

Tuning was performed using `max_samples=10000` to keep experimentation time reasonable.

Tuned validation performance:

| Metric | Baseline | Tuned |
| --- | ---: | ---: |
| MAE | 7,144 | 7,087 |
| RMSLE | 0.293 | 0.292 |
| R² | 0.833 | 0.839 |

The tuned model produced a modest but measurable improvement over the baseline.

---

## Final Model

The final model was trained on the full training dataset using the best-performing hyperparameters.

Final model configuration:

```python
RandomForestRegressor(
    n_estimators=200,
    max_depth=30,
    max_features=0.7,
    min_samples_leaf=1,
    min_samples_split=2,
    max_samples=None,
    random_state=42,
    n_jobs=-1
)
```

Final validation performance:

| Metric | Score |
| --- | ---: |
| MAE | 5,977 |
| RMSLE | 0.248 |
| R² | 0.881 |

Training the final model on the full dataset significantly improved validation performance compared to the subset-trained models.

---

## Results

### Validation Performance

| Model | MAE | RMSLE | R² |
| --- | ---: | ---: | ---: |
| Baseline Random Forest | 7,144 | 0.293 | 0.833 |
| Tuned Random Forest | 7,087 | 0.292 | 0.839 |
| Final Random Forest | 5,977 | 0.248 | 0.881 |

The final Random Forest model achieved the strongest validation performance across all tracked metrics.

---

## Test Predictions

The test dataset does not include target values, so it cannot be evaluated directly in the notebook.

Instead, the final model was used to generate predictions for all test records.

The prediction file was exported in Kaggle submission format:

```text
SalesID,SalePrice
```

---

## Test Feature Alignment

During test preprocessing, the training set contained the following feature:

```text
auctioneerID_is_missing
```

This feature was not automatically created in the test set because the missing-value pattern differed between training and test data.

To solve this, the missing column was manually added and the test dataset was aligned to the training feature order:

```python
df_test["auctioneerID_is_missing"] = False
df_test = df_test.copy()
df_test = df_test[X_train.columns]
```

This step is important because production inference data must match the feature structure used during model training.

---

## Feature Importance

Feature importance was analyzed using the trained Random Forest model.

The most influential features included:

* `YearMade`
* `ProductSize`
* `saleYear`
* `fiSecondaryDesc`
* `Enclosure`
* `fiProductClassDesc`

These results are consistent with the business problem: used equipment prices are strongly influenced by machine age, size, configuration, product class, and market timing.

Feature importance does not prove causality, but it is useful for model inspection and sanity-checking whether the model learned meaningful patterns.

---

## Model Export

The final trained model was exported using Joblib:

```python
joblib.dump(
    final_model,
    "models/bulldozer-price-predictor.joblib"
)
```

The model was then loaded back and evaluated to confirm that the serialized artifact could be reused without retraining.

The trained model artifact is not included in this repository because the full Random Forest model is too large for GitHub.

To regenerate the model artifact, run the notebook from start to finish.

---

## Future Improvements

Potential next steps include:

### Experiment Tracking

Track model runs, metrics, parameters, and artifacts using MLflow.

### Containerization

Package the inference workflow with Docker for reproducible deployment.

### Pipeline Automation

Convert notebook logic into reusable Python scripts and automate preprocessing, training, and inference steps.

### CI/CD

Add automated checks for formatting, dependency installation, and notebook execution.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-Learn
* Joblib
* Jupyter Notebook

---

## Repository Structure

```text
bulldozer-price-prediction/
│
├── data/
│   └── README.md
│
├── models/
│   └── .gitkeep
│
├── notebooks/
│   └── bulldozer_price_predictor.ipynb
│
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

Notes:

* Raw dataset files are excluded from Git.
* Processed dataset files are excluded from Git.
* Model artifacts are excluded from Git due to size.
* The notebook contains the full workflow required to reproduce the project.

---

## Author

Eugene Anufriev, Senior Systems Reliability Engineer at Nutanix, AI/ML, AI Infrastructure and MLOps Enthusiast.
