# ML-Based Production Time Estimation for Factory Process Optimization

> **Case Study No. 100 | Machine Learning Foundations (Sem-5, FY BTech)**  
> **Repository:** [AnoopG7/ML-Production-Time-Prediction](https://github.com/AnoopG7/ML-Production-Time-Prediction)

---

## 1. Project Overview

Accurate production time estimation is vital for manufacturing logistics, delivery scheduling, resource allocation, and shop-floor bottleneck elimination. In traditional manufacturing, plant managers rely on static heuristics or manual planning formulas that fail to account for worker density, operator experience, machine-specific calibration efficiencies, and ambient floor conditions.

This project develops an end-to-end Machine Learning pipeline that predicts continuous batch completion time (`production_time_hrs`) strictly from physical and operational factory variables. The final system is deployed as an interactive, multi-tab Streamlit web application.

### Key Achievements
- **38.7% Error Reduction**: The best machine learning model achieves a Mean Absolute Error (MAE) of **3.35 hours**, substantially outperforming static human planning schedules (**5.46 hours** MAE).
- **Leakage-Free Modeling**: Target leakage was systematically eliminated by excluding initial planning benchmarks (`planned_time_hrs`) from training features, forcing models to learn genuine physical manufacturing dynamics.
- **Ensemble Dominance**: Gradient boosting (**XGBoost**, $R^2 = 0.8764$) and bagging (**Random Forest**, $R^2 = 0.8425$) decisively outperform regularized linear models ($R^2 \approx 0.6011$), proving that non-linear interaction terms dominate discrete manufacturing processes.

---

## 2. Project Architecture & File Structure

The project is structured as a self-contained, single-directory repository with zero external path dependencies (no `/content/` folder required):

```
ML-Production-Time-Prediction/
├── README.md                         # Comprehensive project documentation
├── ProblemStatement.txt              # Official Case Study 100 specifications
├── main.ipynb                        # 33-cell self-contained notebook (EDA to export)
├── app.py                            # Multi-tab Streamlit web application
├── requirements.txt                  # Deployment dependencies for Streamlit Cloud
├── generate_artifacts.py             # Script to regenerate models and data
├── production_time_estimation_plan.md# Architecture & engineering notes
├── production_time_model.pkl         # Trained model artifact
├── scaler.pkl                        # Fitted StandardScaler artifact
├── label_encoders.pkl                # Dictionary of fitted LabelEncoders
├── feature_names.pkl                 # Exact ordered 16-feature signature
├── model_comparison.csv              # Benchmark metrics across all 7 models
└── factory_production_data.csv       # Cleaned dataset for Streamlit Data Explorer
```

---

## 3. Dataset Architecture & Data Quality Engineering

To ensure the project is fully reproducible without requiring external CSV uploads or API keys, the dataset is synthesized inline within [main.ipynb](main.ipynb).

- **Total Records**: 4,525 rows (4,500 base samples + 25 duplicate rows for testing deduplication)
- **Target Variable**: `production_time_hrs` (Continuous actual batch completion time in hours)
- **Underlying Process**: Non-linear multiplicative manufacturing function with 18% stochastic industrial variance, material rework factors, and ambient temperature slowdown penalties.

### Feature Dictionary
| Feature | Data Type | Category | Description |
|---|---|---|---|
| `product_type` | Categorical | Operational | Product variant (Widget_A, B, C, Gadget_X, Y) |
| `batch_size` | Integer | Operational | Total units scheduled in the batch (10 to 500) |
| `material_grade` | Categorical | Material | Raw material quality (Standard, Premium, Economy) |
| `machine_id` | Categorical | Equipment | Assigned machine unit (M01 to M10) |
| `operator_experience_yrs` | Float | Human | Lead operator experience (0.5 to 25.0 years) |
| `setup_time_min` | Float | Operational | Machine calibration time (5 to 60 minutes) |
| `complexity_score` | Integer | Product | Product difficulty rating (1=simple to 5=complex) |
| `ambient_temp_c` | Float | Environmental | Shop floor ambient temperature (15 to 45 deg C) |
| `humidity_pct` | Float | Environmental | Shop floor humidity percentage (20% to 100%) |
| `shift` | Categorical | Operational | Work shift (Morning, Afternoon, Night) |
| `day_of_week` | Categorical | Scheduling | Day scheduled (Monday through Sunday) |
| `num_workers` | Integer | Human | Workforce headcount assigned (1 to 8 workers) |
| `defect_rate_pct` | Float | Quality | Historical batch defect percentage (0% to 15%) |
| `planned_time_hrs` | Float | Benchmark | Static planning estimate (used strictly for business comparison) |
| `batch_per_worker` | Float | Engineered | Workload density: `batch_size / num_workers` |
| `setup_per_batch` | Float | Engineered | Setup overhead ratio: `setup_time_min / batch_size` |

### Injected Data Quality Issues & Cleaning Strategy
| Issue Injected | Prevalence | Cleaning & Resolution Strategy |
|---|---|---|
| Missing numeric values | ~6% experience, ~8% humidity, ~4% defect | Imputed using feature median values |
| Outliers in batch size & time | 30 extreme rows (up to 5,000 units / 200 hrs) | Capped using IQR factor thresholds (1.5x IQR) |
| Negative setup times | 10 negative entries (< 0 min) | Replaced negative entries with median setup time |
| Impossible sensor readings | 8 humidity values > 100% | Capped upper boundary at 100.0% |
| String casing inconsistencies | ~450 rows with mixed casing | Standardized to title case (`str.strip().str.title()`) |
| Categorical typos | 7 typos (e.g., Wiget_A, Gadgt_X) | Fixed via explicit typo dictionary mapping |
| Duplicate rows | 25 fully duplicated records | Identified and dropped (`drop_duplicates()`) |

---

## 4. Machine Learning Pipeline

### Feature Engineering & Preprocessing
1. **Categorical Encoding**: LabelEncoding applied to `product_type`, `machine_id`, `shift`, and `day_of_week`. One-hot encoding applied to `material_grade` (dropping first category to prevent collinearity).
2. **Operational Ratios**:
   - `batch_per_worker = batch_size / num_workers`
   - `setup_per_batch = setup_time_min / batch_size`
3. **Leakage Elimination**: The feature set $X$ strictly drops `['production_time_hrs', 'planned_time_hrs', 'time_deviation']`, retaining only the 16 physical operational inputs.
4. **Standard Scaling**: StandardScaler fit strictly on the 80% training set and applied to the 20% test set to prevent train-test contamination.

---

## 5. Model Evaluation & Benchmark Results

Seven distinct regression architectures were evaluated using 5-Fold Cross-Validation on the test partition:

| Rank | Model | Test R2 | MAE (hrs) | RMSE (hrs) | 5-Fold CV R2 Mean | Model Paradigm |
|:---:|---|:---:|:---:|:---:|:---:|---|
| 1 | **XGBoost** | **0.8764** | **2.90** | **4.33** | **0.8701 +- 0.0135** | Gradient Boosted Trees |
| 2 | **Random Forest (Tuned)** | **0.8425** | **3.35** | **4.90** | **0.8410 +- 0.0110** | Tuned Bagged Ensembles |
| 3 | **Random Forest (Base)** | **0.8357** | **3.39** | **5.00** | **0.8388 +- 0.0106** | Bagged Ensembles |
| 4 | **SVR (RBF Kernel)** | **0.8134** | **3.68** | **5.33** | **0.7729 +- 0.0162** | Non-Linear Support Vector |
| 5 | **Decision Tree** | **0.6894** | **4.48** | **6.87** | **0.7033 +- 0.0260** | Single Tree Baseline |
| 6 | **Linear Regression** | **0.6011** | **5.86** | **7.79** | **0.5618 +- 0.0248** | Ordinary Least Squares |
| 7 | **Ridge Regression** | **0.6011** | **5.86** | **7.79** | **0.5619 +- 0.0248** | L2 Regularized Linear |
| 8 | **Lasso Regression** | **0.6010** | **5.86** | **7.79** | **0.5620 +- 0.0248** | L1 Regularized Linear |

### Feature Importance & Selection Insights
Tree-based feature importance confirms that models prioritize physical operational drivers:
- **`batch_per_worker`**: 31.1%
- **`product_type_encoded`**: 26.3%
- **`batch_size`**: 26.1%
- **`complexity_score`**: 4.3%
- **`operator_experience_yrs`**: 3.2%
- **Noise Features** (`humidity_pct`, `day_encoded`): < 1.0%

### Business Impact Analysis: Human Scheduling vs. ML
```
============================================================
BUSINESS IMPACT ANALYSIS:
  Human Static Planner Schedule MAE: 5.46 hrs (R2: 0.5755)
  ML Best Model MAE:                 3.35 hrs (R2: 0.8425)
  Scheduling Error Reduction:        38.7% improvement!
============================================================
```

---

## 6. Streamlit Web Application

The production application ([app.py](app.py)) provides an operational interface for factory managers:

1. **Predict Tab**:
   - Dynamic inputs for batch size, product type, machine assignment, workers, experience, and environmental conditions.
   - Outputs: Predicted Batch Hours, Schedule Deviation Delta, Throughput (Units/Hour), and On-Time/Delayed Status.
   - Interactive Plotly Gauge chart displaying prediction vs planned deadline threshold.
2. **Model Insights Tab**:
   - Horizontal comparative bar charts across all 7 algorithms.
   - Complete metrics table with sorting by R2, MAE, RMSE, or CV score.
   - Feature importance bar chart for the production model.
3. **Data Explorer Tab**:
   - Preview of the factory dataset with statistical summaries, distribution metrics, and missing value indicators.
   - Custom CSV upload support.

---

## 7. Quickstart Guide

### Prerequisites
- Python 3.10+ (Python 3.12 recommended)

### Local Setup & Launch
```bash
# 1. Clone the repository
git clone https://github.com/AnoopG7/ML-Production-Time-Prediction.git
cd ML-Production-Time-Prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Streamlit application
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

### Running the Notebook
Open [main.ipynb](main.ipynb) in Jupyter Lab, VS Code, or Google Colab and click **Run All**. The notebook runs from end to end in under 2 minutes.

---

## 8. Academic Deliverables Checklist (Case Study 100)

- [x] **Problem Definition**: Formal statement, continuous regression formulation, and factory operational scope.
- [x] **Dataset & Quality Documentation**: Documented synthetic generation, injected data anomalies, and cleaning rationale.
- [x] **Exploratory Data Analysis**: 5 comprehensive visualization plots (target distribution, correlation heatmap, boxplots, categorical breakdown, scatter plots).
- [x] **Preprocessing & Scaling**: Categorical encoding, domain feature engineering, and leakage-free StandardScaler transformation.
- [x] **Model Comparison**: 7 algorithms benchmarked using 5-fold cross-validation.
- [x] **Hyperparameter Optimization**: RandomizedSearchCV tuning for ensemble depth, leaf splits, and estimators.
- [x] **ROC AUC Binary Evaluation**: Operational classification view ("Will the batch finish on time?").
- [x] **Interactive Streamlit App**: Multi-tab user interface with real-time inference, error delta indicators, and data inspection.
