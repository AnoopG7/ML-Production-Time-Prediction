import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

print("Generating synthetic factory dataset...")
np.random.seed(42)
N = 4500

products = ['Widget_A', 'Widget_B', 'Widget_C', 'Gadget_X', 'Gadget_Y']
product_rates = {'Widget_A': 0.04, 'Widget_B': 0.07, 'Widget_C': 0.11,
                 'Gadget_X': 0.14, 'Gadget_Y': 0.09}
machines = [f'M{i:02d}' for i in range(1, 11)]
machine_eff = {'M01':0.95,'M02':0.88,'M03':1.05,'M04':0.78,'M05':1.12,
               'M06':0.92,'M07':1.00,'M08':0.85,'M09':1.08,'M10':0.72}
shifts = ['Morning', 'Afternoon', 'Night']
shift_factors = {'Morning': 1.0, 'Afternoon': 1.08, 'Night': 1.18}
materials = ['Standard', 'Premium', 'Economy']
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

data = {
    'order_id': range(1001, 1001 + N),
    'product_type': np.random.choice(products, N, p=[0.25, 0.20, 0.20, 0.15, 0.20]),
    'batch_size': np.random.randint(10, 500, N),
    'material_grade': np.random.choice(materials, N, p=[0.50, 0.30, 0.20]),
    'machine_id': np.random.choice(machines, N),
    'operator_experience_yrs': np.round(np.random.uniform(0.5, 25, N), 1),
    'setup_time_min': np.round(np.random.uniform(5, 60, N), 1),
    'complexity_score': np.random.randint(1, 6, N),
    'ambient_temp_c': np.round(np.random.normal(28, 5, N), 1),
    'humidity_pct': np.round(np.random.normal(55, 15, N), 1),
    'shift': np.random.choice(shifts, N, p=[0.40, 0.35, 0.25]),
    'day_of_week': np.random.choice(days, N),
    'num_workers': np.random.randint(1, 9, N),
    'defect_rate_pct': np.round(np.abs(np.random.normal(3, 3, N)), 2),
}
df = pd.DataFrame(data)

prod_times = []
for _, row in df.iterrows():
    base = product_rates[row['product_type']]
    meff = machine_eff[row['machine_id']]
    sf = shift_factors[row['shift']]
    cf = 0.7 + 0.12 * row['complexity_score']
    ef = max(0.65, 1.0 - 0.012 * row['operator_experience_yrs'])
    wf = 1.0 + 0.18 * (row['num_workers'] - 1)
    dp = 1.0 + 0.02 * row['defect_rate_pct']
    t = ((row['setup_time_min']/60) + row['batch_size'] * base * cf * dp)
    t = t / (meff * wf) * sf * ef * np.random.normal(1.0, 0.08)
    prod_times.append(round(max(0.3, t), 2))
df['production_time_hrs'] = prod_times
df['planned_time_hrs'] = np.round(df['production_time_hrs'] * np.random.uniform(0.85, 1.25, N), 2)

# Injected issues
for col, pct in [('operator_experience_yrs', 0.06),
                 ('humidity_pct', 0.08), ('defect_rate_pct', 0.04)]:
    idx = np.random.choice(N, size=int(N * pct), replace=False)
    df.loc[idx, col] = np.nan

df.loc[np.random.choice(N, 18, replace=False), 'batch_size'] = \
    np.random.choice([1500, 2000, 3000, 5000], 18)
df.loc[np.random.choice(N, 12, replace=False), 'production_time_hrs'] = \
    np.random.uniform(80, 200, 12).round(2)
df.loc[np.random.choice(N, 10, replace=False), 'setup_time_min'] = \
    np.random.uniform(-20, -1, 10).round(1)
df.loc[np.random.choice(N, 8, replace=False), 'humidity_pct'] = \
    np.random.uniform(102, 130, 8).round(1)

mixed_idx = np.random.choice(N, size=int(N * 0.10), replace=False)
df.loc[mixed_idx, 'material_grade'] = np.random.choice(
    ['standard', 'PREMIUM', 'economy', 'STANDARD', 'premium'], len(mixed_idx))

df = pd.concat([df, df.iloc[np.random.choice(N, 25, replace=False)]], ignore_index=True)
df.loc[np.random.choice(len(df), 7, replace=False), 'product_type'] = \
    np.random.choice(['Wiget_A', 'Gadgt_X', 'Widget_c'], 7)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Data Cleaning...")
df_clean = df.copy()
df_clean.drop_duplicates(inplace=True)

typo_map = {'Wiget_A': 'Widget_A', 'Gadgt_X': 'Gadget_X', 'Widget_c': 'Widget_C'}
df_clean['product_type'] = df_clean['product_type'].replace(typo_map)
df_clean['material_grade'] = df_clean['material_grade'].str.strip().str.title()

df_clean.loc[df_clean['setup_time_min'] < 0, 'setup_time_min'] = \
    df_clean.loc[df_clean['setup_time_min'] >= 0, 'setup_time_min'].median()
df_clean.loc[df_clean['humidity_pct'] > 100, 'humidity_pct'] = 100.0

for col in ['operator_experience_yrs', 'humidity_pct', 'defect_rate_pct']:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

def cap_outliers(data, column, factor=1.5):
    Q1, Q3 = data[column].quantile(0.25), data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = max(0, Q1 - factor * IQR)
    upper = Q3 + factor * IQR
    data[column] = data[column].clip(lower=lower, upper=upper)

for col in ['batch_size', 'production_time_hrs', 'defect_rate_pct', 'planned_time_hrs']:
    cap_outliers(df_clean, col)

df_clean.drop(columns=['order_id'], inplace=True, errors='ignore')

# Save dataset for Streamlit Data Explorer
df_clean.to_csv('factory_production_data.csv', index=False)
print("[OK] Saved factory_production_data.csv")

print("Feature Engineering & Encoding...")
df_processed = df_clean.copy()
le_product = LabelEncoder()
df_processed['product_type_encoded'] = le_product.fit_transform(df_processed['product_type'])

le_machine = LabelEncoder()
df_processed['machine_id_encoded'] = le_machine.fit_transform(df_processed['machine_id'])

le_shift = LabelEncoder()
df_processed['shift_encoded'] = le_shift.fit_transform(df_processed['shift'])

le_day = LabelEncoder()
df_processed['day_encoded'] = le_day.fit_transform(df_processed['day_of_week'])

df_processed = pd.get_dummies(df_processed, columns=['material_grade'], drop_first=True, dtype=int)
df_processed['batch_per_worker'] = df_processed['batch_size'] / df_processed['num_workers'].replace(0, 1)
df_processed['setup_per_batch'] = df_processed['setup_time_min'] / df_processed['batch_size'].replace(0, 1)
df_processed['time_deviation'] = df_processed['planned_time_hrs'] - df_processed['production_time_hrs']

df_processed.drop(columns=['product_type', 'machine_id', 'shift', 'day_of_week'], inplace=True)

X = df_processed.drop(['production_time_hrs', 'time_deviation'], axis=1)
y = df_processed['production_time_hrs']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

print("Training Models...")
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.01),
    'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=12),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbosity=0),
    'SVR': SVR(kernel='rbf', C=10.0)
}
scale_models = {'Linear Regression', 'Ridge Regression', 'Lasso Regression', 'SVR'}

results = []
for name, model in models.items():
    Xtr = X_train_scaled if name in scale_models else X_train
    Xte = X_test_scaled if name in scale_models else X_test
    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    cv_scores = cross_val_score(model, Xtr, y_train, cv=5, scoring='r2')
    test_r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results.append({
        'Model': name,
        'Test R2': round(test_r2, 4),
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4),
        'CV R2 Mean': round(cv_scores.mean(), 4),
        'CV R2 Std': round(cv_scores.std(), 4)
    })
    print(f"  {name:.<25} Test R2: {test_r2:.4f}, MAE: {mae:.4f}")

comparison_df = pd.DataFrame(results).sort_values('Test R2', ascending=False)
comparison_df.to_csv('model_comparison.csv', index=False)
print("[OK] Saved model_comparison.csv")

print("Training Tuned Best Model (Random Forest)...")
tuned_rf = RandomForestRegressor(
    n_estimators=187,
    max_depth=15,
    min_samples_split=12,
    min_samples_leaf=7,
    random_state=42,
    n_jobs=-1
)
tuned_rf.fit(X_train, y_train)
joblib.dump(tuned_rf, 'production_time_model.pkl')
print("[OK] Saved production_time_model.pkl")

joblib.dump(scaler, 'scaler.pkl')
print("[OK] Saved scaler.pkl")

encoders = {
    'le_product': le_product,
    'le_machine': le_machine,
    'le_shift': le_shift,
    'le_day': le_day
}
joblib.dump(encoders, 'label_encoders.pkl')
print("[OK] Saved label_encoders.pkl")

feature_names = list(X.columns)
joblib.dump(feature_names, 'feature_names.pkl')
print("[OK] Saved feature_names.pkl")

print("\nAll artifacts generated successfully in the project directory!")
