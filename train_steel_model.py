import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# Set up paths
DATA_PATH = "c:/Users/Hamza Computer/Downloads/project/project/data/Steel_industry_data.csv"
STATIC_DIR = "c:/Users/Hamza Computer/Downloads/files (2)/fastapi_app/fastapi_app/static"
MODEL_SAVE_PATH = "c:/Users/Hamza Computer/Downloads/files (2)/fastapi_app/fastapi_app/model.joblib"
# Ensure static directory exists
os.makedirs(STATIC_DIR, exist_ok=True)

# 1. Load data
print("Loading data...")
df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'], dayfirst=True)
print("Data loaded. Shape:", df.shape)

# Configure matplotlib for a premium dark dashboard style
plt.style.use('dark_background')
plt.rcParams.update({
    'figure.facecolor': '#05070f',
    'axes.facecolor': '#090d16',
    'savefig.facecolor': '#05070f',
    'axes.edgecolor': '#1e293b',
    'grid.color': '#1e293b',
    'xtick.color': '#94a3b8',
    'ytick.color': '#94a3b8',
    'text.color': '#f8fafc',
    'axes.labelcolor': '#94a3b8',
    'font.family': 'sans-serif',
    'axes.titleweight': 'bold',
    'axes.titlesize': 12,
    'axes.titlepad': 15,
})

# Generate plots
print("Generating plots...")

# Plot 1: Average Usage_kWh by Hour (Bar chart with neon gradient-like colors)
df_temp = df.copy()
df_temp['hour'] = df_temp['date'].dt.hour
fig, ax = plt.subplots(figsize=(10, 5))
usage_by_hour = df_temp.groupby('hour')['Usage_kWh'].mean()
usage_by_hour.plot(kind='bar', ax=ax, color='#00f2fe', edgecolor='#0072ff', linewidth=1, width=0.7)
ax.set_title('Average Usage (kWh) by Hour of Day')
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Usage (kWh)')
ax.grid(True, axis='y', linestyle=':', alpha=0.15)
plt.xticks(rotation=0)
sns.despine(left=True, bottom=True)
plt.tight_layout()
plt.savefig(os.path.join(STATIC_DIR, 'energy_by_hour.png'), dpi=150)
plt.close()

# Plot 2: Usage_kWh by Load_Type (Box plot with modern palette)
fig, ax = plt.subplots(figsize=(8, 5))
palette = {
    "Light_Load": "#10b981",    # Emerald success
    "Medium_Load": "#a855f7",   # Amethyst purple
    "Maximum_Load": "#ef4444"   # Ruby danger
}
sns.boxplot(
    data=df, x='Load_Type', y='Usage_kWh', ax=ax, palette=palette, width=0.55, fliersize=2.5,
    boxprops=dict(alpha=0.85, edgecolor=(1.0, 1.0, 1.0, 0.1), linewidth=1.2),
    whiskerprops=dict(color='#475569'), capprops=dict(color='#475569')
)
ax.set_title('Energy Consumption (kWh) by Load Type')
ax.set_xlabel('Load Type')
ax.set_ylabel('Usage (kWh)')
ax.grid(True, axis='y', linestyle=':', alpha=0.15)
sns.despine(left=True, bottom=True)
plt.tight_layout()
plt.savefig(os.path.join(STATIC_DIR, 'energy_by_load_type.png'), dpi=150)
plt.close()

# Plot 3: Correlation Heatmap (Dark matrix)
numeric_cols = [
    'Usage_kWh',
    'Lagging_Current_Reactive.Power_kVarh',
    'Leading_Current_Reactive_Power_kVarh',
    'CO2(tCO2)',
    'Lagging_Current_Power_Factor',
    'Leading_Current_Power_Factor',
    'NSM'
]
fig, ax = plt.subplots(figsize=(7, 6))
# Create custom correlation matrix labels for cleaner aesthetics
clean_labels = [
    'Usage (kWh)', 'Lagging React. Pwr', 'Leading React. Pwr',
    'CO2 (tCO2)', 'Lagging Pwr Factor', 'Leading Pwr Factor', 'NSM'
]
corr_matrix = df[numeric_cols].corr()
sns.heatmap(
    corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax,
    annot_kws={"size": 10, "weight": "bold", "color": "#ffffff"},
    cbar_kws={"shrink": 0.8}, linewidths=0.75, linecolor='#05070f',
    xticklabels=clean_labels, yticklabels=clean_labels
)
ax.set_title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig(os.path.join(STATIC_DIR, 'correlation_heatmap.png'), dpi=150)
plt.close()
print("Plots generated successfully.")

# Import engineer_features from feature_engineering.py
import sys
sys.path.insert(0, "c:/Users/Hamza Computer/Downloads/files (2)/fastapi_app/fastapi_app")
from feature_engineering import engineer_features

# 2. Train-Test Split (on raw data, before any fitting)
X_raw = df.drop(columns=['Usage_kWh'])
y = df['Usage_kWh']

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=RANDOM_STATE
)
print(f"Train size: {X_train_raw.shape} | Test size: {X_test_raw.shape}")

# Define column names
CATEGORICAL_COLS = ['Load_Type', 'WeekStatus', 'Day_of_week']
NUMERIC_RAW_COLS = [
    'Lagging_Current_Reactive.Power_kVarh',
    'Leading_Current_Reactive_Power_kVarh',
    'CO2(tCO2)',
    'Lagging_Current_Power_Factor',
    'Leading_Current_Power_Factor',
    'NSM',
]
ENGINEERED_NUMERIC_COLS = NUMERIC_RAW_COLS + ['day', 'is_weekend',
                                               'hour_sin', 'hour_cos',
                                               'month_sin', 'month_cos']

preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), CATEGORICAL_COLS),
    ('num', StandardScaler(), ENGINEERED_NUMERIC_COLS),
])

X_train_eng = engineer_features(X_train_raw)
X_test_eng = engineer_features(X_test_raw)

preprocessor.fit(X_train_eng)

X_train_proc = preprocessor.transform(X_train_eng)
X_test_proc = preprocessor.transform(X_test_eng)

# PCA to find components retaining 95% variance
n_features = X_train_proc.shape[1]
pca_full = PCA(n_components=n_features, random_state=RANDOM_STATE)
pca_full.fit(X_train_proc)
cumulative_var = np.cumsum(pca_full.explained_variance_ratio_)
n_components_95 = int(np.argmax(cumulative_var >= 0.95) + 1)
print(f"Components retaining 95% variance: {n_components_95}")

# Hyperparameter search
print("Tuning XGBoost hyper-parameters...")
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
}
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
search = RandomizedSearchCV(
    xgb.XGBRegressor(objective='reg:squarederror', random_state=RANDOM_STATE),
    param_distributions=param_dist,
    n_iter=10, scoring='neg_root_mean_squared_error', cv=cv,
    random_state=RANDOM_STATE, n_jobs=1, verbose=1
)
search.fit(X_train_proc, y_train)
best_params = search.best_params_
print("Best params:", best_params)
print(f"Best CV RMSE: {-search.best_score_:.4f}")

# Train the final pipeline
print("Training final pipeline...")
feature_engineer = FunctionTransformer(engineer_features, validate=False)
final_pipeline = Pipeline(steps=[
    ('feature_engineering', feature_engineer),
    ('preprocessor', preprocessor),
    ('pca', PCA(n_components=n_components_95, random_state=RANDOM_STATE)),
    ('model', xgb.XGBRegressor(objective='reg:squarederror', random_state=RANDOM_STATE, **best_params)),
])

final_pipeline.fit(X_train_raw, y_train)

final_preds = final_pipeline.predict(X_test_raw)
print("Final pipeline test RMSE:", np.sqrt(mean_squared_error(y_test, final_preds)))
print("Final pipeline test R2  :", r2_score(y_test, final_preds))

# Save the model
joblib.dump(final_pipeline, MODEL_SAVE_PATH)
print("Saved final_pipeline to:", MODEL_SAVE_PATH)
