# Copyright [2025] [Dr. Thanh Hoang Nguyen]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#-------------------------------------------------------------------------------------------------------------#
#Training Data Analysis for Ampicillin                                                                        # 
#Multiple Linear Regression (MLR) Model Training                                                              #
#Need to install module/library packages: pandas, numpy, scikit-learn, matplotlib, tabulate, openpyxl by pip  #
#-------------------------------------------------------------------------------------------------------------#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tabulate import tabulate 

#-------------------------------------------------------------------------------------------------------------#
# Define supporting functions 

def load_and_preprocess(file_path):
    """
    Load dataset, rename columns, and calculate variable ratios.
    """
    try:
        df = pd.read_excel(file_path)
        # Assumed column order: C (Concentration), R (Red), G (Green), B (Blue), Cl (Clear/Other)
        df.columns = ['C', 'R', 'G', 'B', 'Cl']
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found. Please check the file path.")
        return None
    except Exception:
        print(f"Error: File '{file_path}' does not match the required 5-column structure or could not be read.")
        return None

    # To ensure data integrity and avoid errors, check for non-zero denominators --> remove rows with 0 in denominators (R, G, B)
    initial_count = len(df)
    df = df[ (df['R'] != 0) & (df['G'] != 0) & (df['B'] != 0) ]
    if len(df) < initial_count:
        print(f"Warning: Removed {initial_count - len(df)} rows containing zero values in R, G, or B columns from '{file_path}'.")

    # Feature Engineering: Calculate 3 ratio variables (Green/Red, Blue/Green, and Blue/Red)
    df['G_R'] = df['G'] / df['R']
    df['B_G'] = df['B'] / df['G']
    df['B_R'] = df['B'] / df['R']

    return df
    
#important values ​​of the model
def calculate_metrics(Y_true, Y_pred):
    """
    Calculate performance metrics: R-squared (R²), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).
    """
    r2 = r2_score(Y_true, Y_pred)
    mae = mean_absolute_error(Y_true, Y_pred)
    rmse = np.sqrt(mean_squared_error(Y_true, Y_pred))
    return r2, mae, rmse
#-------------------------------------------------------------------------------------------------------------#
#DATASET CONFIGURATION

TRAIN_FILE = 'ampicilin-trainning.xlsx'
TEST_FILES = {
    'L1': {'file': 'ampicilin-test-L1.xlsx', 'color': 'red', 'marker': 'D'},  # Diamond
    'L2': {'file': 'ampicilin-test-L2.xlsx', 'color': 'gold', 'marker': '*'}, # Star
    'L3': {'file': 'ampicilin-test-L3.xlsx', 'color': 'purple', 'marker': '^'} # Triangle
}
FEATURE_COLS = ['G_R', 'B_G', 'B_R']
TARGET_COL = 'C'
#-------------------------------------------------------------------------------------------------------------#
#MODEL TRAINING

print("=" * 60)
print(f"STEP 1: MODEL TRAINING WITH DATASET '{TRAIN_FILE}' ")
print("=" * 60)

df_train = load_and_preprocess(TRAIN_FILE)

if df_train is None:
    exit()

X_train = df_train[FEATURE_COLS]
Y_train = df_train[TARGET_COL]

# Initialize and train the Ordinary Least Squares (OLS) Linear Regression model
model = LinearRegression(fit_intercept=True)
model.fit(X_train, Y_train)

#Extracting MLR equation
intercept = model.intercept_
coefficients = model.coef_

# Map internal feature names to display names
display_features_map = {
    'G_R': 'G/R', 
    'B_G': 'B/G', 
    'B_R': 'B/R'
}

equation_parts = [f"C(predicted) = {intercept:.4f}"]
for coef, feature in zip(coefficients, FEATURE_COLS):
    sign = "+" if coef >= 0 else "-"
    display_name = display_features_map[feature]
    equation_parts.append(f" {sign} {abs(coef):.4f} * ({display_name})")

MLR_EQUATION_STRING = "".join(equation_parts)

print("\n--- DERIVED MULTIPLE LINEAR REGRESSION (MLR) EQUATION ---")
print(MLR_EQUATION_STRING)
print("-" * 60)

# Calculate predictions on Training Set to define plot boundaries
Y_train_pred = model.predict(X_train)
Y_train_pred_clipped = np.maximum(0, Y_train_pred) # Ensure non-negative predictions

r2_train, mae_train, rmse_train = calculate_metrics(Y_train, Y_train_pred_clipped)
print(f"TRAINING COMPLETED SUCCESSFULLY.\nPerformance Metrics (Training Set): R² = {r2_train:.4f}, MAE = {mae_train:.4f}, RMSE = {rmse_train:.4f}")
#-------------------------------------------------------------------------------------------------------------#
#MODEL VALIDATION WITH TEST SETS

print("\n" + "=" * 60)
print("STEP 2: MODEL VALIDATION ON TEST SETS")
print("=" * 60)

all_test_results = {}
max_C = Y_train.max()

for name, config in TEST_FILES.items():
    file_path = config['file']
    print(f"Processing test dataset '{name}' ({file_path})...")
    df_test = load_and_preprocess(file_path)

    if df_test is not None:
        X_test = df_test[FEATURE_COLS]
        Y_test = df_test[TARGET_COL]

        # Generate Predictions
        Y_test_pred = model.predict(X_test)
        
        # Create DataFrame for results
        results_df = pd.DataFrame({
            'Actual_Concentration': Y_test.values,
            'Predicted_Concentration': Y_test_pred,
        })
        
        # Clip negative predictions to 0
        results_df['Predicted_Clipped'] = np.maximum(0, results_df['Predicted_Concentration'])
        results_df['Absolute_Error'] = np.abs(results_df['Actual_Concentration'] - results_df['Predicted_Clipped'])
        
        # Calculate validation metrics
        r2_test, mae_test, rmse_test = calculate_metrics(results_df['Actual_Concentration'], results_df['Predicted_Clipped'])
        
        all_test_results[name] = {
            'df': results_df,
            'r2': r2_test,
            'mae': mae_test,
            'rmse': rmse_test,
            'config': config
        }
        
        # Update maximum concentration for plotting limits
        max_C = max(max_C, results_df['Actual_Concentration'].max(), results_df['Predicted_Clipped'].max())
        
        print(f"  > Validation Metrics ({name}): R² = {r2_test:.4f}, MAE = {mae_test:.4f}")
#-------------------------------------------------------------------------------------------------------------#
#Data visualization/plot graph

print("\n" + "=" * 60)
print("STEP 3: GENERATING AGGREGATE PERFORMANCE PLOT")
print("=" * 60)

# Calculate Simple Linear Regression (SLR) line for the plot (Predicted vs Actual)
slr_model = LinearRegression()
slr_model.fit(Y_train.values.reshape(-1, 1), Y_train_pred_clipped)

slr_r2 = slr_model.score(Y_train.values.reshape(-1, 1), Y_train_pred_clipped)
slr_intercept = slr_model.intercept_
slr_slope = slr_model.coef_[0]

# Define Regression Line points
X_line = np.linspace(0, max_C, 100)
Y_slr_line = slr_intercept + slr_slope * X_line

# Plot Style Configuration
try:
    plt.style.use('seaborn-whitegrid')
except OSError:
    plt.style.use('ggplot')
    print("Note: 'seaborn-whitegrid' style not found. Using 'ggplot' fallback.")
    
fig, ax = plt.subplots(figsize=(12, 8))

# Define Plot Limits
padding = max_C * 0.1
min_limit = 0
max_limit = max_C + padding

# 1. Plot Training Data (Solid blue circles)
ax.scatter(
    Y_train, Y_train_pred_clipped,
    color='blue', s=150, alpha=0.9, 
    label=f"Training Set ({TRAIN_FILE}) - $R^2$: {r2_train:.3f}, MAE: {mae_train:.3f}"
)

# 2. Plot Test Data (Hollow markers avoid overlapping) 
for name, res in all_test_results.items():
    if 'df' in res:
        df_plot = res['df']
        config = res['config']
        
        metrics_label = (
            f"Test Set {name} ({config['file']})\n"
            f"  $R^2$: {res['r2']:.3f}, MAE: {res['mae']:.3f}, RMSE: {res['rmse']:.3f}"
        )
        
        ax.scatter(
            df_plot['Actual_Concentration'], df_plot['Predicted_Clipped'],
            facecolors='none',          # Hollow marker
            edgecolors=config['color'], # Colored edge
            marker=config['marker'], 
            s=150,
            linewidths=1.5,
            alpha=0.9,
            label=metrics_label
        )

# 3. Plot Regression Fit Line (Solid Black Line)
ax.plot(
    X_line, Y_slr_line,
    'k-', linewidth=1.5, alpha=0.9, 
    label=f"Linear Fit ($y={slr_intercept:.3f}+{slr_slope:.3f}x$) - $R^2$: {slr_r2:.3f}"
)

# 4. Add MLR Equation to Legend
mlr_for_legend_display = MLR_EQUATION_STRING.replace('* (', '(') 
ax.plot([], [], 'w', label=f"  MLR Eq: {mlr_for_legend_display}")

# 5. Plot Ideal Line (Dashed Red Line)
ax.plot(
    [min_limit, max_limit], [min_limit, max_limit],
    'r--', linewidth=1.5, alpha=0.8, label="Ideal Line ($y=x$)"
)

# Finalize Plot Aesthetics
ax.set_xlabel("Actual Concentration (ppm)", fontsize=14)
ax.set_ylabel("Predicted Concentration (ppm) [≥ 0]", fontsize=14)
ax.set_title("Multiple Linear Regression (MLR) Model Performance: 3-Variable OLS", fontsize=16)

ax.set_xlim(min_limit, max_limit)
ax.set_ylim(min_limit, max_limit)

# Legend Configuration
ax.legend(loc='lower right', fontsize=8, title="Dataset & Model Metrics", title_fontsize=9)
plt.show()

# 6. DISPLAY RESULTS TABLE
print("\n" + "=" * 60)
print("--- STEP 4: PREDICTION RESULTS AND ERROR ANALYSIS ---")
print("=" * 60)

for name, res in all_test_results.items():
    if 'df' in res:
        print(f"\n*** RESULTS FOR TEST SET: {name} ({res['config']['file']}) ***")
        
        display_df = res['df'][['Actual_Concentration', 'Predicted_Clipped', 'Absolute_Error']]
        display_df.columns = ['Actual Concentration (ppm)', 'Predicted Concentration (ppm)', 'Absolute Error']
        
        # Display formatted table
        print(tabulate(display_df, headers='keys', tablefmt='pipe', showindex=True, floatfmt=".4f"))
        
        print(f"\nSUMMARY METRICS FOR {name}:")
        print(f"  R-squared ($R^2$): {res['r2']:.4f}, MAE: {res['mae']:.4f}, RMSE: {res['rmse']:.4f}")

print("\n" + "=" * 60)
print("PROCESS COMPLETED SUCCESSFULLY.")
#END-CODE-------------------------------------------------------------------------------------------------#
