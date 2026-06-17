# Author- Anish Atey
# Modified to include Temperature as an optimization variable
# Modified to use separate XGBoost hyperparameters for each target model

import numpy as np
import pandas as pd
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler

# 1. Load and Preprocess the Data
dataset_path = r"C:\Users\ateya\Downloads\MMPD Github Files\Optimization.xlsx"
dataset = pd.read_excel(dataset_path, sheet_name='new data')

feature_columns = [
    'Fe', 'C', 'Si', 'Mn', 'P', 'S', 'Ni', 'Cr', 'Mo', 'Cu', 'Al', 'N', 'V', 'Nb', 'Ta', 'W', 'Ti', 'B',
    'Sn', 'Zr', 'Co', 'O', 'Temperature'
]

target_columns = ['0.2% Proof Stress', 'Tensile Strength', 'Elongation %']
dataset = dataset.dropna(subset=target_columns)
dataset[feature_columns] = dataset[feature_columns].fillna(0)

scaler = MinMaxScaler()
X = dataset[feature_columns].values
X_scaled = scaler.fit_transform(X)

y_proof      = dataset['0.2% Proof Stress'].values
y_uts        = dataset['Tensile Strength'].values
y_elongation = dataset['Elongation %'].values

# 2. Train XGBoost Models
# ── Edit each model's hyperparameters independently below ────────────────────

# Hyperparameters for 0.2% Proof Stress model
PROOF_STRESS_PARAMS = dict(
    colsample_bytree = 1.0,
    learning_rate    = 0.07,
    max_depth        = 7,
    n_estimators     = 350,
    subsample        = 0.7,
    random_state     = 42,
)

# Hyperparameters for Tensile Strength (UTS) model
UTS_PARAMS = dict(
   colsample_bytree = 0.6,
    learning_rate    = 0.3,
    max_depth        = 5,
    n_estimators     = 1000,
    subsample        = 0.9,
    random_state     = 42,
)

# Hyperparameters for Elongation model
ELONGATION_PARAMS = dict(
    colsample_bytree = 0.7,
    learning_rate    = 0.1,
    max_depth        = 5,
    n_estimators     = 450,
    subsample        = 0.6,
    random_state     = 42,
)

# ─────────────────────────────────────────────────────────────────────────────

def train_xgboost_model(y, params: dict):
    """Train an XGBoost regressor with the given hyperparameters."""
    model = xgb.XGBRegressor(**params)
    model.fit(X_scaled, y)
    return model

model_proof      = train_xgboost_model(y_proof,      PROOF_STRESS_PARAMS)
model_uts        = train_xgboost_model(y_uts,        UTS_PARAMS)
model_elongation = train_xgboost_model(y_elongation, ELONGATION_PARAMS)

# 3. Define the Multi-Objective Optimization
class SteelMultiObjective(Problem):
    def __init__(self, elements_excluding_Fe):
        self.elements_excluding_Fe = elements_excluding_Fe

        min_elements = dataset[elements_excluding_Fe].min().values * 0.80
        max_elements = dataset[elements_excluding_Fe].max().values * 1.20

        min_temp = dataset['Temperature'].min()
        max_temp = dataset['Temperature'].max()

        lower_bounds = np.append(min_elements, min_temp)
        upper_bounds = np.append(max_elements, max_temp)

        self.sum_lower_bound = 1
        self.sum_upper_bound = 75

        super().__init__(n_var=len(elements_excluding_Fe) + 1, n_obj=3, n_constr=2, xl=lower_bounds, xu=upper_bounds)

    def _evaluate(self, x, out, *args, **kwargs):
        f1, f2, f3 = [], [], []
        g1, g2 = [], []

        for xs in x:
            temp_val          = xs[-1]
            element_composition = xs[:-1]
            element_sum       = np.sum(element_composition)
            Fe_percentage     = 100 - element_sum

            g1.append(self.sum_lower_bound - element_sum)
            g2.append(element_sum - self.sum_upper_bound)

            if Fe_percentage <= 0:
                f1.append(np.inf); f2.append(np.inf); f3.append(np.inf)
            else:
                composition_full  = np.hstack((Fe_percentage, element_composition, temp_val))
                scaled_composition = scaler.transform([composition_full])[0]

                f1.append(-model_proof.predict([scaled_composition])[0])
                f2.append(-model_uts.predict([scaled_composition])[0])
                f3.append(-model_elongation.predict([scaled_composition])[0])

        out["F"] = np.column_stack([f1, f2, f3])
        out["G"] = np.column_stack([g1, g2])

# 4. Run Optimization Process
def optimize_steel():
    print("Optimizing Compositions and Temperature...")

    elements_only = feature_columns[1:-1]

    problem     = SteelMultiObjective(elements_only)
    algorithm   = NSGA2(pop_size=50)
    termination = ('n_gen', 50000)

    res = minimize(problem, algorithm, termination, seed=1, verbose=True)

    if res.X is not None:
        print("\nOptimized Results Found:\n")
        for i, (vars, objectives) in enumerate(zip(res.X, res.F)):
            temp   = vars[-1]
            comp   = vars[:-1]
            Fe_val = 100 - np.sum(comp)

            print(f"Solution {i+1}:")
            print(f"  Temperature: {temp} °C")
            print(f"  Fe: {Fe_val}%")
            for name, val in zip(elements_only, comp):
                print(f"  {name}: {val}%")
            print(f"  Proof Stress: {-objectives[0]} MPa")
            print(f"  UTS: {-objectives[1]} MPa")
            print(f"  Elongation: {-objectives[2]}%\n")
    else:
        print("No feasible solutions found.")

optimize_steel()