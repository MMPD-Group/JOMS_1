# Author- Anish Atey

import numpy as np
import pandas as pd
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler

# 1. Load and Preprocess the Data

# Define the dataset path (change to your actual .csv file location)
dataset_path = r"C:\Users\ateya\Downloads\MMPD Github Files\Optimization.xlsx"
# Load dataset
dataset = pd.read_excel(dataset_path, sheet_name='new data')

# Define the feature columns (elements + temperature)
feature_columns = [
    'Fe', 'C', 'Si', 'Mn', 'P', 'S', 'Ni', 'Cr', 'Mo', 'Cu', 'Al', 'N', 'V', 'Nb', 'Ta', 'W', 'Ti', 'B',
    'Sn','Zr', 'Co', 'O', 'Temperature'
]

# Define the target columns (mechanical properties we want to optimize)
target_columns = ['0.2% Proof Stress', 'Tensile Strength', 'Elongation %']

# Remove rows where any target value (proof stress, UTS, elongation) is missing
dataset = dataset.dropna(subset=target_columns)

# Fill missing values in feature columns with 0 to avoid issues in training
dataset[feature_columns] = dataset[feature_columns].fillna(0)

# Initialize MinMaxScaler to scale all feature values between 0 and 1
scaler = MinMaxScaler()

# Extract feature values (X) and apply MinMax Scaling
X = dataset[feature_columns].values
X_scaled = scaler.fit_transform(X)

# Extract target values (y) for training the models
y_proof = dataset['0.2% Proof Stress'].values
y_uts = dataset['Tensile Strength'].values
y_elongation = dataset['Elongation %'].values

# 2. Train XGBoost Models for Prediction
# ── Each model has its own hyperparameters — edit independently below ──────────

# Hyperparameters for Proof Stress model
PROOF_STRESS_PARAMS = dict(
    colsample_bytree = 1.0,
    learning_rate    = 0.07,
    max_depth        = 7,
    n_estimators     = 350,
    subsample        = 0.7,
    random_state     = 42,
)

# Hyperparameters for UTS (Tensile Strength) model
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

# Train each model with its own hyperparameters
model_proof     = train_xgboost_model(y_proof,     PROOF_STRESS_PARAMS)
model_uts       = train_xgboost_model(y_uts,       UTS_PARAMS)
model_elongation= train_xgboost_model(y_elongation, ELONGATION_PARAMS)

# 3. Define the Multi-Objective Optimization

class SteelMultiObjective(Problem):
    def __init__(self, elements_excluding_Fe):
        self.elements_excluding_Fe = elements_excluding_Fe

        min_values = dataset[elements_excluding_Fe].min()
        max_values = dataset[elements_excluding_Fe].max()

        lower_bounds = min_values.values * 0.80
        upper_bounds = max_values.values * 1.20

        self.sum_lower_bound = 1
        self.sum_upper_bound = 75

        super().__init__(n_var=len(elements_excluding_Fe), n_obj=3, n_constr=2, xl=lower_bounds, xu=upper_bounds)

    def _evaluate(self, x, out, *args, **kwargs):
        f1, f2, f3 = [], [], []
        g1 = []

        for xs in x:
            element_sum = np.sum(xs)
            Fe_percentage = 100 - element_sum

            g1.append(self.sum_lower_bound - element_sum)
            g1.append(element_sum - self.sum_upper_bound)

            if Fe_percentage <= 0:
                f1.append(np.inf)
                f2.append(np.inf)
                f3.append(np.inf)
            else:
                composition_with_Fe = np.hstack((Fe_percentage, xs, 1000))
                scaled_composition = scaler.transform([composition_with_Fe])[0]

                predicted_proof     = model_proof.predict([scaled_composition])[0]
                predicted_uts       = model_uts.predict([scaled_composition])[0]
                predicted_elongation= model_elongation.predict([scaled_composition])[0]

                f1.append(-predicted_proof)
                f2.append(-predicted_uts)
                f3.append(-predicted_elongation)

        out["F"] = np.column_stack([f1, f2, f3])
        out["G"] = np.column_stack(g1)

# 4. Run NSGA-II Optimization Algorithm

def optimize_at_1000():
    print("Optimizing Proof Stress, UTS, and Elongation at 1000°C...")

    elements_excluding_Fe = feature_columns[1:-1]
    problem   = SteelMultiObjective(elements_excluding_Fe)
    algorithm = NSGA2(pop_size=50)
    termination = ('n_gen', 50000)

    results = minimize(problem=problem, algorithm=algorithm, termination=termination, seed=1, verbose=True)

    if results.X is not None and len(results.X) > 0:
        print("\nOptimized Compositions at 27°C:\n")

        for i, (composition, value) in enumerate(zip(results.X, results.F)):
            element_sum   = np.sum(composition)
            Fe_percentage = 100 - element_sum
            composition_dict = {element: composition[j] for j, element in enumerate(elements_excluding_Fe)}
            composition_dict['Fe'] = Fe_percentage

            proof_stress = -value[0]
            uts          = -value[1]
            elongation   = -value[2]

            print(f"Composition {i+1}:")
            for element, percentage in composition_dict.items():
                print(f"  {element}: {percentage}%")
            print(f"  Maximized Proof Stress: {proof_stress} MPa")
            print(f"  Maximized UTS: {uts} MPa")
            print(f"  Maximized Elongation: {elongation}%\n")
    else:
        print("No feasible solutions found.")

# Run the optimization process
optimize_at_1000()