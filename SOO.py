# Author- Anish Atey

import numpy as np
import pandas as pd
from pymoo.core.problem import Problem  # Import Problem class from pymoo for optimization
from pymoo.optimize import minimize  # Import minimize function to solve optimization problems
from pymoo.algorithms.moo.nsga2 import NSGA2  # Import NSGA2 algorithm for multi-objective optimization
import xgboost as xgb  # Import XGBoost for regression model
from sklearn.preprocessing import MinMaxScaler  # Import MinMaxScaler for scaling features
from pymoo.core.callback import Callback

class ProgressCallback(Callback):
    def __init__(self, objective_name):
        super().__init__()
        self.objective_name = objective_name
    
    def notify(self, algorithm):
        # Get current generation number
        gen = algorithm.n_gen
        
        # Get best objective value (remember it's negated, so flip it back)
        best_f = -algorithm.opt.get("F").min()
        
        # Print every 100 generations to avoid excessive output
        if gen % 100 == 0 or gen == 1:
            print(f"Generation {gen}: Best {self.objective_name} = {best_f:.2f} MPa")

# Path to the dataset
dataset_path = r"C:\Users\ateya\Downloads\MMPD Github Files\Optimization.xlsx"
# Load dataset
dataset = pd.read_excel(dataset_path, sheet_name='new data')

# Define the feature columns and target columns
feature_columns = [
    'Fe', 'C', 'Si', 'Mn', 'P', 'S', 'Ni', 'Cr', 'Mo', 'Cu', 'Al', 'N', 'V', 'Nb', 'Ta', 'W', 'Ti', 'B',
    'Sn','Zr', 'Co', 'O', 'Temperature'
]
target_columns = {
    'Proof Stress': '0.2% Proof Stress',
    'UTS': 'Tensile Strength',
    '% Elongation': 'Elongation %'
}

# Drop rows with missing target values
dataset = dataset.dropna(subset=target_columns.values())
# Fill missing values in feature columns with 0
dataset[feature_columns] = dataset[feature_columns].fillna(0)

# Scale the features
scaler = MinMaxScaler()  # Initialize scaler
X = dataset[feature_columns].values  # Extract feature values
X_scaled = scaler.fit_transform(X)  # Scale the feature values

# Initialize models for Proof Stress, UTS, and % Elongation
models = {}  # Dictionary to hold models for each target
for target_name, target_column in target_columns.items():
    y = dataset[target_column].values  # Extract target values
    # Initialize XGBoost regressor model with specified hyperparameters
    model = xgb.XGBRegressor(
        colsample_bytree=0.6,
        learning_rate=0.3,
        max_depth=5,
        n_estimators=1000,
        subsample=0.9,
        random_state=42
    )
    """PROOF_STRESS_PARAMS = dict(
    colsample_bytree = 1.0,
    learning_rate    = 0.07,
    max_depth        = 7,
    n_estimators     = 350,
    subsample        = 0.7,
    random_state     = 42,
    
    ELONGATION_PARAMS = dict(
    colsample_bytree = 0.7,
    learning_rate    = 0.1,
    max_depth        = 5,
    n_estimators     = 450,
    subsample        = 0.6,
    random_state     = 42,"""
    model.fit(X_scaled, y)  # Train the model
    models[target_name] = model  # Store model for current target

# Define the optimization problem class
class SteelPropertyMaximizationProblem(Problem):
    def __init__(self, objective, temperature, elements_excluding_Fe):
        self.objective = objective  # Objective (Proof Stress, UTS, etc.)
        self.temperature = temperature  # Temperature value (fixed in optimization)
        self.model = models[objective]  # Load the model for the objective
        self.elements_excluding_Fe = elements_excluding_Fe  # List of elements excluding Fe

        # Set the min and max for the compositions of elements (excluding Fe)
        min_values = dataset[elements_excluding_Fe].min()  # Minimum values for elements
        max_values = dataset[elements_excluding_Fe].max()  # Maximum values for elements

        # Define lower and upper bounds for optimization variables
        lower_bounds = min_values.values * 0.80  # Lower bounds (70% of min values)
        upper_bounds = max_values.values * 1.20  # Upper bounds (140% of max values)

        # Define sum constraints for the element percentages (excludes Fe)
        self.sum_lower_bound = 1  # Minimum sum of element percentages
        self.sum_upper_bound = 75  # Maximum sum of element percentages

        # Initialize the optimization problem with constraints
        super().__init__(n_var=len(elements_excluding_Fe), n_obj=1, n_constr=2, xl=lower_bounds, xu=upper_bounds)

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = []  # List to store objective values (negative of predicted values)
        g1 = []  # List to store constraint violations

        for xs in x:
            element_sum = np.sum(xs)  # Sum of element percentages
            Fe_percentage = 100 - element_sum  # Calculate Fe percentage as the remaining part

            # Check if the sum of elements is within the bounds
            g1.append(self.sum_lower_bound - element_sum)  # Lower bound constraint
            g1.append(element_sum - self.sum_upper_bound)  # Upper bound constraint

            # If Fe percentage is <= 0, assign infinite objective value (invalid solution)
            if Fe_percentage <= 0:
                f1.append(np.inf)  # Invalid composition
            else:
                # Create composition array with Fe and temperature
                composition_with_Fe_and_temp = np.hstack((Fe_percentage, xs, self.temperature))
                # Scale the composition
                scaled_composition = scaler.transform([composition_with_Fe_and_temp])[0]
                # Predict the objective value (negative to maximize)
                predicted_value = self.model.predict([scaled_composition])[0]
                f1.append(-predicted_value)  # Maximize objective by negating predicted value

        # Output the objective values and constraints
        out["F"] = np.column_stack([f1])  # Objective values
        out["G"] = np.column_stack(g1)  # Constraint violations

# Function to run the optimization for a given objective and temperature
def optimize_for_objective_and_temperature(objective, temp):
    print(f"Maximizing {objective} at {temp}°C")

    elements_excluding_Fe = feature_columns[1:-1]
    problem = SteelPropertyMaximizationProblem(objective, temp, elements_excluding_Fe)
    algorithm = NSGA2(pop_size=50)

    termination = ('n_gen', 50000)
    
    # Add the callback here
    callback = ProgressCallback(objective)
    
    results = minimize(
        problem=problem, 
        algorithm=algorithm, 
        termination=termination, 
        seed=1, 
        verbose=True,
        callback=callback  # Add this parameter
    )
    
    if results.X is not None and len(results.X) > 0:
        print(f"FINAL RESULTS: Optimized Compositions for {objective} at {temp}°C")

        if len(results.X.shape) == 1:
            results.X = np.array([results.X])
            results.F = np.array([results.F])

        for i, (composition, value) in enumerate(zip(results.X, results.F)):
            element_sum = np.sum(composition)
            Fe_percentage = 100 - element_sum
            composition_dict = {element: composition[j] for j, element in enumerate(elements_excluding_Fe)}
            composition_dict['Fe'] = Fe_percentage

            print(f"Composition {i + 1}:")
            for element, percentage in composition_dict.items():
                print(f"  {element}: {percentage}%")
            print(f"  Predicted {objective}: {-value[0]:.2f} MPa\n")
    else:
        print(f"No feasible solutions were found for {objective} at {temp}°C.\n")



# Example usage: Run optimization for Proof Stress at 1000°C
objective_to_optimize = 'UTS'  # Change to 'UTS' or '% Elongation' as needed
temperature_to_optimize = 1000  # Change to 400 or 27 as needed

# Run optimization for the selected objective and temperature
optimize_for_objective_and_temperature(objective_to_optimize, temperature_to_optimize)