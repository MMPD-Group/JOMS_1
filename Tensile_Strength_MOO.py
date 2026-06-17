# Author- Anish Atey

import numpy as np
import pandas as pd
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler

# Load and preprocess the dataset
dataset_path = r"C:\Users\ateya\Downloads\MMPD Github Files\Optimization.xlsx"
# Load dataset
dataset = pd.read_excel(dataset_path, sheet_name='new data')

# Define the feature columns (elements and temperature) and the target column for UTS (Tensile Strength)
feature_columns = [
    'Fe', 'C', 'Si', 'Mn', 'P', 'S', 'Ni', 'Cr', 'Mo', 'Cu', 'Al', 'N', 'V', 'Nb', 'Ta', 'W', 'Ti', 'B',
    'Sn', 'Co', 'O', 'Temperature'
]
target_column = 'Tensile Strength'

# Drop rows where the target column has missing values (NaN)
dataset = dataset.dropna(subset=[target_column])

# Fill missing values in the feature columns with 0
dataset[feature_columns] = dataset[feature_columns].fillna(0)

# Scale the features using Min-Max normalization
scaler = MinMaxScaler()
X = dataset[feature_columns].values  # Extract feature columns
X_scaled = scaler.fit_transform(X)  # Normalize features to the range [0, 1]

# Initialize a list of temperatures for which models will be trained (27°C, 400°C, and 1000°C)
temperatures = [27, 400, 1000]
models = {}

# Train an XGBoost regressor model for the target variable (Tensile Strength) at each temperature
y = dataset[target_column].values  # Target values (Tensile Strength)

# Create and configure an XGBoost regressor model
model = xgb.XGBRegressor(
        colsample_bytree=0.6,
        learning_rate=0.3,
        max_depth=5,
        n_estimators=1000,
        subsample=0.9,
        random_state=42
    )
model.fit(X_scaled, y)  # Train the model using the entire scaled dataset

# Store the trained model for each temperature
for temp in temperatures:
    models[temp] = model

# Define the multi-objective optimization problem for UTS (Tensile Strength)
class SteelMultiObjectiveUTS(Problem):
    def __init__(self, elements_excluding_Fe):
        # Initialize the class with the elements (excluding Fe)
        self.elements_excluding_Fe = elements_excluding_Fe

        # Get the minimum and maximum values for the compositions (from the dataset)
        min_values = dataset[elements_excluding_Fe].min()
        max_values = dataset[elements_excluding_Fe].max()

        # Set the lower and upper bounds for the compositions (80% of the min, 120% of the max)
        lower_bounds = min_values.values * 0.80
        upper_bounds = max_values.values * 1.20

        # Set constraints for the sum of elements excluding Fe
        self.sum_lower_bound = 1  # Minimum sum of elements (excluding Fe)
        self.sum_upper_bound = 75  # Maximum sum of elements (excluding Fe)

        # Initialize the problem with number of variables (elements), objectives, and constraints
        super().__init__(n_var=len(elements_excluding_Fe), n_obj=3, n_constr=2, xl=lower_bounds, xu=upper_bounds)

    def _evaluate(self, x, out, *args, **kwargs):
        # This function evaluates the objective values and constraints for the optimization process
        f1, f2, f3 = [], [], []  # Objective values for UTS at 27°C, 400°C, and 1000°C
        g1 = []  # Constraints for the sum of elements

        for xs in x:
            element_sum = np.sum(xs)  # Sum of all elements excluding Fe
            Fe_percentage = 100 - element_sum  # Calculate Fe percentage

            # Apply constraints for sum of elements excluding Fe
            g1.append(self.sum_lower_bound - element_sum)  # Constraint: sum(elements) >= 25
            g1.append(element_sum - self.sum_upper_bound)  # Constraint: sum(elements) <= 75

            # If Fe percentage is <= 0, penalize the composition
            if Fe_percentage <= 0:
                f1.append(np.inf)  # Infeasible solution, penalize
                f2.append(np.inf)
                f3.append(np.inf)
            else:
                # Predict UTS at 27°C using the pre-trained model for 27°C
                composition_with_Fe_and_temp = np.hstack((Fe_percentage, xs, 27))  # Add Fe and temperature (27°C)
                scaled_composition = scaler.transform([composition_with_Fe_and_temp])[0]  # Scale the composition
                predicted_value_27 = models[27].predict([scaled_composition])[0]  # Predict UTS at 27°C
                f1.append(-predicted_value_27)  # Negate to turn maximization into minimization

                # Predict UTS at 400°C (same approach as for 27°C)
                composition_with_Fe_and_temp = np.hstack((Fe_percentage, xs, 400))  # Add Fe and temperature (400°C)
                scaled_composition = scaler.transform([composition_with_Fe_and_temp])[0]
                predicted_value_400 = models[400].predict([scaled_composition])[0]  # Predict UTS at 400°C
                f2.append(-predicted_value_400)

                # Predict UTS at 1000°C (similar to above)
                composition_with_Fe_and_temp = np.hstack((Fe_percentage, xs, 1000))  # Add Fe and temperature (1000°C)
                scaled_composition = scaler.transform([composition_with_Fe_and_temp])[0]
                predicted_value_1000 = models[1000].predict([scaled_composition])[0]  # Predict UTS at 1000°C
                f3.append(-predicted_value_1000)

        # Store the objectives (f1, f2, f3) and constraints (g1) for the optimization process
        out["F"] = np.column_stack([f1, f2, f3])
        out["G"] = np.column_stack(g1)

# Function to perform the multi-objective optimization for UTS at 27°C, 400°C, and 1000°C simultaneously
def optimize_uts_all_temps():
    print("Maximizing UTS at 27°C, 400°C, and 1000°C simultaneously")

    elements_excluding_Fe = feature_columns[1:-1]  # All elements except Fe and Temperature
    problem = SteelMultiObjectiveUTS(elements_excluding_Fe)  # Define the optimization problem

    # Use NSGA2 algorithm for multi-objective optimization (with population size of 50)
    algorithm = NSGA2(pop_size=50)

    # Set termination condition for the optimization (50000 generations)
    termination = ('n_gen', 50000)
    results = minimize(problem=problem, algorithm=algorithm, termination=termination, seed=1, verbose=True)

    # If optimization results are found, process and display them
    if results.X is not None and len(results.X) > 0:
        print(f"Optimized Compositions and Corresponding UTS at 27°C, 400°C, and 1000°C:")

        # Ensure results are in the correct format (reshape if needed)
        if len(results.X.shape) == 1:
            results.X = np.array([results.X])
            results.F = np.array([results.F])

        compositions_with_uts = []  # List to store compositions and UTS values

        for i, (composition, value) in enumerate(zip(results.X, results.F)):
            element_sum = np.sum(composition)  # Calculate sum of elements
            Fe_percentage = 100 - element_sum  # Calculate Fe percentage
            composition_dict = {element: composition[j] for j, element in enumerate(elements_excluding_Fe)}
            composition_dict['Fe'] = Fe_percentage  # Add Fe percentage to the composition dictionary

            # Calculate average UTS
            average_uts = -np.mean(value)  # Negate to get the average UTS
            compositions_with_uts.append((composition_dict, average_uts, value))

        # Sort by average UTS in descending order
        compositions_with_uts.sort(key=lambda x: x[1], reverse=True)

        # Display the sorted results
        for i, (composition_dict, avg_uts, values) in enumerate(compositions_with_uts):
            print(f"Composition {i + 1}:")
            for element, percentage in composition_dict.items():
                print(f"  {element}: {percentage}%")
            print(f"  Average Predicted UTS: {avg_uts} MPa")
            print(f"  Predicted UTS at 27°C: {-values[0]} MPa")
            print(f"  Predicted UTS at 400°C: {-values[1]} MPa")
            print(f"  Predicted UTS at 1000°C: {-values[2]} MPa\n")
    else:
        print(f"No feasible solutions were found by the optimization process.\n")

# Run the multi-objective optimization for UTS at 27°C, 400°C
optimize_uts_all_temps()