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

# Define the feature columns and target column for Elongation
feature_columns = [
    'Fe', 'C', 'Si', 'Mn', 'P', 'S', 'Ni', 'Cr', 'Mo', 'Cu', 'Al', 'N', 'V', 'Nb', 'Ta', 'W', 'Ti', 'B',
    'Sn','Zr', 'Co', 'O', 'Temperature'
]
target_column = 'Elongation %'

# Drop rows with missing values in the target column
dataset = dataset.dropna(subset=[target_column])

# Fill missing feature values with 0
dataset[feature_columns] = dataset[feature_columns].fillna(0)

# Scale the features
scaler = MinMaxScaler()
X = dataset[feature_columns].values
X_scaled = scaler.fit_transform(X)  # Normalize the feature values to a range [0, 1]

# Initialize models for Elongation at different temperatures (27°C, 400°C, 1000°C)
temperatures = [27, 400, 1000]
models = {}

# Train an XGBoost regressor model for each temperature
y = dataset[target_column].values  # The target values (Elongation percentages)

model = xgb.XGBRegressor(
        colsample_bytree=0.7,
        learning_rate=0.1,
        max_depth=5,
        n_estimators=450,
        subsample=0.6,
        random_state=42
    )
model.fit(X_scaled, y)  # Fit the model using the entire scaled dataset

# Store the same model for different temperatures
for temp in temperatures:
    models[temp] = model

# Define the multi-objective optimization problem for Elongation
class SteelMultiObjectiveElongation(Problem):
    def __init__(self, elements_excluding_Fe):
        # The elements used for optimization (excluding Fe)
        self.elements_excluding_Fe = elements_excluding_Fe

        # Set the min and max for the compositions (using dataset to determine element ranges)
        min_values = dataset[elements_excluding_Fe].min() 
        max_values = dataset[elements_excluding_Fe].max() 

        # Set lower and upper bounds for the compositions (80% of the min, 120% of the max)
        lower_bounds = min_values.values * 0.80
        upper_bounds = max_values.values * 1.20

        # Sum constraints for the element compositions
        self.sum_lower_bound = 1  # Minimum sum of elements (excluding Fe)
        self.sum_upper_bound = 75  # Maximum sum of elements (excluding Fe)

        # Initialize the optimization problem (3 objectives, 2 constraints)
        super().__init__(n_var=len(elements_excluding_Fe), n_obj=3, n_constr=2, xl=lower_bounds, xu=upper_bounds)

    def _evaluate(self, x, out, *args, **kwargs):
        # Evaluate the objectives and constraints for the optimization process
        f1, f2, f3 = [], [], []  # These will hold the objective values for each composition
        g1 = []  # Constraints on the sum of elements

        for xs in x:
            element_sum = np.sum(xs)  # Calculate the sum of all elements except Fe
            Fe_percentage = 100 - element_sum  # Calculate the Fe percentage as 100 - sum(elements)

            # Apply constraints for sum of elements excluding Fe
            g1.append(self.sum_lower_bound - element_sum)  # Constraint: sum(elements) >= 25
            g1.append(element_sum - self.sum_upper_bound)  # Constraint: sum(elements) <= 75

            # Penalize if Fe percentage is less than or equal to 0
            if Fe_percentage <= 0:
                f1.append(np.inf)  # Infeasible solution, penalize
                f2.append(np.inf)
                f3.append(np.inf)
            else:
                # Predict Elongation at 27°C (use model for 27°C)
                composition_with_Fe_and_temp = np.hstack((Fe_percentage, xs, 27))  # Add Fe and temp (27°C)
                scaled_composition = scaler.transform([composition_with_Fe_and_temp])[0]  # Scale the composition
                predicted_value_27 = models[27].predict([scaled_composition])[0]  # Predict Elongation at 27°C
                f1.append(-predicted_value_27)  # Negate to turn maximization into minimization

                # Predict Elongation at 400°C (similar to 27°C)
                composition_with_Fe_and_temp = np.hstack((Fe_percentage, xs, 400))  # Add Fe and temp (400°C)
                scaled_composition = scaler.transform([composition_with_Fe_and_temp])[0]
                predicted_value_400 = models[400].predict([scaled_composition])[0]  
                f2.append(-predicted_value_400)

                # Predict Elongation at 1000°C (similar to 27°C)
                composition_with_Fe_and_temp = np.hstack((Fe_percentage, xs, 1000))  # Add Fe and temp (1000°C)
                scaled_composition = scaler.transform([composition_with_Fe_and_temp])[0] 
                predicted_value_1000 = models[1000].predict([scaled_composition])[0] 
                f3.append(-predicted_value_1000)

        # Store the objective values (Elongation at 27°C, 400°C, and 1000°C)
        out["F"] = np.column_stack([f1, f2, f3])
        out["G"] = np.column_stack(g1)  # Store constraint violations (g1)

# Function to optimize Elongation at 27°C, 400°C, and 1000°C simultaneously
def optimize_elongation_all_temps():
    print("Maximizing Elongation at 27°C, 400°C, and 1000°C simultaneously")

    # Specify elements for optimization (excluding Fe and Temperature)
    elements_excluding_Fe = feature_columns[1:-1]  # All elements except Fe and Temperature
    problem = SteelMultiObjectiveElongation(elements_excluding_Fe)  # Define the optimization problem

    # Use NSGA2 algorithm for multi-objective optimization
    algorithm = NSGA2(pop_size=50)  # Population size of 50 for the optimization

    # Set termination criteria (maximum number of generations)
    termination = ('n_gen', 100)  
    results = minimize(problem=problem, algorithm=algorithm, termination=termination, seed=1, verbose=True)

    # Display the results
    if results.X is not None and len(results.X) > 0:
        print(f"Optimized Compositions and Corresponding Elongation at 27°C, 400°C, and 1000°C:")

        # Ensure results are in the correct format (reshape if needed)
        if len(results.X.shape) == 1:
            results.X = np.array([results.X])
            results.F = np.array([results.F])

        compositions_with_elongation = []

        # Process the results and calculate average Elongation
        for i, (composition, value) in enumerate(zip(results.X, results.F)):
            element_sum = np.sum(composition)
            Fe_percentage = 100 - element_sum  # Calculate Fe percentage
            composition_dict = {element: composition[j] for j, element in enumerate(elements_excluding_Fe)}
            composition_dict['Fe'] = Fe_percentage  # Add Fe to the composition

            # Calculate average Elongation (negate the values to get the correct sign)
            average_elongation = -np.mean(value)  # Negate to get the average Elongation
            
            compositions_with_elongation.append((composition_dict, average_elongation, value))

        # Sort by average Elongation in descending order
        compositions_with_elongation.sort(key=lambda x: x[1], reverse=True)

        # Print the results for each optimized composition
        for i, (composition_dict, avg_elongation, values) in enumerate(compositions_with_elongation):
            print(f"Composition {i + 1}:")
            for element, percentage in composition_dict.items():
                print(f"  {element}: {percentage}%")
            print(f"  Average Predicted Elongation: {avg_elongation} %")
            print(f"  Predicted Elongation at 27°C: {-values[0]} %")
            print(f"  Predicted Elongation at 400°C: {-values[1]} %")
            print(f"  Predicted Elongation at 1000°C: {-values[2]} %\n")
    else:
        print(f"No feasible solutions were found by the optimization process.\n")

# Run the multi-objective optimization for Elongation at 27°C, 400°C, and 1000°C
optimize_elongation_all_temps()
