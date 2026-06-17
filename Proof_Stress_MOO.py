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

# Define the feature columns and target column
feature_columns = [
    'Fe', 'C', 'Si', 'Mn', 'P', 'S', 'Ni', 'Cr', 'Mo', 'Cu', 'Al', 'N', 'V', 'Nb', 'Ta', 'W', 'Ti', 'B',
    'Sn', 'Zr', 'Co', 'O', 'Temperature'
]
target_column = '0.2% Proof Stress'

# Drop rows with missing values in the target column
dataset = dataset.dropna(subset=[target_column])

# Fill missing feature values with 0
dataset[feature_columns] = dataset[feature_columns].fillna(0)

# Scale the features
scaler = MinMaxScaler()
X = dataset[feature_columns].values
X_scaled = scaler.fit_transform(X)  # Normalize the feature values to a range [0, 1]

# Define and initialize the XGBoost regressor model
model = xgb.XGBRegressor(
    colsample_bytree=1,
    learning_rate=0.07,
    max_depth=7,
    n_estimators=350,
    subsample=0.7,
    random_state=42
)
model.fit(X_scaled, dataset[target_column].values)  # Train the model on the scaled features

# Define the multi-objective optimization problem for Proof Stress
class SteelMultiObjectiveProofStress(Problem):
    def __init__(self, elements_excluding_Fe):
        # Initialize the problem with the elements to optimize (excluding Fe)
        self.elements_excluding_Fe = elements_excluding_Fe

        # Calculate the min and max values for each element from the dataset
        min_values = dataset[elements_excluding_Fe].min()
        max_values = dataset[elements_excluding_Fe].max()

        # Set lower and upper bounds for the elements based on their min and max values
        lower_bounds = min_values.values * 0.80  # Lower bound is 80% of the min value
        upper_bounds = max_values.values * 1.20  # Upper bound is 120% of the max value

        # Constraints for the sum of the element percentages (excluding Fe)
        self.sum_lower_bound = 1  # Minimum sum for the elements (excluding Fe)
        self.sum_upper_bound = 75  # Maximum sum for the elements (excluding Fe)

        # Initialize the problem with 3 objectives and 2 constraints
        super().__init__(n_var=len(elements_excluding_Fe), n_obj=3, n_constr=2, xl=lower_bounds, xu=upper_bounds)

    def _evaluate(self, x, out, *args, **kwargs):
        # Evaluate the objectives and constraints for each composition in the population
        f1, f2, f3 = [], [], []  # These will hold the objective values (proof stress at different temperatures)
        g1 = []  # These will hold the constraint values (sum of elements)

        for xs in x:
            element_sum = np.sum(xs)  # Calculate the sum of the element percentages
            Fe_percentage = 100 - element_sum  # Calculate the Fe percentage (since total sum should be 100)

            # Apply constraints: ensure the sum of elements is within the specified bounds
            g1.append(self.sum_lower_bound - element_sum)  # Constraint: sum(elements) >= 25
            g1.append(element_sum - self.sum_upper_bound)  # Constraint: sum(elements) <= 75

            # If the Fe percentage is less than or equal to 0, penalize the solution
            if Fe_percentage <= 0:
                f1.append(np.inf)  # Infeasible solution, assign a high penalty
                f2.append(np.inf)
                f3.append(np.inf)
            else:
                # Predict Proof Stress at 27°C by combining the Fe percentage, element composition, and temperature
                composition_with_Fe_and_temp = np.hstack((Fe_percentage, xs, 27))  # Append temperature 27°C
                scaled_composition = scaler.transform([composition_with_Fe_and_temp])[0]  # Scale the composition
                predicted_value_27 = model.predict([scaled_composition])[0]  # Predict the proof stress at 27°C
                f1.append(-predicted_value_27)  # Negate the result to convert maximization to minimization

                # Repeat the same for 400°C and 1000°C
                composition_with_Fe_and_temp = np.hstack((Fe_percentage, xs, 400))  # Append temperature 400°C
                scaled_composition = scaler.transform([composition_with_Fe_and_temp])[0]
                predicted_value_400 = model.predict([scaled_composition])[0]
                f2.append(-predicted_value_400)

                composition_with_Fe_and_temp = np.hstack((Fe_percentage, xs, 1000))  # Append temperature 1000°C
                scaled_composition = scaler.transform([composition_with_Fe_and_temp])[0]
                predicted_value_1000 = model.predict([scaled_composition])[0]
                f3.append(-predicted_value_1000)

        # Store the objective values (proof stress predictions) and constraint values
        out["F"] = np.column_stack([f1, f2, f3])  # Stack the objectives
        out["G"] = np.column_stack(g1)  # Stack the constraints

# Function to optimize proof stress at all temperatures
def optimize_proof_stress_all_temps():
    print("Maximizing Proof Stress at 27°C, 400°C, and 1000°C simultaneously")

    # Define which elements to exclude from optimization (Fe is excluded)
    elements_excluding_Fe = feature_columns[1:-1]  
    problem = SteelMultiObjectiveProofStress(elements_excluding_Fe)

    # Initialize the NSGA2 algorithm with a population size of 50
    algorithm = NSGA2(pop_size=50)

    # Define termination criteria (run for 50000 generations)
    termination = ('n_gen', 50000) 
    results = minimize(problem=problem, algorithm=algorithm, termination=termination, seed=1, verbose=True)

    # Display results if any solutions are found
    if results.X is not None and len(results.X) > 0:
        print(f"Optimized Compositions and Corresponding Proof Stress at 27°C, 400°C, and 1000°C:")

        compositions_list = []  # List to store optimized compositions and their corresponding proof stress values

        # Ensure results are in the correct shape (even if only one composition is returned)
        if len(results.X.shape) == 1:
            results.X = np.array([results.X])
            results.F = np.array([results.F])

        # Process each optimized composition and its predicted proof stress
        for i, (composition, value) in enumerate(zip(results.X, results.F)):
            element_sum = np.sum(composition)
            Fe_percentage = 100 - element_sum  # Calculate the Fe percentage
            composition_dict = {element: composition[j] for j, element in enumerate(elements_excluding_Fe)}
            composition_dict['Fe'] = Fe_percentage  # Add Fe percentage to the composition

            # Calculate the mean predicted proof stress for each composition
            mean_proof_stress = (-value[0] + -value[1] + -value[2]) / 3

            # Append the composition details to the list
            compositions_list.append({
                'Composition': composition_dict,
                'Predicted Proof Stress at 27°C': -value[0],
                'Predicted Proof Stress at 400°C': -value[1],
                'Predicted Proof Stress at 1000°C': -value[2],
                'Mean Predicted Proof Stress': mean_proof_stress
            })

        # Create a DataFrame from the list of compositions
        df_compositions = pd.DataFrame(compositions_list)

        # Sort the DataFrame by Mean Predicted Proof Stress in descending order
        df_sorted = df_compositions.sort_values(by='Mean Predicted Proof Stress', ascending=False).reset_index(drop=True)

        # Display the sorted results
        for i, row in df_sorted.iterrows():
            print(f"Composition {i + 1}:")
            for element, percentage in row['Composition'].items():
                print(f"  {element}: {percentage}%")
            print(f"  Predicted Proof Stress at 27°C: {row['Predicted Proof Stress at 27°C']} MPa")
            print(f"  Predicted Proof Stress at 400°C: {row['Predicted Proof Stress at 400°C']} MPa")
            print(f"  Predicted Proof Stress at 1000°C: {row['Predicted Proof Stress at 1000°C']} MPa")
            print(f"  Mean Predicted Proof Stress: {row['Mean Predicted Proof Stress']} MPa\n")
    else:
        print(f"No feasible solutions were found by the optimization process.\n")

# Run the multi-objective optimization for Proof Stress at 27°C, 400°C, and 1000°C
optimize_proof_stress_all_temps()
