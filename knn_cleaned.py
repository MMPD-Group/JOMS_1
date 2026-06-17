import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

# ==========================================
# CONFIGURATION
# ==========================================
dataset_path = r"C:\Users\ateya\OneDrive - Indian Institute of Science\Optimization.xlsx"

feature_columns = ['Fe', 'C', 'Si', 'Mn', 'P', 'S', 'Ni', 'Cr', 'Mo', 'Cu', 'Al', 'N', 'V',
                   'Nb', 'Ta', 'W', 'Ti', 'B', 'Sn', 'Zr', 'Co', 'O', 'Temperature']
k = 5

# Each entry: (sheet, temperature, opt_target_col, train_target_col)
sheet_config = [
    # --- SOO 1.x: Proof Stress ---
    ("SOO 1.1", 27,   "Predicted Proof Stress (MPa)",  "0.2% Proof Stress"),
    ("SOO 1.2", 400,  "Predicted Proof Stress (MPa)",  "0.2% Proof Stress"),
    ("SOO 1.3", 1000, "Predicted Proof Stress (MPa)",  "0.2% Proof Stress"),
    # --- SOO 2.x: UTS ---
    ("SOO 2.1", 27,   "Predicted UTS",                 "Tensile Strength"),
    ("SOO 2.2", 400,  "Predicted UTS",                 "Tensile Strength"),
    ("SOO 2.3", 1000, "Predicted UTS (MPa)",            "Tensile Strength"),
    # --- SOO 3.x: Elongation ---
    ("SOO 3.1", 27,   "Elongation %",                  "Elongation %"),
    ("SOO 3.2", 400,  "Predicted % Elongation",         "Elongation %"),
    ("SOO 3.3", 1000, "Predicted % Elongation",         "Elongation %"),
    # --- MOO 5.x: all 3 targets, one temp each ---
    ("MOO 5.1", 27,   "Maximized Proof Stress (MPa)",  "0.2% Proof Stress"),
    ("MOO 5.1", 27,   "Maximized UTS (MPa)",           "Tensile Strength"),
    ("MOO 5.1", 27,   "Maximized Elongation (%)",      "Elongation %"),
    ("MOO 5.2", 400,  "Maximized Proof Stress (MPa)",  "0.2% Proof Stress"),
    ("MOO 5.2", 400,  "Maximized UTS (MPa)",           "Tensile Strength"),
    ("MOO 5.2", 400,  "Maximized Elongation (%)",      "Elongation %"),
    ("MOO 5.3", 1000, "Maximized Proof Stress (MPa)",  "0.2% Proof Stress"),
    ("MOO 5.3", 1000, "Maximized UTS (MPa)",           "Tensile Strength"),
    ("MOO 5.3", 1000, "Maximized Elongation (%)",      "Elongation %"),
]

# ==========================================
# LOAD TRAINING DATA
# ==========================================
train_df = pd.read_excel(dataset_path, sheet_name="new data")
train_df[feature_columns] = train_df[feature_columns].fillna(0)

# Cache opt sheets so we don't reload the same sheet multiple times
opt_cache = {}

# ==========================================
# RUN kNN FOR EACH CONFIG ENTRY
# ==========================================
def run_knn(sheet_name, temperature, opt_target_col, train_target_col):
    if sheet_name not in opt_cache:
        df = pd.read_excel(dataset_path, sheet_name=sheet_name)
        df = df.dropna(how='all')
        df[feature_columns[:-1]] = df[feature_columns[:-1]].fillna(0)
        opt_cache[sheet_name] = df

    opt_df = opt_cache[sheet_name].copy()
    opt_df['Temperature'] = temperature

    train_sub = train_df[train_df['Temperature'] == temperature].copy()
    train_sub = train_sub.dropna(subset=[train_target_col])

    y_opt = opt_df[[opt_target_col]].dropna()
    X_opt = opt_df[feature_columns].loc[y_opt.index]
    X_train = train_sub[feature_columns]
    y_train = train_sub[[train_target_col]]

    # --- NEW: Find the Champion alloy using idxmax ---
    # This grabs the index of the row with the maximum value in the target column
    champion_idx = y_opt[opt_target_col].idxmax()
    
    # Extract the features and predicted value for just this one champion alloy
    X_opt_champion = X_opt.loc[[champion_idx]]
    y_opt_pred_champion = y_opt.loc[champion_idx, opt_target_col]

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_opt_scaled_champion = scaler.transform(X_opt_champion)

    knn = NearestNeighbors(n_neighbors=min(k, len(X_train)), metric='euclidean')
    knn.fit(X_train_scaled)
    
    # We only pass the single champion composition
    distances, indices = knn.kneighbors(X_opt_scaled_champion)

    # Force float to ensure .mean() and .max() work properly
    neighbor_props = y_train.iloc[indices[0]].astype(float)

    results = [{
        "Sheet":         sheet_name,
        "Temperature":   temperature,
        "Property":      train_target_col,
        "Optimized":     round(y_opt_pred_champion, 2),
        "Neighbor_Mean": round(neighbor_props.mean().values[0], 2),
        "Neighbor_Max":  round(neighbor_props.max().values[0], 2),
    }]
    return results

all_results = []
for (sheet, temp, opt_col, train_col) in sheet_config:
    print(f"Processing {sheet} | {temp}°C | {train_col}...")
    try:
        rows = run_knn(sheet, temp, opt_col, train_col)
        all_results.extend(rows)
    except Exception as e:
        print(f"  ERROR: {e}")

final_df = pd.DataFrame(all_results)
print("\n", final_df.to_string())

# Save directly to a clean Excel file
final_df.to_excel("kNN_Champion_Table.xlsx", index=False)
print("\nSaved to kNN_Champion_Table.xlsx")