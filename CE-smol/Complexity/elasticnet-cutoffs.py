import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import ElasticNet
from smol.cofe import ClusterSubspace, StructureWrangler
from itertools import product
from monty.serialization import loadfn
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from monty.serialization import loadfn
from smol.cofe import StructureWrangler
from monty.serialization import loadfn
from smol.cofe import ClusterSubspace, StructureWrangler, ClusterExpansion, RegressionData
import os 
from pymatgen.core import Lattice, Structure
from monty.serialization import loadfn
from ase.io import read
from smol.cofe import ClusterSubspace
from sklearn.linear_model import ElasticNet
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import cross_val_score

cwd = os.getcwd()
directory = os.path.join(cwd, "CE-smol")

# 1. STRUCTURE: first create the primitive lattice: use fractional occupancies to have a disordered primitive cell
species = [{'Mn': 0.60, 'Ni': 0.4},{'As':1.0}]
my_lattice = Lattice.from_parameters(3.64580405, 3.64580405, 5.04506600, 90, 90, 120)
prim = Structure.from_spacegroup(194, my_lattice,  species, coords=[[0, 0, 0],[0.33333333, 0.66666667, 0.25]], site_properties={"oxidation": [0, 0]})
supercell = prim *(8,8,8)
# 1. STRUCTURE: We read the MnNiAs supercell from a cif file to get the structure of interest 
atoms = read("MnNiAs.cif", format="cif")



# Define the range of cutoffs for pairs (2-body), triplets (3-body), and quadruplets (4-body)
cutoff_ranges = {
    2: np.arange(7, 8),  # Pair cutoffs of 7 
    3: np.arange(6, 8),  # Triplet cutoffs from 3 to 8 Angstroms
    4: np.arange(0, 8)   # Quadruplet cutoffs, assumed to be either 0 (disabled) or other value (if applicable)
}

print(cutoff_ranges)

# Generate all possible combinations of cutoffs
from itertools import product
cutoff_combinations = list(product(cutoff_ranges[2], cutoff_ranges[3], cutoff_ranges[4]))
energies_file = os.path.join(directory,"TrainingSet/MnNiAs-initstruct-energy-new.json" )
entries = loadfn(energies_file)

cutoff_keys = sorted(cutoff_ranges.keys())  # Sorted keys: [2, 3, 4]
cutoff_combinations = [
    dict(zip(cutoff_keys, combination))
    for combination in product(*(cutoff_ranges[key] for key in cutoff_keys))
]

def generate_data_for_cutoffs(cutoffs):
    subspace = ClusterSubspace.from_cutoffs(prim, cutoffs=cutoffs)
    # TRAINING SET: The Wrangler object will contain the training structures with their corresponding ground-state energies 
    wrangler = StructureWrangler(subspace)
    for entry in entries:
        wrangler.add_entry(entry)
    X_train =  wrangler.feature_matrix
    y =  wrangler.get_property_vector("energy")
    return X_train, y

rmse_values = []
cv_scores = []
labels = []

# Define your scorer for cross-validation
cv_scorer = make_scorer(mean_squared_error, greater_is_better=False)


def optimize_cutoffs(entries, cutoff_ranges, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    best_rmse = float('inf')
    best_cutoffs = None
    results = []

    for cutoffs in product(*[cutoff_ranges[key] for key in sorted(cutoff_ranges.keys())]):
        cutoff_dict = dict(zip(sorted(cutoff_ranges.keys()), cutoffs))
        X, y = generate_data_for_cutoffs(cutoff_dict)
        rmse_list = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model = ElasticNet(alpha=10**(-3.667), l1_ratio=0.4, fit_intercept=True)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mse(y_test, y_pred))
            rmse_list.append(rmse)

        mean_rmse = np.mean(rmse_list)
        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_cutoffs = cutoff_dict

        results.append((cutoff_dict, mean_rmse))

    return best_cutoffs, best_rmse, results


cutoff_ranges = {
    2: np.arange(7, 8),
    3: np.arange(6, 8),
    4: np.arange(0, 8)
}
best_cutoffs, best_rmse, all_results = optimize_cutoffs(entries, cutoff_ranges)
print("Best Cutoffs:", best_cutoffs)
print("Best RMSE:", best_rmse)
