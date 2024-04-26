
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
    2: np.arange(5, 9),  # Pair cutoffs of 7 
    3: np.arange(5, 9),  # Triplet cutoffs from 3 to 8 Angstroms
    4: np.arange(0, 9)   # Quadruplet cutoffs
}



# Generate all possible combinations of cutoffs
from itertools import product
cutoff_combinations = list(product(cutoff_ranges[2], cutoff_ranges[3], cutoff_ranges[4]))
energies_file = os.path.join(directory,"TrainingSet/MnNiAs-initstruct-energy-new.json" )
entries = loadfn(energies_file)

cutoff_keys = sorted(cutoff_ranges.keys())  
cutoff_combinations = [
    dict(zip(cutoff_keys, combination))
    for combination in product(*(cutoff_ranges[key] for key in cutoff_keys))
]
cutoff_ranges = {
    2: sorted(set(d[2] for d in cutoff_combinations)),
    3: sorted(set(d[3] for d in cutoff_combinations)),
    4: sorted(set(d[4] for d in cutoff_combinations))
}

# Create matrices to store the scores
cv_score_matrix = np.zeros((len(cutoff_ranges[2]), len(cutoff_ranges[3]), len(cutoff_ranges[4])))
rmse_score_matrix = np.zeros_like(cv_score_matrix)

# Function to convert cutoff values to matrix indices
def get_indices(cutoff_dict):
    return (cutoff_ranges[2].index(cutoff_dict[2]),
            cutoff_ranges[3].index(cutoff_dict[3]),
            cutoff_ranges[4].index(cutoff_dict[4]))

# Load the work containing the fitted Cluster Expansion
cwd = os.getcwd()
file_path = os.path.join(cwd, 'CE-smol/Regularization/fitted-ce/lasso766.mson')
work = loadfn(file_path)
wrangler = work["StructureWrangler"]
X = wrangler.feature_matrix
y = wrangler.get_property_vector(key="energy")


def generate_data_for_cutoffs(cutoffs):
    subspace = ClusterSubspace.from_cutoffs(prim, cutoffs=cutoffs)
    # TRAINING SET: The Wrangler object will contain the training structures with their corresponding ground-state energies 
    wrangler = StructureWrangler(subspace)
    for entry in entries:
        wrangler.add_entry(entry)
    X_train =  wrangler.feature_matrix
    y =  wrangler.get_property_vector("energy")
    print("The dimension of the feature matrix is equal to")
    print(cutoffs)
    print(X_train.shape)
    return X_train, y

rmse_values = []
cv_scores = []
labels = []
LAMBDA = 10**(-3.667)
L1_RATIO =  0.4
cv_scorer = make_scorer(mean_squared_error, greater_is_better=False)

num_features_list = []
rmse_list = []
loocv_error_list = []
loo = Leave
# Calculate the scores and store results
for cutoff_dict in cutoff_combinations:
    X, y = generate_data_for_cutoffs(cutoff_dict)
    num_features = X.shape[1]
    model = ElasticNet(alpha=LAMBDA, l1_ratio=L1_RATIO, fit_intercept=True)

    # Fit the model and calculate RMSE for the test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mse(y_test, y_pred))

    # Perform LOOCV and calculate the error
    neg_mse_scores = cross_val_score(model, X, y, cv=loo, scoring='neg_mean_squared_error')
    loocv_error = -neg_mse_scores.mean()  # Convert to positive MSE score
    
    # Record the results
    num_features_list.append(num_features)
    rmse_list.append(rmse)
    loocv_error_list.append(loocv_error)

# Plot RMSE as a function of the number of features
plt.figure(figsize=(10, 6))
plt.plot(num_features_list, rmse_list, label='RMSE', marker='o')
plt.plot(num_features_list, loocv_error_list, label='LOOCV Error', marker='s')
plt.xlabel('Number of Features', fontsize=20)
plt.ylabel('Error',fontsize=20)
plt.title('Error as a Function of the Number of Features',fontsize=20)

plt.legend()
plt.savefig(".././CPSC440-Project/figs/CountFeatures.png")
plt.show()
