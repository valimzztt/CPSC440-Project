
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
LAMBDA = 10**(-3.667)
L1_RATIO =  0.4
cv_scorer = make_scorer(mean_squared_error, greater_is_better=False)

# Calculate the scores and populate the matrices
for cutoff_dict in cutoff_combinations:
    X, y = generate_data_for_cutoffs(cutoff_dict)
    print(X.shape[1])
    model = ElasticNet(alpha=LAMBDA, l1_ratio=L1_RATIO, fit_intercept=True)
    
    # Perform 5-fold cross-validation
    neg_mse_scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5)
    cv_scores = -neg_mse_scores  # Convert to positive MSE scores
    mean_cv_score = cv_scores.mean()
    
    # Fit the model and calculate RMSE for the test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse_score = np.sqrt(mse(y_test, y_pred))
    
    # Get matrix indices for the current cutoffs
    indices = get_indices(cutoff_dict)
    
    # Store the scores in the matrices
    cv_score_matrix[indices] = mean_cv_score
    rmse_score_matrix[indices] = rmse_score

cutoff_2_body, cutoff_3_body, cutoff_4_body = np.meshgrid(cutoff_ranges[2], cutoff_ranges[3], cutoff_ranges[4], indexing='ij')

# Flatten the meshgrid matrices and the score matrices for plotting
cutoff_2_body_flat = cutoff_2_body.flatten()
cutoff_3_body_flat = cutoff_3_body.flatten()
cutoff_4_body_flat = cutoff_4_body.flatten()
cv_score_flat = cv_score_matrix.flatten()
rmse_score_flat = rmse_score_matrix.flatten()

# Plot RMSE scores in a separate figure
fig_rmse = plt.figure(figsize=(8, 6))
ax_rmse = fig_rmse.add_subplot(111, projection='3d')
sc_rmse = ax_rmse.scatter(cutoff_2_body_flat, cutoff_3_body_flat, cutoff_4_body_flat, c=rmse_score_flat, cmap='plasma')
plt.colorbar(sc_rmse, label='RMSE (eV/atom)', shrink=0.5, aspect=10)  # Adjust size and aspect of colorbar
ax_rmse.set_xlabel('2-body cluster cutoff (Å)', fontsize = 15)
ax_rmse.set_ylabel('3-body cluster cutoff (Å)',fontsize = 15)
ax_rmse.set_zlabel('4-body cluster cutoff (Å)',fontsize = 15)
ax_rmse.set_title('RMSE as a Function of Cluster Diameter Cutoff (Å)',fontsize = 25)
fig_rmse.savefig(".././CPSC440-Project/figs/RMSE_DiameterCutoff.png")

# Plot CV scores in a separate figure
fig_cv = plt.figure(figsize=(8, 6))
ax_cv = fig_cv.add_subplot(111, projection='3d')
sc_cv = ax_cv.scatter(cutoff_2_body_flat, cutoff_3_body_flat, cutoff_4_body_flat, c=cv_score_flat, cmap='plasma')
plt.colorbar(sc_cv, label='CV Score (eV/atom)', shrink=0.5, aspect=10)  # Adjust size and aspect of colorbar
ax_cv.set_xlabel('2-body cluster cutoff (Å)', fontsize = 15)
ax_cv.set_ylabel('3-body cluster cutoff (Å)', fontsize = 15)
ax_cv.set_zlabel('4-body cluster cutoff (Å)', fontsize = 15)
ax_cv.set_title('CV Score as a Function of Cluster Diameter Cutoff (Å)', fontsize = 25)
fig_cv.savefig(".././CPSC440-Project/figs/CVScore_DiameterCutoff.png")

plt.show()
