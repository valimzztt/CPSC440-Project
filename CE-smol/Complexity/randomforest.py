import numpy as np
from monty.serialization import loadfn
from smol.cofe import StructureWrangler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
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
from sklearn.metrics import r2_score

cwd = os.getcwd()
directory = os.path.join(cwd, "CE-smol")


# 1. STRUCTURE: first create the primitive lattice: use fractional occupancies to have a disordered primitive cell
species = [{'Mn': 0.60, 'Ni': 0.4},{'As':1.0}]
my_lattice = Lattice.from_parameters(3.64580405, 3.64580405, 5.04506600, 90, 90, 120)
prim = Structure.from_spacegroup(194, my_lattice,  species, coords=[[0, 0, 0],[0.33333333, 0.66666667, 0.25]], site_properties={"oxidation": [0, 0]})
supercell = prim *(8,8,8)
# 1. STRUCTURE: We read the MnNiAs supercell from a cif file to get the structure of interest 
atoms = read("MnNiAs.cif", format="cif")


# 2. CLUSTER SUBSPACE: Specify cluster subspace information: create a cluster subspace including pair, triplet and quadruplet clusters up to given cluster diameter cutoffs.
# They keys are the number of atoms in a cluster (pair, triplet, quadruplet) and the values are the maximum diameters of the given n-body cluster (in Angstrom = 10^-10 m)

cutoffs = {2: 7, 3: 5, 4: 0}   
subspace = ClusterSubspace.from_cutoffs(prim, cutoffs=cutoffs)

# 3. TRAINING SET: The json file contains the structures with their corresponding ground-state energies 
energies_file = os.path.join(directory,"TrainingSet\MnNiAs-initstruct-energy-new.json" )
entries = loadfn(energies_file)
TEST_SIZE = 0.20
PROPERTY = 'energy'

# TRAINING SET: The Wrangler object will contain the training structures with their corresponding ground-state energies 
wrangler = StructureWrangler(subspace)
for entry in entries:
    wrangler.add_entry(entry)


# Split the data into training and test sets
X = wrangler.feature_matrix
y = wrangler.get_property_vector(key="energy")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate RMSE for train and test sets
train_rmse = np.sqrt(mse(y_train, y_train_pred))
test_rmse = np.sqrt(mse(y_test, y_test_pred))

print(f'In-sample RMSE: {train_rmse} eV/prim')
print(f'Out-of-sample RMSE: {test_rmse} eV/prim')


r2_scores = []
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
r2_scores.append((r2_train, r2_test))



plt.figure(figsize=(10, 8)) 
plt.scatter(y_test, y_test_pred, label='Predictions', s=100)  # Increase scatter point size with `s=100`)
plt.xlabel('DFT Energy (eV)', fontsize=24)
plt.ylabel('CE Predicted Energy (eV)', fontsize=24)
plt.plot(y_test, y_test, 'k--', label="Line of perfect agreement", color="red") # Line of perfect agreement
plt.title(f'OLS',  fontsize=25)
plt.text(0.05, 0.9, f'Train $R^2"$: {r2_train:.3f}', transform=plt.gca().transAxes, fontsize=20)
plt.text(0.05, 0.85, f'Test $R^2"$: {r2_test:.3f}', transform=plt.gca().transAxes, fontsize=20)
plt.legend(loc='lower right')
plt.savefig(".././CPSC440-Project/figs/RandomForest750.png")

# 4.2 Generate the Cluster Expansion Object: 
from smol.cofe import RegressionData, ClusterExpansion
from random import choice

reg_data = RegressionData.from_sklearn(
    model, wrangler.feature_matrix, wrangler.get_property_vector(key=PROPERTY)
)
wvec = np.concatenate(
    (np.array([model.intercept_]), model.coef_),
    axis=0
)

expansion = ClusterExpansion(
    subspace, coefficients=wvec, regression_data=reg_data
)


structure = choice(wrangler.structures)
prediction = expansion.predict(structure, normalized=True)
print(f"Structure with composition {structure.composition} has predicted energy {prediction} eV/prim")

print(expansion)
# We have built the cluster expansion: Now let’s run canonical MC on Mn0.6Ni0.4As 

from smol.io import save_work

file_path = os.path.join(cwd, 'CE-smol/Regularization/fitted-ce/RandomForest.mson')
# we can save the subspace as well, but since both the wrangler
# and the expansion have it, there is no need to do so.
save_work(file_path, wrangler, expansion)