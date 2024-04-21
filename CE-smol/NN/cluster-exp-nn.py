import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from monty.serialization import loadfn
from smol.cofe import StructureWrangler
from monty.serialization import loadfn
from smol.cofe import ClusterSubspace, StructureWrangler, ClusterExpansion, RegressionData
import os 
from pymatgen.core import Lattice, Structure
from monty.serialization import loadfn
from ase.io import read
from smol.cofe import ClusterSubspace

"""
    Scripts that demonstrate a basic CLUSTER EXPANSION using SMOL and using a NN
"""

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
# The keys are the number of atoms in a cluster (pair, triplet, quadruplet) and the values are the maximum diameters of the given n-body cluster (in Angstrom = 10^-10 m)

cutoffs = {2: 7, 3: 6, 4: 0}   # This is equal to CLEASE'S max_cluster_diameter =(7,6,6)  
subspace = ClusterSubspace.from_cutoffs(prim, cutoffs=cutoffs)


# 3. TRAINING SET: The json file contains the structures with their corresponding ground-state energies 
energies_file = os.path.join(directory,"comp-struct-energy.json" )
entries = loadfn(energies_file)

# TRAINING SET: The Wrangler object will contain the training structures with their corresponding ground-state energies 
wrangler = StructureWrangler(subspace)
for entry in entries:
    wrangler.add_entry(entry)



# 4. MACHINE LEARNING STEP: All training examples are in the wrangler object
#   a) wrangler.feature_matrix contains the feature matrix of training examples: dim(wrangler.feature_matrix)=clusters we are using to describe energy 
#   b) wrangler.get_property_vector("energy") is a vector containing the target property = energy (in eV)
    
input_shape = wrangler.feature_matrix.shape[0]

# Neural network architecture
model = Sequential([
    Dense(128, activation='relu', input_shape=(input_shape,)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')


from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
X =  wrangler.feature_matrix
y =  wrangler.get_property_vector("energy")

def create_model():
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Wrap the model so it can be used by scikit-learn
keras_model = KerasRegressor(build_fn=create_model, epochs=100, batch_size=10, verbose=0)

# Define the grid search parameters
param_grid = {'batch_size': [10, 20, 40],
              'epochs': [50, 100, 150]}

grid = GridSearchCV(estimator=keras_model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, y)  # X and y are your datasets

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
