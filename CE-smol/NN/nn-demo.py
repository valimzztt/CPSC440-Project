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

"""
    Scripts that demonstrate a basic CLUSTER EXPANSION using SMOL and using a NN
"""
import matplotlib.pyplot as plt
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

cutoffs = {2: 7, 3: 6, 4: 6}   # This is equal to CLEASE'S max_cluster_diameter =(7,6,6)  
subspace = ClusterSubspace.from_cutoffs(prim, cutoffs=cutoffs)


# 3. TRAINING SET: The json file contains the structures with their corresponding ground-state energies 
energies_file = os.path.join(directory,"TrainingSet/MnNiAs-initstruct-energy-new.json" )
entries = loadfn(energies_file)

# TRAINING SET: The Wrangler object will contain the training structures with their corresponding ground-state energies 
wrangler = StructureWrangler(subspace)
for entry in entries:
    wrangler.add_entry(entry)



# 4. MACHINE LEARNING STEP: All training examples are in the wrangler object
#   a) wrangler.feature_matrix contains the feature matrix of training examples: dim(wrangler.feature_matrix)=clusters we are using to describe energy 
#   b) wrangler.get_property_vector("energy") is a vector containing the target property = energy (in eV)
    
# We are using a Neural Network 
print("TensorFlow version:", tf.__version__)
X_train =  wrangler.feature_matrix
y =  wrangler.get_property_vector("energy")
input_shape = X_train.shape
input_dim = X_train.shape[1]
print(input_dim )
print(X_train.shape[0])
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
   tf.keras.layers.Dense(1) 
])

predictions = model(X_train[:1]).numpy()
print(predictions)
    
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
print(model.summary())
history = model.fit(X_train, y, epochs=50, batch_size=32, validation_split=0.2)

model.save('cluster_exp_model.h5')

from keras.models import load_model
loaded_model = load_model('cluster_exp_model.h5')

# Evaluate the model
y_pred = model.predict(X_test)
print(f'R^2 Score: {r2_score(y_test, y_pred)}')

# Save the best model
model.save(os.path.join(cwd, 'best_nn_model.h5'))
y_train_pred =model.predict(X_train)
y_test_pred = model.predict(X_test)


r2_scores = []
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
r2_scores.append((r2_train, r2_test))

plt.figure(figsize=(10, 8)) 
plt.scatter(y_test, y_test_pred, label='Predictions', s=100)  # Increase scatter point size with `s=100`)
plt.xlabel('DFT Energy (eV)', fontsize=24)
plt.ylabel('CE Predicted Energy (eV)', fontsize=24)
plt.plot(y_test, y_test, 'k--', label="Line of perfect agreement", color="red") # Line of perfect agreement
plt.title(f'Lasso',  fontsize=25)
plt.text(0.05, 0.9, f'Train $R^2"$: {r2_train:.3f}', transform=plt.gca().transAxes, fontsize=20)
plt.text(0.05, 0.85, f'Test $R^2"$: {r2_test:.3f}', transform=plt.gca().transAxes, fontsize=20)
plt.legend(loc='lower right')
plt.savefig(".././CPSC440-Project/figs/BestNN766.png")
