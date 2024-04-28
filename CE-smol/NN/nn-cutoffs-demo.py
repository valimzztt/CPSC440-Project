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
    2: np.arange(6, 8),  # Pair cutoffs from 3 to 8 Angstroms
    3: np.arange(6, 8),  # Triplet cutoffs from 3 to 8 Angstroms
    4: np.arange(6, 8)   # Quadruplet cutoffs, assumed to be either 0 (disabled) or other value (if applicable)
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
test_losses = []
val_losses = []
cutoff_labels = []

# Generate data for each cutoff combination
for cutoffs in cutoff_combinations:
    X, y = generate_data_for_cutoffs(cutoffs)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define and compile the neural network model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        # tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)  # Output layer for regression
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

    # Evaluate the model on the test set
    test_loss, test_mae = model.evaluate(X_test, y_test)
    print(f"Cutoffs: {cutoffs}, Test Loss: {test_loss}, Test MAE: {test_mae}")

    test_losses.append(test_loss)
    val_losses.append(min(history.history['val_loss']))  # Store minimum validation loss during training
    cutoff_labels.append(f"2:{cutoffs[2]}, 3:{cutoffs[3]}, 4:{cutoffs[4]}")

# Plot test and validation losses
plt.figure(figsize=(12, 6))
plt.plot(cutoff_labels, test_losses, label='Test Loss')
plt.plot(cutoff_labels, val_losses, label='Min Validation Loss')
plt.title('Test and Validation Loss for Different Cutoff Combinations')
plt.xlabel('Cutoff Combinations')
plt.ylabel('Loss')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
