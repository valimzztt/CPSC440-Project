import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from monty.serialization import loadfn
from smol.cofe import StructureWrangler
from monty.serialization import loadfn
import warnings
from smol.cofe import ClusterSubspace, StructureWrangler, ClusterExpansion, RegressionData
import os 
from pymatgen.core import Lattice, Structure
from monty.serialization import loadfn
from ase.io import read
from smol.cofe import ClusterSubspace
from sklearn.exceptions import ConvergenceWarning
"""
    Scripts that demonstrates CLUSTER EXPANSION using SMOL and using a NN where we are testing NN with different layers and (hyper-parameter):  the goal is to get a 
    R^2 value that is between 0 and 1 (ideally 1)
"""

import wandb

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

cutoffs = {2: 7, 3: 7, 4: 7}   # This is equal to CLEASE'S max_cluster_diameter =(7,6,6)  
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

# Weight & Biases setup 
os.environ["WANDB_SILENT"] = "true"
TRIALS = 10
TEST_SIZE = 0.20
PROPERTY = 'energy'

# (initialize wandb project and add hyperparameters)
# NOTE: wandb API key: 966b532ea49a07fa69b9a1e34f47bc02865ea9ff

wandb.login()
wandb.init(project='cpsc440 ML for cluster expansion', entity="cpsc440-ml-cluster-expansion", config = {
    "species" : species,
    "cluster_info" : cutoffs,
    "property" : PROPERTY,
    "model" : "Lasso"})


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    X =  wrangler.feature_matrix
    y =  wrangler.get_property_vector("energy")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    r2_scores = []

    # Range of layers to train separate models
    layer_configs = [10, 20, 30]

    for layers in layer_configs:
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(X.shape[1],)))

        # Adjusting the number of units and adding dropout and batch normalization
        for _ in range(layers):
            model.add(tf.keras.layers.Dense(units=64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
            #model.add(tf.keras.layers.BatchNormalization())
            # model.add(tf.keras.layers.Dropout(0.5))

        model.add(tf.keras.layers.Dense(units=1, activation='linear'))

        # Let's try a learning rate scheduler
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=10000,
            decay_rate=0.9)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        #optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)       
        model.compile(optimizer=optimizer, loss='mean_squared_error')

        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        # Time to fit! 
        model.fit(
            X_train, y_train, 
            epochs=300, 
            batch_size=32, 
            verbose=0,
            validation_split=0.2,
            # callbacks=[early_stopping]
        )
        
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        

        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)
        r2_scores.append((r2_train, r2_test))

        plt.figure()
        plt.scatter(y_test, y_test_pred, label=f'{layers} Layers')
        plt.xlabel('DFT Energy (eV)')
        plt.ylabel('CE Predicted Energy (eV)')
        plt.plot(y_test, y_test, 'k--')  # Line of perfect agreement
        plt.title(f'{layers} Layers Neural Network')
        plt.legend()
        plt.show()
wandb.finish()


for i, scores in enumerate(r2_scores):
    print(f"Model with {layer_configs[i]} layers - R^2 Train: {scores[0]}, R^2 Test: {scores[1]}")
