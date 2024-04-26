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
from sklearn.model_selection import train_test_split


"""
    Scripts that demonstrate a basic CLUSTER EXPANSION using SMOL and using a Convolutional Neural Network
"""

from smol.io import load_work
import os 
import keras_tuner
import keras
from keras import layers

"""
    Scripts that demonstrate how to load a fitted Cluster Expansion saved to file or to load the dataset faster
"""

cwd = os.getcwd()
file_path = os.path.join(cwd, 'CE-smol/Regularization/fitted-ce/lasso766.mson')
work = load_work(file_path)
PROPERTY = "energy"
for name, obj in work.items():
    print(f'{name}: {type(obj)}\n')

wrangler = work["StructureWrangler"]
X  = wrangler.feature_matrix
y = wrangler.get_property_vector(key=PROPERTY)

subspace = wrangler.cluster_subspace
print(subspace)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Flatten())
    model.add(
        layers.Dense(
            # Define the hyperparameter.
            units=hp.Int("units", min_value=32, max_value=512, step=32),
            activation="relu",
        )
    )
    model.add(layers.Dense(10, activation="softmax"))
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

tuner = keras_tuner.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=5)

tuner.search(X, y, epochs=5, validation_data=(X_test, y_test))
print(tuner.search_space_summary())
best_model = tuner.get_best_models()[0]