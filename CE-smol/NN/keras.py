import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from keras_tuner import RandomSearch, HyperParameters
from monty.serialization import loadfn
from smol.cofe import StructureWrangler, ClusterSubspace, ClusterExpansion, RegressionData
from pymatgen.core import Lattice, Structure
from ase.io import read

# Load the work containing the fitted Cluster Expansion
cwd = os.getcwd()
file_path = os.path.join(cwd, 'CE-smol/Regularization/fitted-ce/lasso766.mson')
work = loadfn(file_path)
wrangler = work["StructureWrangler"]
X = wrangler.feature_matrix
y = wrangler.get_property_vector(key="energy")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def build_model(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=hp.Int("units", min_value=32, max_value=512, step=32), activation="relu", input_shape=(X.shape[1],)))
    for i in range(hp.Int('layers', 1, 5)):
        model.add(tf.keras.layers.Dense(units=hp.Int(f'units_{i}', min_value=32, max_value=512, step=32), activation='relu'))
        model.add(tf.keras.layers.Dropout(rate=hp.Float('dropout_' + str(i), 0.0, 0.5, step=0.1)))
    model.add(tf.keras.layers.Dense(1))
    optimizer = tf.keras.optimizers.Adam(learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, sampling='log'))
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[tf.keras.metrics.MeanSquaredError()])
    return model

tuner = RandomSearch(
    hypermodel=build_model,
    objective="val_mean_squared_error",
    max_trials=10,
    executions_per_trial=2,
    overwrite=True,
    directory=os.path.join(cwd, "my_dir"),
    project_name="helloworld"
)

tuner.search(X_train, y_train, epochs=50, validation_split=0.2)
best_model = tuner.get_best_models(num_models=1)[0]

# Evaluate the model
y_pred = best_model.predict(X_test)
print(f'R^2 Score: {r2_score(y_test, y_pred)}')

# Save the best model
best_model.save(os.path.join(cwd, 'best_nn_model.h5'))
