import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import keras
from keras import layers

import keras_tuner
from monty.serialization import loadfn

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
    for i in range(hp.Int('layers', 10, 20)):
        model.add(tf.keras.layers.Dense(
                # Tune number of units separately.
                units=hp.Int(f"units_{i}", min_value=32, max_value=512, step=32),
                activation=hp.Choice("activation", ["relu", "tanh"]),
            ))
    if hp.Boolean("dropout"):
        model.add(tf.keras.layers.Dropout(rate=0.25))
    model.add(tf.keras.layers.Dense(1))
    optimizer = tf.keras.optimizers.Adam(learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, sampling='log'))
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[tf.keras.metrics.MeanSquaredError()])
    return model

directory = os.path.join(cwd, r"CE-smol\NN")
tuner = keras_tuner.RandomSearch(
    hypermodel=build_model,
    objective="val_mean_squared_error",
    max_trials=10,
    executions_per_trial=2,
    overwrite=True,
    directory=os.path.join(directory, "NN_search"),
    project_name="helloworld"
)

tuner.search(X_train, y_train, epochs=50, validation_split=0.2)
best_model = tuner.get_best_models(num_models=1)[0]

# Evaluate the model
y_pred = best_model.predict(X_test)
print(f'R^2 Score: {r2_score(y_test, y_pred)}')

# Save the best model
best_model.save(os.path.join(cwd, 'best_nn_model.h5'))
