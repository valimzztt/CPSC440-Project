import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import keras
from keras import layers
from sklearn.metrics import r2_score
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
    # Tune the initial number of units in the first Dense layer
    model.add(tf.keras.layers.Dense(
        units=hp.Int("units", min_value=32, max_value=512, step=32),
        activation=hp.Choice("activation", ["relu", "tanh"]),
        kernel_regularizer=tf.keras.regularizers.l2(hp.Float('l2', 1e-5, 1e-2, sampling='log')),
        input_shape=(X.shape[1],)
    ))

    for i in range(hp.Int('layers', 10, 20)):
        model.add(tf.keras.layers.Dense(
            units=hp.Int(f"units_{i}", min_value=32, max_value=512, step=32),
            activation=hp.Choice(f"activation_{i}", ["relu", "tanh"]),
            kernel_regularizer=tf.keras.regularizers.l2(hp.Float(f'l2_{i}', 1e-5, 1e-2, sampling='log'))
        ))
        if hp.Boolean(f"dropout_{i}"):
            model.add(tf.keras.layers.Dropout(rate=hp.Float(f'dropout_rate_{i}', 0.0, 0.5, step=0.1)))

    model.add(tf.keras.layers.Dense(1, activation=hp.Choice('final_activation', ['relu', 'linear'])))

    optimizer_name = hp.Choice('optimizer', ['adam', 'sgd'])
    lr = hp.Float('lr', 1e-4, 1e-2, sampling='log')
    if optimizer_name == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=hp.Float('momentum', 0.5, 0.9))

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
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)


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
