import tensorflow as tf
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
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
"""
    Scripts that demonstrate a basic CLUSTER EXPANSION using SMOL and using a Convolutional Neural Network
"""

from smol.io import load_work
import os 
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape[0])
# Define input shape based on your reshaped data
sequence_length = X_train.shape[0]
num_features = X_train.shape[1]
input_shape = (num_features, 1) 

model = tf.keras.models.Sequential([
    # Add a 1D Convolutional layer
    tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1) 
])

# Assuming X_train has been reshaped appropriately for the CNN input
predictions = model(X_train[:1]).numpy()
print(predictions)

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
print(model.summary())

# Fit the model - ensure your y is correctly shaped as well, typically (num_samples, )
history = model.fit(X_train, y, epochs=50, batch_size=32, validation_split=0.2)

# Save the model
model.save('cluster_exp_cnn_model.h5')
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
print(r2_test )
plt.figure()
plt.scatter(y_test, y_test_pred, label=f'{layers} Layers')
plt.xlabel('DFT Energy (eV)')
plt.ylabel('CE Predicted Energy (eV)')
plt.plot(y_test, y_test, 'k--')  # Line of perfect agreement
plt.title(f'{layers} Layers Neural Network')
plt.legend()
plt.show()



